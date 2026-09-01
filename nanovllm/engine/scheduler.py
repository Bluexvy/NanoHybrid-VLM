from time import monotonic
from collections import deque
from dataclasses import dataclass

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.hybrid_state import (
    StateSlotAllocator,
)
from nanovllm.engine.prefix_cache import (
    PrefixStateCache,
    PrefixStateEntry,
)

@dataclass(slots=True)
class SchedulePlan:
    """
    Scheduler 一轮逻辑调度的执行计划。

    Engine 按以下顺序执行：
        1. Decode microbatch
        2. Prefill microbatch
    """

    decode_seqs: list[Sequence]
    prefill_seqs: list[Sequence]

    num_decode_tokens: int
    num_prefill_tokens: int
    
    preempted_seq_ids: list[int]

    @property
    def total_num_tokens(self) -> int:
        return (
            self.num_decode_tokens
            + self.num_prefill_tokens
        )

    @property
    def is_empty(self) -> bool:
        return (
            not self.decode_seqs
            and not self.prefill_seqs
        )

class Scheduler:

    def __init__(
        self,
        config: Config,
    ):
        self.scheduler_policy = (
            config.scheduler_policy
        )

        self.max_num_seqs = (
            config.max_num_seqs
        )

        self.max_num_batched_tokens = (
            config.max_num_batched_tokens
        )

        self.max_prefill_wait_ms = (
            config.max_prefill_wait_ms
        )

        self.eos = config.eos

        self.block_size = (
            config.kvcache_block_size
        )

        # 先判断模型中是否存在 GDN 层。
        layer_types = getattr(
            config.text_config,
            "layer_types",
            (),
        )

        self.requires_gdn_state = any(
            layer_type == "linear_attention"
            for layer_type in layer_types
        )

        # 普通 Qwen3：
        #     config.enable_prefix_cache=True
        #     允许传统 KV Prefix Cache lookup。
        #
        # Qwen3.5：
        #     config.enable_prefix_cache=False
        #     当前禁止 KV-only lookup。
        self.enable_kv_prefix_lookup = bool(
            config.enable_prefix_cache
        )

        # 普通 Qwen3 开启传统 Prefix Cache 时需要记录。
        #
        # Qwen3.5 为了开发联合 KV + GDN Prefix Cache，
        # 即使暂时禁止 KV-only lookup，也需要记录
        # 完整 block 的 token/hash 元数据。
        self.record_prefix_metadata = (
            self.enable_kv_prefix_lookup
            or self.requires_gdn_state
        )

        self.block_manager = BlockManager(
            num_blocks=config.num_kvcache_blocks,
            block_size=config.kvcache_block_size,
            enable_kv_prefix_lookup=(
                self.enable_kv_prefix_lookup
            ),
            record_prefix_metadata=(
                self.record_prefix_metadata
            ),
        )
        
        # PrefixStateCache 由 LLMEngine 创建后注入。
        #
        # Scheduler 本身不能创建它，因为创建
        # PrefixStateCache 还需要 ModelRunner 中的
        # HybridStateManager。
        self.prefix_state_cache: (
            PrefixStateCache | None
        ) = None

        # Prefix Cache 命中统计。
        self.num_prefix_hit_requests = 0
        self.num_prefix_hit_tokens = 0

        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

        # seq_id -> 本次进入 waiting 队列的时间。
        self.waiting_since: dict[
            int,
            float,
        ] = {}

        self.num_preemptions = 0
        self.num_recomputed_tokens = 0

        if self.requires_gdn_state:
            if config.num_state_slots <= 0:
                raise ValueError(
                    "ModelRunner must calculate a positive "
                    "num_state_slots before Scheduler "
                    "construction"
                )

            self.state_slot_allocator = (
                StateSlotAllocator(
                    config.num_state_slots
                )
            )

        else:
            self.state_slot_allocator = None

    def set_prefix_state_cache(
        self,
        cache: PrefixStateCache | None,
    ) -> None:
        """
        将 LLMEngine 创建的 Hybrid Prefix State
        Cache 注入 Scheduler。
        """

        if cache is not None:
            if not self.requires_gdn_state:
                raise RuntimeError(
                    "Hybrid Prefix State Cache can only "
                    "be attached to a model with GDN layers"
                )

            if (
                cache.block_manager
                is not self.block_manager
            ):
                raise RuntimeError(
                    "Scheduler and PrefixStateCache must "
                    "share the same BlockManager"
                )

        self.prefix_state_cache = cache

    def _lookup_prefix_for_new_request(
        self,
        seq: Sequence,
    ) -> PrefixStateEntry | None:
        """
        为尚未分配运行时资源的新请求执行一次
        Hybrid Prefix State Cache lookup。

        同一个 waiting 周期只扫描一次 Prompt。
        """

        cache = self.prefix_state_cache

        # Cache 未开启时走普通 Prefill。
        if cache is None:
            seq.prefix_lookup_completed = True
            return None

        # 首版不缓存图文前缀。
        #
        # 仅比较 image placeholder token IDs 无法区分
        # 两张内容不同、但视觉 token 数量相同的图片。
        if seq.is_multimodal:
            seq.prefix_lookup_completed = True
            return None

        # 第一次处理这个新请求时，扫描 Prompt 的
        # 完整 token blocks，寻找最长命中。
        if not seq.prefix_lookup_completed:
            entry = cache.lookup_longest(
                seq.prompt_token_ids
            )

            seq.prefix_lookup_completed = True

            if entry is None:
                seq.prefix_cache_key = None
                return None

            seq.prefix_cache_key = entry.key
            return entry

        # 请求之前已经 lookup 过但没有命中。
        if seq.prefix_cache_key is None:
            return None

        # 请求可能因为资源不足而在 waiting 中停留多轮。
        #
        # 不需要每轮重新计算整个 Prompt 的链式 Hash，
        # 直接使用保存的 PrefixKey 取 Entry。
        entry = cache.get_resident_entry(
            seq.prefix_cache_key
        )

        # 当前版本还没有 LRU，正常情况下不会进入这里。
        #
        # 但提前处理 Entry 被删除的情况，可以避免以后
        # 加入显存预算和 LRU 后出现悬空引用。
        if entry is None:
            seq.prefix_cache_key = None
            return None

        return entry

    def is_finished(self):
        return not self.waiting and not self.running

    def add(
        self,
        seq: Sequence,
    ) -> None:
        if seq.seq_id in self.waiting_since:
            raise RuntimeError(
                f"Sequence {seq.seq_id} is already "
                "tracked as waiting"
            )

        self.waiting.append(seq)

        self.waiting_since[seq.seq_id] = (
            monotonic()
        )
        
    def _mark_waiting(
        self,
        seq: Sequence,
    ) -> None:
        """
        记录 Sequence 本次开始连续等待的时间。
        """

        self.waiting_since[seq.seq_id] = (
            monotonic()
        )
        
    def _clear_waiting(
        self,
        seq: Sequence,
    ) -> None:
        """
        Sequence 离开 waiting 队列时删除计时信息。
        """

        if seq.seq_id not in self.waiting_since:
            raise RuntimeError(
                f"Sequence {seq.seq_id} has no "
                "waiting timestamp"
            )

        del self.waiting_since[seq.seq_id]
        
    def _waiting_time_ms(
        self,
        seq: Sequence,
        now: float | None = None,
    ) -> float:
        """
        返回 Sequence 本次连续等待的毫秒数。
        """

        waiting_since = self.waiting_since.get(
            seq.seq_id
        )

        if waiting_since is None:
            raise RuntimeError(
                f"Sequence {seq.seq_id} has no "
                "waiting timestamp"
            )

        if now is None:
            now = monotonic()

        return (
            now - waiting_since
        ) * 1000.0
        
    def _should_reserve_prefill(
        self,
        now: float | None = None,
    ) -> bool:
        """
        waiting 队首请求是否已经等待超时。
        """

        if not self.waiting:
            return False

        if now is None:
            now = monotonic()

        oldest_seq = self.waiting[0]

        return (
            self._waiting_time_ms(
                oldest_seq,
                now,
            )
            >= self.max_prefill_wait_ms
        )     
        
    def _select_preemption_victim(
        self,
        excluded_seq_ids: set[int] | None = None,
    ) -> Sequence | None:
        """
        从 running 中选择重算成本最低的请求。

        已进入本轮 Decode microbatch 的请求不能抢占。
        """

        if excluded_seq_ids is None:
            excluded_seq_ids = set()

        candidates = [
            seq
            for seq in self.running
            if seq.seq_id not in excluded_seq_ids
        ]

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda seq: (
                seq.num_cached_tokens,
                seq.seq_id,
            ),
        )        

    def schedule(self) -> SchedulePlan:
        """
        根据 scheduler_policy 构造一轮执行计划。

        decode_first：
            优先调度 Decode，再用剩余预算调度 Prefill。

        prefill_first：
            只要 waiting 中存在请求，本轮就暂停 Decode，
            使用完整预算调度 Prefill。
        """

        decode_seqs: list[Sequence] = []
        prefill_seqs: list[Sequence] = []

        num_decode_tokens = 0
        num_prefill_tokens = 0

        preempted_seq_ids: list[int] = []
        
        # 一轮调度只读取一次当前时间，
        # 保证本轮所有等待时间判断基于同一时刻。
        now = monotonic()

        # 当前是否进入“严格 Prefill-first”状态。
        #
        # 必须同时满足：
        # 1. 用户配置了 prefill_first；
        # 2. waiting 队列中确实有 Prefill 请求。
        strict_prefill_first = (
            self.scheduler_policy
            == "prefill_first"
            and bool(self.waiting)
        )

        # reserve_prefill 表示：
        # 本轮必须保证 Prefill 得到执行机会。
        #
        # Decode-first：
        #   只有最老 Prefill 等待超时才为它保留资源。
        #
        # Prefill-first：
        #   只要 waiting 不为空，就立即保留资源。
        reserve_prefill = (
            strict_prefill_first
            or self._should_reserve_prefill(
                now
            )
        )

        # Decode-first 下，如果 Prefill 等待超时，
        # 至少给 Prefill 留出一个 Sequence 位置。
        reserved_seq_slots = (
            1
            if reserve_prefill
            else 0
        )

        # Decode-first 下，如果 Prefill 等待超时，
        # 至少给 Prefill 留出一个 token 的计算预算。
        reserved_prefill_tokens = (
            1
            if reserve_prefill
            else 0
        )

        if strict_prefill_first:
            # 严格 Prefill-first：
            #
            # waiting 中还有请求时，本轮完全不调度 Decode。
            #
            # 后面的 Decode while 会因为两个 limit 都是0
            # 而直接跳过。
            decode_seq_limit = 0
            decode_token_limit = 0

        else:
            # Decode-first：
            #
            # 正常情况下 Decode 可以使用全部资源。
            #
            # 如果 Prefill 已经等待超时，则分别保留：
            # 1个 Sequence 位置
            # 1个 token 预算。
            decode_seq_limit = max(
                0,
                self.max_num_seqs
                - reserved_seq_slots,
            )

            decode_token_limit = max(
                0,
                self.max_num_batched_tokens
                - reserved_prefill_tokens,
            )

        # =====================================
        # 第一阶段：Decode-first
        # =====================================

        while (
            self.running
            and len(decode_seqs) < decode_seq_limit
            and num_decode_tokens < decode_token_limit
        ):
            seq = self.running.popleft()

            # Decode 的新 token 可能正好需要一个
            # 新的物理 KV block。
            #
            # 如果没有可用 block，则抢占其他请求，
            # 释放它占用的 KV blocks 和 GDN slot。
            while not self.block_manager.can_append(seq):
                victim = self._select_preemption_victim()

                if victim is None:
                    # 没有其他未调度请求可以释放资源，
                    # 只能抢占当前请求自己。
                    preempted_seq_ids.append(
                        seq.seq_id
                    )
                    self.preempt(seq)
                    seq = None
                    break

                # preempt() 假设调用者已经把 victim
                # 从 running 队列移除。
                preempted_seq_ids.append(
                    victim.seq_id
                )
                self.running.remove(victim)
                self.preempt(victim)

            if seq is None:
                continue

            # Decode 每条请求每轮只处理一个 token。
            seq.num_scheduled_tokens = 1
            seq.is_prefill = False

            # 如果当前 token 落在一个新逻辑 block
            # 的开头，则为它分配新的物理 KV block。
            self.block_manager.may_append(seq)

            decode_seqs.append(seq)
            num_decode_tokens += 1

        # 上面为了遍历 running，临时把被调度请求
        # 从 deque 中取了出来。
        #
        # 现在按原顺序放回。
        self.running.extendleft(
            reversed(decode_seqs)
        )
        # 本轮已经承诺执行 Decode 的请求不能再被抢占。
        scheduled_decode_ids = {
            seq.seq_id
            for seq in decode_seqs
}

        # =====================================
        # 第二阶段：计算剩余预算
        # =====================================

        remaining_token_budget = (
            self.max_num_batched_tokens
            - num_decode_tokens
        )

        remaining_seq_budget = (
            self.max_num_seqs
            - len(decode_seqs)
        )

        # active sequence 不只包括已经完成 Prefill
        # 的 running 请求。
        #
        # 已经执行过一个 Prefill chunk、但尚未完成
        # Prefill 的请求也持有 KV blocks 和 state slot。
        num_active_seqs = len(self.running)

        num_active_seqs += sum(
            1
            for seq in self.waiting
            if seq.block_table
        )

        # =====================================
        # 第三阶段：使用剩余预算执行 Prefill
        # =====================================

        while (
            self.waiting
            and len(prefill_seqs)
            < remaining_seq_budget
            and remaining_token_budget > 0
        ):
            # 当前保持 FIFO：
            # 只检查等待队列最前面的请求。
            seq = self.waiting[0]

            is_new_request = not seq.block_table

            # 只有新请求才可能在这里查到 Prefix Entry。
            #
            # 已经执行过 Prefill chunk 的请求已经拥有
            # block_table 和 state slot，不应再次 lookup。
            prefix_entry: (
                PrefixStateEntry | None
            ) = None

            if is_new_request:
                # =================================
                # 1. 查询联合 KV + GDN Prefix
                # =================================

                prefix_entry = (
                    self._lookup_prefix_for_new_request(
                        seq
                    )
                )

                if prefix_entry is not None:
                    # Entry 中保存的每个物理 Block，
                    # 分别对应一个完整的逻辑 token block。
                    num_cached_blocks = len(
                        prefix_entry.kv_block_ids
                    )

                    # 命中时不能使用普通 can_allocate()。
                    #
                    # 普通 can_allocate() 会从全局
                    # hash_to_block_id 中自行查找 KV，
                    # 但联合 Prefix Cache 必须使用 Entry
                    # 明确保存的那组 KV Block。
                    has_kv_blocks = (
                        self.block_manager
                        .can_allocate_from_prefix(
                            seq,
                            prefix_entry.kv_block_ids,
                        )
                    )

                    expected_cached_tokens = (
                        num_cached_blocks
                        * self.block_size
                    )

                    if (
                        expected_cached_tokens
                        != prefix_entry
                        .key.num_cached_tokens
                    ):
                        raise RuntimeError(
                            "Prefix Entry token boundary "
                            "does not match its KV blocks"
                        )

                else:
                    # Prefix miss，沿用普通分配路径。
                    num_cached_blocks = (
                        self.block_manager
                        .can_allocate(seq)
                    )

                    has_kv_blocks = (
                        num_cached_blocks != -1
                    )

                def resources_available() -> bool:
                    """
                    新请求只有同时获得三类资源，
                    才能进入 Prefill。
                    """

                    has_active_capacity = (
                        num_active_seqs
                        < self.max_num_seqs
                    )

                    has_state_slot = (
                        self.can_allocate_state_slot(seq)
                    )

                    return (
                        has_active_capacity
                        and has_kv_blocks
                        and has_state_slot
                    )

                # 普通 Prefill 资源不足时继续等待。
                #
                # 等待超时后，可以抢占没有进入本轮
                # Decode microbatch 的运行中请求。
                while (
                    reserve_prefill
                    and not resources_available()
                ):
                    victim = (
                        self._select_preemption_victim(
                            excluded_seq_ids=(
                                scheduled_decode_ids
                            ),
                        )
                    )

                    if victim is None:
                        break

                    preempted_seq_ids.append(
                        victim.seq_id
                    )

                    self.running.remove(victim)
                    self.preempt(victim)

                    num_active_seqs -= 1

                    # 抢占释放了 KV blocks，因此重新检查。
                    #
                    # Prefix hit 与 miss 必须分别调用各自
                    # 对应的资源检查函数。
                    if prefix_entry is not None:
                        has_kv_blocks = (
                            self.block_manager
                            .can_allocate_from_prefix(
                                seq,
                                prefix_entry.kv_block_ids,
                            )
                        )

                    else:
                        num_cached_blocks = (
                            self.block_manager
                            .can_allocate(seq)
                        )

                        has_kv_blocks = (
                            num_cached_blocks != -1
                        )

                if not resources_available():
                    break

                # =================================
                # 2. 计算真正需要执行的 Prompt tokens
                # =================================

                num_cached_tokens = (
                    num_cached_blocks
                    * self.block_size
                )

                num_tokens = (
                    seq.num_tokens
                    - num_cached_tokens
                )

            else:
                # 该请求已经执行过至少一个
                # Chunked Prefill。
                #
                # 它已经持有 KV blocks 和 GDN state slot，
                # 不需要重新执行 admission。
                num_tokens = (
                    seq.num_tokens
                    - seq.num_cached_tokens
                )

            if num_tokens <= 0:
                raise RuntimeError(
                    f"Sequence {seq.seq_id} has no "
                    "Prefill tokens remaining"
                )

            if is_new_request:
                # =================================
                # 3. 真正占用 KV Block
                # =================================

                if prefix_entry is not None:
                    reused_blocks = (
                        self.block_manager
                        .allocate_from_prefix(
                            seq,
                            prefix_entry.kv_block_ids,
                        )
                    )

                    if (
                        reused_blocks
                        != num_cached_blocks
                    ):
                        raise RuntimeError(
                            "Allocated Prefix KV block "
                            "count changed unexpectedly"
                        )

                else:
                    # Prefix miss，走原来的普通分配路径。
                    self.block_manager.allocate(
                        seq,
                        num_cached_blocks,
                    )

                # =================================
                # 4. 分配 active GDN state slot
                # =================================

                self.allocate_state_slot(seq)

                # =================================
                # 5. 记录 Prefix 命中状态
                # =================================

                if prefix_entry is not None:
                    if seq.state_slot is None:
                        raise RuntimeError(
                            "Prefix hit request did not "
                            "receive a GDN state slot"
                        )

                    if (
                        seq.num_cached_tokens
                        != prefix_entry
                        .key.num_cached_tokens
                    ):
                        raise RuntimeError(
                            "Sequence cached-token boundary "
                            "does not match Prefix Entry"
                        )

                    # Scheduler 已经完成：
                    #
                    # 1. KV Block attach；
                    # 2. state slot 分配。
                    #
                    # 但 GDN Snapshot 还没有复制进 slot。
                    # Engine 会在模型执行前完成复制。
                    seq.prefix_restore_pending = True

                    seq.num_prefix_hit_tokens = (
                        prefix_entry
                        .key.num_cached_tokens
                    )

                    self.num_prefix_hit_requests += 1

                    self.num_prefix_hit_tokens += (
                        seq.num_prefix_hit_tokens
                    )

                num_active_seqs += 1

            # 如果剩余预算不够处理完整 prompt，
            # 就只处理一个 chunk。
            seq.num_scheduled_tokens = min(
                num_tokens,
                remaining_token_budget,
            )

            seq.is_prefill = True

            prefill_seqs.append(seq)

            num_prefill_tokens += (
                seq.num_scheduled_tokens
            )

            remaining_token_budget -= (
                seq.num_scheduled_tokens
            )

            prefill_will_finish = (
                seq.num_cached_tokens
                + seq.num_scheduled_tokens
                == seq.num_tokens
            )

            if prefill_will_finish:
                seq.status = SequenceStatus.RUNNING

                removed_seq = self.waiting.popleft()

                if removed_seq is not seq:
                    raise RuntimeError(
                        "Prefill queue order changed "
                        "unexpectedly"
                    )

                self._clear_waiting(seq)
                self.running.append(seq)
                
            else:
                # 请求得到了一个 Prefill chunk，
                # 因此它不再算作持续饥饿。
                #
                # 从本 chunk 执行后重新开始计时。
                self._mark_waiting(seq)

            # 如果 Prefill 没完成，该请求仍然停留在
            # waiting[0]。
            #
            # 此时它一定已经消耗完 remaining token
            # budget，因此 while 会自然结束。

        # =====================================
        # 第四阶段：生成 SchedulePlan
        # =====================================

        plan = SchedulePlan(
            decode_seqs=decode_seqs,
            prefill_seqs=prefill_seqs,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_prefill_tokens,
            preempted_seq_ids=preempted_seq_ids,
        )

        if plan.is_empty:
            raise RuntimeError(
                "Scheduler cannot make progress: "
                "no Decode or Prefill request could "
                "be scheduled"
            )
        return plan
    
    def preempt(
        self,
        seq: Sequence,
    ) -> None:
        """
        释放请求的全部运行时状态。

        调用者必须先将 seq 从 running 队列移除。
        """

        recomputed_tokens = (
            seq.num_cached_tokens
        )

        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        seq.num_scheduled_tokens = 0
        
        # 被抢占请求重新回到 waiting 后，需要重新进行
        # Prefix lookup。
        #
        # 它可能再次命中同一个 Entry，也可能因为未来的
        # LRU 淘汰而发生 miss。
        seq.prefix_lookup_completed = False
        seq.prefix_cache_key = None
        seq.prefix_restore_pending = False
        seq.num_prefix_hit_tokens = 0

        # 释放 Full Attention KV。
        self.block_manager.deallocate(seq)

        # 释放 GDN state slot。
        self.release_state_slot(seq)

        self.num_preemptions += 1

        self.num_recomputed_tokens += (
            recomputed_tokens
        )

        # 放到 waiting 队尾，不能插到已经等待
        # 更久的请求前面。
        self.waiting.append(seq)
        self._mark_waiting(seq)

    def record_computed_block_metadata(
        self,
        seqs: list[Sequence],
    ) -> None:
        """
        模型 Forward 成功后，为本轮刚计算完成的
        完整 KV blocks 记录 token IDs 和链式 Hash。

        该阶段不推进 num_cached_tokens，
        也不释放请求资源。
        """

        for seq in seqs:
            if seq.num_scheduled_tokens <= 0:
                raise RuntimeError(
                    f"Sequence {seq.seq_id} has no "
                    "completed scheduled tokens"
                )

            self.block_manager.hash_blocks(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if ((not seq.ignore_eos and token_id == self.eos) or (seq.num_completion_tokens == seq.max_tokens)):
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.release_state_slot(seq)
                self.running.remove(seq)

    def can_allocate_state_slot(
        self,
        seq: Sequence,
    ) -> bool:

        if not self.requires_gdn_state:
            return True

        # 已经做过一个 Prefill chunk 的请求
        # 会继续持有原来的 slot。
        if seq.state_slot is not None:
            return True

        return (
            self.state_slot_allocator
            .can_allocate()
        )


    def allocate_state_slot(
        self,
        seq: Sequence,
    ) -> None:

        if not self.requires_gdn_state:
            return

        if seq.state_slot is not None:
            raise RuntimeError(
                f"Sequence {seq.seq_id} already "
                "owns state slot {seq.state_slot}"
            )

        seq.state_slot = (
            self.state_slot_allocator.allocate()
        )


    def release_state_slot(
        self,
        seq: Sequence,
    ) -> None:

        if not self.requires_gdn_state:
            return

        if seq.state_slot is None:
            raise RuntimeError(
                f"Sequence {seq.seq_id} has no "
                "state slot to release"
            )

        self.state_slot_allocator.release(
            seq.state_slot
        )

        seq.state_slot = None