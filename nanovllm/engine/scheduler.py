from time import monotonic
from collections import deque
from dataclasses import dataclass

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.hybrid_state import (
    StateSlotAllocator,
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

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_prefill_wait_ms = (config.max_prefill_wait_ms)
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(
            num_blocks=config.num_kvcache_blocks,
            block_size=config.kvcache_block_size,
            enable_prefix_cache=(
                config.enable_prefix_cache
            ),
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        # seq_id -> 本次进入 waiting 队列的单调时间。
        self.waiting_since: dict[int, float] = {}
        # Scheduler 生命周期内的累计抢占次数。
        self.num_preemptions = 0
        # 被抢占请求需要重新计算的累计 token 数。
        self.num_recomputed_tokens = 0
        layer_types = getattr(
        config.text_config,
        "layer_types",
            (),
        )

        self.requires_gdn_state = any(
            layer_type == "linear_attention"
            for layer_type in layer_types
        )

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
        构造一轮 Decode-first 执行计划。

        调度顺序：

            1. 判断 Prefill 是否等待超时。
            2. 优先选择 Decode 请求。
            3. 如果 Prefill 超时，为它保留预算。
            4. 使用剩余 token 和 sequence budget
            调度 Prefill/Chunked Prefill。

        当 Prefill 等待超时时，Scheduler 会同时
        预留计算预算，并从未进入本轮 Decode 的
        running 请求中选择重算成本最低的 victim，
        释放其 KV blocks 和 GDN state slot。
        """

        decode_seqs: list[Sequence] = []
        prefill_seqs: list[Sequence] = []

        num_decode_tokens = 0
        num_prefill_tokens = 0

        preempted_seq_ids: list[int] = []
        
        # 一轮调度只读取一次当前时间，
        # 保证本轮所有等待时间判断基于同一时刻。
        now = monotonic()

        reserve_prefill = (
            self._should_reserve_prefill(now)
        )

        reserved_seq_slots = (
            1
            if reserve_prefill
            else 0
        )

        reserved_prefill_tokens = (
            1
            if reserve_prefill
            else 0
        )

        # Prefill 超时后，Decode 最多只能使用
        # max_num_seqs - 1 个请求位置。
        decode_seq_limit = max(
            0,
            self.max_num_seqs
            - reserved_seq_slots,
        )

        # Prefill 超时后，Decode 最多只能消耗
        # max_num_batched_tokens - 1 个 token。
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

            if is_new_request:
                # 先检查当前 KV blocks 是否足够。
                num_cached_blocks = (
                    self.block_manager.can_allocate(seq)
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

                    has_kv_blocks = (
                        num_cached_blocks != -1
                    )

                    has_state_slot = (
                        self.can_allocate_state_slot(seq)
                    )

                    return (
                        has_active_capacity
                        and has_kv_blocks
                        and has_state_slot
                    )

                # 普通 Prefill：
                # 资源不足就继续等待。
                #
                # 超时 Prefill：
                # 尝试抢占未进入本轮 Decode 的请求，
                # 直到资源足够或者没有合法 victim。
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
                    # victim 还在 running 中，
                    # 必须先将其移出。
                    self.running.remove(victim)

                    # 释放 victim 的 KV blocks 和
                    # GDN state slot，并放回 waiting 队尾。
                    self.preempt(victim)

                    num_active_seqs -= 1

                    # 抢占释放了 KV blocks，
                    # 因此必须重新检查目标请求能否分配。
                    num_cached_blocks = (
                        self.block_manager.can_allocate(seq)
                    )

                # while 可能因为没有合法 victim 而退出，
                # 所以退出后必须再次检查资源。
                if not resources_available():
                    break

                # admission 成功。
                #
                # 对 Qwen3.5，Prefix Cache 关闭，
                # num_cached_blocks 通常等于 0。
                num_tokens = (
                    seq.num_tokens
                    - num_cached_blocks * self.block_size
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
                # can_allocate() 只做检查，
                # allocate() 才真正占用 KV blocks。
                self.block_manager.allocate(
                    seq,
                    num_cached_blocks,
                )

                # Qwen3.5 在这里获得一个固定 state slot。
                self.allocate_state_slot(seq)

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

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
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