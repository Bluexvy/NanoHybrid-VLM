from dataclasses import dataclass
from collections import OrderedDict
from math import prod

import torch

from nanovllm.engine.block_manager import (
    BlockManager,
)
from nanovllm.engine.hybrid_state import (
    HybridCacheSpec,
    HybridStateManager,
)

@dataclass(
    frozen=True,
    slots=True,
)
class PrefixKey:
    """
    一个纯文本完整前缀边界的缓存身份。

    block_hash 是链式 Hash，已经包含此前所有
    完整 token blocks 的有序历史。
    """

    model_namespace: str
    block_hash: int
    num_cached_tokens: int

    def __post_init__(self) -> None:
        if not self.model_namespace:
            raise ValueError(
                "model_namespace must not be empty"
            )

        if self.block_hash < 0:
            raise ValueError(
                "block_hash must be non-negative"
            )

        if self.num_cached_tokens <= 0:
            raise ValueError(
                "num_cached_tokens must be positive"
            )
            
            
@dataclass(slots=True)
class PrefixStateEntry:
    """
    一个完整纯文本 token-block 边界对应的
    Full Attention KV 和 GDN 状态快照。

    Tensor 在语义上只读：
    请求恢复时只能复制到 active state slot，
    不能直接原地修改 snapshot。
    """

    key: PrefixKey

    # 从 token 0 到 key.num_cached_tokens
    # 所使用的全部物理 KV blocks。
    kv_block_ids: tuple[int, ...]

    # 所有 GDN 层的独立快照：
    #
    # [num_gdn_layers, conv_dim, kernel_size]
    conv_state_snapshot: torch.Tensor

    # dtype 由 Prefix Cache 配置决定：
    #
    # correctness mode:
    #     torch.float32
    #
    # compressed mode:
    #     torch.bfloat16
    #
    # 恢复到 active state slot 时，目标状态池
    # 仍然使用 spec.recurrent_dtype，也就是 FP32。
    recurrent_state_snapshot: torch.Tensor

    @property
    def gdn_snapshot_bytes(self) -> int:
        """
        该 Entry 的 GDN snapshot 显存。
        不包含 KV blocks，因为 KV blocks 位于
        全局 Paged KV Cache 中。
        """

        conv_bytes = (
            self.conv_state_snapshot.numel()
            * self.conv_state_snapshot.element_size()
        )

        recurrent_bytes = (
            self.recurrent_state_snapshot.numel()
            * self.recurrent_state_snapshot.element_size()
        )

        return conv_bytes + recurrent_bytes

    def validate(
        self,
        spec: HybridCacheSpec,
        block_size: int,
        expected_recurrent_snapshot_dtype: (
            torch.dtype
        ),
    ) -> None:
        """
        检查 Entry 的边界、KV block 数量以及
        GDN snapshot 的 shape/dtype/device。
        """

        supported_snapshot_dtypes = {
            torch.float32,
            torch.bfloat16,
        }

        if (
            expected_recurrent_snapshot_dtype
            not in supported_snapshot_dtypes
        ):
            raise ValueError(
                "recurrent snapshot dtype must be "
                "torch.float32 or torch.bfloat16"
            )

        if block_size <= 0:
            raise ValueError(
                "block_size must be positive"
            )

        if (
            self.key.num_cached_tokens
            % block_size
            != 0
        ):
            raise ValueError(
                "Prefix entry must end at a complete "
                "token block boundary"
            )

        expected_num_blocks = (
            self.key.num_cached_tokens
            // block_size
        )

        if (
            len(self.kv_block_ids)
            != expected_num_blocks
        ):
            raise ValueError(
                "kv_block_ids length does not match "
                "num_cached_tokens"
            )

        if any(
            block_id < 0
            for block_id in self.kv_block_ids
        ):
            raise ValueError(
                "KV block IDs must be non-negative"
            )

        if (
            len(set(self.kv_block_ids))
            != len(self.kv_block_ids)
        ):
            raise ValueError(
                "A prefix must not contain duplicate "
                "physical KV block IDs"
            )

        expected_conv_shape = (
            spec.conv_state_shape_per_slot
        )

        if (
            tuple(self.conv_state_snapshot.shape)
            != expected_conv_shape
        ):
            raise ValueError(
                "Invalid conv snapshot shape: "
                f"expected {expected_conv_shape}, "
                "got "
                f"{tuple(self.conv_state_snapshot.shape)}"
            )

        expected_recurrent_shape = (
            spec.recurrent_state_shape_per_slot
        )

        if (
            tuple(
                self.recurrent_state_snapshot.shape
            )
            != expected_recurrent_shape
        ):
            raise ValueError(
                "Invalid recurrent snapshot shape: "
                f"expected "
                f"{expected_recurrent_shape}, got "
                f"{tuple(self.recurrent_state_snapshot.shape)}"
            )

        if (
            self.conv_state_snapshot.dtype
            != spec.conv_dtype
        ):
            raise TypeError(
                "conv snapshot has the wrong dtype"
            )

        if (
            self.recurrent_state_snapshot.dtype
            != expected_recurrent_snapshot_dtype
        ):
            raise TypeError(
                "recurrent snapshot has the wrong dtype: "
                f"expected "
                f"{expected_recurrent_snapshot_dtype}, got "
                f"{self.recurrent_state_snapshot.dtype}"
            )

        if (
            self.conv_state_snapshot.device
            != self.recurrent_state_snapshot.device
        ):
            raise ValueError(
                "GDN snapshots must be on the same "
                "device"
            )

        if self.conv_state_snapshot.requires_grad:
            raise ValueError(
                "conv snapshot must not require grad"
            )

        if (
            self.recurrent_state_snapshot
            .requires_grad
        ):
            raise ValueError(
                "recurrent snapshot must not require "
                "grad"
            )
            
            
class PrefixStateCache:
    """
    管理纯文本 Qwen3.5 Prefix Entries。

    当前 Part 只实现：
        原子 commit
        精确重复检测
        最小 discard

    显存预算、LRU、热度和 lookup/restore
    在后续 Part 接入。
    """

    def __init__(
        self,
        block_manager: BlockManager,
        state_manager: HybridStateManager,
        model_namespace: str,
        recurrent_snapshot_dtype: torch.dtype,
        kv_block_bytes: int,
        capacity_bytes: int,
        admission_policy: str = "always",
        admission_min_observations: int = 2,
        admission_max_candidates: int = 4096,
    ) -> None:
        if not model_namespace:
            raise ValueError(
                "model_namespace must not be empty"
            )

        supported_snapshot_dtypes = {
            torch.float32,
            torch.bfloat16,
        }

        if capacity_bytes <= 0:
            raise ValueError(
                "capacity_bytes must be positive"
            )

        if kv_block_bytes <= 0:
            raise ValueError(
                "kv_block_bytes must be positive"
            )
        if (
            recurrent_snapshot_dtype
            not in supported_snapshot_dtypes
        ):
            raise ValueError(
                "recurrent_snapshot_dtype must be "
                "torch.float32 or torch.bfloat16"
            )

        supported_admission_policies = {
            "always",
            "frequency",
        }

        if (
            admission_policy
            not in supported_admission_policies
        ):
            raise ValueError(
                "admission_policy must be "
                "'always' or 'frequency'"
            )

        if admission_min_observations <= 0:
            raise ValueError(
                "admission_min_observations "
                "must be positive"
            )

        if admission_max_candidates <= 0:
            raise ValueError(
                "admission_max_candidates "
                "must be positive"
            )

        self.block_manager = block_manager
        self.state_manager = state_manager

        self.spec = state_manager.spec
        self.block_size = (
            block_manager.block_size
        )

        self.model_namespace = model_namespace
        self.recurrent_snapshot_dtype = (
            recurrent_snapshot_dtype
        )
        self.kv_block_bytes = kv_block_bytes
        self.capacity_bytes = capacity_bytes

        self.admission_policy = (
            admission_policy
        )

        self.admission_min_observations = (
            admission_min_observations
        )

        self.admission_max_candidates = (
            admission_max_candidates
        )

        # CPU侧热度历史。
        #
        # 左端：最久未观察
        # 右端：最近观察
        #
        # value是观察次数，达到阈值后停止增长。
        self.admission_history: OrderedDict[
            PrefixKey,
            int,
        ] = OrderedDict()

        # 顺序含义：
        #
        # 最左边：最久未使用 LRU
        # 最右边：最近使用 MRU
        self.entries: OrderedDict[
            PrefixKey,
            PrefixStateEntry,
        ] = OrderedDict()
        
        # Entry 被命中或重复提交时更新 LRU 的次数。
        self.num_lru_touches = 0

        # 因LRU容量不足被自动淘汰的Entry数量。
        self.num_evictions = 0

        # 候选Entry自身就大于预算，无法被缓存的次数。
        self.num_capacity_rejections = 0

        # 历次LRU淘汰在当时释放的缓存容量。
        self.total_evicted_capacity_bytes = 0

        self.total_gdn_snapshot_bytes = 0
        self.num_commits = 0
        self.num_duplicate_commits = 0
        
        self.num_lookups = 0
        self.num_hits = 0
        self.num_misses = 0
        self.num_hash_collisions = 0
        self.num_gdn_restores = 0

        # 到达合法checkpoint的候选观察次数。
        self.num_admission_observations = 0

        # 达到准入条件的次数。
        #
        # 注意：被准入后仍可能因为GPU容量不足而无法commit。
        self.num_admission_accepts = 0

        # 因热度不足而暂缓缓存的次数。
        self.num_admission_deferrals = 0

        # CPU候选历史超过上限后淘汰的记录数。
        self.num_admission_candidate_evictions = 0

        # Resident Entry命中时刷新热度历史的次数。
        self.num_admission_hit_touches = 0

    def _validate_admission_key(
        self,
        key: PrefixKey,
    ) -> None:
        if (
            key.model_namespace
            != self.model_namespace
        ):
            raise ValueError(
                "Admission PrefixKey model namespace "
                "does not match this cache"
            )


    def _remember_admission_count(
        self,
        key: PrefixKey,
        count: int,
    ) -> None:
        """
        写入或更新一条CPU热度历史。

        admission_history只控制准入元数据容量，
        不影响GPU Prefix Entry的LRU顺序。
        """

        self._validate_admission_key(key)

        if count <= 0:
            raise ValueError(
                "Admission observation count "
                "must be positive"
            )

        if key in self.admission_history:
            self.admission_history[key] = count

            self.admission_history.move_to_end(
                key,
                last=True,
            )

            return

        # 新候选进入历史表前，先保证CPU元数据有空间。
        if (
            len(self.admission_history)
            >= self.admission_max_candidates
        ):
            self.admission_history.popitem(
                last=False,
            )

            self.num_admission_candidate_evictions += 1

        self.admission_history[key] = count


    def observe_and_should_admit(
        self,
        key: PrefixKey,
    ) -> bool:
        """
        观察一个已经实际计算到达的Prefix checkpoint，
        并决定是否允许创建GPU Prefix Entry。

        always：
            立即允许。

        frequency：
            observation count达到阈值后允许。
        """

        self._validate_admission_key(key)

        self.num_admission_observations += 1

        if self.admission_policy == "always":
            self.num_admission_accepts += 1
            return True

        previous_count = (
            self.admission_history.get(
                key,
                0,
            )
        )

        # 达到阈值后饱和，不让整数无限增长。
        new_count = min(
            previous_count + 1,
            self.admission_min_observations,
        )

        self._remember_admission_count(
            key,
            new_count,
        )

        if (
            new_count
            >= self.admission_min_observations
        ):
            self.num_admission_accepts += 1
            return True

        self.num_admission_deferrals += 1
        return False


    def record_admission_hit(
        self,
        key: PrefixKey,
    ) -> None:
        """
        Resident Entry真正命中时，刷新其CPU热度历史。

        这样一个经常命中的Prefix即使之后因GPU容量不足
        被驱逐，其热度历史仍可能保留，下一次可快速重新准入。
        """

        if self.admission_policy != "frequency":
            return

        self._validate_admission_key(key)

        self._remember_admission_count(
            key,
            self.admission_min_observations,
        )

        self.num_admission_hit_touches += 1

    def _touch_entry(
        self,
        entry: PrefixStateEntry,
    ) -> None:
        """
        将一个 resident Entry 移动到 MRU 端。

        OrderedDict：
            左端是 LRU；
            右端是 MRU。
        """

        resident_entry = self.entries.get(
            entry.key
        )

        if resident_entry is None:
            raise RuntimeError(
                "Cannot touch a non-resident "
                "Prefix Entry"
            )

        if resident_entry is not entry:
            raise RuntimeError(
                "Prefix Entry object changed before "
                "LRU touch"
            )

        self.entries.move_to_end(
            entry.key,
            last=True,
        )

        self.num_lru_touches += 1

    def lookup_longest(
        self,
        prompt_token_ids: list[int],
    ) -> PrefixStateEntry | None:
        """
        查询严格短于完整 Prompt 的最长联合 Prefix Entry。

        Entry 不保存 logits，因此必须至少保留一个
        Prompt token 给模型真正执行。
        """

        self.num_lookups += 1

        if len(prompt_token_ids) <= self.block_size:
            self.num_misses += 1
            return None

        # 例如：
        #
        # prompt length = 1165
        # block_size = 256
        #
        # (1165 - 1) // 256 = 4
        #
        # 最多允许查到 1024-token 边界。
        max_cacheable_blocks = (
            (len(prompt_token_ids) - 1)
            // self.block_size
        )

        current_hash = -1

        longest_entry: (
            PrefixStateEntry | None
        ) = None

        for block_index in range(
            max_cacheable_blocks
        ):
            start = (
                block_index
                * self.block_size
            )

            end = start + self.block_size

            token_block = prompt_token_ids[
                start:end
            ]

            if len(token_block) != self.block_size:
                raise RuntimeError(
                    "Prefix lookup encountered an "
                    "incomplete token block"
                )

            current_hash = (
                self.block_manager.compute_hash(
                    token_block,
                    current_hash,
                )
            )

            num_cached_tokens = end

            key = PrefixKey(
                model_namespace=(
                    self.model_namespace
                ),
                block_hash=current_hash,
                num_cached_tokens=(
                    num_cached_tokens
                ),
            )

            entry = self.entries.get(key)

            if entry is None:
                continue

            # 验证 Entry 对应的物理 KV Blocks
            # 仍然处于 resident、被缓存引用持有的状态。
            self.block_manager.validate_prefix_blocks(
                entry.kv_block_ids,
                expected_block_hash=(
                    entry.key.block_hash
                ),
                require_request_owner=False,
            )

            if (
                len(entry.kv_block_ids)
                != block_index + 1
            ):
                raise RuntimeError(
                    "Prefix Entry KV block count "
                    "does not match lookup boundary"
                )

            # Hash 只是快速索引，不是最终正确性证明。
            #
            # 逐 Block 比较真实 token IDs，
            # 防止极低概率的 xxhash64 碰撞。
            tokens_match = all(
                (
                    self.block_manager
                    .blocks[physical_block_id]
                    .token_ids
                    == prompt_token_ids[
                        logical_block_index
                        * self.block_size:
                        (logical_block_index + 1)
                        * self.block_size
                    ]
                )
                for (
                    logical_block_index,
                    physical_block_id,
                )
                in enumerate(
                    entry.kv_block_ids
                )
            )

            if not tokens_match:
                self.num_hash_collisions += 1
                continue

            longest_entry = entry

        if longest_entry is None:
            self.num_misses += 1
            return None

        self.num_hits += 1

        self._touch_entry(
            longest_entry
        )

        # GPU Entry命中，同时刷新CPU热度历史。
        self.record_admission_hit(
            longest_entry.key
        )

        return longest_entry
    
    def get_resident_entry(
        self,
        key: PrefixKey,
    ) -> PrefixStateEntry | None:
        """
        根据已经保存的 PrefixKey，重新取得仍然
        resident 的 PrefixStateEntry。

        这个接口不算作一次新的 Prefix lookup，
        因为它不会重新扫描 Prompt token blocks。
        """

        if (
            key.model_namespace
            != self.model_namespace
        ):
            raise ValueError(
                "PrefixKey model namespace does not "
                "match this PrefixStateCache"
            )

        entry = self.entries.get(key)

        if entry is None:
            return None

        # Entry 仍然存在于字典里，并不一定能完全证明
        # 它引用的物理 KV blocks 仍然有效。
        #
        # 所以这里重新检查：
        # 1. Block 仍然位于 used_block_ids；
        # 2. Block 仍然有 cache owner；
        # 3. 最后一个 Block 的链式 Hash 没有变化。
        self.block_manager.validate_prefix_blocks(
            entry.kv_block_ids,
            expected_block_hash=(
                entry.key.block_hash
            ),
            require_request_owner=False,
        )

        return entry
 
    def restore_gdn_state(
        self,
        entry: PrefixStateEntry,
        state_slot: int,
    ) -> None:
        """
        将仍然 resident 的 Prefix Entry GDN Snapshot
        恢复到一个 active state slot。
        """

        resident_entry = self.entries.get(
            entry.key
        )

        if resident_entry is None:
            raise RuntimeError(
                "Cannot restore an evicted "
                "Prefix Entry"
            )

        if resident_entry is not entry:
            raise RuntimeError(
                "Prefix Entry object changed before "
                "GDN state restore"
            )

        entry.validate(
            spec=self.spec,
            block_size=self.block_size,
            expected_recurrent_snapshot_dtype=(
                self.recurrent_snapshot_dtype
            ),
        )

        self.block_manager.validate_prefix_blocks(
            entry.kv_block_ids,
            expected_block_hash=(
                entry.key.block_hash
            ),
            require_request_owner=False,
        )

        self.state_manager.restore_slot(
            slot=state_slot,
            conv_state_snapshot=(
                entry.conv_state_snapshot
            ),
            recurrent_state_snapshot=(
                entry.recurrent_state_snapshot
            ),
        )

        self.num_gdn_restores += 1
 
    @property
    def estimated_gdn_snapshot_bytes_per_entry(
        self,
    ) -> int:
        """
        不创建GPU Tensor，直接根据shape和dtype
        计算一份Prefix GDN Snapshot的字节数。
        """

        conv_bytes = (
            prod(
                self.spec
                .conv_state_shape_per_slot
            )
            * self.spec.dtype_nbytes(
                self.spec.conv_dtype
            )
        )

        recurrent_bytes = (
            prod(
                self.spec
                .recurrent_state_shape_per_slot
            )
            * self.spec.dtype_nbytes(
                self.recurrent_snapshot_dtype
            )
        )

        return (
            conv_bytes
            + recurrent_bytes
        )
 
    def _additional_capacity_bytes(
        self,
        kv_block_ids: tuple[int, ...],
    ) -> int:
        """
        如果现在提交候选Entry，它会增加多少缓存容量。

        已经被其他Entry pin住的共享KV Block不重复计费。
        """

        additional_unique_kv_blocks = 0

        for block_id in kv_block_ids:
            block = self.block_manager.blocks[
                block_id
            ]

            if block_id not in (
                self.block_manager.used_block_ids
            ):
                raise RuntimeError(
                    "Candidate Prefix references a free "
                    "KV Block"
                )

            if block.ref_count <= 0:
                raise RuntimeError(
                    "Candidate Prefix KV Block has no "
                    "owner"
                )

            # cache_ref_count == 0：
            # 当前没有任何Prefix Entry持有它。
            #
            # 候选Entry提交后，它会成为新的唯一
            # pinned KV Block，需要计入新增容量。
            if block.cache_ref_count == 0:
                additional_unique_kv_blocks += 1

        additional_kv_bytes = (
            additional_unique_kv_blocks
            * self.kv_block_bytes
        )

        return (
            self.estimated_gdn_snapshot_bytes_per_entry
            + additional_kv_bytes
        ) 

    def _ensure_capacity_for(
        self,
        kv_block_ids: tuple[int, ...],
    ) -> bool:
        """
        在创建候选Entry之前，通过LRU淘汰确保预算充足。

        返回False表示候选Entry单独存在时就超过预算，
        因此不能缓存，但不影响当前请求继续推理。
        """

        # 假设Cache完全为空时，候选Entry独占所有
        # KV Blocks所需的最大容量。
        standalone_capacity_bytes = (
            self.estimated_gdn_snapshot_bytes_per_entry
            + len(kv_block_ids)
            * self.kv_block_bytes
        )

        # 单个Entry自身就超过预算时，不应该先把已有
        # Cache全部清空再发现仍然放不下。
        if (
            standalone_capacity_bytes
            > self.capacity_bytes
        ):
            self.num_capacity_rejections += 1
            return False

        while True:
            # 必须每轮重新计算。
            #
            # 被淘汰Entry可能和候选Entry共享KV Blocks；
            # 淘汰后候选Entry需要重新承担这些Block容量。
            additional_capacity_bytes = (
                self._additional_capacity_bytes(
                    kv_block_ids
                )
            )

            projected_capacity_bytes = (
                self.current_prefix_cache_capacity_bytes
                + additional_capacity_bytes
            )

            if (
                projected_capacity_bytes
                <= self.capacity_bytes
            ):
                return True

            victim = self.peek_lru_entry()

            if victim is None:
                # 理论上standalone检查通过后不应进入这里。
                self.num_capacity_rejections += 1
                return False

            reclaimable_bytes = (
                self.reclaimable_capacity_bytes(
                    victim
                )
            )

            victim_key = victim.key

            discarded = self.discard(
                victim_key
            )

            if not discarded:
                raise RuntimeError(
                    "LRU victim disappeared during "
                    "capacity eviction"
                )

            self.num_evictions += 1

            self.total_evicted_capacity_bytes += (
                reclaimable_bytes
            )

    def commit(
        self,
        key: PrefixKey,
        kv_block_ids: (
            list[int]
            | tuple[int, ...]
        ),
        state_slot: int,
    ) -> tuple[
        PrefixStateEntry | None,
        bool,
    ]:
        """
        
        entry=None、created=False：
            候选Entry超过容量预算，没有写入Cache，
            但当前请求可以继续正常运行。
            
        原子提交一个 Prefix Entry。

        返回：
            entry
            created

        created=True：
            本次创建了新 Entry。

        created=False：
            相同 Entry 已经存在，没有重复 Snapshot。
        """

        if (
            key.model_namespace
            != self.model_namespace
        ):
            raise ValueError(
                "PrefixKey model namespace does not "
                "match this PrefixStateCache"
            )

        if (
            key.num_cached_tokens
            % self.block_size
            != 0
        ):
            raise ValueError(
                "Prefix commit must end at a complete "
                "token block boundary"
            )

        expected_num_blocks = (
            key.num_cached_tokens
            // self.block_size
        )

        if expected_num_blocks <= 0:
            raise ValueError(
                "Prefix commit must contain at least "
                "one complete token block"
            )

        if (
            len(kv_block_ids)
            != expected_num_blocks
        ):
            raise ValueError(
                "KV block count does not match "
                "PrefixKey.num_cached_tokens"
            )

        normalized_block_ids = (
            self.block_manager
            .validate_prefix_blocks(
                kv_block_ids,
                expected_block_hash=(
                    key.block_hash
                ),
            )
        )

        existing_entry = self.entries.get(
            key
        )

        if existing_entry is not None:
            same_tokens = (
                self.block_manager
                .prefix_blocks_have_same_tokens(
                    existing_entry.kv_block_ids,
                    normalized_block_ids,
                )
            )

            if not same_tokens:
                raise RuntimeError(
                    "Prefix hash collision detected: "
                    "the same PrefixKey refers to "
                    "different token blocks"
                )

            self.num_duplicate_commits += 1
            self._touch_entry(
                existing_entry
            )

            return existing_entry, False

        has_capacity = self._ensure_capacity_for(
            normalized_block_ids
        )

        if not has_capacity:
            return None, False

        (
            conv_state_snapshot,
            recurrent_state_snapshot,
        ) = self.state_manager.snapshot_slot(
            slot=state_slot,
            recurrent_snapshot_dtype=(
                self.recurrent_snapshot_dtype
            ),
        )

        entry = PrefixStateEntry(
            key=key,
            kv_block_ids=normalized_block_ids,
            conv_state_snapshot=(
                conv_state_snapshot
            ),
            recurrent_state_snapshot=(
                recurrent_state_snapshot
            ),
        )

        entry.validate(
            spec=self.spec,
            block_size=self.block_size,
            expected_recurrent_snapshot_dtype=(
                self.recurrent_snapshot_dtype
            ),
        )

        pinned = False

        try:
            pinned_block_ids = (
                self.block_manager.pin_blocks(
                    normalized_block_ids
                )
            )

            pinned = True

            if pinned_block_ids != normalized_block_ids:
                raise RuntimeError(
                    "BlockManager changed KV block order "
                    "while pinning Prefix Entry"
                )

            # 字典插入是提交点。
            #
            # 在此之前，外部 lookup 看不到该 Entry。
            self.entries[key] = entry

        except Exception:
            if pinned:
                self.block_manager.unpin_blocks(
                    normalized_block_ids
                )

            raise

        self.total_gdn_snapshot_bytes += (
            entry.gdn_snapshot_bytes
        )

        self.num_commits += 1

        if (
            self.current_prefix_cache_capacity_bytes
            > self.capacity_bytes
        ):
            # 这是内部计算错误，不是普通容量不足。
            #
            # 先回滚刚创建的Entry，再抛出异常。
            self.discard(key)

            raise RuntimeError(
                "Prefix Cache exceeded its capacity "
                "after a successful commit"
            )

        return entry, True
        
    def discard(
        self,
        key: PrefixKey,
    ) -> bool:
        """
        删除一个 Prefix Entry，并释放它持有的
        KV cache-owner 引用。

        返回 False 表示 Entry 不存在。
        """

        entry = self.entries.get(key)

        if entry is None:
            return False

        # 先释放外部资源。
        #
        # 如果 unpin 失败，Entry 仍然保留在字典中，
        # 不会变成“资源还在但索引丢失”。
        self.block_manager.unpin_blocks(
            entry.kv_block_ids
        )

        removed_entry = self.entries.pop(key)

        if removed_entry is not entry:
            raise RuntimeError(
                "Prefix Entry changed during discard"
            )

        self.total_gdn_snapshot_bytes -= (
            entry.gdn_snapshot_bytes
        )

        if self.total_gdn_snapshot_bytes < 0:
            raise RuntimeError(
                "Prefix snapshot byte accounting "
                "became negative"
            )

        return True
    
    def reclaimable_capacity_bytes(
        self,
        entry: PrefixStateEntry,
    ) -> int:
        """
        淘汰指定 Entry 后可以真正回收的缓存容量。

        GDN Snapshot 一定属于单个 Entry，可以全部释放。

        KV Block 只有在该 Entry 是最后一个 cache owner
        时才能释放缓存容量。
        """

        resident_entry = self.entries.get(
            entry.key
        )

        if resident_entry is not entry:
            raise RuntimeError(
                "Cannot calculate reclaimable bytes "
                "for a non-resident Prefix Entry"
            )

        uniquely_owned_kv_blocks = 0

        for block_id in entry.kv_block_ids:
            block = self.block_manager.blocks[
                block_id
            ]

            if block.cache_ref_count <= 0:
                raise RuntimeError(
                    "Resident Prefix Entry references "
                    "a KV Block without a cache owner"
                )

            # cache_ref_count == 1：
            # 当前 Entry 是最后一个缓存 owner。
            #
            # 删除它后，这个 Block 不再被任何 Entry pin。
            if block.cache_ref_count == 1:
                uniquely_owned_kv_blocks += 1

        reclaimable_kv_bytes = (
            uniquely_owned_kv_blocks
            * self.kv_block_bytes
        )

        return (
            entry.gdn_snapshot_bytes
            + reclaimable_kv_bytes
        )
        
    @property
    def num_entries(self) -> int:
        return len(self.entries)


    @property
    def current_gdn_snapshot_bytes(
        self,
    ) -> int:
        return self.total_gdn_snapshot_bytes
    
    @property
    def pinned_kv_block_ids(
        self,
    ) -> frozenset[int]:
        """
        返回全部 Prefix Entries 当前共同持有的唯一
        物理 KV Block IDs。

        多个 Entry 共享的物理 Block 只出现一次。
        """

        block_ids: set[int] = set()

        for entry in self.entries.values():
            block_ids.update(
                entry.kv_block_ids
            )

        # 返回不可变集合，防止调用者意外修改
        # Prefix Cache 的统计结果。
        return frozenset(block_ids)
    
    @property
    def num_unique_pinned_kv_blocks(
        self,
    ) -> int:
        return len(
            self.pinned_kv_block_ids
        )
        
    @property
    def current_pinned_kv_capacity_bytes(
        self,
    ) -> int:
        """
        Prefix Entries 占用的唯一 KV Block 容量。

        这不是新分配的 CUDA Tensor bytes，而是已经
        预分配的 KV Cache 中不能再供其他请求使用的容量。
        """

        return (
            self.num_unique_pinned_kv_blocks
            * self.kv_block_bytes
        )
        
    @property
    def current_prefix_cache_capacity_bytes(
        self,
    ) -> int:
        """
        Prefix Cache 的总容量成本：

            动态分配的 GDN Snapshot
            +
            被缓存占住的 Paged KV Block 容量
        """

        return (
            self.current_gdn_snapshot_bytes
            + self.current_pinned_kv_capacity_bytes
        )
        

    @property
    def entry_keys_lru_to_mru(
        self,
    ) -> tuple[PrefixKey, ...]:
        """
        返回从最久未使用到最近使用的 Entry keys。
        """

        return tuple(
            self.entries.keys()
        )
        
    def peek_lru_entry(
        self,
    ) -> PrefixStateEntry | None:
        """
        返回当前最久未使用 Entry，但不删除它。
        """

        if not self.entries:
            return None

        first_key = next(
            iter(self.entries)
        )

        return self.entries[first_key]
    
    @property
    def remaining_capacity_bytes(
        self,
    ) -> int:
        remaining = (
            self.capacity_bytes
            - self.current_prefix_cache_capacity_bytes
        )

        if remaining < 0:
            raise RuntimeError(
                "Prefix Cache capacity accounting "
                "became negative"
            )

        return remaining
    
    @property
    def capacity_utilization(
        self,
    ) -> float:
        return (
            self.current_prefix_cache_capacity_bytes
            / self.capacity_bytes
        )
        
    @property
    def num_admission_candidates(
        self,
    ) -> int:
        return len(
            self.admission_history
        )


    @property
    def admission_keys_lru_to_mru(
        self,
    ) -> tuple[PrefixKey, ...]:
        return tuple(
            self.admission_history.keys()
        )


    def admission_observation_count(
        self,
        key: PrefixKey,
    ) -> int:
        return self.admission_history.get(
            key,
            0,
        )