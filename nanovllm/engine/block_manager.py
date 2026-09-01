from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(
        self,
        block_id: int,
    ) -> None:
        self.block_id = block_id

        # 请求引用 + Prefix Entry 引用。
        self.ref_count = 0

        # ref_count 中有多少引用属于 Prefix Entry。
        self.cache_ref_count = 0

        self.hash = -1
        self.token_ids = []

    @property
    def request_ref_count(self) -> int:
        return (
            self.ref_count
            - self.cache_ref_count
        )

    def update(
        self,
        hash: int,
        token_ids: list[int],
    ) -> None:
        self.hash = hash
        self.token_ids = token_ids

    def reset(self) -> None:
        # 新分配的 block 首先属于一个请求。
        self.ref_count = 1
        self.cache_ref_count = 0
        self.hash = -1
        self.token_ids = []
        
class BlockManager:

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        enable_kv_prefix_lookup: bool = True,
        record_prefix_metadata: bool | None = None,
    ):
        self.block_size = block_size

        # 是否允许 can_allocate() 根据历史 Hash
        # 直接复用旧的物理 KV blocks。
        #
        # 普通 Qwen3：
        #     可以开启。
        #
        # 当前 Qwen3.5：
        #     必须关闭，因为只有 KV，没有 GDN state
        #     的命中是不完整的。
        self.enable_kv_prefix_lookup = (
            enable_kv_prefix_lookup
        )

        # 是否为已经计算完成的完整 KV blocks
        # 记录 token_ids 和链式 Hash。
        #
        # Qwen3.5 即使暂时不能进行 KV-only lookup，
        # 也需要这些元数据，用来建立联合
        # KV + GDN State Prefix Entry。
        if record_prefix_metadata is None:
            record_prefix_metadata = (
                enable_kv_prefix_lookup
            )

        self.record_prefix_metadata = (
            record_prefix_metadata
        )

        self.blocks: list[Block] = [
            Block(i)
            for i in range(num_blocks)
        ]

        self.hash_to_block_id: dict[
            int,
            int,
        ] = {}

        self.free_block_ids: deque[int] = deque(
            range(num_blocks)
        )

        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        assert block.cache_ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(
        self,
        block_id: int,
    ) -> None:
        block = self.blocks[block_id]

        assert block.ref_count == 0
        assert block.cache_ref_count == 0

        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def _normalize_block_ids(
        self,
        block_ids: (
            list[int]
            | tuple[int, ...]
        ),
    ) -> tuple[int, ...]:
        """
        将 block IDs 固定为不可变 tuple，并在修改
        引用计数前完成全部结构检查。
        """

        normalized_block_ids = tuple(
            block_ids
        )

        if not normalized_block_ids:
            raise ValueError(
                "block_ids must not be empty"
            )

        for block_id in normalized_block_ids:
            if not isinstance(block_id, int):
                raise TypeError(
                    "Every block ID must be an int"
                )

            if not 0 <= block_id < len(self.blocks):
                raise IndexError(
                    f"KV block ID {block_id} is outside "
                    f"[0, {len(self.blocks)})"
                )

        if (
            len(set(normalized_block_ids))
            != len(normalized_block_ids)
        ):
            raise ValueError(
                "block_ids must not contain duplicates"
            )

        return normalized_block_ids

    def pin_blocks(
        self,
        block_ids: (
            list[int]
            | tuple[int, ...]
        ),
    ) -> tuple[int, ...]:
        """
        为一个 Prefix Entry 增加 KV block
        cache-owner 引用。

        调用前，这些 blocks 必须已经属于至少一个
        活跃请求或其他缓存 Entry。
        """

        normalized_block_ids = (
            self._normalize_block_ids(
                block_ids
            )
        )

        # 第一遍只验证，不修改。
        for block_id in normalized_block_ids:
            block = self.blocks[block_id]

            if block_id not in self.used_block_ids:
                raise RuntimeError(
                    f"Cannot pin free KV block "
                    f"{block_id}"
                )

            if block.ref_count <= 0:
                raise RuntimeError(
                    f"Used KV block {block_id} has "
                    "non-positive ref_count"
                )

            if (
                block.cache_ref_count < 0
                or block.cache_ref_count
                > block.ref_count
            ):
                raise RuntimeError(
                    f"KV block {block_id} has invalid "
                    "cache reference accounting"
                )

        # 全部验证成功后再修改。
        for block_id in normalized_block_ids:
            block = self.blocks[block_id]

            block.ref_count += 1
            block.cache_ref_count += 1

        return normalized_block_ids

    def unpin_blocks(
        self,
        block_ids: (
            list[int]
            | tuple[int, ...]
        ),
    ) -> None:
        """
        释放一个 Prefix Entry 持有的 KV block
        cache-owner 引用。
        """

        normalized_block_ids = (
            self._normalize_block_ids(
                block_ids
            )
        )

        # 第一遍只验证。
        for block_id in normalized_block_ids:
            block = self.blocks[block_id]

            if block_id not in self.used_block_ids:
                raise RuntimeError(
                    f"Cannot unpin free KV block "
                    f"{block_id}"
                )

            if block.cache_ref_count <= 0:
                raise RuntimeError(
                    f"KV block {block_id} has no "
                    "Prefix Cache reference"
                )

            if (
                block.ref_count
                < block.cache_ref_count
            ):
                raise RuntimeError(
                    f"KV block {block_id} has invalid "
                    "reference accounting"
                )

        # 全部验证成功后再修改。
        for block_id in normalized_block_ids:
            block = self.blocks[block_id]

            block.cache_ref_count -= 1
            block.ref_count -= 1

            if block.ref_count == 0:
                self._deallocate_block(
                    block_id
                )

    def validate_prefix_blocks(
        self,
        block_ids: (
            list[int]
            | tuple[int, ...]
        ),
        expected_block_hash: int,
        *,
        require_request_owner: bool = True,
    ) -> tuple[int, ...]:
        """
        验证一组有序物理 KV blocks：

        1. 全部处于 used 状态；
        2. 至少有一个请求正在持有；
        3. 每个 block 都是完整 token block；
        4. 每一级链式 Hash 与 block.hash 一致；
        5. 最终 Hash 等于 PrefixKey.block_hash。
        """

        normalized_block_ids = (
            self._normalize_block_ids(
                block_ids
            )
        )

        if expected_block_hash < 0:
            raise ValueError(
                "expected_block_hash must be "
                "non-negative"
            )

        current_hash = -1

        for block_id in normalized_block_ids:
            block = self.blocks[block_id]

            if block_id not in self.used_block_ids:
                raise RuntimeError(
                    f"Prefix references free KV block "
                    f"{block_id}"
                )

            if require_request_owner:
                if block.request_ref_count <= 0:
                    raise RuntimeError(
                        f"KV block {block_id} is not owned "
                        "by an active request during commit"
                    )

            else:
                if block.cache_ref_count <= 0:
                    raise RuntimeError(
                        f"Resident Prefix KV block "
                        f"{block_id} has no cache-owner "
                        "reference"
                    )
                    
            if (
                len(block.token_ids)
                != self.block_size
            ):
                raise RuntimeError(
                    f"KV block {block_id} is not a "
                    "complete token block"
                )

            current_hash = self.compute_hash(
                block.token_ids,
                current_hash,
            )

            if block.hash != current_hash:
                raise RuntimeError(
                    f"KV block {block_id} has an "
                    "inconsistent chained hash"
                )

        if current_hash != expected_block_hash:
            raise RuntimeError(
                "KV block chain does not match "
                "PrefixKey.block_hash"
            )

        return normalized_block_ids

    def prefix_blocks_have_same_tokens(
        self,
        first_block_ids: (
            list[int]
            | tuple[int, ...]
        ),
        second_block_ids: (
            list[int]
            | tuple[int, ...]
        ),
    ) -> bool:
        """
        在 Hash 相同后，逐 block 比较真实 token IDs。
        """

        first_ids = self._normalize_block_ids(
            first_block_ids
        )

        second_ids = self._normalize_block_ids(
            second_block_ids
        )

        if len(first_ids) != len(second_ids):
            return False

        return all(
            self.blocks[first_id].token_ids
            == self.blocks[second_id].token_ids
            for first_id, second_id
            in zip(first_ids, second_ids)
        )

    def prefix_metadata_at_boundary(
        self,
        seq: Sequence,
        num_cached_tokens: int,
    ) -> tuple[
        tuple[int, ...],
        int,
    ]:
        """
        读取某个已经计算完成的完整 token-block
        边界对应的：

            1. 有序物理 KV block IDs
            2. 最终链式 block hash

        该方法只读取和验证，不增加引用计数。
        """

        if num_cached_tokens <= 0:
            raise ValueError(
                "Prefix boundary must be positive"
            )

        if (
            num_cached_tokens
            % self.block_size
            != 0
        ):
            raise ValueError(
                "Prefix boundary must be aligned "
                "to block_size"
            )

        computed_end = (
            seq.num_cached_tokens
            + seq.num_scheduled_tokens
        )

        if num_cached_tokens > computed_end:
            raise ValueError(
                "Prefix boundary has not been "
                "computed yet"
            )

        if num_cached_tokens > seq.num_tokens:
            raise ValueError(
                "Prefix boundary exceeds Sequence "
                "token count"
            )

        num_blocks = (
            num_cached_tokens
            // self.block_size
        )

        if num_blocks > len(seq.block_table):
            raise RuntimeError(
                "Sequence block table is shorter "
                "than the requested prefix boundary"
            )

        kv_block_ids = tuple(
            seq.block_table[:num_blocks]
        )

        if not kv_block_ids:
            raise RuntimeError(
                "Prefix boundary does not contain "
                "any KV blocks"
            )

        last_block_id = kv_block_ids[-1]

        block_hash = self.blocks[
            last_block_id
        ].hash

        if block_hash < 0:
            raise RuntimeError(
                "Prefix block metadata has not been "
                "recorded at this boundary"
            )

        validated_block_ids = (
            self.validate_prefix_blocks(
                kv_block_ids,
                expected_block_hash=block_hash,
            )
        )

        return (
            validated_block_ids,
            block_hash,
        )

    def _validate_prefix_attachment(
        self,
        seq: Sequence,
        prefix_block_ids: (
            list[int]
            | tuple[int, ...]
        ),
    ) -> tuple[int, ...]:
        """
        校验一组 resident Prefix KV blocks 能否作为
        新请求 block_table 的开头。

        只验证，不修改引用计数和 Sequence。
        """

        if seq.block_table:
            raise RuntimeError(
                f"Sequence {seq.seq_id} already owns "
                "KV blocks"
            )

        normalized_block_ids = (
            self._normalize_block_ids(
                prefix_block_ids
            )
        )

        num_cached_blocks = len(
            normalized_block_ids
        )

        # Entry 不保存 logits，所以合法命中必须保证
        # 新请求至少还有一个 Prompt token 真正执行。
        #
        # 因此缓存 Block 数必须严格小于请求总 Block 数。
        if num_cached_blocks >= seq.num_blocks:
            raise ValueError(
                "Prefix attachment must leave at "
                "least one uncached logical block"
            )

        last_block_hash = self.blocks[
            normalized_block_ids[-1]
        ].hash

        if last_block_hash < 0:
            raise RuntimeError(
                "Resident Prefix KV chain does not "
                "contain a valid final hash"
            )

        # 这里校验的是 resident Entry，不再要求源请求
        # 仍然持有这些 Blocks。
        self.validate_prefix_blocks(
            normalized_block_ids,
            expected_block_hash=(
                last_block_hash
            ),
            require_request_owner=False,
        )

        # Hash 命中后，仍逐 Block 对比真实 token IDs。
        for (
            logical_block_index,
            physical_block_id,
        ) in enumerate(
            normalized_block_ids
        ):
            expected_token_ids = seq.block(
                logical_block_index
            )

            actual_token_ids = self.blocks[
                physical_block_id
            ].token_ids

            if actual_token_ids != expected_token_ids:
                raise RuntimeError(
                    "Prefix KV block tokens do not "
                    "match the target Sequence"
                )

        return normalized_block_ids

    def can_allocate_from_prefix(
        self,
        seq: Sequence,
        prefix_block_ids: (
            list[int]
            | tuple[int, ...]
        ),
    ) -> bool:
        """
        检查复用 Prefix KV blocks 后，剩余逻辑
        Blocks 是否能从 free_block_ids 中分配。
        """

        normalized_block_ids = (
            self._validate_prefix_attachment(
                seq,
                prefix_block_ids,
            )
        )

        num_new_blocks = (
            seq.num_blocks
            - len(normalized_block_ids)
        )

        return (
            len(self.free_block_ids)
            >= num_new_blocks
        )

    def allocate_from_prefix(
        self,
        seq: Sequence,
        prefix_block_ids: (
            list[int]
            | tuple[int, ...]
        ),
    ) -> int:
        """
        将 resident Prefix KV blocks 作为新请求
        block_table 的开头，并分配剩余 Blocks。

        返回实际复用的完整 KV block 数量。
        """

        normalized_block_ids = (
            self._validate_prefix_attachment(
                seq,
                prefix_block_ids,
            )
        )

        num_cached_blocks = len(
            normalized_block_ids
        )

        num_new_blocks = (
            seq.num_blocks
            - num_cached_blocks
        )

        if (
            len(self.free_block_ids)
            < num_new_blocks
        ):
            raise RuntimeError(
                "Insufficient free KV blocks for "
                "Prefix attachment"
            )

        # =====================================
        # 第一阶段：给命中 Blocks 增加请求引用
        # =====================================

        for block_id in normalized_block_ids:
            block = self.blocks[block_id]

            # cache_ref_count 保持不变。
            #
            # ref_count 增加 1 后：
            #
            # request_ref_count
            # = ref_count - cache_ref_count
            # 会自然增加 1。
            block.ref_count += 1

        # =====================================
        # 第二阶段：构造新请求 block_table
        # =====================================

        seq.block_table.extend(
            normalized_block_ids
        )

        for _ in range(num_new_blocks):
            seq.block_table.append(
                self._allocate_block()
            )

        # 这个请求已经具备相同边界的：
        #
        # 1. Full Attention KV
        # 2. 即将在 Engine 中恢复的 GDN state
        seq.num_cached_tokens = (
            num_cached_blocks
            * self.block_size
        )

        return num_cached_blocks


    def can_allocate(self, seq: Sequence) -> int:
        
        if not self.enable_kv_prefix_lookup:
        # 不查找任何历史 hash。
        #
        # 请求需要的所有逻辑块都必须重新分配。
            if (
            len(self.free_block_ids)
            < seq.num_blocks
        ):
                return -1

        # 0 表示复用了 0 个缓存 block。
            return 0 
        
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                if block.cache_ref_count != 0:
                    raise RuntimeError(
                        f"Free KV block {block_id} still has "
                        "Prefix Cache references"
                    )
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(
        self,
        seq: Sequence,
    ) -> None:
        block_ids = tuple(
            reversed(seq.block_table)
        )

        if (
            len(set(block_ids))
            != len(block_ids)
        ):
            raise RuntimeError(
                f"Sequence {seq.seq_id} contains "
                "duplicate KV block IDs"
            )

        # 第一遍验证每个 block 确实还有请求引用。
        for block_id in block_ids:
            if not 0 <= block_id < len(self.blocks):
                raise IndexError(
                    f"Sequence {seq.seq_id} references "
                    f"invalid KV block {block_id}"
                )

            block = self.blocks[block_id]

            if block_id not in self.used_block_ids:
                raise RuntimeError(
                    f"Sequence {seq.seq_id} references "
                    f"free KV block {block_id}"
                )

            if block.request_ref_count <= 0:
                raise RuntimeError(
                    f"KV block {block_id} has no request "
                    f"reference for Sequence {seq.seq_id}"
                )

        # 全部合法后，再释放该请求的引用。
        for block_id in block_ids:
            block = self.blocks[block_id]

            block.ref_count -= 1

            if block.ref_count == 0:
                self._deallocate_block(
                    block_id
                )

        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        
        if not self.record_prefix_metadata:
            return
        
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
