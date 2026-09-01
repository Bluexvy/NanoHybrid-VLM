from dataclasses import dataclass
from math import prod
from collections import deque

import torch

# slots=True 表示这个对象只允许存在声明过的属性
@dataclass(slots=True)
class GDNLayerState:
    """
    一条 Sequence 在某一个 GDN 层上的状态。
    """

    conv_state: torch.Tensor | None = None

    recurrent_state: torch.Tensor | None = None
    
    
class StateSlotAllocator:
    """
    只管理 state slot 整数编号。

    不持有 GPU Tensor。
    """

    def __init__(
        self,
        num_slots: int,
    ) -> None:

        if num_slots <= 0:
            raise ValueError(
                "num_slots must be positive"
            )

        self.num_slots = num_slots

        self.free_slots: deque[int] = deque(
            range(num_slots)
        )

        self.used_slots: set[int] = set()

    @property
    def num_free_slots(self) -> int:
        return len(self.free_slots)

    @property
    def num_used_slots(self) -> int:
        return len(self.used_slots)

    def can_allocate(self) -> bool:
        return bool(self.free_slots)

    def allocate(self) -> int:
        if not self.free_slots:
            raise RuntimeError(
                "No free GDN state slots"
            )

        slot = self.free_slots.popleft()

        if slot in self.used_slots:
            raise RuntimeError(
                f"State slot {slot} is already used"
            )

        self.used_slots.add(slot)

        return slot

    def release(
        self,
        slot: int,
    ) -> None:

        if slot not in self.used_slots:
            raise RuntimeError(
                f"State slot {slot} is not allocated"
            )

        self.used_slots.remove(slot)
        self.free_slots.append(slot)


@dataclass(frozen=True, slots=True)
class HybridCacheSpec:
    """
    从 Qwen3.5 text_config 推导出的
    KV Cache 和 GDN State 形状。
    """
    
    """
    模型有多少层
    哪些层是 GDN
    哪些层是 Full Attention
    它们之间如何映射
    KV Cache 每层是什么形状
    conv_state 是什么形状
    recurrent_state 是什么形状
    使用什么 dtype
    每个 slot 需要多少字节
    一个 KV block 需要多少字节
    """

    num_hidden_layers: int

    # 全局 DecoderLayer 编号。
    full_attention_layer_indices: tuple[int, ...]
    gdn_layer_indices: tuple[int, ...]

    # global layer_idx -> compact cache/state index。
    #
    # 不属于相应类型的层保存 -1。
    full_attention_index_by_layer: tuple[int, ...]
    gdn_index_by_layer: tuple[int, ...]

    # Full Attention KV Cache 参数。
    num_kv_heads: int
    attention_head_dim: int

    # GDN conv_state 参数。
    conv_dim: int
    conv_kernel_size: int

    # GDN recurrent_state 参数。
    num_gdn_value_heads: int
    gdn_key_head_dim: int
    gdn_value_head_dim: int

    # conv_state 通常使用模型 BF16 dtype。
    conv_dtype: torch.dtype

    # recurrent state 按方案固定使用 FP32。
    recurrent_dtype: torch.dtype = torch.float32

    @classmethod
    def from_text_config(
        cls,
        text_config,
        tensor_parallel_size: int = 1,
    ) -> "HybridCacheSpec":

        # 当前 Qwen3.5 Runtime 明确只支持 TP=1。
        if tensor_parallel_size != 1:
            raise NotImplementedError(
                "Qwen3.5 Hybrid Runtime currently "
                "supports only tensor_parallel_size=1"
            )

        # 区分 GDN 层和 Attention 层
        layer_types = tuple(
            text_config.layer_types
        )

        if (
            len(layer_types)
            != text_config.num_hidden_layers
        ):
            raise ValueError(
                "layer_types length must equal "
                "num_hidden_layers"
            )

        supported_layer_types = {
            "linear_attention",
            "full_attention",
        }

        unsupported_layer_types = (
            set(layer_types)
            - supported_layer_types
        )

        if unsupported_layer_types:
            raise ValueError(
                "Unsupported Qwen3.5 layer types: "
                f"{sorted(unsupported_layer_types)}"
            )

        # 找出full attention 和 GDN
        full_attention_layer_indices = tuple(
            layer_idx
            for layer_idx, layer_type
            in enumerate(layer_types)
            if layer_type == "full_attention"
        )

        gdn_layer_indices = tuple(
            layer_idx
            for layer_idx, layer_type
            in enumerate(layer_types)
            if layer_type == "linear_attention"
        )

        if not full_attention_layer_indices:
            raise ValueError(
                "Qwen3.5 must contain at least one "
                "Full Attention layer"
            )

        if not gdn_layer_indices:
            raise ValueError(
                "Qwen3.5 must contain at least one "
                "GDN layer"
            )
        # -1 表示：这个全局层不属于当前类型
        full_attention_index_by_layer = [
            -1
            for _ in range(
                text_config.num_hidden_layers
            )
        ]
        # 保存正反两种映射 已知 compact index，寻找全局层；已知全局层，寻找 Cache 位置
        for compact_idx, layer_idx in enumerate(
            full_attention_layer_indices
        ):
            full_attention_index_by_layer[
                layer_idx
            ] = compact_idx

        gdn_index_by_layer = [
            -1
            for _ in range(
                text_config.num_hidden_layers
            )
        ]

        for compact_idx, layer_idx in enumerate(
            gdn_layer_indices
        ):
            gdn_index_by_layer[
                layer_idx
            ] = compact_idx

        attention_head_dim = getattr(
            text_config,
            "head_dim",
            None,
        )

        if attention_head_dim is None:
            attention_head_dim = (
                text_config.hidden_size
                // text_config.num_attention_heads
            )

        key_dim = (
            text_config.linear_num_key_heads
            * text_config.linear_key_head_dim
        )

        value_dim = (
            text_config.linear_num_value_heads
            * text_config.linear_value_head_dim
        )

        # mixed_qkv = Q + K + V
        conv_dim = (
            key_dim
            + key_dim
            + value_dim
        )

        model_dtype = text_config.dtype

        if not isinstance(model_dtype, torch.dtype):
            raise TypeError(
                "text_config.dtype must be a "
                "torch.dtype"
            )

        return cls(
            num_hidden_layers=(
                text_config.num_hidden_layers
            ),
            full_attention_layer_indices=(
                full_attention_layer_indices
            ),
            gdn_layer_indices=gdn_layer_indices,
            full_attention_index_by_layer=tuple(
                full_attention_index_by_layer
            ),
            gdn_index_by_layer=tuple(
                gdn_index_by_layer
            ),
            num_kv_heads=(
                text_config.num_key_value_heads
            ),
            attention_head_dim=(
                attention_head_dim
            ),
            conv_dim=conv_dim,
            conv_kernel_size=(
                text_config.linear_conv_kernel_dim
            ),
            num_gdn_value_heads=(
                text_config.linear_num_value_heads
            ),
            gdn_key_head_dim=(
                text_config.linear_key_head_dim
            ),
            gdn_value_head_dim=(
                text_config.linear_value_head_dim
            ),
            conv_dtype=model_dtype,
            recurrent_dtype=torch.float32,
        )

    @property
    def num_full_attention_layers(self) -> int:
        return len(
            self.full_attention_layer_indices
        )

    @property
    def num_gdn_layers(self) -> int:
        return len(
            self.gdn_layer_indices
        )


    @property
    def conv_state_shape_per_slot(
        self,
    ) -> tuple[int, ...]:
        # 一个slot就代表一条Sequence 不需要再单独写batch=1
        """一条请求 slot的全部 conv state"""
        return (
            self.num_gdn_layers,
            self.conv_dim,
            self.conv_kernel_size,
        )

    @property
    def recurrent_state_shape_per_slot(
        self,
    ) -> tuple[int, ...]:
        return (
            self.num_gdn_layers,
            self.num_gdn_value_heads,
            self.gdn_key_head_dim,
            self.gdn_value_head_dim,
        )

    @staticmethod
    def dtype_nbytes(
        dtype: torch.dtype,
    ) -> int:
        """创建一个指定 dtype 的标量 Tensor，然后询问：每个元素占多少字节"""
        return torch.empty(
            (),
            dtype=dtype,
        ).element_size()

    @property
    def conv_state_bytes_per_slot(self) -> int:
        """每个 slot 的 conv state 占多少显存"""
        return (
            # prod 会把一组数字全部相乘
            prod(self.conv_state_shape_per_slot)
            * self.dtype_nbytes(
                self.conv_dtype
            )
        )

    @property
    def recurrent_state_bytes_per_slot(
        self,
    ) -> int:
        """每个 slot 的 recurrent state 占多少显存"""
        return (
            prod(
                self.recurrent_state_shape_per_slot
            )
            * self.dtype_nbytes(
                self.recurrent_dtype
            )
        )

    @property
    def state_bytes_per_slot(self) -> int:
        return (
            self.conv_state_bytes_per_slot
            + self.recurrent_state_bytes_per_slot
        )

    def kv_block_bytes(
        self,
        block_size: int,
    ) -> int:
        """
        一个逻辑 KV block 在所有 Full Attention
        层上占用的总字节数。
        """

        if block_size <= 0:
            raise ValueError(
                "block_size must be positive"
            )

        return (
            2  # K 和 V
            * self.num_full_attention_layers
            * block_size
            * self.num_kv_heads
            * self.attention_head_dim
            * self.dtype_nbytes(
                self.conv_dtype
            )
        )
        
        
class HybridStateManager:
    """
    管理所有 GDN 层的 GPU 状态 Tensor。

    这里只管理 slot 对应的 Tensor，
    slot 的调度与分配稍后由 Scheduler 负责。
    """

    def __init__(
        self,
        spec: HybridCacheSpec,
        num_slots: int,
        device: torch.device | str,
    ) -> None:
        """在 GPU 上创建可以容纳 num_slots 条活跃 Sequence 的 GDN 状态池"""
        if num_slots <= 0:
            raise ValueError(
                "num_slots must be positive"
            )

        self.spec = spec
        self.num_slots = num_slots
        self.device = torch.device(device)

        # [slot, gdn_layer, conv_dim, kernel]
        self.conv_state_pool = torch.empty(
            (
                # * 代表把 tuple 展开
                num_slots,
                *spec.conv_state_shape_per_slot,
            ),
            dtype=spec.conv_dtype,
            device=self.device,
        )

        # [slot, gdn_layer, heads, Dk, Dv]
        self.recurrent_state_pool = torch.empty(
            (
                num_slots,
                *spec.recurrent_state_shape_per_slot,
            ),
            dtype=spec.recurrent_dtype,
            device=self.device,
        )

        # torch.empty() 中是未初始化数据。
        #
        # 只有 initialized_slots[slot] 为 True 时，
        # 才允许读取对应状态。
        self.initialized_slots = [
            False
            for _ in range(num_slots)
        ]

    def _validate_slot(
        self,
        slot: int,
    ) -> None:
        if not 0 <= slot < self.num_slots:
            raise IndexError(
                f"State slot {slot} is outside "
                f"[0, {self.num_slots})"
            )
            
    def _prepare_slot_indices(
        self,
        slots: list[int],
    ) -> tuple[tuple[int, ...], torch.Tensor]:

        if not slots:
            raise ValueError(
                "slots must not be empty"
            )

        normalized_slots = tuple(slots)

        for slot in normalized_slots:
            if not isinstance(slot, int):
                raise TypeError(
                    "Every state slot must be an int"
                )

            self._validate_slot(slot)

        if (
            len(set(normalized_slots))
            != len(normalized_slots)
        ):
            raise ValueError(
                "A batched state operation cannot "
                "contain duplicate slots"
            )

        slot_indices = torch.tensor(
            normalized_slots,
            dtype=torch.long,
            device=self.device,
        )

        return normalized_slots, slot_indices

    def is_slot_initialized(
        self,
        slot: int,
    ) -> bool:
        self._validate_slot(slot)

        return self.initialized_slots[slot]

    def reset_slot(
        self,
        slot: int,
    ) -> None:
        """
        将 slot 标记为新请求状态。

        不需要立即 zero_()，因为未初始化 slot
        不会被读取。首次 Prefill 会向模型传 None。
        """

        self._validate_slot(slot)
        self.initialized_slots[slot] = False

    @torch.inference_mode()
    def snapshot_slot(
        self,
        slot: int,
        recurrent_snapshot_dtype: torch.dtype,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        为一个已经初始化的 active state slot
        创建独立的 GDN Prefix Snapshot。

        conv snapshot 始终保持 spec.conv_dtype。

        recurrent snapshot 支持：
            torch.float32：正确性模式
            torch.bfloat16：压缩模式
        """

        self._validate_slot(slot)

        supported_snapshot_dtypes = {
            torch.float32,
            torch.bfloat16,
        }

        if (
            recurrent_snapshot_dtype
            not in supported_snapshot_dtypes
        ):
            raise ValueError(
                "recurrent_snapshot_dtype must be "
                "torch.float32 or torch.bfloat16"
            )

        if not self.initialized_slots[slot]:
            raise RuntimeError(
                f"Cannot snapshot uninitialized "
                f"GDN state slot {slot}"
            )

        conv_state_snapshot = (
            self.conv_state_pool[slot]
            .detach()
            .clone()
        )

        recurrent_state_source = (
            self.recurrent_state_pool[slot]
            .detach()
        )

        if (
            recurrent_snapshot_dtype
            == recurrent_state_source.dtype
        ):
            recurrent_state_snapshot = (
                recurrent_state_source.clone()
            )
        else:
            recurrent_state_snapshot = (
                recurrent_state_source.to(
                    dtype=recurrent_snapshot_dtype
                )
            )

        return (
            conv_state_snapshot,
            recurrent_state_snapshot,
        )
        
    @torch.inference_mode()
    def restore_slot(
        self,
        slot: int,
        conv_state_snapshot: torch.Tensor,
        recurrent_state_snapshot: torch.Tensor,
    ) -> None:
        """
        将一个独立的 GDN Prefix Snapshot 恢复到
        active state slot。

        conv snapshot 必须与 active conv dtype 一致。

        recurrent snapshot 可以是 FP32 或 BF16；
        copy_() 会在写入 FP32 active pool 时完成转换。
        """

        self._validate_slot(slot)

        expected_conv_shape = (
            self.spec.conv_state_shape_per_slot
        )

        if (
            tuple(conv_state_snapshot.shape)
            != expected_conv_shape
        ):
            raise ValueError(
                "Invalid conv snapshot shape: "
                f"expected {expected_conv_shape}, "
                f"got {tuple(conv_state_snapshot.shape)}"
            )

        expected_recurrent_shape = (
            self.spec
            .recurrent_state_shape_per_slot
        )

        if (
            tuple(recurrent_state_snapshot.shape)
            != expected_recurrent_shape
        ):
            raise ValueError(
                "Invalid recurrent snapshot shape: "
                f"expected {expected_recurrent_shape}, "
                "got "
                f"{tuple(recurrent_state_snapshot.shape)}"
            )

        if (
            conv_state_snapshot.dtype
            != self.spec.conv_dtype
        ):
            raise TypeError(
                "conv snapshot dtype does not match "
                "active conv state dtype"
            )

        supported_recurrent_dtypes = {
            torch.float32,
            torch.bfloat16,
        }

        if (
            recurrent_state_snapshot.dtype
            not in supported_recurrent_dtypes
        ):
            raise TypeError(
                "recurrent snapshot must use "
                "torch.float32 or torch.bfloat16"
            )

        if (
            conv_state_snapshot.device
            != self.device
        ):
            raise ValueError(
                "conv snapshot must be on the same "
                "device as the active state pool"
            )

        if (
            recurrent_state_snapshot.device
            != self.device
        ):
            raise ValueError(
                "recurrent snapshot must be on the same "
                "device as the active state pool"
            )

        if conv_state_snapshot.requires_grad:
            raise ValueError(
                "conv snapshot must not require grad"
            )

        if recurrent_state_snapshot.requires_grad:
            raise ValueError(
                "recurrent snapshot must not require grad"
            )

        # 在两份 Tensor 都成功写入前，不能把 slot
        # 标记成可读取状态。
        self.initialized_slots[slot] = False

        try:
            self.conv_state_pool[slot].copy_(
                conv_state_snapshot
            )

            # active recurrent pool 是 FP32。
            #
            # 如果 Snapshot 是 BF16，copy_() 会在这里
            # 将其转换回 FP32。
            self.recurrent_state_pool[slot].copy_(
                recurrent_state_snapshot
            )

        except Exception:
            self.initialized_slots[slot] = False
            raise

        self.initialized_slots[slot] = True

    def read_states(
        self,
        slot: int,
    ) -> list[GDNLayerState | None]:
        """
        返回一条 Sequence 在全部 DecoderLayer
        上的 GDN state view。
        """

        self._validate_slot(slot)

        # 首次 Prefill：所有 GDN 层都从 None 开始。
        if not self.initialized_slots[slot]:
            return [
                None
                for _ in range(
                    self.spec.num_hidden_layers
                )
            ]

        states: list[
            GDNLayerState | None
        ] = [
            None
            for _ in range(
                self.spec.num_hidden_layers
            )
        ]

        for (
            gdn_index,
            global_layer_idx,
        ) in enumerate(
            self.spec.gdn_layer_indices
        ):
            # 池中没有 batch 维。
            #
            # GDNLayer 当前需要：
            # conv      [1,C,K]
            # recurrent [1,H,Dk,Dv]
            states[global_layer_idx] = (
                GDNLayerState(
                    conv_state=(
                        self.conv_state_pool[
                            slot,
                            gdn_index,
                        ].unsqueeze(0)
                    ),
                    recurrent_state=(
                        self.recurrent_state_pool[
                            slot,
                            gdn_index,
                        ].unsqueeze(0)
                    ),
                )
            )

        return states

    def read_batched_states(
        self,
        slots: list[int],
    ) -> list[GDNLayerState | None]:
        """
        按 slots 给出的顺序 Gather 多条 Sequence
        的全部 GDN states。

        返回的 GDN state 具有 batch 维：
            conv      [B, C, K]
            recurrent [B, H, Dk, Dv]
        """

        (
            normalized_slots,
            slot_indices,
        ) = self._prepare_slot_indices(slots)

        batch_size = len(normalized_slots)

        states: list[
            GDNLayerState | None
        ] = [
            None
            for _ in range(
                self.spec.num_hidden_layers
            )
        ]

        initialized = [
            self.initialized_slots[slot]
            for slot in normalized_slots
        ]

        # 只在 batch 中存在新请求时创建 mask。
        initialized_mask = None

        if not all(initialized):
            initialized_mask = torch.tensor(
                initialized,
                dtype=torch.bool,
                device=self.device,
            )

        for (
            gdn_index,
            global_layer_idx,
        ) in enumerate(
            self.spec.gdn_layer_indices
        ):
            batched_conv_state = torch.index_select(
                self.conv_state_pool[
                    :,
                    gdn_index,
                ],
                dim=0,
                index=slot_indices,
            )

            batched_recurrent_state = (
                torch.index_select(
                    self.recurrent_state_pool[
                        :,
                        gdn_index,
                    ],
                    dim=0,
                    index=slot_indices,
                )
            )

            if initialized_mask is not None:
                # torch.empty() 中未初始化 slot 的内容不能读取。
                #
                # 对新请求，将 Gather 出来的对应行清零。
                batched_conv_state.masked_fill_(
                    ~initialized_mask.view(
                        batch_size,
                        1,
                        1,
                    ),
                    0,
                )

                batched_recurrent_state.masked_fill_(
                    ~initialized_mask.view(
                        batch_size,
                        1,
                        1,
                        1,
                    ),
                    0,
                )

            states[global_layer_idx] = (
                GDNLayerState(
                    conv_state=batched_conv_state,
                    recurrent_state=(
                        batched_recurrent_state
                    ),
                )
            )

        return states


    def write_states(
        self,
        slot: int,
        states: list[
            GDNLayerState | None
        ],
    ) -> None:
        """
        把一次模型前向得到的最终 GDN states
        写回固定 GPU slot。
        """

        self._validate_slot(slot)

        if (
            len(states)
            != self.spec.num_hidden_layers
        ):
            raise ValueError(
                "states must contain one entry "
                "per DecoderLayer"
            )

        expected_conv_shape = (
            1,
            self.spec.conv_dim,
            self.spec.conv_kernel_size,
        )

        expected_recurrent_shape = (
            1,
            self.spec.num_gdn_value_heads,
            self.spec.gdn_key_head_dim,
            self.spec.gdn_value_head_dim,
        )

        for (
            gdn_index,
            global_layer_idx,
        ) in enumerate(
            self.spec.gdn_layer_indices
        ):
            state = states[global_layer_idx]

            if (
                state is None
                or state.conv_state is None
                or state.recurrent_state is None
            ):
                raise ValueError(
                    "GDN layer "
                    f"{global_layer_idx} did not "
                    "return complete state"
                )

            if (
                tuple(state.conv_state.shape)
                != expected_conv_shape
            ):
                raise ValueError(
                    "Invalid conv_state shape for "
                    f"layer {global_layer_idx}: "
                    f"{tuple(state.conv_state.shape)}"
                )

            if (
                tuple(state.recurrent_state.shape)
                != expected_recurrent_shape
            ):
                raise ValueError(
                    "Invalid recurrent_state shape "
                    f"for layer {global_layer_idx}: "
                    f"{tuple(state.recurrent_state.shape)}"
                )

            if (
                state.conv_state.device
                != self.device
            ):
                raise ValueError(
                    "conv_state is on the wrong "
                    "device"
                )

            if (
                state.recurrent_state.device
                != self.device
            ):
                raise ValueError(
                    "recurrent_state is on the "
                    "wrong device"
                )

            if (
                state.conv_state.dtype
                != self.spec.conv_dtype
            ):
                raise ValueError(
                    "conv_state has the wrong dtype"
                )

            if (
                state.recurrent_state.dtype
                != self.spec.recurrent_dtype
            ):
                raise ValueError(
                    "recurrent_state must use FP32"
                )

            self.conv_state_pool[
                slot,
                gdn_index,
            ].copy_(
                state.conv_state[0]
            )

            self.recurrent_state_pool[
                slot,
                gdn_index,
            ].copy_(
                state.recurrent_state[0]
            )

        # 只有全部 18 个 GDN 层都成功写入后，
        # 才将 slot 标记为可读。
        self.initialized_slots[slot] = True

    def write_batched_states(
        self,
        slots: list[int],
        states: list[
            GDNLayerState | None
        ],
    ) -> None:
        """
        把一次 Batched 模型前向得到的最终状态
        Scatter 回对应的固定 state slots。
        """

        (
            normalized_slots,
            slot_indices,
        ) = self._prepare_slot_indices(slots)

        batch_size = len(normalized_slots)

        if (
            len(states)
            != self.spec.num_hidden_layers
        ):
            raise ValueError(
                "states must contain one entry "
                "per DecoderLayer"
            )

        # Full Attention 层不应该返回 GDN 状态。
        for global_layer_idx in (
            self.spec.full_attention_layer_indices
        ):
            if states[global_layer_idx] is not None:
                raise ValueError(
                    "Full Attention layer "
                    f"{global_layer_idx} returned "
                    "a GDN state"
                )

        expected_conv_shape = (
            batch_size,
            self.spec.conv_dim,
            self.spec.conv_kernel_size,
        )

        expected_recurrent_shape = (
            batch_size,
            self.spec.num_gdn_value_heads,
            self.spec.gdn_key_head_dim,
            self.spec.gdn_value_head_dim,
        )

        validated_states = []

        # 第一遍只做验证，暂时不修改状态池。
        for (
            gdn_index,
            global_layer_idx,
        ) in enumerate(
            self.spec.gdn_layer_indices
        ):
            state = states[global_layer_idx]

            if (
                state is None
                or state.conv_state is None
                or state.recurrent_state is None
            ):
                raise ValueError(
                    "GDN layer "
                    f"{global_layer_idx} did not "
                    "return complete batched state"
                )

            if (
                tuple(state.conv_state.shape)
                != expected_conv_shape
            ):
                raise ValueError(
                    "Invalid batched conv_state shape "
                    f"for layer {global_layer_idx}: "
                    f"{tuple(state.conv_state.shape)}"
                )

            if (
                tuple(state.recurrent_state.shape)
                != expected_recurrent_shape
            ):
                raise ValueError(
                    "Invalid batched recurrent_state "
                    f"shape for layer "
                    f"{global_layer_idx}: "
                    f"{tuple(state.recurrent_state.shape)}"
                )

            if state.conv_state.device != self.device:
                raise ValueError(
                    "batched conv_state is on the "
                    "wrong device"
                )

            if (
                state.recurrent_state.device
                != self.device
            ):
                raise ValueError(
                    "batched recurrent_state is on "
                    "the wrong device"
                )

            if (
                state.conv_state.dtype
                != self.spec.conv_dtype
            ):
                raise ValueError(
                    "batched conv_state has the "
                    "wrong dtype"
                )

            if (
                state.recurrent_state.dtype
                != self.spec.recurrent_dtype
            ):
                raise ValueError(
                    "batched recurrent_state must "
                    "use FP32"
                )

            validated_states.append(
                (
                    gdn_index,
                    state,
                )
            )

        # 所有层都验证成功后，才开始真正写入状态池。
        for gdn_index, state in validated_states:
            self.conv_state_pool[
                :,
                gdn_index,
            ].index_copy_(
                0,
                slot_indices,
                state.conv_state,
            )

            self.recurrent_state_pool[
                :,
                gdn_index,
            ].index_copy_(
                0,
                slot_indices,
                state.recurrent_state,
            )

        # 只有全部层都写回成功后，才允许后续读取。
        for slot in normalized_slots:
            self.initialized_slots[slot] = True

    @property
    def allocated_bytes(self) -> int:
        return (
            self.conv_state_pool.numel()
            * self.conv_state_pool.element_size()
            + self.recurrent_state_pool.numel()
            * self.recurrent_state_pool.element_size()
        )