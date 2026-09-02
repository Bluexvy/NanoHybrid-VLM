from __future__ import annotations

from dataclasses import dataclass

import torch

from nanovllm.engine.hybrid_state import (
    GDNLayerState,
    HybridCacheSpec,
)


@dataclass(slots=True)
class HybridDecodeStaticWorkspace:
    """
    Qwen3.5 Hybrid Decode CUDA Graph 使用的
    固定地址 GPU Tensor。

    Workspace 按最大 Graph batch 分配一次。
    不同 bucket 通过 [:batch_size] 获得固定 Shape View。
    """

    spec: HybridCacheSpec

    max_batch_size: int
    max_num_blocks: int
    hidden_size: int
    device: torch.device

    # Decode 基础输入。
    input_ids: torch.Tensor
    positions: torch.Tensor

    # Full Attention/Paged KV 元数据。
    slot_mapping: torch.Tensor
    context_lens: torch.Tensor
    block_tables: torch.Tensor

    # GDN 请求状态槽编号。
    state_slot_ids: torch.Tensor

    # Graph 使用的 GDN 输入状态。
    #
    # [max_batch, num_gdn_layers, C, K]
    gdn_conv_states: torch.Tensor

    # [max_batch, num_gdn_layers, H, Dk, Dv]
    gdn_recurrent_states: torch.Tensor

    # Graph 输出状态。
    updated_gdn_conv_states: torch.Tensor
    updated_gdn_recurrent_states: torch.Tensor

    # Graph 最终 Decoder hidden states。
    hidden_states: torch.Tensor

    @classmethod
    def allocate(
        cls,
        *,
        spec: HybridCacheSpec,
        max_batch_size: int,
        max_num_blocks: int,
        hidden_size: int,
        device: torch.device | str,
    ) -> "HybridDecodeStaticWorkspace":
        """
        一次性分配全部固定地址 Tensor。
        """

        if max_batch_size <= 0:
            raise ValueError(
                "max_batch_size must be positive"
            )

        if max_num_blocks <= 0:
            raise ValueError(
                "max_num_blocks must be positive"
            )

        if hidden_size <= 0:
            raise ValueError(
                "hidden_size must be positive"
            )

        normalized_device = torch.device(
            device
        )

        input_ids = torch.zeros(
            max_batch_size,
            dtype=torch.int64,
            device=normalized_device,
        )

        # Qwen3.5 Decode 使用 T/H/W 三轴位置。
        positions = torch.zeros(
            3,
            max_batch_size,
            dtype=torch.int64,
            device=normalized_device,
        )

        # -1 表示当前行不应该写入任何 KV slot。
        slot_mapping = torch.full(
            (max_batch_size,),
            -1,
            dtype=torch.int32,
            device=normalized_device,
        )

        context_lens = torch.zeros(
            max_batch_size,
            dtype=torch.int32,
            device=normalized_device,
        )

        # -1 表示不存在对应物理 KV block。
        block_tables = torch.full(
            (
                max_batch_size,
                max_num_blocks,
            ),
            -1,
            dtype=torch.int32,
            device=normalized_device,
        )

        # 首版 Gather/Scatter 在 Graph 外，
        # 但仍然提前固定 state_slot_ids 接口。
        state_slot_ids = torch.full(
            (max_batch_size,),
            -1,
            dtype=torch.long,
            device=normalized_device,
        )

        gdn_conv_shape = (
            max_batch_size,
            spec.num_gdn_layers,
            spec.conv_dim,
            spec.conv_kernel_size,
        )

        gdn_recurrent_shape = (
            max_batch_size,
            spec.num_gdn_layers,
            spec.num_gdn_value_heads,
            spec.gdn_key_head_dim,
            spec.gdn_value_head_dim,
        )

        gdn_conv_states = torch.empty(
            gdn_conv_shape,
            dtype=spec.conv_dtype,
            device=normalized_device,
        )

        gdn_recurrent_states = torch.empty(
            gdn_recurrent_shape,
            dtype=spec.recurrent_dtype,
            device=normalized_device,
        )

        updated_gdn_conv_states = torch.empty(
            gdn_conv_shape,
            dtype=spec.conv_dtype,
            device=normalized_device,
        )

        updated_gdn_recurrent_states = (
            torch.empty(
                gdn_recurrent_shape,
                dtype=spec.recurrent_dtype,
                device=normalized_device,
            )
        )

        hidden_states = torch.empty(
            max_batch_size,
            hidden_size,
            dtype=spec.conv_dtype,
            device=normalized_device,
        )

        return cls(
            spec=spec,
            max_batch_size=max_batch_size,
            max_num_blocks=max_num_blocks,
            hidden_size=hidden_size,
            device=normalized_device,
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            state_slot_ids=state_slot_ids,
            gdn_conv_states=gdn_conv_states,
            gdn_recurrent_states=(
                gdn_recurrent_states
            ),
            updated_gdn_conv_states=(
                updated_gdn_conv_states
            ),
            updated_gdn_recurrent_states=(
                updated_gdn_recurrent_states
            ),
            hidden_states=hidden_states,
        )

    def validate_batch_size(
        self,
        batch_size: int,
    ) -> None:
        if not (
            0
            < batch_size
            <= self.max_batch_size
        ):
            raise ValueError(
                "batch_size must be in "
                f"[1, {self.max_batch_size}], "
                f"got {batch_size}"
            )

    def input_gdn_state_views(
        self,
        batch_size: int,
    ) -> list[GDNLayerState | None]:
        """
        把紧凑 GDN Tensor 转换成模型需要的：

            list[GDNLayerState | None]

        列表长度等于全部 Decoder 层数。

        Full Attention 层对应 None；
        GDN 层对应固定地址 Tensor View。
        """

        self.validate_batch_size(
            batch_size
        )

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
            states[global_layer_idx] = (
                GDNLayerState(
                    conv_state=(
                        self.gdn_conv_states[
                            :batch_size,
                            gdn_index,
                        ]
                    ),
                    recurrent_state=(
                        self.gdn_recurrent_states[
                            :batch_size,
                            gdn_index,
                        ]
                    ),
                )
            )

        return states

    def output_gdn_state_views(
        self,
        batch_size: int,
    ) -> list[GDNLayerState | None]:
        """
        返回 Graph 输出状态的固定地址 View。

        之后可以直接交给：
            HybridStateManager.write_batched_states()
        """

        self.validate_batch_size(
            batch_size
        )

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
            states[global_layer_idx] = (
                GDNLayerState(
                    conv_state=(
                        self.updated_gdn_conv_states[
                            :batch_size,
                            gdn_index,
                        ]
                    ),
                    recurrent_state=(
                        self
                        .updated_gdn_recurrent_states[
                            :batch_size,
                            gdn_index,
                        ]
                    ),
                )
            )

        return states

    @torch.inference_mode()
    def copy_decode_inputs(
        self,
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        context_lens: torch.Tensor,
        block_tables: torch.Tensor,
        state_slot_ids: list[int],
        gdn_states: list[
            GDNLayerState | None
        ],
    ) -> int:
        """
        把一轮 Eager 准备出的 Decode 数据复制到
        固定地址 Workspace。

        返回真实 batch_size。
        """

        if input_ids.ndim != 1:
            raise ValueError(
                "input_ids must have shape [B]"
            )

        batch_size = input_ids.shape[0]

        self.validate_batch_size(
            batch_size
        )

        if positions.shape != (
            3,
            batch_size,
        ):
            raise ValueError(
                "positions must have shape [3, B]"
            )

        expected_vector_shape = (
            batch_size,
        )

        if slot_mapping.shape != (
            expected_vector_shape
        ):
            raise ValueError(
                "slot_mapping must have shape [B]"
            )

        if context_lens.shape != (
            expected_vector_shape
        ):
            raise ValueError(
                "context_lens must have shape [B]"
            )

        if (
            block_tables.ndim != 2
            or block_tables.shape[0]
            != batch_size
        ):
            raise ValueError(
                "block_tables must have shape "
                "[B, num_blocks]"
            )

        if (
            block_tables.shape[1]
            > self.max_num_blocks
        ):
            raise ValueError(
                "block_tables exceed the "
                "Workspace block capacity"
            )

        if len(state_slot_ids) != batch_size:
            raise ValueError(
                "state_slot_ids length must equal B"
            )

        if len(set(state_slot_ids)) != batch_size:
            raise ValueError(
                "state_slot_ids must be unique"
            )

        if any(
            (
                not isinstance(slot, int)
                or slot < 0
            )
            for slot in state_slot_ids
        ):
            raise ValueError(
                "state_slot_ids must contain "
                "non-negative integers"
            )

        if (
            len(gdn_states)
            != self.spec.num_hidden_layers
        ):
            raise ValueError(
                "gdn_states must contain one "
                "entry per Decoder layer"
            )

        # 清理小型 metadata Buffer，防止上一轮
        # 较长 block table 的尾部残留。
        self.input_ids.zero_()
        self.positions.zero_()
        self.slot_mapping.fill_(-1)
        self.context_lens.zero_()
        self.block_tables.fill_(-1)
        self.state_slot_ids.fill_(-1)

        self.input_ids[
            :batch_size
        ].copy_(
            input_ids
        )

        self.positions[
            :,
            :batch_size,
        ].copy_(
            positions
        )

        self.slot_mapping[
            :batch_size
        ].copy_(
            slot_mapping
        )

        self.context_lens[
            :batch_size
        ].copy_(
            context_lens
        )

        num_block_columns = (
            block_tables.shape[1]
        )

        self.block_tables[
            :batch_size,
            :num_block_columns,
        ].copy_(
            block_tables
        )

        slot_tensor = torch.tensor(
            state_slot_ids,
            dtype=torch.long,
            device=self.device,
        )

        self.state_slot_ids[
            :batch_size
        ].copy_(
            slot_tensor
        )

        # 把 Eager Gather 得到的状态复制进
        # Graph 固定输入状态。
        for (
            gdn_index,
            global_layer_idx,
        ) in enumerate(
            self.spec.gdn_layer_indices
        ):
            state = gdn_states[
                global_layer_idx
            ]

            if (
                state is None
                or state.conv_state is None
                or state.recurrent_state is None
            ):
                raise ValueError(
                    f"GDN layer {global_layer_idx} "
                    "is missing input state"
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

            if (
                tuple(state.conv_state.shape)
                != expected_conv_shape
            ):
                raise ValueError(
                    "Invalid conv state shape for "
                    f"layer {global_layer_idx}: "
                    f"{tuple(state.conv_state.shape)}"
                )

            if (
                tuple(
                    state.recurrent_state.shape
                )
                != expected_recurrent_shape
            ):
                raise ValueError(
                    "Invalid recurrent state shape "
                    f"for layer {global_layer_idx}: "
                    f"{tuple(state.recurrent_state.shape)}"
                )

            if (
                state.conv_state.dtype
                != self.spec.conv_dtype
            ):
                raise TypeError(
                    "Graph conv input has the "
                    "wrong dtype"
                )

            if (
                state.recurrent_state.dtype
                != self.spec.recurrent_dtype
            ):
                raise TypeError(
                    "Graph recurrent input must "
                    "use FP32"
                )

            self.gdn_conv_states[
                :batch_size,
                gdn_index,
            ].copy_(
                state.conv_state
            )

            self.gdn_recurrent_states[
                :batch_size,
                gdn_index,
            ].copy_(
                state.recurrent_state
            )

        return batch_size

    @torch.inference_mode()
    def copy_model_outputs(
        self,
        *,
        batch_size: int,
        hidden_states: torch.Tensor,
        updated_gdn_states: list[
            GDNLayerState | None
        ],
    ) -> None:
        """
        在 Capture 区域中调用。

        将模型产生的临时 Tensor 复制到长期存在的
        Graph 输出 Buffer。GPU copy_ 会被 Graph 记录。
        """

        self.validate_batch_size(
            batch_size
        )

        if hidden_states.shape != (
            batch_size,
            self.hidden_size,
        ):
            raise ValueError(
                "hidden_states must have shape "
                f"[{batch_size}, "
                f"{self.hidden_size}]"
            )

        if (
            len(updated_gdn_states)
            != self.spec.num_hidden_layers
        ):
            raise ValueError(
                "updated_gdn_states must contain "
                "one entry per Decoder layer"
            )

        self.hidden_states[
            :batch_size
        ].copy_(
            hidden_states
        )

        for (
            gdn_index,
            global_layer_idx,
        ) in enumerate(
            self.spec.gdn_layer_indices
        ):
            state = updated_gdn_states[
                global_layer_idx
            ]

            if (
                state is None
                or state.conv_state is None
                or state.recurrent_state is None
            ):
                raise ValueError(
                    f"GDN layer {global_layer_idx} "
                    "is missing output state"
                )

            self.updated_gdn_conv_states[
                :batch_size,
                gdn_index,
            ].copy_(
                state.conv_state
            )

            self.updated_gdn_recurrent_states[
                :batch_size,
                gdn_index,
            ].copy_(
                state.recurrent_state
            )

    @property
    def allocated_bytes(self) -> int:
        tensors = (
            self.input_ids,
            self.positions,
            self.slot_mapping,
            self.context_lens,
            self.block_tables,
            self.state_slot_ids,
            self.gdn_conv_states,
            self.gdn_recurrent_states,
            self.updated_gdn_conv_states,
            self.updated_gdn_recurrent_states,
            self.hidden_states,
        )

        return sum(
            tensor.numel()
            * tensor.element_size()
            for tensor in tensors
        )

    def storage_pointers(
        self,
    ) -> dict[str, int]:
        """
        用于测试 Capture 前后 Tensor 地址是否稳定。
        """

        return {
            "input_ids": (
                self.input_ids.data_ptr()
            ),
            "positions": (
                self.positions.data_ptr()
            ),
            "slot_mapping": (
                self.slot_mapping.data_ptr()
            ),
            "context_lens": (
                self.context_lens.data_ptr()
            ),
            "block_tables": (
                self.block_tables.data_ptr()
            ),
            "state_slot_ids": (
                self.state_slot_ids.data_ptr()
            ),
            "gdn_conv_states": (
                self.gdn_conv_states.data_ptr()
            ),
            "gdn_recurrent_states": (
                self.gdn_recurrent_states
                .data_ptr()
            ),
            "updated_gdn_conv_states": (
                self.updated_gdn_conv_states
                .data_ptr()
            ),
            "updated_gdn_recurrent_states": (
                self
                .updated_gdn_recurrent_states
                .data_ptr()
            ),
            "hidden_states": (
                self.hidden_states.data_ptr()
            ),
        }
        
@dataclass(frozen=True, slots=True)
class HybridDecodeGraphRoute:
    """
    一次 Hybrid Decode 应该走哪条执行路径。
    """

    # True：
    #     使用已经捕获的 CUDA Graph。
    #
    # False：
    #     使用普通 Eager forward。
    use_graph: bool

    # 当前真实 Decode 请求数。
    batch_size: int

    # use_graph=True 时，表示使用哪个 Graph bucket。
    #
    # 当前采用精确匹配，因此：
    #     bucket_size == batch_size
    #
    # Eager fallback 时为 None。
    bucket_size: int | None

    # 路由原因，供调试和性能统计使用。
    reason: str
    
    
@dataclass(frozen=True, slots=True)
class HybridDecodeGraphPolicy:
    """
    决定 Hybrid Decode 使用 CUDA Graph
    还是回退 Eager。
    """

    batch_sizes: tuple[int, ...]
    max_num_seqs: int

    def __post_init__(self) -> None:
        if self.max_num_seqs <= 0:
            raise ValueError(
                "max_num_seqs must be positive"
            )

        if not self.batch_sizes:
            raise ValueError(
                "batch_sizes must not be empty"
            )

        if any(
            (
                not isinstance(batch_size, int)
                or isinstance(batch_size, bool)
            )
            for batch_size in self.batch_sizes
        ):
            raise TypeError(
                "Every graph batch size must "
                "be an integer"
            )

        if any(
            batch_size <= 0
            for batch_size in self.batch_sizes
        ):
            raise ValueError(
                "Every graph batch size must "
                "be positive"
            )

        if tuple(
            sorted(set(self.batch_sizes))
        ) != self.batch_sizes:
            raise ValueError(
                "batch_sizes must be unique "
                "and strictly increasing"
            )

        if (
            self.batch_sizes[-1]
            > self.max_num_seqs
        ):
            raise ValueError(
                "The largest graph batch size "
                "must not exceed max_num_seqs"
            )

    def route(
        self,
        *,
        batch_size: int,
        is_prefill: bool,
        enforce_eager: bool,
    ) -> HybridDecodeGraphRoute:
        """
        为一次模型执行选择 Graph 或 Eager。

        当前规则：

        1. Prefill 永远走 Eager。
        2. enforce_eager=True 时永远走 Eager。
        3. Decode batch 精确命中 bucket 时走 Graph。
        4. 其他 Decode batch 回退 Eager。
        """

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive"
            )

        if batch_size > self.max_num_seqs:
            raise ValueError(
                "batch_size exceeds max_num_seqs"
            )

        if is_prefill:
            return HybridDecodeGraphRoute(
                use_graph=False,
                batch_size=batch_size,
                bucket_size=None,
                reason="prefill_uses_eager",
            )

        if enforce_eager:
            return HybridDecodeGraphRoute(
                use_graph=False,
                batch_size=batch_size,
                bucket_size=None,
                reason="enforce_eager",
            )

        if batch_size in self.batch_sizes:
            return HybridDecodeGraphRoute(
                use_graph=True,
                batch_size=batch_size,
                bucket_size=batch_size,
                reason="exact_graph_bucket",
            )

        return HybridDecodeGraphRoute(
            use_graph=False,
            batch_size=batch_size,
            bucket_size=None,
            reason="unsupported_batch_size",
        )