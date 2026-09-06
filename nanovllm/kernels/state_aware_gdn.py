from __future__ import annotations

import torch
import triton
import triton.language as tl


def _l2_normalize(
    tensor: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    沿最后一个维度执行 L2 Normalization。

    输入：
        [B, H, D]

    输出：
        [B, H, D]
    """

    inverse_norm = torch.rsqrt(
        tensor.square().sum(
            dim=-1,
            keepdim=True,
        )
        + eps
    )

    return tensor * inverse_norm


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state_pool: torch.Tensor,
    state_slot_ids: torch.Tensor,
    gdn_index: int,
) -> tuple[int, int, int, int]:
    """
    检查 State-aware Decode 箨子的输入契约。

    返回：
        batch_size
        num_heads
        key_dim
        value_dim
    """

    if query.ndim != 4:
        raise ValueError(
            "query must have shape [B, 1, H, Dk]"
        )

    (
        batch_size,
        sequence_length,
        num_heads,
        key_dim,
    ) = query.shape

    if sequence_length != 1:
        raise ValueError(
            "State-aware GDN Decode only supports "
            "sequence_length == 1"
        )

    if key.shape != query.shape:
        raise ValueError(
            "key must have the same shape as query"
        )

    if value.ndim != 4:
        raise ValueError(
            "value must have shape [B, 1, H, Dv]"
        )

    if value.shape[:3] != (
        batch_size,
        1,
        num_heads,
    ):
        raise ValueError(
            "value must have shape [B, 1, H, Dv]"
        )

    value_dim = value.shape[-1]

    expected_gate_shape = (
        batch_size,
        1,
        num_heads,
    )

    if g.shape != expected_gate_shape:
        raise ValueError(
            f"g must have shape {expected_gate_shape}"
        )

    if beta.shape != expected_gate_shape:
        raise ValueError(
            f"beta must have shape {expected_gate_shape}"
        )

    if recurrent_state_pool.ndim != 5:
        raise ValueError(
            "recurrent_state_pool must have shape "
            "[num_slots, num_gdn_layers, H, Dk, Dv]"
        )

    expected_state_tail = (
        num_heads,
        key_dim,
        value_dim,
    )

    if tuple(recurrent_state_pool.shape[2:]) != (
        expected_state_tail
    ):
        raise ValueError(
            "recurrent_state_pool has an incompatible "
            "H/Dk/Dv shape"
        )

    if recurrent_state_pool.dtype != torch.float32:
        raise TypeError(
            "recurrent_state_pool must use FP32"
        )

    if state_slot_ids.shape != (batch_size,):
        raise ValueError(
            "state_slot_ids must have shape [B]"
        )

    if state_slot_ids.dtype != torch.long:
        raise TypeError(
            "state_slot_ids must use torch.long"
        )

    if state_slot_ids.device != query.device:
        raise ValueError(
            "state_slot_ids and query must be on "
            "the same device"
        )

    if recurrent_state_pool.device != query.device:
        raise ValueError(
            "recurrent_state_pool and query must "
            "be on the same device"
        )

    num_slots = recurrent_state_pool.shape[0]
    num_gdn_layers = recurrent_state_pool.shape[1]

    if not 0 <= gdn_index < num_gdn_layers:
        raise IndexError(
            f"gdn_index {gdn_index} is out of range"
        )

    # Reference 路径允许 CPU 同步，因为它只用于验证，
    # 不会进入最终高性能执行路径。
    slot_list = (
        state_slot_ids
        .detach()
        .cpu()
        .tolist()
    )

    if any(
        slot < 0 or slot >= num_slots
        for slot in slot_list
    ):
        raise IndexError(
            "state_slot_ids contains an invalid slot"
        )

    # 同一 Decode batch 中，两条请求不能使用同一个
    # state slot，否则二者会同时修改同一块状态。
    if len(set(slot_list)) != len(slot_list):
        raise ValueError(
            "state_slot_ids must be unique within "
            "one Decode batch"
        )

    return (
        batch_size,
        num_heads,
        key_dim,
        value_dim,
    )


@torch.inference_mode()
def state_aware_gdn_decode_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state_pool: torch.Tensor,
    state_slot_ids: torch.Tensor,
    gdn_index: int,
    *,
    scale: float | None = None,
    use_qk_l2norm: bool = True,
) -> torch.Tensor:
    """
    State-aware Gated Delta Rule Decode Reference。

    只支持 Decode：
        sequence_length == 1

    输入：
        query:
            [B, 1, H, Dk]

        key:
            [B, 1, H, Dk]

        value:
            [B, 1, H, Dv]

        g:
            [B, 1, H]

        beta:
            [B, 1, H]

        recurrent_state_pool:
            [num_slots, num_gdn_layers, H, Dk, Dv]

        state_slot_ids:
            [B]

        gdn_index:
            当前 GDN 层在紧凑 GDN 层维度中的编号。

    输出：
        [B, 1, H, Dv]

    副作用：
        直接原地更新：
            recurrent_state_pool[
                state_slot_ids,
                gdn_index,
            ]
    """

    (
        batch_size,
        num_heads,
        key_dim,
        value_dim,
    ) = _validate_inputs(
        query=query,
        key=key,
        value=value,
        g=g,
        beta=beta,
        recurrent_state_pool=(
            recurrent_state_pool
        ),
        state_slot_ids=state_slot_ids,
        gdn_index=gdn_index,
    )

    if scale is None:
        scale = key_dim ** -0.5

    output_dtype = value.dtype

    # Decode 的 sequence_length 固定为 1，
    # 因此去掉长度维度。
    #
    # [B, 1, H, Dk] -> [B, H, Dk]
    query_fp32 = query[:, 0].float()
    key_fp32 = key[:, 0].float()

    # [B, 1, H, Dv] -> [B, H, Dv]
    value_fp32 = value[:, 0].float()

    # [B, 1, H] -> [B, H]
    g_fp32 = g[:, 0].float()
    beta_fp32 = beta[:, 0].float()

    if use_qk_l2norm:
        query_fp32 = _l2_normalize(
            query_fp32
        )

        key_fp32 = _l2_normalize(
            key_fp32
        )

    # 与当前 FLA 调用保持一致：
    #
    # scale = head_k_dim ** -0.5
    query_fp32 = query_fp32 * scale

    output = torch.empty(
        (
            batch_size,
            1,
            num_heads,
            value_dim,
        ),
        dtype=output_dtype,
        device=query.device,
    )

    # Reference 使用 Python 循环逐请求处理。
    #
    # 最终 Triton Kernel 会把 B 和 H 映射到
    # 并行 Program，不会保留这个 Python 循环。
    slot_list = (
        state_slot_ids
        .detach()
        .cpu()
        .tolist()
    )

    for batch_index, slot in enumerate(
        slot_list
    ):
        # 这里是 basic indexing，返回的是 View。
        #
        # state 的修改会直接反映到
        # recurrent_state_pool 中。
        #
        # state.shape:
        # [H, Dk, Dv]
        state = recurrent_state_pool[
            slot,
            gdn_index,
        ]

        # 每条请求、每个 Head 有一个衰减系数。
        #
        # [H] -> [H, 1, 1]
        decay = (
            torch.exp(g_fp32[batch_index])
            .view(
                num_heads,
                1,
                1,
            )
        )

        # S_decay = exp(g) * S_old
        #
        # 这是直接对状态池中的状态执行原地衰减。
        state.mul_(decay)

        # remembered_value = k^T @ S_decay
        #
        # state: [H, Dk, Dv]
        # key:   [H, Dk]
        # result:[H, Dv]
        remembered_value = torch.einsum(
            "hkv,hk->hv",
            state,
            key_fp32[batch_index],
        )

        # delta = beta * (v - remembered_value)
        #
        # beta 控制本轮修正写入多少。
        delta = (
            value_fp32[batch_index]
            - remembered_value
        ) * beta_fp32[
            batch_index
        ].unsqueeze(-1)

        # S_new = S_decay + k outer delta
        #
        # key:
        #   [H, Dk, 1]
        #
        # delta:
        #   [H, 1, Dv]
        #
        # 外积结果：
        #   [H, Dk, Dv]
        state.add_(
            key_fp32[
                batch_index
            ].unsqueeze(-1)
            * delta.unsqueeze(-2)
        )

        # output = q^T @ S_new
        #
        # 必须读取更新后的 S_new，而不是旧状态。
        output_fp32 = torch.einsum(
            "hkv,hk->hv",
            state,
            query_fp32[batch_index],
        )

        output[
            batch_index,
            0,
        ].copy_(
            output_fp32.to(output_dtype)
        )

    return output

@triton.jit
def _state_aware_gdn_decode_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    g_ptr,
    beta_ptr,
    state_pool_ptr,
    state_slot_ids_ptr,
    output_ptr,

    # query/key stride
    stride_q_batch,
    stride_q_head,
    stride_q_dim,

    stride_k_batch,
    stride_k_head,
    stride_k_dim,

    # value stride
    stride_v_batch,
    stride_v_head,
    stride_v_dim,

    # g/beta stride
    stride_g_batch,
    stride_g_head,

    stride_beta_batch,
    stride_beta_head,

    # state pool stride:
    # [num_slots, num_gdn_layers, H, Dk, Dv]
    stride_state_slot,
    stride_state_layer,
    stride_state_head,
    stride_state_key,
    stride_state_value,

    # output stride
    stride_output_batch,
    stride_output_head,
    stride_output_dim,

    gdn_index,
    scale,

    KEY_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_KEY: tl.constexpr,
    BLOCK_VALUE: tl.constexpr,
    EPSILON: tl.constexpr,
):
    """
    一个 Triton Program 处理：

        一条 Sequence
        × 一个 GDN Head
        × 一组连续的 Value 列

    Grid：

        axis 0: batch_index
        axis 1: head_index
        axis 2: value_tile_index
    """

    batch_index = tl.program_id(axis=0)
    head_index = tl.program_id(axis=1)
    value_tile_index = tl.program_id(axis=2)

    # 当前 Triton Program 处理全部 Dk。
    key_offsets = tl.arange(
        0,
        BLOCK_KEY,
    )

    key_mask = key_offsets < KEY_DIM

    # 当前 Triton Program 只处理一部分 Dv。
    value_offsets = (
        value_tile_index * BLOCK_VALUE
        + tl.arange(0, BLOCK_VALUE)
    )

    value_mask = value_offsets < VALUE_DIM

    # 根据 batch row 找到请求真正使用的
    # state pool slot。
    #
    # 例如：
    #
    # state_slot_ids = [7, 2, 13]
    #
    # batch row 0 -> slot 7
    # batch row 1 -> slot 2
    # batch row 2 -> slot 13
    state_slot = tl.load(
        state_slot_ids_ptr + batch_index
    )

    # -----------------------------------------------------
    # 1. 加载 Query
    # -----------------------------------------------------

    query_offsets = (
        batch_index * stride_q_batch
        + head_index * stride_q_head
        + key_offsets * stride_q_dim
    )

    query = tl.load(
        query_ptr + query_offsets,
        mask=key_mask,
        other=0.0,
    ).to(tl.float32)

    # -----------------------------------------------------
    # 2. 加载 Key
    # -----------------------------------------------------

    key_input_offsets = (
        batch_index * stride_k_batch
        + head_index * stride_k_head
        + key_offsets * stride_k_dim
    )

    key = tl.load(
        key_ptr + key_input_offsets,
        mask=key_mask,
        other=0.0,
    ).to(tl.float32)

    # -----------------------------------------------------
    # 3. 在 Kernel 内完成 Q/K L2Norm
    # -----------------------------------------------------

    query_squared_sum = tl.sum(
        query * query,
        axis=0,
    )

    key_squared_sum = tl.sum(
        key * key,
        axis=0,
    )

    query_inverse_norm = (
        1.0
        / tl.sqrt(
            query_squared_sum + EPSILON
        )
    )

    key_inverse_norm = (
        1.0
        / tl.sqrt(
            key_squared_sum + EPSILON
        )
    )

    query = (
        query
        * query_inverse_norm
        * scale
    )

    key = key * key_inverse_norm

    # -----------------------------------------------------
    # 4. 计算状态池中当前 Tile 的物理地址
    # -----------------------------------------------------

    # state_tile 的逻辑形状：
    #
    # [BLOCK_KEY, BLOCK_VALUE]
    #
    # 对应状态矩阵：
    #
    # state_pool[
    #     state_slot,
    #     gdn_index,
    #     head_index,
    #     key_offsets,
    #     value_offsets,
    # ]
    state_offsets = (
        state_slot * stride_state_slot
        + gdn_index * stride_state_layer
        + head_index * stride_state_head
        + key_offsets[:, None]
        * stride_state_key
        + value_offsets[None, :]
        * stride_state_value
    )

    state_mask = (
        key_mask[:, None]
        & value_mask[None, :]
    )

    # recurrent_state_pool 是 FP32，
    # 因此 state 也在 FP32 中计算。
    state = tl.load(
        state_pool_ptr + state_offsets,
        mask=state_mask,
        other=0.0,
    ).to(tl.float32)

    # -----------------------------------------------------
    # 5. 加载 g，衰减旧状态
    # -----------------------------------------------------

    g_offset = (
        batch_index * stride_g_batch
        + head_index * stride_g_head
    )

    g_value = tl.load(
        g_ptr + g_offset
    ).to(tl.float32)

    decay = tl.exp(g_value)

    # S_decay = exp(g) * S_old
    decayed_state = state * decay

    # -----------------------------------------------------
    # 6. 使用 Key 从旧状态读取 remembered_value
    # -----------------------------------------------------

    # decayed_state:
    # [Dk, BLOCK_VALUE]
    #
    # key[:, None]:
    # [Dk, 1]
    #
    # 沿 Dk 归约后：
    # [BLOCK_VALUE]
    remembered_value = tl.sum(
        decayed_state * key[:, None],
        axis=0,
    )

    # -----------------------------------------------------
    # 7. 加载当前 Value 和 beta
    # -----------------------------------------------------

    value_input_offsets = (
        batch_index * stride_v_batch
        + head_index * stride_v_head
        + value_offsets * stride_v_dim
    )

    current_value = tl.load(
        value_ptr + value_input_offsets,
        mask=value_mask,
        other=0.0,
    ).to(tl.float32)

    beta_offset = (
        batch_index * stride_beta_batch
        + head_index * stride_beta_head
    )

    beta_value = tl.load(
        beta_ptr + beta_offset
    ).to(tl.float32)

    # delta = beta * (v - k^T S_decay)
    delta = (
        current_value - remembered_value
    ) * beta_value

    # -----------------------------------------------------
    # 8. 原地计算新状态
    # -----------------------------------------------------

    # key[:, None]:
    # [Dk, 1]
    #
    # delta[None, :]:
    # [1, BLOCK_VALUE]
    #
    # 外积：
    # [Dk, BLOCK_VALUE]
    updated_state = (
        decayed_state
        + key[:, None] * delta[None, :]
    )

    # -----------------------------------------------------
    # 9. 使用 Query 从新状态读取输出
    # -----------------------------------------------------

    # output_tile:
    # [BLOCK_VALUE]
    output_tile = tl.sum(
        updated_state * query[:, None],
        axis=0,
    )

    # -----------------------------------------------------
    # 10. 将新状态直接写回 State Pool
    # -----------------------------------------------------

    tl.store(
        state_pool_ptr + state_offsets,
        updated_state,
        mask=state_mask,
    )

    # -----------------------------------------------------
    # 11. 保存当前 Value Tile 的输出
    # -----------------------------------------------------

    output_offsets = (
        batch_index * stride_output_batch
        + head_index * stride_output_head
        + value_offsets * stride_output_dim
    )

    tl.store(
        output_ptr + output_offsets,
        output_tile,
        mask=value_mask,
    )


@torch.inference_mode()
def state_aware_gdn_decode_triton(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state_pool: torch.Tensor,
    state_slot_ids: torch.Tensor,
    gdn_index: int,
    *,
    output: torch.Tensor | None = None,
    scale: float | None = None,
    block_value: int = 32,
) -> torch.Tensor:
    """
    State-aware GDN Decode Triton 实现。

    输入：

        query:
            [B, 1, H, Dk]

        key:
            [B, 1, H, Dk]

        value:
            [B, 1, H, Dv]

        g:
            [B, 1, H]

        beta:
            [B, 1, H]

        recurrent_state_pool:
            [num_slots, num_gdn_layers, H, Dk, Dv]

        state_slot_ids:
            [B]

    返回：

        output:
            [B, 1, H, Dv]

    副作用：

        直接更新状态池对应 slot 和 GDN layer，
        不创建 Batched recurrent state 和 final_state。
    """

    if not query.is_cuda:
        raise RuntimeError(
            "State-aware Triton GDN requires CUDA"
        )

    if query.ndim != 4:
        raise ValueError(
            "query must have shape [B, 1, H, Dk]"
        )

    (
        batch_size,
        sequence_length,
        num_heads,
        key_dim,
    ) = query.shape

    if sequence_length != 1:
        raise ValueError(
            "State-aware Triton GDN only supports "
            "Decode with sequence_length == 1"
        )

    if key.shape != query.shape:
        raise ValueError(
            "key must have the same shape as query"
        )

    if value.ndim != 4:
        raise ValueError(
            "value must have shape [B, 1, H, Dv]"
        )

    if value.shape[:3] != (
        batch_size,
        1,
        num_heads,
    ):
        raise ValueError(
            "value must have shape [B, 1, H, Dv]"
        )

    value_dim = value.shape[-1]

    expected_gate_shape = (
        batch_size,
        1,
        num_heads,
    )

    if g.shape != expected_gate_shape:
        raise ValueError(
            f"g must have shape {expected_gate_shape}"
        )

    if beta.shape != expected_gate_shape:
        raise ValueError(
            f"beta must have shape {expected_gate_shape}"
        )

    if recurrent_state_pool.ndim != 5:
        raise ValueError(
            "recurrent_state_pool must have shape "
            "[num_slots, num_gdn_layers, H, Dk, Dv]"
        )

    if (
        tuple(recurrent_state_pool.shape[2:])
        != (
            num_heads,
            key_dim,
            value_dim,
        )
    ):
        raise ValueError(
            "recurrent_state_pool has incompatible "
            "H/Dk/Dv dimensions"
        )

    if recurrent_state_pool.dtype != torch.float32:
        raise TypeError(
            "recurrent_state_pool must use FP32"
        )

    if state_slot_ids.shape != (batch_size,):
        raise ValueError(
            "state_slot_ids must have shape [B]"
        )

    if state_slot_ids.dtype != torch.long:
        raise TypeError(
            "state_slot_ids must use torch.long"
        )

    if state_slot_ids.device != query.device:
        raise ValueError(
            "state_slot_ids must be on the same "
            "device as query"
        )

    if recurrent_state_pool.device != query.device:
        raise ValueError(
            "recurrent_state_pool must be on the "
            "same device as query"
        )

    num_gdn_layers = recurrent_state_pool.shape[1]

    if not 0 <= gdn_index < num_gdn_layers:
        raise IndexError(
            f"gdn_index {gdn_index} is out of range"
        )

    if block_value not in {
        16,
        32,
        64,
    }:
        raise ValueError(
            "block_value must be 16, 32 or 64"
        )

    if scale is None:
        scale = key_dim ** -0.5

    if output is None:
        output = torch.empty_like(value)
    else:
        if output.shape != value.shape:
            raise ValueError(
                "output must have the same shape "
                "as value"
            )

        if output.dtype != value.dtype:
            raise TypeError(
                "output must have the same dtype "
                "as value"
            )

        if output.device != value.device:
            raise ValueError(
                "output must be on the same device "
                "as value"
            )

    block_key = triton.next_power_of_2(
        key_dim
    )

    # Grid：
    #
    # [B, H, ceil(Dv / BLOCK_VALUE)]
    grid = (
        batch_size,
        num_heads,
        triton.cdiv(
            value_dim,
            block_value,
        ),
    )

    _state_aware_gdn_decode_kernel[grid](
        query,
        key,
        value,
        g,
        beta,
        recurrent_state_pool,
        state_slot_ids,
        output,

        query.stride(0),
        query.stride(2),
        query.stride(3),

        key.stride(0),
        key.stride(2),
        key.stride(3),

        value.stride(0),
        value.stride(2),
        value.stride(3),

        g.stride(0),
        g.stride(2),

        beta.stride(0),
        beta.stride(2),

        recurrent_state_pool.stride(0),
        recurrent_state_pool.stride(1),
        recurrent_state_pool.stride(2),
        recurrent_state_pool.stride(3),
        recurrent_state_pool.stride(4),

        output.stride(0),
        output.stride(2),
        output.stride(3),

        gdn_index,
        scale,

        KEY_DIM=key_dim,
        VALUE_DIM=value_dim,
        BLOCK_KEY=block_key,
        BLOCK_VALUE=block_value,
        EPSILON=1e-6,

        # 128 × 32 的状态 Tile 分给 8 个 Warp。
        num_warps=8,
        num_stages=1,
    )

    return output