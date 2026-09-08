from __future__ import annotations

import torch
import torch.nn.functional as F

from causal_conv1d import causal_conv1d_update


BATCH_SIZES = (1, 2, 4, 8, 16)

NUM_GDN_LAYERS = 24
NUM_CHANNELS = 8192
KERNEL_SIZE = 4
DECODE_STEPS = 8


@torch.inference_mode()
def state_aware_causal_conv_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state_pool: torch.Tensor,
    state_slot_ids: torch.Tensor,
    gdn_index: int,
) -> torch.Tensor:
    """
    x:
        [B, C, 1], BF16

    weight:
        [C, K], BF16

    conv_state_pool:
        [num_slots, num_gdn_layers, C, K], BF16

    state_slot_ids:
        [B], INT64

    返回:
        [B, C, 1], BF16

    副作用:
        直接更新 conv_state_pool 中对应的物理状态。
    """

    batch_size = x.shape[0]
    output = torch.empty_like(x)

    for batch_index in range(batch_size):
        state_slot = int(state_slot_ids[batch_index].item())
        current_state = conv_state_pool[state_slot, gdn_index]

        new_state = torch.empty_like(current_state)
        new_state[:, :-1] = current_state[:, 1:]
        new_state[:, -1] = x[batch_index, :, 0]

        current_state.copy_(new_state)

        convolution = (new_state.float() * weight.float()).sum(dim=-1)
        activated = F.silu(convolution)

        output[batch_index, :, 0] = activated.to(output.dtype)

    return output


@torch.inference_mode()
def run_case(batch_size: int) -> None:
    torch.manual_seed(2026 + batch_size)
    torch.cuda.manual_seed_all(2026 + batch_size)

    num_slots = batch_size + 4
    gdn_index = 7

    state_slot_ids = torch.arange(batch_size - 1, -1, -1, dtype=torch.long, device="cuda") + 2

    weight = torch.randn(
        NUM_CHANNELS,
        KERNEL_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
    ) * 0.05

    initial_state_pool = torch.randn(
        num_slots,
        NUM_GDN_LAYERS,
        NUM_CHANNELS,
        KERNEL_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
    ) * 0.05

    reference_state = initial_state_pool.index_select(dim=0, index=state_slot_ids)[:, gdn_index].clone()
    state_aware_pool = initial_state_pool.clone()

    maximum_state_error = 0.0
    maximum_output_error = 0.0

    for decode_step in range(DECODE_STEPS):
        x = torch.randn(
            batch_size,
            NUM_CHANNELS,
            1,
            dtype=torch.bfloat16,
            device="cuda",
        ) * 0.05

        expected_output = causal_conv1d_update(
            x=x,
            conv_state=reference_state,
            weight=weight,
            bias=None,
            activation="silu",
        )

        actual_output = state_aware_causal_conv_reference(
            x=x,
            weight=weight,
            conv_state_pool=state_aware_pool,
            state_slot_ids=state_slot_ids,
            gdn_index=gdn_index,
        )

        actual_state = state_aware_pool.index_select(dim=0, index=state_slot_ids)[:, gdn_index]

        state_error = (actual_state.float() - reference_state.float()).abs().max().item()
        output_error = (actual_output.float() - expected_output.float()).abs().max().item()

        maximum_state_error = max(maximum_state_error, state_error)
        maximum_output_error = max(maximum_output_error, output_error)

        torch.testing.assert_close(actual_state, reference_state, rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual_output.float(), expected_output.float(), rtol=0.02, atol=0.02)

    expected_final_pool = initial_state_pool.clone()
    expected_final_pool[:, gdn_index].index_copy_(dim=0, index=state_slot_ids, source=reference_state)

    untouched_error = (state_aware_pool.float() - expected_final_pool.float()).abs().max().item()

    torch.testing.assert_close(state_aware_pool, expected_final_pool, rtol=0.0, atol=0.0)

    print(
        f"B={batch_size}: "
        f"state_error={maximum_state_error:.9e}, "
        f"output_error={maximum_output_error:.9e}, "
        f"untouched_error={untouched_error:.9e}"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Shape: C={NUM_CHANNELS}, K={KERNEL_SIZE}")
    print(f"GDN layers: {NUM_GDN_LAYERS}")
    print(f"Decode steps per case: {DECODE_STEPS}")

    for batch_size in BATCH_SIZES:
        run_case(batch_size)

    print(
        "Part Conv-1 PASSED: direct slot/layer addressing, "
        "causal state shift, SiLU output and untouched-state "
        "protection are correct."
    )


if __name__ == "__main__":
    main()