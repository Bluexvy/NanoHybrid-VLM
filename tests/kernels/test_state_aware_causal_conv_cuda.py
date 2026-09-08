from __future__ import annotations

import torch

from causal_conv1d import causal_conv1d_update

from nanovllm.kernels.state_aware_gdn_cuda import (
    state_aware_causal_conv1d_cuda,
)


BATCH_SIZES = (1, 2, 4, 8, 16)

NUM_CHANNELS = 8192
KERNEL_SIZE = 4
NUM_GDN_LAYERS = 24
GDN_INDEX = 7
DECODE_STEPS = 8


@torch.inference_mode()
def run_case(batch_size: int) -> None:
    num_slots = batch_size + 4

    state_slot_ids = torch.arange(
        batch_size - 1,
        -1,
        -1,
        dtype=torch.long,
        device="cuda",
    ) + 2

    weight = torch.randn(
        NUM_CHANNELS,
        KERNEL_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
    ) * 0.05

    initial_pool = torch.randn(
        num_slots,
        NUM_GDN_LAYERS,
        NUM_CHANNELS,
        KERNEL_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
    ) * 0.05

    reference_state = initial_pool.index_select(
        dim=0,
        index=state_slot_ids,
    )[:, GDN_INDEX].clone()

    cuda_pool = initial_pool.clone()

    output_workspace = torch.empty(
        batch_size,
        NUM_CHANNELS,
        1,
        dtype=torch.bfloat16,
        device="cuda",
    )

    output_pointer = output_workspace.data_ptr()

    maximum_state_error = 0.0
    maximum_output_error = 0.0

    for _ in range(DECODE_STEPS):
        x = torch.randn(
            batch_size,
            NUM_CHANNELS,
            1,
            dtype=torch.bfloat16,
            device="cuda",
        ) * 0.05

        reference_output = causal_conv1d_update(
            x=x,
            conv_state=reference_state,
            weight=weight,
            bias=None,
            activation="silu",
        )

        cuda_output = state_aware_causal_conv1d_cuda(
            x=x,
            weight=weight,
            conv_state_pool=cuda_pool,
            state_slot_ids=state_slot_ids,
            gdn_index=GDN_INDEX,
            output=output_workspace,
        )

        assert cuda_output.data_ptr() == output_pointer

        cuda_state = cuda_pool.index_select(
            dim=0,
            index=state_slot_ids,
        )[:, GDN_INDEX]

        state_error = (cuda_state.float() - reference_state.float()).abs().max().item()
        output_error = (cuda_output.float() - reference_output.float()).abs().max().item()

        maximum_state_error = max(maximum_state_error, state_error)
        maximum_output_error = max(maximum_output_error, output_error)

        torch.testing.assert_close(cuda_state, reference_state, rtol=0.0, atol=0.0)
        torch.testing.assert_close(cuda_output.float(), reference_output.float(), rtol=0.02, atol=0.02)

    expected_pool = initial_pool.clone()
    expected_pool[:, GDN_INDEX].index_copy_(dim=0, index=state_slot_ids, source=reference_state)

    untouched_error = (cuda_pool.float() - expected_pool.float()).abs().max().item()

    torch.testing.assert_close(cuda_pool, expected_pool, rtol=0.0, atol=0.0)

    print(
        f"B={batch_size}: "
        f"state_error={maximum_state_error:.9e}, "
        f"output_error={maximum_output_error:.9e}, "
        f"untouched_error={untouched_error:.9e}"
    )


def main() -> None:
    torch.manual_seed(20260908)
    torch.cuda.manual_seed_all(20260908)

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Shape: C={NUM_CHANNELS}, K={KERNEL_SIZE}")
    print(f"GDN layers: {NUM_GDN_LAYERS}")
    print(f"Decode steps: {DECODE_STEPS}")

    for batch_size in BATCH_SIZES:
        run_case(batch_size)

    print(
        "Part Conv CUDA-1 PASSED: direct Conv State Pool "
        "addressing, BF16 vector state update, SiLU output "
        "and output workspace reuse are correct."
    )


if __name__ == "__main__":
    main()