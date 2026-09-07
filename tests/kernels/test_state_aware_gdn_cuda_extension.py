from __future__ import annotations

import torch

from nanovllm.kernels.state_aware_gdn import (
    state_aware_gdn_decode_reference,
)

from nanovllm.kernels.state_aware_gdn_cuda import (
    state_aware_gdn_decode_cuda,
)


NUM_HEADS = 32
KEY_DIM = 128
VALUE_DIM = 128
NUM_GDN_LAYERS = 3
GDN_INDEX = 1
NUM_DECODE_STEPS = 8

def make_inputs(
    batch_size: int,
    num_slots: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    q_width = NUM_HEADS * KEY_DIM
    k_width = NUM_HEADS * KEY_DIM
    v_width = NUM_HEADS * VALUE_DIM

    mixed_qkv = torch.randn(
        batch_size,
        1,
        q_width + k_width + v_width,
        device="cuda",
        dtype=torch.bfloat16,
    )

    mixed_qkv.mul_(0.02)

    query_flat, key_flat, value_flat = torch.split(
        mixed_qkv,
        [
            q_width,
            k_width,
            v_width,
        ],
        dim=-1,
    )

    query = query_flat.reshape(
        batch_size,
        1,
        NUM_HEADS,
        KEY_DIM,
    )

    key = key_flat.reshape(
        batch_size,
        1,
        NUM_HEADS,
        KEY_DIM,
    )

    value = value_flat.reshape(
        batch_size,
        1,
        NUM_HEADS,
        VALUE_DIM,
    )
    g = -(
        0.05
        + torch.rand(
            batch_size,
            1,
            NUM_HEADS,
            device="cuda",
            dtype=torch.float32,
        ) * 0.02
    )

    beta = torch.sigmoid(
        torch.randn(
            batch_size,
            1,
            NUM_HEADS,
            device="cuda",
            dtype=torch.bfloat16,
        )
    )

    state_slot_ids = (
        torch.arange(
            batch_size,
            device="cuda",
            dtype=torch.long,
        ) * 3 + 1
    ) % num_slots

    initial_state = torch.randn(
        num_slots,
        NUM_GDN_LAYERS,
        NUM_HEADS,
        KEY_DIM,
        VALUE_DIM,
        device="cuda",
        dtype=torch.float32,
    ) * 0.001

    return (
        query,
        key,
        value,
        g.contiguous(),
        beta.contiguous(),
        state_slot_ids.contiguous(),
        initial_state.contiguous(),
    )
    

def compute_untouched_error(
    initial_state: torch.Tensor,
    cuda_state: torch.Tensor,
    state_slot_ids: torch.Tensor,
) -> float:
    selected_slots = set(
        state_slot_ids.cpu().tolist()
    )

    max_error = 0.0

    for slot_index in range(initial_state.shape[0]):
        for layer_index in range(initial_state.shape[1]):
            selected = slot_index in selected_slots and layer_index == GDN_INDEX

            if selected:
                continue

            error = (
                cuda_state[slot_index, layer_index]
                - initial_state[slot_index, layer_index]
            ).abs().max().item()

            max_error = max(
                max_error,
                error,
            )

    return max_error


@torch.inference_mode()
def run_case(
    batch_size: int,
) -> None:
    num_slots = batch_size + 3

    (
        query,
        key,
        value,
        g,
        beta,
        state_slot_ids,
        initial_state,
    ) = make_inputs(
        batch_size=batch_size,
        num_slots=num_slots,
    )
    if batch_size == 2:
        print(f"query shape: {tuple(query.shape)}")
        print(f"query stride: {query.stride()}")
        print(f"query contiguous: {query.is_contiguous()}")

        assert not query.is_contiguous()
        assert query.stride(-1) == 1

    reference_state = initial_state.clone()
    cuda_state = initial_state.clone()

    cuda_output_workspace = torch.empty_like(
        value
    )

    reference_output = None
    cuda_output = None

    original_output_pointer = (
        cuda_output_workspace.data_ptr()
    )

    for _ in range(NUM_DECODE_STEPS):
        reference_output = (
            state_aware_gdn_decode_reference(
                query=query,
                key=key,
                value=value,
                g=g,
                beta=beta,
                recurrent_state_pool=reference_state,
                state_slot_ids=state_slot_ids,
                gdn_index=GDN_INDEX,
                scale=KEY_DIM ** -0.5,
            )
        )

        cuda_output = (
            state_aware_gdn_decode_cuda(
                query=query,
                key=key,
                value=value,
                g=g,
                beta=beta,
                recurrent_state_pool=cuda_state,
                state_slot_ids=state_slot_ids,
                gdn_index=GDN_INDEX,
                output=cuda_output_workspace,
                scale=KEY_DIM ** -0.5,
            )
        )

        assert (
            cuda_output.data_ptr()
            == original_output_pointer
        ), (
            "CUDA Extension did not reuse the "
            "preallocated output workspace"
        )

    torch.cuda.synchronize()

    assert reference_output is not None
    assert cuda_output is not None

    state_error = (
        reference_state
        - cuda_state
    ).abs().max().item()

    output_error = (
        reference_output.float()
        - cuda_output.float()
    ).abs().max().item()

    untouched_error = compute_untouched_error(
        initial_state=initial_state,
        cuda_state=cuda_state,
        state_slot_ids=state_slot_ids,
    )

    print(
        f"B={batch_size}: "
        f"state_error={state_error:.9e}, "
        f"output_error={output_error:.9e}, "
        f"untouched_error={untouched_error:.9e}"
    )

    assert state_error <= 5e-5, (
        f"State error is too large: {state_error}"
    )

    assert output_error <= 2e-2, (
        f"Output error is too large: {output_error}"
    )

    assert untouched_error == 0.0, (
        "CUDA Kernel modified an unrelated "
        "State Slot or GDN layer"
    )


@torch.inference_mode()
def main() -> None:
    torch.manual_seed(20260907)
    torch.cuda.manual_seed_all(20260907)

    properties = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    )

    print(f"GPU: {properties.name}")
    print(
        "Testing Python -> C++ -> CUDA Extension"
    )

    print(
        "Shape: "
        f"H={NUM_HEADS}, "
        f"Dk={KEY_DIM}, "
        f"Dv={VALUE_DIM}"
    )

    print(
        f"Decode steps per case: "
        f"{NUM_DECODE_STEPS}"
    )

    for batch_size in (
        1,
        2,
        4,
        8,
        16,
    ):
        run_case(
            batch_size=batch_size
        )

    print(
        "Part CUDA Extension-1 PASSED: "
        "binding, BF16 inputs, FP32 state, "
        "multi-batch slot addressing, recurrent "
        "updates and output workspace reuse are correct."
    )


if __name__ == "__main__":
    main()