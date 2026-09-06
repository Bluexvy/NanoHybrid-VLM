from __future__ import annotations

import torch
import torch.nn.functional as F

from fla.ops.gated_delta_rule import (
    fused_recurrent_gated_delta_rule,
)

from nanovllm.kernels.state_aware_gdn import (
    state_aware_gdn_decode_reference,
)


DEVICE = torch.device("cuda")
INPUT_DTYPE = torch.bfloat16

NUM_HEADS = 32
KEY_DIM = 128
VALUE_DIM = 128

NUM_GDN_LAYERS = 1
NUM_STEPS = 8


def make_decode_inputs(
    batch_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    query = torch.randn(
        batch_size,
        1,
        NUM_HEADS,
        KEY_DIM,
        device=DEVICE,
        dtype=INPUT_DTYPE,
    )

    key = torch.randn(
        batch_size,
        1,
        NUM_HEADS,
        KEY_DIM,
        device=DEVICE,
        dtype=INPUT_DTYPE,
    )

    value = torch.randn(
        batch_size,
        1,
        NUM_HEADS,
        VALUE_DIM,
        device=DEVICE,
        dtype=INPUT_DTYPE,
    )

    # Qwen3.5 中 g 是负数。
    #
    # softplus(x) > 0
    # 因此：
    #     -softplus(x) < 0
    #
    # exp(g) 会位于 0 和 1 之间。
    g = -F.softplus(
        torch.randn(
            batch_size,
            1,
            NUM_HEADS,
            device=DEVICE,
            dtype=torch.float32,
        )
    )

    # beta 是写入门，位于 0 和 1 之间。
    beta = torch.sigmoid(
        torch.randn(
            batch_size,
            1,
            NUM_HEADS,
            device=DEVICE,
            dtype=torch.float32,
        )
    ).to(INPUT_DTYPE)

    return (
        query,
        key,
        value,
        g,
        beta,
    )


def selected_states(
    state_pool: torch.Tensor,
    state_slot_ids: torch.Tensor,
) -> torch.Tensor:
    return torch.index_select(
        state_pool[:, 0],
        dim=0,
        index=state_slot_ids,
    )


def run_case(
    batch_size: int,
) -> None:
    print()
    print("=" * 72)
    print(f"Testing batch size B={batch_size}")
    print("=" * 72)

    num_slots = batch_size + 8

    # 从 num_slots 中随机挑选 B 个不同的 slot。
    #
    # randperm 本身不会产生重复值，
    # 同时得到的 slot 通常也是乱序、不连续的。
    state_slot_ids = torch.randperm(
        num_slots,
        device=DEVICE,
        dtype=torch.long,
    )[:batch_size]

    if (
        torch.unique(state_slot_ids).numel()
        != batch_size
    ):
        raise RuntimeError(
            "Test generated duplicate slots"
        )

    initial_pool = (
        torch.randn(
            num_slots,
            NUM_GDN_LAYERS,
            NUM_HEADS,
            KEY_DIM,
            VALUE_DIM,
            device=DEVICE,
            dtype=torch.float32,
        )
        * 0.01
    )

    fla_pool = initial_pool.clone()
    reference_pool = initial_pool.clone()

    selected_slot_set = set(
        state_slot_ids.cpu().tolist()
    )

    untouched_slots = [
        slot
        for slot in range(num_slots)
        if slot not in selected_slot_set
    ]

    maximum_output_error = 0.0
    maximum_state_error = 0.0

    for step in range(NUM_STEPS):
        (
            query,
            key,
            value,
            g,
            beta,
        ) = make_decode_inputs(batch_size)

        # -------------------------------------------------
        # 旧路径：
        #
        # State Pool
        #   -> Gather
        #   -> FLA
        #   -> final_state
        #   -> Scatter
        # -------------------------------------------------

        fla_initial_state = selected_states(
            fla_pool,
            state_slot_ids,
        )

        (
            fla_output,
            fla_final_state,
        ) = fused_recurrent_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            scale=KEY_DIM ** -0.5,
            initial_state=fla_initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

        fla_pool[:, 0].index_copy_(
            0,
            state_slot_ids,
            fla_final_state,
        )

        # -------------------------------------------------
        # 新接口的 PyTorch Reference：
        #
        # 直接根据 slot ID 读写状态池。
        # -------------------------------------------------

        reference_output = (
            state_aware_gdn_decode_reference(
                query=query,
                key=key,
                value=value,
                g=g,
                beta=beta,
                recurrent_state_pool=(
                    reference_pool
                ),
                state_slot_ids=state_slot_ids,
                gdn_index=0,
                scale=KEY_DIM ** -0.5,
                use_qk_l2norm=True,
            )
        )

        output_error = (
            reference_output.float()
            - fla_output.float()
        ).abs().max().item()

        state_error = (
            selected_states(
                reference_pool,
                state_slot_ids,
            )
            - selected_states(
                fla_pool,
                state_slot_ids,
            )
        ).abs().max().item()

        maximum_output_error = max(
            maximum_output_error,
            output_error,
        )

        maximum_state_error = max(
            maximum_state_error,
            state_error,
        )

        print(
            f"step={step:2d}  "
            f"output_max_abs={output_error:.6e}  "
            f"state_max_abs={state_error:.6e}"
        )

        torch.testing.assert_close(
            reference_output.float(),
            fla_output.float(),
            rtol=3e-2,
            atol=3e-2,
        )

        torch.testing.assert_close(
            selected_states(
                reference_pool,
                state_slot_ids,
            ),
            selected_states(
                fla_pool,
                state_slot_ids,
            ),
            rtol=3e-2,
            atol=5e-3,
        )

    # 没进入本轮 Batch 的 slot 必须完全不变。
    if untouched_slots:
        untouched_indices = torch.tensor(
            untouched_slots,
            device=DEVICE,
            dtype=torch.long,
        )

        actual_untouched = torch.index_select(
            reference_pool,
            dim=0,
            index=untouched_indices,
        )

        expected_untouched = torch.index_select(
            initial_pool,
            dim=0,
            index=untouched_indices,
        )

        if not torch.equal(
            actual_untouched,
            expected_untouched,
        ):
            raise AssertionError(
                "State-aware reference modified an "
                "unselected state slot"
            )

    print(
        f"B={batch_size} passed: "
        f"max output error={maximum_output_error:.6e}, "
        f"max state error={maximum_state_error:.6e}"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this test"
        )

    torch.manual_seed(20260903)
    torch.cuda.manual_seed_all(20260903)

    for batch_size in (1, 4, 16):
        run_case(batch_size)

    torch.cuda.synchronize()

    print()
    print(
        "State-aware GDN Decode Reference passed "
        "for B=1/4/16."
    )


if __name__ == "__main__":
    main()