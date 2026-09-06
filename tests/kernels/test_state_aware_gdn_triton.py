from __future__ import annotations

import gc

import torch
import torch.nn.functional as F

from fla.ops.gated_delta_rule import (
    fused_recurrent_gated_delta_rule,
)

from nanovllm.kernels.state_aware_gdn import (
    state_aware_gdn_decode_triton,
)


DEVICE = torch.device("cuda")
INPUT_DTYPE = torch.bfloat16

NUM_HEADS = 32
KEY_DIM = 128
VALUE_DIM = 128
NUM_GDN_LAYERS = 1

CORRECTNESS_STEPS = 8
WARMUP_STEPS = 20
BENCHMARK_STEPS = 100


def make_inputs(
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

    g = -F.softplus(
        torch.randn(
            batch_size,
            1,
            NUM_HEADS,
            device=DEVICE,
            dtype=torch.float32,
        )
    )

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


def gather_states(
    state_pool: torch.Tensor,
    state_slot_ids: torch.Tensor,
) -> torch.Tensor:
    return torch.index_select(
        state_pool[:, 0],
        dim=0,
        index=state_slot_ids,
    )


@torch.inference_mode()
def run_fla_path(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    state_slot_ids: torch.Tensor,
) -> torch.Tensor:
    """
    旧路径：

        Gather
        -> FLA
        -> final_state
        -> Scatter
    """

    batched_state = gather_states(
        state_pool,
        state_slot_ids,
    )

    (
        output,
        final_state,
    ) = fused_recurrent_gated_delta_rule(
        q=query,
        k=key,
        v=value,
        g=g,
        beta=beta,
        scale=KEY_DIM ** -0.5,
        initial_state=batched_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    if final_state is None:
        raise RuntimeError(
            "FLA did not return final_state"
        )

    state_pool[:, 0].index_copy_(
        0,
        state_slot_ids,
        final_state,
    )

    return output


def benchmark_cuda(
    function,
    *,
    warmup_steps: int,
    benchmark_steps: int,
) -> float:
    """
    使用 CUDA Event 统计单次调用的平均 GPU 时间。

    返回单位：
        milliseconds
    """

    for _ in range(warmup_steps):
        function()

    torch.cuda.synchronize()

    start_event = torch.cuda.Event(
        enable_timing=True
    )

    end_event = torch.cuda.Event(
        enable_timing=True
    )

    start_event.record()

    for _ in range(benchmark_steps):
        function()

    end_event.record()

    torch.cuda.synchronize()

    total_ms = start_event.elapsed_time(
        end_event
    )

    return total_ms / benchmark_steps


@torch.inference_mode()
def run_case(
    batch_size: int,
) -> None:
    print()
    print("=" * 80)
    print(f"Testing B={batch_size}")
    print("=" * 80)

    num_slots = batch_size + 8

    state_slot_ids = torch.randperm(
        num_slots,
        device=DEVICE,
        dtype=torch.long,
    )[:batch_size]

    print(
        "state_slot_ids:",
        state_slot_ids.cpu().tolist(),
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
    triton_pool = initial_pool.clone()

    selected_slots = set(
        state_slot_ids.cpu().tolist()
    )

    untouched_slots = [
        slot
        for slot in range(num_slots)
        if slot not in selected_slots
    ]

    maximum_output_error = 0.0
    maximum_state_error = 0.0

    # =====================================================
    # 正确性：连续执行多个 Decode step
    # =====================================================

    for step in range(CORRECTNESS_STEPS):
        (
            query,
            key,
            value,
            g,
            beta,
        ) = make_inputs(batch_size)

        fla_output = run_fla_path(
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            state_pool=fla_pool,
            state_slot_ids=state_slot_ids,
        )

        triton_output = (
            state_aware_gdn_decode_triton(
                query=query,
                key=key,
                value=value,
                g=g,
                beta=beta,
                recurrent_state_pool=(
                    triton_pool
                ),
                state_slot_ids=state_slot_ids,
                gdn_index=0,
                scale=KEY_DIM ** -0.5,
                block_value=32,
            )
        )

        output_error = (
            triton_output.float()
            - fla_output.float()
        ).abs().max().item()

        state_error = (
            gather_states(
                triton_pool,
                state_slot_ids,
            )
            - gather_states(
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
            triton_output.float(),
            fla_output.float(),
            rtol=4e-2,
            atol=4e-2,
        )

        torch.testing.assert_close(
            gather_states(
                triton_pool,
                state_slot_ids,
            ),
            gather_states(
                fla_pool,
                state_slot_ids,
            ),
            rtol=4e-2,
            atol=8e-3,
        )

    # 检查未选中的 slots 没有被 Triton 误写。
    if untouched_slots:
        untouched_indices = torch.tensor(
            untouched_slots,
            device=DEVICE,
            dtype=torch.long,
        )

        actual_untouched = torch.index_select(
            triton_pool,
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
                "Triton Kernel modified an "
                "unselected state slot"
            )

    print()
    print(
        "Correctness passed:"
    )

    print(
        f"  maximum output error: "
        f"{maximum_output_error:.6e}"
    )

    print(
        f"  maximum state error:  "
        f"{maximum_state_error:.6e}"
    )

    # =====================================================
    # 性能测试
    # =====================================================

    (
        query,
        key,
        value,
        g,
        beta,
    ) = make_inputs(batch_size)

    benchmark_fla_pool = (
        initial_pool.clone()
    )

    benchmark_triton_pool = (
        initial_pool.clone()
    )

    # 预分配输出，避免把 torch.empty_like()
    # 的 Python/Allocator 开销算进 Triton 路径。
    triton_output_workspace = (
        torch.empty_like(value)
    )

    fla_ms = benchmark_cuda(
        lambda: run_fla_path(
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            state_pool=benchmark_fla_pool,
            state_slot_ids=state_slot_ids,
        ),
        warmup_steps=WARMUP_STEPS,
        benchmark_steps=BENCHMARK_STEPS,
    )

    triton_ms = benchmark_cuda(
        lambda: state_aware_gdn_decode_triton(
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            recurrent_state_pool=(
                benchmark_triton_pool
            ),
            state_slot_ids=state_slot_ids,
            gdn_index=0,
            output=triton_output_workspace,
            scale=KEY_DIM ** -0.5,
            block_value=32,
        ),
        warmup_steps=WARMUP_STEPS,
        benchmark_steps=BENCHMARK_STEPS,
    )

    speedup = fla_ms / triton_ms

    # 每个请求、每个 GDN 层的状态大小。
    state_bytes = (
        NUM_HEADS
        * KEY_DIM
        * VALUE_DIM
        * torch.tensor(
            [],
            dtype=torch.float32,
        ).element_size()
    )

    state_mib = state_bytes / 1024**2

    print()
    print("Performance:")
    print(
        f"  state per sequence/layer: "
        f"{state_mib:.2f} MiB"
    )
    print(
        f"  Gather + FLA + Scatter: "
        f"{fla_ms:.6f} ms"
    )
    print(
        f"  State-aware Triton:      "
        f"{triton_ms:.6f} ms"
    )
    print(
        f"  Combined-path speedup:   "
        f"{speedup:.3f}x"
    )

    del initial_pool
    del fla_pool
    del triton_pool
    del benchmark_fla_pool
    del benchmark_triton_pool
    del triton_output_workspace

    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required"
        )

    torch.manual_seed(20260903)
    torch.cuda.manual_seed_all(20260903)

    for batch_size in (
        1,
        4,
        16,
    ):
        run_case(batch_size)

    print()
    print(
        "State-aware Triton GDN Decode passed "
        "for B=1/4/16."
    )


if __name__ == "__main__":
    main()