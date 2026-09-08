from __future__ import annotations

import statistics

import torch

from causal_conv1d import causal_conv1d_update

from nanovllm.kernels.state_aware_gdn_cuda import (
    state_aware_causal_conv1d_cuda,
)


BATCH_SIZES = (1, 2, 4, 8, 16)
BLOCK_SIZES = (64, 128, 256, 512)

NUM_GDN_LAYERS = 24
NUM_CHANNELS = 8192
KERNEL_SIZE = 4

WARMUP_STEPS = 10
MEASURE_STEPS = 50
REPEATS = 5


@torch.inference_mode()
def run_original_engine_step(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    conv_state_pool: torch.Tensor,
    state_slot_ids: torch.Tensor,
) -> None:
    """
    模拟当前引擎真实的 Conv Decode 状态路径：

        1. 一次 Gather 全部 GDN 层的 conv_state；
        2. 每层 clone 出 new_conv_state；
        3. 调用 causal_conv1d_update；
        4. 每层 index_copy_ 回 State Pool。
    """

    gathered_states = torch.index_select(
        conv_state_pool,
        dim=0,
        index=state_slot_ids,
    )

    updated_states = []

    for gdn_index in range(NUM_GDN_LAYERS):
        layer_state = gathered_states[:, gdn_index].clone()

        causal_conv1d_update(
            x=inputs[gdn_index],
            conv_state=layer_state,
            weight=weights[gdn_index],
            bias=None,
            activation="silu",
        )

        updated_states.append(layer_state)

    for gdn_index in range(NUM_GDN_LAYERS):
        conv_state_pool[:, gdn_index].index_copy_(
            dim=0,
            index=state_slot_ids,
            source=updated_states[gdn_index],
        )


@torch.inference_mode()
def run_state_aware_step(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    conv_state_pool: torch.Tensor,
    state_slot_ids: torch.Tensor,
    output_workspace: torch.Tensor,
    block_size: int,
) -> None:
    """
    新路径：

        不建立 Batched Conv State；
        不 clone；
        不执行 index_copy_；
        每层 CUDA Kernel 直接更新 State Pool。
    """

    for gdn_index in range(NUM_GDN_LAYERS):
        state_aware_causal_conv1d_cuda(
            x=inputs[gdn_index],
            weight=weights[gdn_index],
            conv_state_pool=conv_state_pool,
            state_slot_ids=state_slot_ids,
            gdn_index=gdn_index,
            output=output_workspace[gdn_index],
            block_size=block_size,
        )


def measure_step(function) -> float:
    for _ in range(WARMUP_STEPS):
        function()

    torch.cuda.synchronize()

    measurements = []

    for _ in range(REPEATS):
        start_event = torch.cuda.Event(enable_timing=True)
        stop_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        for _ in range(MEASURE_STEPS):
            function()

        stop_event.record()
        stop_event.synchronize()

        average_step_us = start_event.elapsed_time(stop_event) * 1000.0 / MEASURE_STEPS
        measurements.append(average_step_us)

    return statistics.median(measurements)


@torch.inference_mode()
def run_case(batch_size: int) -> None:
    torch.manual_seed(20260908 + batch_size)
    torch.cuda.manual_seed_all(20260908 + batch_size)

    num_slots = batch_size + 4

    state_slot_ids = torch.arange(
        batch_size - 1,
        -1,
        -1,
        dtype=torch.long,
        device="cuda",
    ) + 2

    inputs = torch.randn(
        NUM_GDN_LAYERS,
        batch_size,
        NUM_CHANNELS,
        1,
        dtype=torch.bfloat16,
        device="cuda",
    ) * 0.05

    weights = torch.randn(
        NUM_GDN_LAYERS,
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

    original_pool = initial_pool.clone()
    state_aware_pool = initial_pool.clone()

    output_workspace = torch.empty(
        NUM_GDN_LAYERS,
        batch_size,
        NUM_CHANNELS,
        1,
        dtype=torch.bfloat16,
        device="cuda",
    )

    original_step_us = measure_step(
        lambda: run_original_engine_step(
            inputs=inputs,
            weights=weights,
            conv_state_pool=original_pool,
            state_slot_ids=state_slot_ids,
        )
    )

    block_results = {}

    for block_size in BLOCK_SIZES:
        state_aware_pool.copy_(initial_pool)

        step_us = measure_step(
            lambda block_size=block_size: run_state_aware_step(
                inputs=inputs,
                weights=weights,
                conv_state_pool=state_aware_pool,
                state_slot_ids=state_slot_ids,
                output_workspace=output_workspace,
                block_size=block_size,
            )
        )

        block_results[block_size] = step_us

    best_block_size = min(
        block_results,
        key=block_results.get,
    )

    best_step_us = block_results[best_block_size]
    speedup = original_step_us / best_step_us
    reduction = (1.0 - best_step_us / original_step_us) * 100.0

    print(
        f"{batch_size:2d} | "
        f"{original_step_us:12.3f} | "
        f"{block_results[64]:10.3f} | "
        f"{block_results[128]:11.3f} | "
        f"{block_results[256]:11.3f} | "
        f"{block_results[512]:11.3f} | "
        f"{best_block_size:4d} | "
        f"{speedup:7.3f}x | "
        f"{reduction:7.2f}%"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Shape: C={NUM_CHANNELS}, K={KERNEL_SIZE}")
    print(f"GDN layers per Decode step: {NUM_GDN_LAYERS}")
    print(f"Warmup steps: {WARMUP_STEPS}")
    print(f"Measured steps per repeat: {MEASURE_STEPS}")
    print(f"Repeats: {REPEATS}")

    print()
    print(
        " B | Original us | CUDA-64 us | CUDA-128 us | "
        "CUDA-256 us | CUDA-512 us | Best | Speedup | Reduction"
    )
    print(
        "---+-------------+------------+-------------+-------------+"
        "-------------+------+---------+----------"
    )

    for batch_size in BATCH_SIZES:
        run_case(batch_size)


if __name__ == "__main__":
    main()