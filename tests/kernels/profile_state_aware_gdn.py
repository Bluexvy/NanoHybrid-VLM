from __future__ import annotations

import argparse
import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    schedule,
)

from fla.ops.gated_delta_rule import (
    fused_recurrent_gated_delta_rule,
)

from nanovllm.kernels.state_aware_gdn import (
    state_aware_gdn_decode_triton,
)


H = 32
DK = 128
DV = 128
NUM_GDN_LAYERS = 24


@contextmanager
def profile_range(name: str):
    """
    在 PyTorch Profiler 时间线中增加一个语义区间。

    例如：
        nano::old_state_gather
        nano::old_fla_recurrence
        nano::old_state_scatter
        nano::state_aware_triton
    """
    with record_function(name):
        yield


def create_inputs(
    batch_size: int,
    num_slots: int,
    device: torch.device,
):
    torch.manual_seed(1234)

    q = torch.randn(
        batch_size,
        H,
        DK,
        device=device,
        dtype=torch.bfloat16,
    )

    k = torch.randn(
        batch_size,
        H,
        DK,
        device=device,
        dtype=torch.bfloat16,
    )

    v = torch.randn(
        batch_size,
        H,
        DV,
        device=device,
        dtype=torch.bfloat16,
    )

    beta = torch.sigmoid(
        torch.randn(
            batch_size,
            H,
            device=device,
            dtype=torch.float32,
        )
    )

    g = -torch.nn.functional.softplus(
        torch.randn(
            batch_size,
            H,
            device=device,
            dtype=torch.float32,
        )
    )

    state_pool = torch.zeros(
        num_slots,
        NUM_GDN_LAYERS,
        H,
        DK,
        DV,
        device=device,
        dtype=torch.float32,
    )

    # 当前算子接口要求 slot ID 使用 int64。
    state_slot_ids = torch.arange(
        batch_size,
        device=device,
        dtype=torch.long,
    ) * 2

    # output 必须与传入算子的 value
    # 保持相同的形状和 dtype。
    #
    # v 原本为 [B, H, Dv]、BF16；
    # 添加 Decode 的 T=1 维度后为 [B, 1, H, Dv]。
    output = torch.empty_like(
        v.unsqueeze(1)
    )

    return {
        "q": q,
        "k": k,
        "v": v,
        "beta": beta,
        "g": g,
        "state_pool": state_pool,
        "state_slot_ids": state_slot_ids,
        "output": output,
    }


def run_old_fla_step(
    tensors: dict[str, torch.Tensor],
    gdn_layer_idx: int,
):
    """
    原来的执行路径：

        State Pool
            ↓ index_select
        Batched recurrent_state
            ↓ FLA
        final_state
            ↓ index_copy_
        State Pool
    """

    state_pool = tensors["state_pool"]
    state_slot_ids = tensors["state_slot_ids"]

    # 原始输入：
    #
    # q/k:  [B, H, Dk]
    # v:    [B, H, Dv]
    # g:    [B, H]
    # beta: [B, H]
    #
    # FLA 需要显式的序列长度维度：
    #
    # q/k:  [B, T, H, Dk]
    # v:    [B, T, H, Dv]
    # g:    [B, T, H]
    # beta: [B, T, H]
    #
    # Decode 每次只处理一个 token，所以 T=1。
    q = tensors["q"].unsqueeze(1)
    k = tensors["k"].unsqueeze(1)
    v = tensors["v"].unsqueeze(1)

    beta = tensors["beta"].unsqueeze(1)
    g = tensors["g"].unsqueeze(1)

    with profile_range("nano::old_state_gather"):
        batched_state = torch.index_select(
            state_pool[:, gdn_layer_idx],
            dim=0,
            index=state_slot_ids.long(),
        )

    with profile_range("nano::old_fla_recurrence"):
        output, final_state = (
            fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                beta=beta,
                g=g,

                # 和项目正式运行路径保持一致。
                scale=DK ** -0.5,

                initial_state=batched_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        )

    with profile_range("nano::old_state_scatter"):
        state_pool[:, gdn_layer_idx].index_copy_(
            dim=0,
            index=state_slot_ids.long(),
            source=final_state,
        )

    return output
def run_state_aware_step(
    tensors: dict[str, torch.Tensor],
    gdn_layer_idx: int,
    block_value: int,
):
    # 与 FLA 路径使用相同的输入布局。
    #
    # query/key: [B, 1, H, Dk]
    # value:     [B, 1, H, Dv]
    # g/beta:    [B, 1, H]
    query = tensors["q"].unsqueeze(1)
    key = tensors["k"].unsqueeze(1)
    value = tensors["v"].unsqueeze(1)

    g = tensors["g"].unsqueeze(1)
    beta = tensors["beta"].unsqueeze(1)

    with profile_range(
        "nano::state_aware_triton"
    ):
        output = state_aware_gdn_decode_triton(
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            recurrent_state_pool=(
                tensors["state_pool"]
            ),
            state_slot_ids=(
                tensors["state_slot_ids"]
            ),

            # 左边是算子定义的参数名；
            # 右边是本函数中的局部变量。
            gdn_index=gdn_layer_idx,

            output=tensors["output"],
            scale=DK ** -0.5,
            block_value=block_value,
        )

    return output

def run_one_step(
    backend: str,
    tensors: dict[str, torch.Tensor],
    gdn_layer_idx: int,
    block_value: int,
):
    if backend == "fla":
        return run_old_fla_step(
            tensors=tensors,
            gdn_layer_idx=gdn_layer_idx,
        )

    if backend == "triton":
        return run_state_aware_step(
            tensors=tensors,
            gdn_layer_idx=gdn_layer_idx,
            block_value=block_value,
        )

    raise ValueError(
        f"Unsupported backend: {backend}"
    )


def benchmark_with_cuda_events(
    backend: str,
    tensors: dict[str, torch.Tensor],
    gdn_layer_idx: int,
    block_value: int,
    warmup_steps: int = 20,
    benchmark_steps: int = 100,
):
    """
    用 CUDA Event 计算平均 GPU 延迟。

    Profiler 用于回答“时间花在哪里”；
    CUDA Event 用于回答“总共花了多少时间”。
    """

    for _ in range(warmup_steps):
        run_one_step(
            backend=backend,
            tensors=tensors,
            gdn_layer_idx=gdn_layer_idx,
            block_value=block_value,
        )

    torch.cuda.synchronize()

    start = torch.cuda.Event(
        enable_timing=True
    )

    end = torch.cuda.Event(
        enable_timing=True
    )

    start.record()

    for _ in range(benchmark_steps):
        run_one_step(
            backend=backend,
            tensors=tensors,
            gdn_layer_idx=gdn_layer_idx,
            block_value=block_value,
        )

    end.record()
    end.synchronize()

    total_ms = start.elapsed_time(end)

    return {
        "benchmark_steps": benchmark_steps,
        "total_ms": total_ms,
        "mean_ms": total_ms / benchmark_steps,
    }


def get_event_cuda_time_us(event) -> float:
    """
    兼容不同 PyTorch 版本的字段命名。
    """

    if hasattr(event, "device_time_total"):
        return float(event.device_time_total)

    if hasattr(event, "cuda_time_total"):
        return float(event.cuda_time_total)

    return 0.0


def get_event_self_cuda_time_us(event) -> float:
    if hasattr(event, "self_device_time_total"):
        return float(
            event.self_device_time_total
        )

    if hasattr(event, "self_cuda_time_total"):
        return float(
            event.self_cuda_time_total
        )

    return 0.0


def collect_profiler_summary(prof):
    key_averages = prof.key_averages()

    semantic_ranges = {}
    raw_cuda_events = []

    for event in key_averages:
        cuda_total_us = (
            get_event_cuda_time_us(event)
        )

        self_cuda_us = (
            get_event_self_cuda_time_us(event)
        )

        if event.key.startswith("nano::"):
            semantic_ranges[event.key] = {
                "count": int(event.count),
                "cuda_total_us": (
                    cuda_total_us
                ),
                "cuda_mean_us": (
                    cuda_total_us
                    / max(int(event.count), 1)
                ),
            }

        elif self_cuda_us > 0:
            raw_cuda_events.append(
                {
                    "name": event.key,
                    "count": int(event.count),
                    "self_cuda_total_us": (
                        self_cuda_us
                    ),
                    "self_cuda_mean_us": (
                        self_cuda_us
                        / max(int(event.count), 1)
                    ),
                }
            )

    raw_cuda_events.sort(
        key=lambda item: (
            item["self_cuda_total_us"]
        ),
        reverse=True,
    )

    return {
        "semantic_ranges": semantic_ranges,
        "top_cuda_events": (
            raw_cuda_events[:20]
        ),
    }


def profile_case(
    backend: str,
    batch_size: int,
    block_value: int,
    output_dir: Path,
):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    device = torch.device("cuda")

    # 采用非连续 slot：
    # B=16 时实际访问 0,2,4,...,30。
    num_slots = batch_size * 2

    tensors = create_inputs(
        batch_size=batch_size,
        num_slots=num_slots,
        device=device,
    )

    gdn_layer_idx = 7

    # 先触发 Triton JIT、FLA 初始化和显存分配。
    for _ in range(10):
        run_one_step(
            backend=backend,
            tensors=tensors,
            gdn_layer_idx=gdn_layer_idx,
            block_value=block_value,
        )

    torch.cuda.synchronize()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trace_path = (
        output_dir / "trace.json"
    )

    table_path = (
        output_dir / "operator_table.txt"
    )

    summary_path = (
        output_dir / "summary.json"
    )

    # wait=1：第一步不记录
    # warmup=2：后两步用于 Profiler 热身
    # active=8：真正记录八个 Decode step
    profiler_schedule = schedule(
        wait=1,
        warmup=2,
        active=8,
        repeat=1,
    )

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        schedule=profiler_schedule,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(11):
            with profile_range(
                "nano::gdn_decode_step"
            ):
                run_one_step(
                    backend=backend,
                    tensors=tensors,
                    gdn_layer_idx=(
                        gdn_layer_idx
                    ),
                    block_value=block_value,
                )

            prof.step()

    prof.export_chrome_trace(
        str(trace_path)
    )

    operator_table = (
        prof.key_averages().table(
            sort_by=(
                "self_device_time_total"
            ),
            row_limit=50,
        )
    )

    table_path.write_text(
        operator_table,
        encoding="utf-8",
    )

    profiler_summary = (
        collect_profiler_summary(prof)
    )

    event_benchmark = (
        benchmark_with_cuda_events(
            backend=backend,
            tensors=tensors,
            gdn_layer_idx=gdn_layer_idx,
            block_value=block_value,
        )
    )

    summary = {
        "backend": backend,
        "batch_size": batch_size,
        "block_value": block_value,
        "num_slots": num_slots,
        "gdn_layer_idx": gdn_layer_idx,
        "state_pool_shape": list(
            tensors["state_pool"].shape
        ),
        "state_pool_dtype": str(
            tensors["state_pool"].dtype
        ),
        "slot_ids": (
            tensors["state_slot_ids"]
            .cpu()
            .tolist()
        ),
        "event_benchmark": event_benchmark,
        **profiler_summary,
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("Backend:", backend)
    print("Batch size:", batch_size)
    print("BLOCK_VALUE:", block_value)
    print(
        "Mean CUDA Event latency:",
        f"{event_benchmark['mean_ms']:.6f} ms",
    )

    print()
    print("Semantic ranges:")

    for name, values in (
        summary["semantic_ranges"].items()
    ):
        print(
            f"  {name}: "
            f"{values['cuda_total_us']:.3f} us"
        )

    print()
    print("Top CUDA events:")

    for item in summary[
        "top_cuda_events"
    ][:10]:
        print(
            f"  {item['name']}: "
            f"{item['self_cuda_total_us']:.3f} us"
        )

    print()
    print("Trace:", trace_path)
    print("Table:", table_path)
    print("Summary:", summary_path)


def run_all_cases(
    batch_size: int,
    output_root: Path,
):
    cases = [
        ("fla", 32),
        ("triton", 16),
        ("triton", 32),
        ("triton", 64),
    ]

    current_file = Path(__file__).resolve()

    for backend, block_value in cases:
        case_name = (
            f"{backend}_b{batch_size}"
            f"_tile{block_value}"
        )

        case_dir = output_root / case_name

        command = [
            sys.executable,
            str(current_file),
            "--backend",
            backend,
            "--batch-size",
            str(batch_size),
            "--block-value",
            str(block_value),
            "--output-dir",
            str(case_dir),
        ]

        print()
        print("=" * 72)
        print("Running:", " ".join(command))
        print("=" * 72)

        subprocess.run(
            command,
            check=True,
        )

    summaries = {}

    for backend, block_value in cases:
        case_name = (
            f"{backend}_b{batch_size}"
            f"_tile{block_value}"
        )

        summary_path = (
            output_root
            / case_name
            / "summary.json"
        )

        summaries[case_name] = json.loads(
            summary_path.read_text(
                encoding="utf-8"
            )
        )

    fla_name = (
        f"fla_b{batch_size}_tile32"
    )

    fla_ms = summaries[fla_name][
        "event_benchmark"
    ]["mean_ms"]

    print()
    print("=" * 72)
    print("Final comparison")
    print("=" * 72)
    print(
        f"FLA Gather + FLA + Scatter: "
        f"{fla_ms:.6f} ms"
    )

    best_name = None
    best_ms = float("inf")

    for block_value in (16, 32, 64):
        name = (
            f"triton_b{batch_size}"
            f"_tile{block_value}"
        )

        triton_ms = summaries[name][
            "event_benchmark"
        ]["mean_ms"]

        speedup = fla_ms / triton_ms

        reduction = (
            (fla_ms - triton_ms)
            / fla_ms
            * 100.0
        )

        print(
            f"Triton tile={block_value}: "
            f"{triton_ms:.6f} ms, "
            f"speedup={speedup:.3f}x, "
            f"latency reduction="
            f"{reduction:.2f}%"
        )

        if triton_ms < best_ms:
            best_ms = triton_ms
            best_name = name

    print()
    print("Best configuration:", best_name)
    print("Best latency:", f"{best_ms:.6f} ms")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backend",
        choices=[
            "all",
            "fla",
            "triton",
        ],
        default="all",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--block-value",
        type=int,
        choices=[16, 32, 64],
        default=32,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/kernels/"
            "state_aware_profile"
        ),
    )

    args = parser.parse_args()

    if args.backend == "all":
        run_all_cases(
            batch_size=args.batch_size,
            output_root=args.output_dir,
        )
        return

    profile_case(
        backend=args.backend,
        batch_size=args.batch_size,
        block_value=args.block_value,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()