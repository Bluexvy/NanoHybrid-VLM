from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


REPO_ROOT = Path(
    "/workspace/nano-vllm"
)

MODEL_PATH = (
    "/workspace/models/Qwen3.5-9B"
)

OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts"
    / "cuda_graph"
    / "benchmark"
)

EAGER_PATH = (
    OUTPUT_DIR / "eager.json"
)

GRAPH_PATH = (
    OUTPUT_DIR / "graph.json"
)
BATCH_SIZES = (
    1,
    2,
    4,
    8,
    16,
)

def percentile(
    values: list[float],
    ratio: float,
) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)

    position = (
        len(ordered) - 1
    ) * ratio

    lower = int(position)
    upper = min(
        lower + 1,
        len(ordered) - 1,
    )

    weight = position - lower

    return (
        ordered[lower] * (1.0 - weight)
        + ordered[upper] * weight
    )


def make_prompt(
    tokenizer,
) -> str:
    return tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": (
                    "请用一段话解释线性注意力的"
                    "基本原理和主要优点。"
                ),
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def run_generation_case(
    *,
    llm: LLM,
    prompt: str,
    batch_size: int,
    output_tokens: int,
) -> dict[str, object]:
    """
    手动调用 add_request() 和 step()，
    单独累计 Decode 阶段数据。
    """

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=output_tokens,
        ignore_eos=True,
    )

    start_graph_replays = (
        llm.model_runner
        .num_hybrid_graph_replays
    )

    start_eager_fallbacks = (
        llm.model_runner
        .num_hybrid_graph_eager_fallbacks
    )

    start_time = perf_counter()

    for _ in range(batch_size):
        llm.add_request(
            prompt,
            sampling_params,
        )

    decode_step_ms: list[float] = []

    num_decode_tokens = 0
    num_prefill_tokens = 0

    while not llm.is_finished():
        _, stats = llm.step()

        num_prefill_tokens += (
            stats.num_prefill_tokens
        )

        if stats.num_decode_tokens > 0:
            num_decode_tokens += (
                stats.num_decode_tokens
            )

            decode_step_ms.append(
                stats.decode_elapsed
                * 1000.0
            )

    torch.cuda.synchronize()

    e2e_seconds = (
        perf_counter() - start_time
    )

    decode_elapsed_seconds = (
        sum(decode_step_ms)
        / 1000.0
    )

    num_decode_steps = len(
        decode_step_ms
    )

    if num_decode_steps == 0:
        raise RuntimeError(
            "No Decode steps were executed"
        )

    # 对固定并发请求而言，每轮每条请求生成一个token。
    #
    # 因此每个Decode step的耗时，也近似对应
    # 单条请求相邻两个token之间的时间。
    average_tpot_ms = (
        decode_elapsed_seconds
        / num_decode_steps
        * 1000.0
    )

    decode_tokens_per_second = (
        num_decode_tokens
        / decode_elapsed_seconds
    )

    graph_replays = (
        llm.model_runner
        .num_hybrid_graph_replays
        - start_graph_replays
    )

    eager_fallbacks = (
        llm.model_runner
        .num_hybrid_graph_eager_fallbacks
        - start_eager_fallbacks
    )

    return {
        "batch_size": batch_size,
        "output_tokens": output_tokens,
        "num_prefill_tokens": (
            num_prefill_tokens
        ),
        "num_decode_tokens": (
            num_decode_tokens
        ),
        "num_decode_steps": (
            num_decode_steps
        ),
        "decode_elapsed_seconds": (
            decode_elapsed_seconds
        ),
        "decode_tokens_per_second": (
            decode_tokens_per_second
        ),
        "average_tpot_ms": (
            average_tpot_ms
        ),
        "step_latency_ms": {
            "mean": statistics.mean(
                decode_step_ms
            ),
            "p50": percentile(
                decode_step_ms,
                0.50,
            ),
            "p95": percentile(
                decode_step_ms,
                0.95,
            ),
            "p99": percentile(
                decode_step_ms,
                0.99,
            ),
            "max": max(
                decode_step_ms
            ),
        },
        "e2e_seconds": e2e_seconds,
        "graph_replays": graph_replays,
        "eager_fallbacks": (
            eager_fallbacks
        ),
        "raw_step_latencies": (
            decode_step_ms
        ),
    }


def aggregate_case(
    iterations: list[
        dict[str, object]
    ],
) -> dict[str, object]:
    total_decode_tokens = sum(
        int(item["num_decode_tokens"])
        for item in iterations
    )

    total_decode_elapsed = sum(
        float(
            item["decode_elapsed_seconds"]
        )
        for item in iterations
    )

    all_step_latencies = [
        float(latency)
        for item in iterations
        for latency in item[
            "raw_step_latencies"
        ]
    ]

    total_steps = len(
        all_step_latencies
    )

    return {
        "batch_size": (
            iterations[0]["batch_size"]
        ),
        "output_tokens": (
            iterations[0]["output_tokens"]
        ),
        "repeats": len(iterations),
        "total_decode_tokens": (
            total_decode_tokens
        ),
        "total_decode_steps": (
            total_steps
        ),
        "decode_tokens_per_second": (
            total_decode_tokens
            / total_decode_elapsed
        ),
        "average_tpot_ms": (
            total_decode_elapsed
            / total_steps
            * 1000.0
        ),
        "step_latency_ms": {
            "mean": statistics.mean(
                all_step_latencies
            ),
            "p50": percentile(
                all_step_latencies,
                0.50,
            ),
            "p95": percentile(
                all_step_latencies,
                0.95,
            ),
            "p99": percentile(
                all_step_latencies,
                0.99,
            ),
            "max": max(
                all_step_latencies
            ),
        },
        "mean_e2e_seconds": (
            statistics.mean(
                float(item["e2e_seconds"])
                for item in iterations
            )
        ),
        "graph_replays": sum(
            int(item["graph_replays"])
            for item in iterations
        ),
        "eager_fallbacks": sum(
            int(item["eager_fallbacks"])
            for item in iterations
        ),
    }


def run_child(
    *,
    mode: str,
    output_path: Path,
    repeats: int,
    output_tokens: int,
) -> None:
    if mode not in {
        "eager",
        "graph",
    }:
        raise ValueError(mode)

    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_PATH
        )
    )

    prompt = make_prompt(
        tokenizer
    )

    initialization_start = (
        perf_counter()
    )

    llm = LLM(
        MODEL_PATH,
        enforce_eager=(
            mode == "eager"
        ),
        tensor_parallel_size=1,
        max_model_len=512,
        max_num_batched_tokens=1024,
        # 最大 Decode Graph bucket 是 B=8，
        # 因此 Scheduler 至少允许8条活跃请求。
        max_num_seqs=max(BATCH_SIZES),

        # 每条活跃 Qwen3.5 Sequence 都需要
        # 独占一个 GDN state slot。
        num_state_slots=max(BATCH_SIZES),

        gpu_memory_utilization=0.78,

        # 初始化期间分别捕获四张 Decode Graph。
        hybrid_cuda_graph_batch_sizes=(
            BATCH_SIZES
        ),
    )

    torch.cuda.synchronize()

    initialization_seconds = (
        perf_counter()
        - initialization_start
    )
    # 分别预热 B=1/2/4/8。
    #
    # Eager 模式用于消除首次算子调用影响；
    # Graph 模式用于让四个 bucket 都进入稳定状态。
    # 预热数据不计入正式结果。
    for batch_size in BATCH_SIZES:
        run_generation_case(
            llm=llm,
            prompt=prompt,
            batch_size=batch_size,
            output_tokens=16,
        )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    case_results: dict[
        str,
        object,
    ] = {}

    for batch_size in BATCH_SIZES:
        iterations = []

        for repeat_index in range(
            repeats
        ):
            result = run_generation_case(
                llm=llm,
                prompt=prompt,
                batch_size=batch_size,
                output_tokens=output_tokens,
            )

            iterations.append(
                result
            )

            print(
                f"{mode} B={batch_size} "
                f"repeat={repeat_index + 1}: "
                f"{result['decode_tokens_per_second']:.2f} tok/s, "
                f"TPOT={result['average_tpot_ms']:.3f} ms"
            )

        case_results[
            f"batch_{batch_size}"
        ] = aggregate_case(
            iterations
        )

    torch.cuda.synchronize()

    workspace = (
        llm.model_runner
        .hybrid_graph_workspace
    )

    payload = {
        "mode": mode,
        "batch_sizes": list(
            BATCH_SIZES
        ),
        "repeats": repeats,
        "output_tokens": output_tokens,
        "initialization_seconds": (
            initialization_seconds
        ),
        "cases": case_results,
        "memory": {
            "current_allocated_mib": (
                torch.cuda.memory_allocated()
                / 1024**2
            ),
            "peak_allocated_mib": (
                torch.cuda.max_memory_allocated()
                / 1024**2
            ),
            "current_reserved_mib": (
                torch.cuda.memory_reserved()
                / 1024**2
            ),
            "workspace_mib": (
                0.0
                if workspace is None
                else (
                    workspace.allocated_bytes
                    / 1024**2
                )
            ),
            "capture_allocated_delta_mib": (
                llm.model_runner
                .hybrid_graph_capture_allocated_bytes
                / 1024**2
            ),
        },
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nSaved {mode} result to "
        f"{output_path}"
    )


def compare_results() -> None:
    with EAGER_PATH.open(
        encoding="utf-8",
    ) as file:
        eager = json.load(file)

    with GRAPH_PATH.open(
        encoding="utf-8",
    ) as file:
        graph = json.load(file)

    print(
        "\n"
        + "=" * 72
    )
    print(
        "Hybrid CUDA Graph Benchmark"
    )
    print(
        "=" * 72
    )

    for batch_size in BATCH_SIZES:
        key = f"batch_{batch_size}"

        eager_case = eager["cases"][key]
        graph_case = graph["cases"][key]

        eager_throughput = float(
            eager_case[
                "decode_tokens_per_second"
            ]
        )

        graph_throughput = float(
            graph_case[
                "decode_tokens_per_second"
            ]
        )

        eager_tpot = float(
            eager_case[
                "average_tpot_ms"
            ]
        )

        graph_tpot = float(
            graph_case[
                "average_tpot_ms"
            ]
        )

        throughput_change = (
            graph_throughput
            / eager_throughput
            - 1.0
        ) * 100.0

        tpot_change = (
            graph_tpot
            / eager_tpot
            - 1.0
        ) * 100.0

        print(
            f"\nBatch size {batch_size}:"
        )

        print(
            f"  Eager throughput: "
            f"{eager_throughput:.2f} tok/s"
        )

        print(
            f"  Graph throughput: "
            f"{graph_throughput:.2f} tok/s"
        )

        print(
            f"  Throughput change: "
            f"{throughput_change:+.2f}%"
        )

        print(
            f"  Eager TPOT: "
            f"{eager_tpot:.3f} ms"
        )

        print(
            f"  Graph TPOT: "
            f"{graph_tpot:.3f} ms"
        )

        print(
            f"  TPOT change: "
            f"{tpot_change:+.2f}%"
        )

        print(
            f"  Graph replays: "
            f"{graph_case['graph_replays']}"
        )

    eager_memory = eager["memory"]
    graph_memory = graph["memory"]

    print(
        "\nMemory:"
    )

    print(
        "  Eager current allocated: "
        f"{eager_memory['current_allocated_mib']:.2f} MiB"
    )

    print(
        "  Graph current allocated: "
        f"{graph_memory['current_allocated_mib']:.2f} MiB"
    )

    print(
        "  Graph Workspace: "
        f"{graph_memory['workspace_mib']:.2f} MiB"
    )

    print(
        "  Graph capture delta: "
        f"{graph_memory['capture_allocated_delta_mib']:.2f} MiB"
    )


def run_all(
    repeats: int,
    output_tokens: int,
) -> None:
    script_path = Path(
        __file__
    ).resolve()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for mode, output_path in [
        ("eager", EAGER_PATH),
        ("graph", GRAPH_PATH),
    ]:
        command = [
            sys.executable,
            str(script_path),
            "--mode",
            mode,
            "--output",
            str(output_path),
            "--repeats",
            str(repeats),
            "--output-tokens",
            str(output_tokens),
        ]

        print(
            "\nRunning:",
            " ".join(command),
        )

        subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
        )

    compare_results()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "all",
            "eager",
            "graph",
        ],
        default="all",
    )

    parser.add_argument(
        "--output",
        type=Path,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--output-tokens",
        type=int,
        default=128,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "all":
        run_all(
            repeats=args.repeats,
            output_tokens=(
                args.output_tokens
            ),
        )
        return

    if args.output is None:
        raise ValueError(
            "--output is required "
            "for child mode"
        )

    run_child(
        mode=args.mode,
        output_path=args.output,
        repeats=args.repeats,
        output_tokens=args.output_tokens,
    )


if __name__ == "__main__":
    main()