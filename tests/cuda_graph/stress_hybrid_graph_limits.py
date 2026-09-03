from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoTokenizer

from nanovllm import LLM

from benchmark_hybrid_graph import (
    aggregate_case,
    make_prompt,
    run_generation_case,
)


REPO_ROOT = Path(
    "/workspace/nano-vllm"
)

MODEL_PATH = (
    "/workspace/models/Qwen3.5-9B"
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts"
    / "cuda_graph"
    / "stress_limits"
)

BASE_GRAPH_BUCKETS = (
    1,
    2,
    4,
    8,
    16,
)


def parse_batch_sizes(
    text: str,
) -> tuple[int, ...]:
    values = tuple(
        int(item.strip())
        for item in text.split(",")
        if item.strip()
    )

    if not values:
        raise ValueError(
            "At least one batch size is required"
        )

    if any(value <= 0 for value in values):
        raise ValueError(
            "Batch sizes must be positive"
        )

    if tuple(sorted(set(values))) != values:
        raise ValueError(
            "Batch sizes must be unique and "
            "strictly increasing"
        )

    return values


def mib(value: int) -> float:
    return value / 1024**2


def safe_cuda_memory() -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {
            "current_allocated_mib": None,
            "peak_allocated_mib": None,
            "current_reserved_mib": None,
            "peak_reserved_mib": None,
            "free_mib": None,
            "total_mib": None,
        }

    try:
        free_bytes, total_bytes = (
            torch.cuda.mem_get_info()
        )

        return {
            "current_allocated_mib": mib(
                torch.cuda.memory_allocated()
            ),
            "peak_allocated_mib": mib(
                torch.cuda.max_memory_allocated()
            ),
            "current_reserved_mib": mib(
                torch.cuda.memory_reserved()
            ),
            "peak_reserved_mib": mib(
                torch.cuda.max_memory_reserved()
            ),
            "free_mib": mib(free_bytes),
            "total_mib": mib(total_bytes),
        }

    except Exception:
        return {
            "current_allocated_mib": None,
            "peak_allocated_mib": None,
            "current_reserved_mib": None,
            "peak_reserved_mib": None,
            "free_mib": None,
            "total_mib": None,
        }


def write_json(
    path: Path,
    payload: dict[str, object],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )


def graph_buckets_for_worker(
    *,
    backend: str,
    batch_size: int,
    isolated_bucket: bool,
) -> tuple[int, ...]:
    if backend == "eager":
        return (1,)

    if isolated_bucket:
        return (batch_size,)

    return tuple(
        sorted(
            set(BASE_GRAPH_BUCKETS)
            | {batch_size}
        )
    )


def run_worker(
    args: argparse.Namespace,
) -> int:
    if args.worker_output is None:
        raise ValueError(
            "--worker-output is required in worker mode"
        )

    output_path = args.worker_output

    graph_buckets = graph_buckets_for_worker(
        backend=args.backend,
        batch_size=args.batch_size,
        isolated_bucket=args.isolated_bucket,
    )

    payload: dict[str, object] = {
        "status": "started",
        "backend": args.backend,
        "batch_size": args.batch_size,
        "graph_buckets": list(graph_buckets),
        "output_tokens": args.output_tokens,
        "repeats": args.repeats,
        "gpu_memory_utilization": (
            args.gpu_memory_utilization
        ),
        "max_model_len": args.max_model_len,
        "token_budget": args.token_budget,
    }

    try:
        torch.manual_seed(2026)
        torch.cuda.manual_seed_all(2026)

        tokenizer = (
            AutoTokenizer.from_pretrained(
                MODEL_PATH
            )
        )

        prompt = make_prompt(tokenizer)

        prompt_tokens = len(
            tokenizer.encode(prompt)
        )

        payload["prompt_tokens"] = (
            prompt_tokens
        )

        initialization_start = (
            perf_counter()
        )

        llm = LLM(
            MODEL_PATH,
            enforce_eager=(
                args.backend == "eager"
            ),
            tensor_parallel_size=1,
            max_model_len=(
                args.max_model_len
            ),
            max_num_batched_tokens=max(
                args.token_budget,
                args.batch_size
                * prompt_tokens,
            ),
            max_num_seqs=args.batch_size,
            num_state_slots=args.batch_size,
            gpu_memory_utilization=(
                args.gpu_memory_utilization
            ),
            hybrid_cuda_graph_batch_sizes=(
                graph_buckets
            ),
        )

        torch.cuda.synchronize()

        payload["initialization_seconds"] = (
            perf_counter()
            - initialization_start
        )

        print(
            f"Initialized {args.backend} "
            f"B={args.batch_size} with buckets "
            f"{graph_buckets}"
        )

        # 使用完整目标 batch 预热，确保正式数据不包含
        # 首次算子调用、首次 Graph replay 等一次性开销。
        run_generation_case(
            llm=llm,
            prompt=prompt,
            batch_size=args.batch_size,
            output_tokens=args.warmup_tokens,
        )

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        iterations: list[
            dict[str, object]
        ] = []

        for repeat_index in range(
            args.repeats
        ):
            result = run_generation_case(
                llm=llm,
                prompt=prompt,
                batch_size=args.batch_size,
                output_tokens=(
                    args.output_tokens
                ),
            )

            iterations.append(result)

            print(
                f"B={args.batch_size} "
                f"repeat={repeat_index + 1}/"
                f"{args.repeats}: "
                f"{result['decode_tokens_per_second']:.2f} "
                "tok/s, "
                f"TPOT={result['average_tpot_ms']:.3f} ms"
            )

        aggregate = aggregate_case(
            iterations
        )

        runner = llm.model_runner
        scheduler = llm.scheduler

        state_allocator = (
            scheduler.state_slot_allocator
        )

        used_state_slots = (
            0
            if state_allocator is None
            else state_allocator.num_used_slots
        )

        used_kv_blocks = len(
            scheduler
            .block_manager
            .used_block_ids
        )

        if used_state_slots != 0:
            raise RuntimeError(
                "GDN state slots leaked after run: "
                f"{used_state_slots}"
            )

        if used_kv_blocks != 0:
            raise RuntimeError(
                "KV blocks leaked after run: "
                f"{used_kv_blocks}"
            )

        if args.backend == "graph":
            if int(
                aggregate["graph_replays"]
            ) <= 0:
                raise RuntimeError(
                    "The stress case did not replay "
                    "a CUDA Graph"
                )

            if int(
                aggregate["eager_fallbacks"]
            ) != 0:
                raise RuntimeError(
                    "The exact stress bucket "
                    "unexpectedly fell back to Eager"
                )

        workspace = (
            runner.hybrid_graph_workspace
        )

        payload.update(
            {
                "status": "success",
                "aggregate": aggregate,
                "captured_buckets": sorted(
                    runner.hybrid_graphs.keys()
                ),
                "workspace_mib": (
                    0.0
                    if workspace is None
                    else mib(
                        workspace.allocated_bytes
                    )
                ),
                "capture_allocated_delta_mib": (
                    mib(
                        runner
                        .hybrid_graph_capture_allocated_bytes
                    )
                ),
                "fallback_reasons": dict(
                    runner
                    .hybrid_graph_fallback_reasons
                ),
                "used_state_slots_after_run": (
                    used_state_slots
                ),
                "used_kv_blocks_after_run": (
                    used_kv_blocks
                ),
                "memory": safe_cuda_memory(),
            }
        )

        write_json(output_path, payload)

        print(
            f"SUCCESS: {args.backend} "
            f"B={args.batch_size}"
        )

        print(
            "Peak allocated: "
            f"{payload['memory']['peak_allocated_mib']:.2f} MiB"
        )

        print(
            "Peak reserved: "
            f"{payload['memory']['peak_reserved_mib']:.2f} MiB"
        )

        return 0

    except BaseException as exception:
        payload.update(
            {
                "status": "failed",
                "error_type": type(
                    exception
                ).__name__,
                "error": str(exception),
                "traceback": traceback.format_exc(),
                "memory": safe_cuda_memory(),
            }
        )

        write_json(output_path, payload)

        print(
            f"FAILED: {args.backend} "
            f"B={args.batch_size}: "
            f"{type(exception).__name__}: "
            f"{exception}",
            file=sys.stderr,
        )

        traceback.print_exc()

        return 1


def run_subprocess_case(
    *,
    args: argparse.Namespace,
    batch_size: int,
    output_tokens: int,
    repeats: int,
    label: str,
) -> dict[str, object]:
    output_dir = args.output_dir
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = (
        f"{label}_{args.backend}_b{batch_size}"
    )

    json_path = output_dir / f"{stem}.json"
    log_path = output_dir / f"{stem}.log"

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--backend",
        args.backend,
        "--batch-size",
        str(batch_size),
        "--output-tokens",
        str(output_tokens),
        "--warmup-tokens",
        str(args.warmup_tokens),
        "--repeats",
        str(repeats),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--token-budget",
        str(args.token_budget),
        "--worker-output",
        str(json_path),
    ]

    if args.isolated_bucket:
        command.append(
            "--isolated-bucket"
        )

    print("\n" + "=" * 72)
    print(
        f"Running {label}: "
        f"{args.backend} B={batch_size}"
    )
    print("=" * 72)

    output_lines: list[str] = []

    with subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as process:
        if process.stdout is None:
            raise RuntimeError(
                "Failed to capture child output"
            )

        for line in process.stdout:
            print(line, end="")
            output_lines.append(line)

        return_code = process.wait()

    log_path.write_text(
        "".join(output_lines),
        encoding="utf-8",
    )

    if json_path.exists():
        with json_path.open(
            encoding="utf-8",
        ) as file:
            payload = json.load(file)
    else:
        payload = {
            "status": "failed",
            "backend": args.backend,
            "batch_size": batch_size,
            "error_type": "ChildProcessError",
            "error": (
                "Worker exited without writing JSON"
            ),
        }

    payload["return_code"] = return_code
    payload["log_path"] = str(log_path)
    payload["json_path"] = str(json_path)

    return payload


def format_number(
    value: object,
    digits: int = 2,
) -> str:
    if value is None:
        return "-"

    return f"{float(value):.{digits}f}"


def write_summary(
    *,
    args: argparse.Namespace,
    probe_results: list[
        dict[str, object]
    ],
    sustained_result: (
        dict[str, object] | None
    ),
) -> None:
    successful = [
        result
        for result in probe_results
        if result.get("status") == "success"
    ]

    failed = [
        result
        for result in probe_results
        if result.get("status") != "success"
    ]

    max_successful_batch = (
        max(
            int(result["batch_size"])
            for result in successful
        )
        if successful
        else None
    )

    peak_throughput_result = (
        max(
            successful,
            key=lambda result: float(
                result["aggregate"][
                    "decode_tokens_per_second"
                ]
            ),
        )
        if successful
        else None
    )
    summary_payload = {
        "backend": args.backend,
        "tested_batch_sizes": list(
            args.batch_sizes
        ),
        "max_successful_tested_batch": (
            max_successful_batch
        ),
        "first_failed_tested_batch": (
            int(failed[0]["batch_size"])
            if failed
            else None
        ),
        "peak_throughput_batch": (
            int(
                peak_throughput_result[
                    "batch_size"
                ]
            )
            if peak_throughput_result
            else None
        ),
        "peak_decode_tokens_per_second": (
            float(
                peak_throughput_result[
                    "aggregate"
                ]["decode_tokens_per_second"]
            )
            if peak_throughput_result
            else None
        ),
        "probe_results": probe_results,
        "sustained_result": sustained_result,
    }

    write_json(
        args.output_dir / "summary.json",
        summary_payload,
    )

    lines = [
        "# Hybrid CUDA Graph 极限压测",
        "",
        f"- Backend: `{args.backend}`",
        (
            "- Bucket mode: `isolated`"
            if args.isolated_bucket
            else "- Bucket mode: `base + candidate`"
        ),
        (
            "- 最大成功测试 Batch: "
            f"`{max_successful_batch}`"
        ),
        (
            "- 首个失败测试 Batch: "
            f"`{summary_payload['first_failed_tested_batch']}`"
        ),
        (
            "- 峰值吞吐 Batch: "
            f"`{summary_payload['peak_throughput_batch']}`"
        ),
        (
            "- 峰值 Decode 吞吐: "
            f"`{format_number(summary_payload['peak_decode_tokens_per_second'])} tok/s`"
        ),
        "",
        "## 探测结果",
        "",
        "| B | 状态 | Decode tok/s | TPOT ms | Peak allocated MiB | Peak reserved MiB |",
        "|---:|---|---:|---:|---:|---:|",
    ]

    for result in probe_results:
        if result.get("status") == "success":
            aggregate = result["aggregate"]
            memory = result["memory"]

            lines.append(
                "| "
                f"{result['batch_size']} | success | "
                f"{format_number(aggregate['decode_tokens_per_second'])} | "
                f"{format_number(aggregate['average_tpot_ms'], 3)} | "
                f"{format_number(memory['peak_allocated_mib'])} | "
                f"{format_number(memory['peak_reserved_mib'])} |"
            )
        else:
            lines.append(
                "| "
                f"{result['batch_size']} | failed: "
                f"{result.get('error_type', 'unknown')} | "
                "- | - | - | - |"
            )

    if sustained_result is not None:
        lines.extend(
            [
                "",
                "## 最大成功 Batch 持续压测",
                "",
                "```json",
                json.dumps(
                    sustained_result,
                    indent=2,
                    ensure_ascii=False,
                ),
                "```",
            ]
        )

    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            (
                "这里的最大 Batch 只表示给定候选集合、模型、"
                "显存比例、上下文长度和 Graph bucket 配置下的"
                "最大成功点，不代表硬件或引擎的数学绝对上限。"
            ),
            "",
        ]
    )

    (args.output_dir / "summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run_controller(
    args: argparse.Namespace,
) -> int:
    probe_results: list[
        dict[str, object]
    ] = []

    for batch_size in args.batch_sizes:
        result = run_subprocess_case(
            args=args,
            batch_size=batch_size,
            output_tokens=(
                args.probe_output_tokens
            ),
            repeats=args.probe_repeats,
            label="probe",
        )

        probe_results.append(result)

        if (
            result.get("status")
            != "success"
            and not args.continue_after_failure
        ):
            print(
                "Stopping after first failed "
                "candidate. Use "
                "--continue-after-failure to "
                "test larger values."
            )
            break

    successful = [
        result
        for result in probe_results
        if result.get("status") == "success"
    ]

    sustained_result = None

    if successful and not args.skip_sustained:
        max_successful_batch = max(
            int(result["batch_size"])
            for result in successful
        )

        sustained_result = run_subprocess_case(
            args=args,
            batch_size=max_successful_batch,
            output_tokens=(
                args.sustained_output_tokens
            ),
            repeats=args.sustained_repeats,
            label="sustained",
        )

    write_summary(
        args=args,
        probe_results=probe_results,
        sustained_result=sustained_result,
    )

    print("\n" + "=" * 72)
    print("Stress search completed")
    print("=" * 72)
    print(
        "Summary: "
        f"{args.output_dir / 'summary.md'}"
    )

    return 0 if successful else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search the maximum tested Qwen3.5 "
            "Hybrid Decode batch and then run a "
            "sustained stress case in an isolated "
            "child process."
        )
    )

    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--backend",
        choices=("graph", "eager"),
        default="graph",
    )

    parser.add_argument(
        "--batch-sizes",
        type=parse_batch_sizes,
        default=parse_batch_sizes(
            "16,24,32,48,64,96,128"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--probe-output-tokens",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--probe-repeats",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--sustained-output-tokens",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--sustained-repeats",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--output-tokens",
        type=int,
        default=128,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.78,
    )

    parser.add_argument(
        "--max-model-len",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--token-budget",
        type=int,
        default=32768,
    )

    parser.add_argument(
        "--isolated-bucket",
        action="store_true",
        help=(
            "Capture only the candidate bucket. "
            "By default the worker captures the "
            "existing 1/2/4/8/16 buckets plus "
            "the candidate."
        ),
    )

    parser.add_argument(
        "--continue-after-failure",
        action="store_true",
    )

    parser.add_argument(
        "--skip-sustained",
        action="store_true",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--worker-output",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    positive_fields = (
        "batch_size",
        "probe_output_tokens",
        "probe_repeats",
        "sustained_output_tokens",
        "sustained_repeats",
        "output_tokens",
        "repeats",
        "warmup_tokens",
        "max_model_len",
        "token_budget",
    )

    for field in positive_fields:
        if getattr(args, field) <= 0:
            raise ValueError(
                f"{field} must be positive"
            )

    if not (
        0.0
        < args.gpu_memory_utilization
        < 1.0
    ):
        raise ValueError(
            "gpu-memory-utilization must be "
            "between 0 and 1"
        )

    return args


def main() -> int:
    args = parse_args()

    if args.worker:
        return run_worker(args)

    return run_controller(args)


if __name__ == "__main__":
    raise SystemExit(main())

