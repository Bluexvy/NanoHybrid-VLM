from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams

from suite_common import (
    build_exact_token_prompt,
    build_synthetic_image,
    close_llm,
    memory_stats_mib,
    run_requests,
    write_json,
)


def parse_case(value: str) -> tuple[int, int]:
    try:
        prompt_text, batch_text = value.lower().replace("x", ":").split(":")
        prompt_tokens = int(prompt_text)
        batch_size = int(batch_text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "Cases must use PROMPT_TOKENS:BATCH_SIZE, for example 128:16"
        ) from error

    if prompt_tokens <= 0 or batch_size <= 0:
        raise argparse.ArgumentTypeError("Case values must be positive")
    return prompt_tokens, batch_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--tp-size", type=int, choices=(1, 2), required=True)
    parser.add_argument("--mode", choices=("eager", "graph"), required=True)
    parser.add_argument(
        "--backend",
        choices=("fla", "state_aware_triton", "state_aware_cuda"),
        required=True,
    )
    parser.add_argument("--cases", type=parse_case, nargs="+", required=True)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup-output-tokens", type=int, default=8)
    parser.add_argument("--token-budget", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--graph-buckets", type=int, nargs="+", default=(1, 2, 4, 8, 16))
    parser.add_argument("--run-dynamic", action="store_true")
    parser.add_argument("--run-vision", action="store_true")
    parser.add_argument("--vision-size", type=int, default=448)
    return parser.parse_args()


def run_dynamic_case(llm: LLM) -> dict[str, object]:
    incumbent_ids = [
        llm.add_request(
            build_exact_token_prompt(llm, 128),
            SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True),
        )
        for _ in range(4)
    ]

    completed: dict[int, list[int]] = {}
    timeline: list[dict[str, object]] = []

    def execute_step(phase: str) -> None:
        torch.cuda.synchronize()
        start = perf_counter()
        outputs, stats = llm.step()
        torch.cuda.synchronize()
        elapsed_ms = (perf_counter() - start) * 1000.0
        for seq_id, token_ids in outputs:
            completed[seq_id] = list(token_ids)
        timeline.append(
            {
                "phase": phase,
                "prefill_tokens": stats.num_prefill_tokens,
                "decode_tokens": stats.num_decode_tokens,
                "elapsed_ms": elapsed_ms,
            }
        )

    while any(
        llm.request_metrics[seq_id].first_token_time is None
        for seq_id in incumbent_ids
    ):
        execute_step("start_incumbents")

    for _ in range(4):
        execute_step("incumbent_decode")

    late_specs = ((300, 40), (900, 40))
    late_ids = [
        llm.add_request(
            build_exact_token_prompt(llm, prompt_tokens),
            SamplingParams(temperature=0.0, max_tokens=output_tokens, ignore_eos=True),
        )
        for prompt_tokens, output_tokens in late_specs
    ]

    while not llm.is_finished():
        execute_step("late_arrival")

    all_ids = incumbent_ids + late_ids
    missing = [seq_id for seq_id in all_ids if seq_id not in completed]
    if missing:
        raise RuntimeError(f"Dynamic requests did not complete: {missing}")

    expected_lengths = [64] * len(incumbent_ids) + [40] * len(late_ids)
    actual_lengths = [len(completed[seq_id]) for seq_id in all_ids]
    if actual_lengths != expected_lengths:
        raise RuntimeError(
            f"Dynamic completion lengths mismatch: {actual_lengths} != {expected_lengths}"
        )

    return {
        "incumbent_ids": incumbent_ids,
        "late_ids": late_ids,
        "outputs": [completed[seq_id] for seq_id in all_ids],
        "completion_lengths": actual_lengths,
        "steps_with_decode_and_prefill": sum(
            bool(item["decode_tokens"] and item["prefill_tokens"])
            for item in timeline
        ),
        "timeline": timeline,
    }


def run_vision_case(llm: LLM, image_size: int) -> dict[str, object]:
    if llm.input_processor.processor is None:
        return {"skipped": True, "reason": "Model has no vision processor"}

    prompt = {
        "prompt": "Describe the two colored shapes and their relative positions.",
        "multi_modal_data": {"image": build_synthetic_image(image_size)},
    }
    result = run_requests(
        llm,
        [prompt],
        SamplingParams(temperature=0.0, max_tokens=24, ignore_eos=True),
    )
    result["skipped"] = False
    result["image_size"] = image_size
    return result


def main() -> None:
    args = parse_args()
    if args.output_tokens <= 1:
        raise ValueError("output-tokens must be greater than one")
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")

    torch.manual_seed(20260925)
    torch.cuda.manual_seed_all(20260925)

    max_batch_size = max(
        max(batch_size for _, batch_size in args.cases),
        6 if args.run_dynamic else 1,
    )
    max_prompt_tokens = max(
        max(prompt_tokens for prompt_tokens, _ in args.cases),
        900 if args.run_dynamic else 1,
    )
    graph_buckets = tuple(sorted(set(args.graph_buckets)))
    if graph_buckets[-1] > max_batch_size:
        max_batch_size = graph_buckets[-1]

    llm: LLM | None = None
    payload: dict[str, object] = {
        "label": args.label,
        "config": vars(args) | {"output": str(args.output)},
        "cases": {},
    }

    try:
        initialization_start = perf_counter()
        llm = LLM(
            args.model,
            enforce_eager=args.mode == "eager",
            tensor_parallel_size=args.tp_size,
            gdn_decode_backend=args.backend,
            max_model_len=max(4096, max_prompt_tokens + args.output_tokens + 32),
            max_num_batched_tokens=args.token_budget,
            max_num_seqs=max_batch_size,
            num_state_slots=max_batch_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            hybrid_cuda_graph_batch_sizes=graph_buckets,
            hybrid_prefix_cache_mode="disabled",
        )
        torch.cuda.synchronize()
        payload["initialization_seconds"] = perf_counter() - initialization_start

        unique_batches = sorted({batch_size for _, batch_size in args.cases})
        if args.warmup_output_tokens > 0:
            for batch_size in unique_batches:
                run_requests(
                    llm,
                    [build_exact_token_prompt(llm, 32)] * batch_size,
                    SamplingParams(
                        temperature=0.0,
                        max_tokens=args.warmup_output_tokens,
                        ignore_eos=True,
                    ),
                )

        start_replays = llm.model_runner.num_hybrid_graph_replays
        start_fallbacks = llm.model_runner.num_hybrid_graph_eager_fallbacks

        torch.cuda.nvtx.range_push("nano_tp2_measure")
        for prompt_tokens, batch_size in args.cases:
            key = f"p{prompt_tokens}_b{batch_size}"
            iterations = []
            for repeat_index in range(args.repeats):
                prompt = build_exact_token_prompt(llm, prompt_tokens)
                result = run_requests(
                    llm,
                    [prompt] * batch_size,
                    SamplingParams(
                        temperature=0.0,
                        max_tokens=args.output_tokens,
                        ignore_eos=True,
                    ),
                )
                result["repeat"] = repeat_index
                iterations.append(result)
                print(
                    f"{args.label} {key} repeat={repeat_index + 1}: "
                    f"decode={result['decode_tokens_per_second']:.2f} tok/s, "
                    f"TPOT={result['tpot_ms']['mean']:.3f} ms, "
                    f"TTFT={result['ttft_ms']['mean']:.3f} ms"
                )
            payload["cases"][key] = {
                "prompt_tokens": prompt_tokens,
                "batch_size": batch_size,
                "iterations": iterations,
            }

        if args.run_dynamic:
            payload["dynamic"] = run_dynamic_case(llm)

        if args.run_vision:
            payload["vision"] = run_vision_case(llm, args.vision_size)
        torch.cuda.nvtx.range_pop()

        payload["scheduler"] = {
            "num_preemptions": llm.scheduler.num_preemptions,
            "num_recomputed_tokens": llm.scheduler.num_recomputed_tokens,
            "num_prefix_hit_requests": llm.scheduler.num_prefix_hit_requests,
            "num_prefix_hit_tokens": llm.scheduler.num_prefix_hit_tokens,
        }
        payload["graph"] = {
            "replays": llm.model_runner.num_hybrid_graph_replays - start_replays,
            "eager_fallbacks": (
                llm.model_runner.num_hybrid_graph_eager_fallbacks - start_fallbacks
            ),
            "fallback_reasons": dict(llm.model_runner.hybrid_graph_fallback_reasons),
        }
        payload["rank0_memory_mib"] = memory_stats_mib(llm)
        write_json(args.output, payload)
        print(json.dumps(payload["graph"], ensure_ascii=False, indent=2))
    finally:
        close_llm(llm)


if __name__ == "__main__":
    main()
