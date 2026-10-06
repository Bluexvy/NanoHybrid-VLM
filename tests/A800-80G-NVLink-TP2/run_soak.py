from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams
from suite_common import build_exact_token_prompt, close_llm, memory_stats_mib, run_requests, write_json


def parse_lengths(value: str) -> tuple[int, ...]:
    lengths = tuple(int(item) for item in value.split(",") if item.strip())
    if not lengths or min(lengths) <= 0:
        raise argparse.ArgumentTypeError("prompt lengths must be positive comma-separated integers")
    return lengths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--mode", choices=("eager", "graph"), default="graph")
    parser.add_argument("--backend", default="state_aware_cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--prompt-lengths", type=parse_lengths, default=(128, 512, 2048, 8192, 16384))
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--token-budget", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--graph-buckets", type=int, nargs="+", default=(8, 16, 32, 64, 128))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.rounds <= 0:
        raise ValueError("batch-size and rounds must be positive")
    torch.manual_seed(20260930)
    torch.cuda.manual_seed_all(20260930)

    llm: LLM | None = None
    started = perf_counter()
    try:
        llm = LLM(
            args.model,
            enforce_eager=args.mode == "eager",
            tensor_parallel_size=args.tp_size,
            gdn_decode_backend=args.backend,
            max_model_len=max(args.prompt_lengths) + args.output_tokens + 32,
            max_num_batched_tokens=args.token_budget,
            max_num_seqs=max(args.batch_size, max(args.graph_buckets)),
            num_state_slots=max(args.batch_size, max(args.graph_buckets)),
            gpu_memory_utilization=args.gpu_memory_utilization,
            hybrid_cuda_graph_batch_sizes=tuple(sorted(set(args.graph_buckets))),
            hybrid_prefix_cache_mode="disabled",
        )
        initialization_seconds = perf_counter() - started
        prompt_bank = {
            length: build_exact_token_prompt(llm, length) for length in args.prompt_lengths
        }
        prompts = [
            prompt_bank[args.prompt_lengths[index % len(args.prompt_lengths)]]
            for index in range(args.batch_size)
        ]
        params = SamplingParams(temperature=0.0, max_tokens=args.output_tokens, ignore_eos=True)

        run_requests(
            llm,
            [prompt_bank[min(args.prompt_lengths)]] * args.batch_size,
            SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True),
        )
        memory_before = memory_stats_mib(llm)
        start_replays = llm.model_runner.num_hybrid_graph_replays
        start_fallbacks = llm.model_runner.num_hybrid_graph_eager_fallbacks
        rounds: list[dict[str, object]] = []

        torch.cuda.nvtx.range_push("a800_tp2_soak")
        for round_index in range(args.rounds):
            result = run_requests(llm, prompts, params)
            output_digest = hashlib.sha256(
                json.dumps(result.pop("outputs"), separators=(",", ":")).encode()
            ).hexdigest()
            result["round"] = round_index
            result["output_sha256"] = output_digest
            rounds.append(result)
            print(
                f"soak {round_index + 1}/{args.rounds}: "
                f"decode={result['decode_tokens_per_second']:.2f} tok/s "
                f"TPOT={result['tpot_ms']['mean']:.3f} ms "
                f"TTFT={result['ttft_ms']['mean']:.3f} ms",
                flush=True,
            )
        torch.cuda.nvtx.range_pop()

        decode_rates = [float(item["decode_tokens_per_second"]) for item in rounds]
        tpot = [float(item["tpot_ms"]["mean"]) for item in rounds]
        ttft = [float(item["ttft_ms"]["mean"]) for item in rounds]
        memory_after = memory_stats_mib(llm)
        payload = {
            "config": vars(args) | {"output": str(args.output)},
            "initialization_seconds": initialization_seconds,
            "total_requests": args.batch_size * args.rounds,
            "total_generated_tokens": args.batch_size * args.rounds * args.output_tokens,
            "aggregate": {
                "decode_tokens_per_second_mean": statistics.mean(decode_rates),
                "decode_tokens_per_second_min": min(decode_rates),
                "decode_tokens_per_second_max": max(decode_rates),
                "tpot_ms_mean": statistics.mean(tpot),
                "ttft_ms_mean": statistics.mean(ttft),
                "throughput_cv": statistics.pstdev(decode_rates) / statistics.mean(decode_rates),
            },
            "rounds": rounds,
            "memory_before_mib": memory_before,
            "memory_after_mib": memory_after,
            "graph": {
                "replays": llm.model_runner.num_hybrid_graph_replays - start_replays,
                "eager_fallbacks": llm.model_runner.num_hybrid_graph_eager_fallbacks - start_fallbacks,
                "fallback_reasons": dict(llm.model_runner.hybrid_graph_fallback_reasons),
            },
            "scheduler": {
                "num_preemptions": llm.scheduler.num_preemptions,
                "num_recomputed_tokens": llm.scheduler.num_recomputed_tokens,
            },
            "passed": True,
        }
        write_json(args.output, payload)
        print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))
    finally:
        close_llm(llm)


if __name__ == "__main__":
    main()
