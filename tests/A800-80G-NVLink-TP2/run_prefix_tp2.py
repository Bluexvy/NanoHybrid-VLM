from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams

from suite_common import (
    build_exact_token_prompt,
    close_llm,
    memory_stats_mib,
    run_requests,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument(
        "--backend",
        choices=("fla", "state_aware_triton", "state_aware_cuda"),
        default="state_aware_cuda",
    )
    parser.add_argument("--checkpoint-tokens", type=int, default=1024)
    parser.add_argument("--suffix-tokens", type=int, default=141)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--hot-repeats", type=int, default=3)
    parser.add_argument("--cache-capacity-mib", type=int, default=4096)
    return parser.parse_args()


def cache_stats(llm: LLM) -> dict[str, object]:
    cache = llm.prefix_state_cache
    if cache is None:
        raise RuntimeError("Hybrid Prefix Cache is not initialized")
    return {
        "num_entries": cache.num_entries,
        "num_commits": cache.num_commits,
        "num_duplicate_commits": cache.num_duplicate_commits,
        "num_lookups": cache.num_lookups,
        "num_hits": cache.num_hits,
        "num_misses": cache.num_misses,
        "num_gdn_restores": cache.num_gdn_restores,
        "num_evictions": cache.num_evictions,
        "num_capacity_rejections": cache.num_capacity_rejections,
        "num_unique_pinned_kv_blocks": cache.num_unique_pinned_kv_blocks,
        "current_capacity_bytes": cache.current_prefix_cache_capacity_bytes,
    }


def main() -> None:
    args = parse_args()
    block_size = 256
    if args.checkpoint_tokens % block_size != 0:
        raise ValueError("checkpoint-tokens must be divisible by 256")
    interval_blocks = args.checkpoint_tokens // block_size
    prompt_tokens = args.checkpoint_tokens + args.suffix_tokens

    torch.manual_seed(20260925)
    torch.cuda.manual_seed_all(20260925)

    llm: LLM | None = None
    try:
        initialization_start = perf_counter()
        llm = LLM(
            args.model,
            enforce_eager=True,
            tensor_parallel_size=args.tp_size,
            gdn_decode_backend=args.backend,
            max_model_len=max(2048, prompt_tokens + args.output_tokens + 16),
            max_num_batched_tokens=args.checkpoint_tokens,
            max_num_seqs=1,
            num_state_slots=1,
            gpu_memory_utilization=args.gpu_memory_utilization,
            hybrid_prefix_cache_mode="opportunistic",
            prefix_checkpoint_interval_blocks=interval_blocks,
            prefix_recurrent_snapshot_dtype="float32",
            max_new_prefix_snapshots_per_request=1,
            hybrid_prefix_cache_capacity_mib=args.cache_capacity_mib,
            prefix_admission_policy="always",
            hybrid_cuda_graph_batch_sizes=(1,),
        )
        initialization_seconds = perf_counter() - initialization_start

        prompt = build_exact_token_prompt(llm, prompt_tokens)
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=args.output_tokens,
            ignore_eos=True,
        )

        cold = run_requests(llm, [prompt], sampling)
        after_cold = cache_stats(llm)
        hot_runs = [
            run_requests(llm, [prompt], sampling)
            for _ in range(args.hot_repeats)
        ]
        hot = hot_runs[0]
        after_hot = cache_stats(llm)

        if cold["total_prefill_tokens"] != prompt_tokens:
            raise RuntimeError(
                "Cold request did not execute the full prompt: "
                f"{cold['total_prefill_tokens']} != {prompt_tokens}"
            )
        if hot["total_prefill_tokens"] != args.suffix_tokens:
            raise RuntimeError(
                "Hot request did not skip the cached prefix: "
                f"{hot['total_prefill_tokens']} != {args.suffix_tokens}"
            )
        if any(cold["outputs"] != item["outputs"] for item in hot_runs):
            raise RuntimeError("Cold and hot greedy outputs differ")
        if after_cold["num_entries"] != 1:
            raise RuntimeError("Cold request did not create exactly one entry")
        if after_hot["num_hits"] < 1 or after_hot["num_gdn_restores"] < 1:
            raise RuntimeError("Hot request did not restore the joint prefix state")

        cache = llm.prefix_state_cache
        assert cache is not None
        key, entry = next(iter(cache.entries.items()))
        if key.num_cached_tokens != args.checkpoint_tokens:
            raise RuntimeError("PrefixKey cached-token boundary is incorrect")

        payload = {
            "config": vars(args) | {"output": str(args.output)},
            "initialization_seconds": initialization_seconds,
            "prompt_tokens": prompt_tokens,
            "expected_cached_tokens": args.checkpoint_tokens,
            "expected_suffix_tokens": args.suffix_tokens,
            "prefix_key": {
                "model_namespace": key.model_namespace,
                "block_hash": key.block_hash,
                "num_cached_tokens": key.num_cached_tokens,
            },
            "entry": {
                "kv_block_ids": list(entry.kv_block_ids),
                "conv_snapshot_shape": list(entry.conv_state_snapshot.shape),
                "recurrent_snapshot_shape": list(entry.recurrent_state_snapshot.shape),
                "recurrent_snapshot_dtype": str(entry.recurrent_state_snapshot.dtype),
                "gdn_snapshot_mib_rank0": entry.gdn_snapshot_bytes / 1024**2,
            },
            "cold": cold,
            "hot": hot,
            "hot_runs": hot_runs,
            "request_hit_rate": args.hot_repeats / (args.hot_repeats + 1),
            "after_cold": after_cold,
            "after_hot": after_hot,
            "scheduler": {
                "num_prefix_hit_requests": llm.scheduler.num_prefix_hit_requests,
                "num_prefix_hit_tokens": llm.scheduler.num_prefix_hit_tokens,
            },
            "rank0_memory_mib": memory_stats_mib(llm),
            "passed": True,
        }
        write_json(args.output, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        close_llm(llm)


if __name__ == "__main__":
    main()
