import json
import statistics

from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


MODEL_PATH = "/workspace/models/Qwen3.5-9B"
OUTPUT_PATH = Path("artifacts/bench/long_context/prefix_16k_fp32.json")

BLOCK_SIZE = 256

MAX_MODEL_LEN = 16384
PROMPT_TOKENS = 16128
OUTPUT_TOKENS = 256

PREFIX_TOKENS = 15360
SUFFIX_TOKENS = PROMPT_TOKENS - PREFIX_TOKENS

PREFILL_CHUNK_TOKENS = 1024
MEASURED_REPEATS = 3


def repeat_to_length(token_ids: list[int], target_length: int) -> list[int]:
    repeat_count = (target_length + len(token_ids) - 1) // len(token_ids)
    return (token_ids * repeat_count)[:target_length]


def build_exact_prompt(tokenizer) -> list[int]:
    prefix_seed = tokenizer.encode(
        "你是一个负责代码分析、工具调用和任务规划的智能体，请严格遵守系统指令。",
        add_special_tokens=False,
    )

    suffix_seed = tokenizer.encode(
        "请根据以上规则分析当前请求，并给出准确、清晰且可执行的回答。",
        add_special_tokens=False,
    )

    prefix_ids = repeat_to_length(prefix_seed, PREFIX_TOKENS)
    suffix_ids = repeat_to_length(suffix_seed, SUFFIX_TOKENS)
    prompt_ids = prefix_ids + suffix_ids

    assert len(prompt_ids) == PROMPT_TOKENS
    return prompt_ids


def compare_outputs(cold_ids: list[int], hot_ids: list[int]) -> dict:
    compared_tokens = min(len(cold_ids), len(hot_ids))
    matching_tokens = sum(cold_ids[index] == hot_ids[index] for index in range(compared_tokens))

    return {
        "exact_match": cold_ids == hot_ids,
        "compared_tokens": compared_tokens,
        "matching_tokens": matching_tokens,
        "matching_ratio": matching_tokens / max(len(cold_ids), len(hot_ids), 1),
    }


def run_request(llm: LLM, prompt_ids: list[int], sampling_params: SamplingParams) -> dict:
    start_prefix_hits = llm.scheduler.num_prefix_hit_requests
    start_prefix_hit_tokens = llm.scheduler.num_prefix_hit_tokens
    start_graph_replays = llm.model_runner.num_hybrid_graph_replays

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start_time = perf_counter()
    seq_id = llm.add_request(prompt_ids, sampling_params)

    output_ids = None
    prefill_tokens = 0
    decode_tokens = 0
    prefill_elapsed = 0.0
    decode_elapsed = 0.0

    while not llm.is_finished():
        outputs, stats = llm.step()

        prefill_tokens += stats.num_prefill_tokens
        decode_tokens += stats.num_decode_tokens
        prefill_elapsed += stats.prefill_elapsed
        decode_elapsed += stats.decode_elapsed

        for finished_seq_id, finished_token_ids in outputs:
            if finished_seq_id == seq_id:
                output_ids = finished_token_ids

    torch.cuda.synchronize()
    wall_time_ms = (perf_counter() - start_time) * 1000.0

    metrics = next(
        item
        for item in llm.get_completed_request_metrics()
        if item.seq_id == seq_id
    )

    memory = llm.model_runner.get_memory_stats()

    return {
        "seq_id": seq_id,
        "output_ids": output_ids,
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
        "prefill_tokens_per_second": prefill_tokens / prefill_elapsed,
        "decode_tokens_per_second": decode_tokens / decode_elapsed,
        "ttft_ms": metrics.ttft_ms,
        "tpot_ms": metrics.tpot_ms,
        "e2e_ms": metrics.e2e_ms,
        "wall_time_ms": wall_time_ms,
        "prefix_hit_requests": llm.scheduler.num_prefix_hit_requests - start_prefix_hits,
        "prefix_hit_tokens": llm.scheduler.num_prefix_hit_tokens - start_prefix_hit_tokens,
        "graph_replays": llm.model_runner.num_hybrid_graph_replays - start_graph_replays,
        "peak_allocated_gib": memory["cuda_peak_allocated_bytes"] / 1024**3,
        "runtime_peak_extra_gib": memory["runtime_peak_extra_bytes"] / 1024**3,
    }


def run_cycle(llm: LLM, prompt_ids: list[int], sampling_params: SamplingParams) -> dict:
    cache = llm.prefix_state_cache
    assert cache is not None
    assert cache.num_entries == 0

    cold = run_request(llm, prompt_ids, sampling_params)

    assert cold["prefill_tokens"] == PROMPT_TOKENS
    assert cache.num_entries == 1

    key, entry = next(iter(cache.entries.items()))

    cache_stats = {
        "cached_prefix_tokens": key.num_cached_tokens,
        "gdn_snapshot_mib": entry.gdn_snapshot_bytes / 1024**2,
        "unique_pinned_kv_blocks": cache.num_unique_pinned_kv_blocks,
        "pinned_kv_capacity_mib": cache.current_pinned_kv_capacity_bytes / 1024**2,
        "total_prefix_capacity_mib": cache.current_prefix_cache_capacity_bytes / 1024**2,
    }

    hot = run_request(llm, prompt_ids, sampling_params)

    assert hot["prefill_tokens"] == SUFFIX_TOKENS
    assert hot["prefix_hit_tokens"] == PREFIX_TOKENS

    comparison = compare_outputs(cold["output_ids"], hot["output_ids"])

    cache.discard(key)

    return {
        "cold": cold,
        "hot": hot,
        "output_comparison": comparison,
        "cache": cache_stats,
    }


def median_metric(cycles: list[dict], request_type: str, metric: str) -> float:
    return statistics.median(cycle[request_type][metric] for cycle in cycles)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    prompt_ids = build_exact_prompt(tokenizer)

    print("16K Long-Context Prefix Cache Benchmark")
    print(f"Total sequence limit: {MAX_MODEL_LEN}")
    print(f"Prompt tokens: {PROMPT_TOKENS}")
    print(f"Shared prefix tokens: {PREFIX_TOKENS}")
    print(f"Uncached suffix tokens: {SUFFIX_TOKENS}")
    print(f"Output tokens: {OUTPUT_TOKENS}")
    print(f"Prefill chunk tokens: {PREFILL_CHUNK_TOKENS}")

    llm = LLM(
        MODEL_PATH,
        enforce_eager=False,
        tensor_parallel_size=1,
        gdn_decode_backend="state_aware_cuda",
        hybrid_cuda_graph_batch_sizes=(1,),
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=PREFILL_CHUNK_TOKENS,
        max_num_seqs=1,
        num_state_slots=1,
        gpu_memory_utilization=0.9,
        hybrid_prefix_cache_mode="opportunistic",
        prefix_checkpoint_interval_blocks=PREFIX_TOKENS // BLOCK_SIZE,
        prefix_recurrent_snapshot_dtype="float32",
        max_new_prefix_snapshots_per_request=1,
        hybrid_prefix_cache_capacity_mib=1024,
        prefix_admission_policy="always",
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=OUTPUT_TOKENS,
        ignore_eos=True,
    )

    print("\nRunning one warm-up cycle...")
    run_cycle(llm, prompt_ids, sampling_params)

    measured_cycles = []

    for repeat_index in range(MEASURED_REPEATS):
        print(f"\nRunning measured cycle {repeat_index + 1}/{MEASURED_REPEATS}...")
        cycle = run_cycle(llm, prompt_ids, sampling_params)
        measured_cycles.append(cycle)

        print(
            f"Cold TTFT={cycle['cold']['ttft_ms']:.3f} ms, "
            f"Hot TTFT={cycle['hot']['ttft_ms']:.3f} ms, "
            f"Hot Prefill={cycle['hot']['prefill_tokens']}, "
            f"Match={cycle['output_comparison']['matching_ratio']:.4f}"
        )

    cold_ttft = median_metric(measured_cycles, "cold", "ttft_ms")
    hot_ttft = median_metric(measured_cycles, "hot", "ttft_ms")
    cold_tpot = median_metric(measured_cycles, "cold", "tpot_ms")
    hot_tpot = median_metric(measured_cycles, "hot", "tpot_ms")
    cold_e2e = median_metric(measured_cycles, "cold", "e2e_ms")
    hot_e2e = median_metric(measured_cycles, "hot", "e2e_ms")

    summary = {
        "max_model_len": MAX_MODEL_LEN,
        "prompt_tokens": PROMPT_TOKENS,
        "output_tokens": OUTPUT_TOKENS,
        "shared_prefix_tokens": PREFIX_TOKENS,
        "suffix_tokens": SUFFIX_TOKENS,
        "prefill_chunk_tokens": PREFILL_CHUNK_TOKENS,
        "skipped_prefill_percent": PREFIX_TOKENS / PROMPT_TOKENS * 100.0,
        "cold_ttft_median_ms": cold_ttft,
        "hot_ttft_median_ms": hot_ttft,
        "ttft_reduction_percent": (cold_ttft - hot_ttft) / cold_ttft * 100.0,
        "cold_tpot_median_ms": cold_tpot,
        "hot_tpot_median_ms": hot_tpot,
        "cold_e2e_median_ms": cold_e2e,
        "hot_e2e_median_ms": hot_e2e,
        "e2e_reduction_percent": (cold_e2e - hot_e2e) / cold_e2e * 100.0,
        "all_hot_outputs_exact": all(cycle["output_comparison"]["exact_match"] for cycle in measured_cycles),
        "minimum_output_match_ratio": min(cycle["output_comparison"]["matching_ratio"] for cycle in measured_cycles),
        "cache": measured_cycles[0]["cache"],
    }

    payload = {
        "summary": summary,
        "cycles": measured_cycles,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nBenchmark summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nResult saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()