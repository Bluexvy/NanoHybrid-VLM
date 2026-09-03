import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
from transformers import AutoTokenizer

from nanovllm import (
    LLM,
    SamplingParams,
)

from test_prefix_commit import (
    MODEL_PATH,
    BLOCK_SIZE,
    build_long_prompt,
    assert_entry_owns_kv_blocks,
)

from test_prefix_hit import (
    run_one_request,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint-tokens",
        type=int,
        default=8192,
    )

    parser.add_argument(
        "--suffix-tokens",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--output-tokens",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--hot-runs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--snapshot-dtype",
        choices=[
            "float32",
            "bfloat16",
        ],
        default="bfloat16",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def timed_request(
    llm: LLM,
    prompt: str,
    sampling_params: SamplingParams,
):
    torch.cuda.synchronize()
    start = perf_counter()

    result = run_one_request(
        llm,
        prompt,
        sampling_params,
    )

    torch.cuda.synchronize()

    elapsed_ms = (
        perf_counter() - start
    ) * 1000.0

    return result, elapsed_ms


def compare_tokens(
    reference: list[int],
    candidate: list[int],
) -> dict:
    matching_tokens = sum(
        left == right
        for left, right in zip(
            reference,
            candidate,
        )
    )

    first_mismatch = None

    for index, (left, right) in enumerate(
        zip(reference, candidate)
    ):
        if left != right:
            first_mismatch = index
            break

    if (
        first_mismatch is None
        and len(reference) != len(candidate)
    ):
        first_mismatch = min(
            len(reference),
            len(candidate),
        )

    denominator = max(
        len(reference),
        len(candidate),
        1,
    )

    return {
        "exact_match": (
            reference == candidate
        ),
        "matching_tokens": (
            matching_tokens
        ),
        "match_ratio": (
            matching_tokens / denominator
        ),
        "first_mismatch_index": (
            first_mismatch
        ),
    }


def main():
    args = parse_args()

    if args.checkpoint_tokens <= 0:
        raise ValueError(
            "checkpoint-tokens must be positive"
        )

    if (
        args.checkpoint_tokens
        % BLOCK_SIZE
        != 0
    ):
        raise ValueError(
            "checkpoint-tokens must be divisible "
            f"by block_size={BLOCK_SIZE}"
        )

    if args.suffix_tokens <= 0:
        raise ValueError(
            "suffix-tokens must be positive"
        )

    if args.output_tokens <= 0:
        raise ValueError(
            "output-tokens must be positive"
        )

    if args.hot_runs <= 0:
        raise ValueError(
            "hot-runs must be positive"
        )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_PATH
        )
    )

    target_prompt_tokens = (
        args.checkpoint_tokens
        + args.suffix_tokens
    )

    prompt, num_prompt_tokens = (
        build_long_prompt(
            tokenizer,
            target_tokens=(
                target_prompt_tokens
            ),
        )
    )

    if (
        num_prompt_tokens
        <= args.checkpoint_tokens
    ):
        raise RuntimeError(
            "Generated Prompt does not contain "
            "a suffix after the checkpoint"
        )

    checkpoint_interval_blocks = (
        args.checkpoint_tokens
        // BLOCK_SIZE
    )

    # 给完整 Prompt、输出 token 和少量安全空间。
    max_model_len = (
        num_prompt_tokens
        + args.output_tokens
        + BLOCK_SIZE
    )

    print(
        "Snapshot dtype:",
        args.snapshot_dtype,
    )

    print(
        "Prompt tokens:",
        num_prompt_tokens,
    )

    print(
        "Checkpoint tokens:",
        args.checkpoint_tokens,
    )

    print(
        "Expected suffix tokens:",
        (
            num_prompt_tokens
            - args.checkpoint_tokens
        ),
    )

    print(
        "Output tokens:",
        args.output_tokens,
    )

    llm = LLM(
        MODEL_PATH,
        enforce_eager=True,
        tensor_parallel_size=1,

        max_model_len=max_model_len,

        # Cold 请求第一轮自然停止在指定 checkpoint。
        max_num_batched_tokens=(
            args.checkpoint_tokens
        ),

        max_num_seqs=1,
        num_state_slots=1,
        gpu_memory_utilization=0.9,

        hybrid_prefix_cache_mode=(
            "opportunistic"
        ),

        # 例如 checkpoint=8192：
        #
        # interval_blocks = 8192 / 256 = 32
        #
        # 第一处合法 Snapshot 边界就是 8192。
        prefix_checkpoint_interval_blocks=(
            checkpoint_interval_blocks
        ),

        prefix_recurrent_snapshot_dtype=(
            args.snapshot_dtype
        ),

        max_new_prefix_snapshots_per_request=1,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.output_tokens,
        ignore_eos=True,
    )

    cache = llm.prefix_state_cache

    assert cache is not None

    # =========================================
    # Cold 请求
    # =========================================

    print("\nRunning cold request...")

    (
        (
            cold_output_ids,
            cold_prefill_tokens,
            cold_decode_tokens,
        ),
        cold_elapsed_ms,
    ) = timed_request(
        llm,
        prompt,
        sampling_params,
    )

    assert (
        cold_prefill_tokens
        == num_prompt_tokens
    )

    assert (
        cold_decode_tokens
        == args.output_tokens - 1
    )

    assert cache.num_entries == 1

    key, entry = next(
        iter(cache.entries.items())
    )

    assert (
        key.num_cached_tokens
        == args.checkpoint_tokens
    )

    assert (
        len(entry.kv_block_ids)
        == checkpoint_interval_blocks
    )

    expected_hot_prefill_tokens = (
        num_prompt_tokens
        - args.checkpoint_tokens
    )

    print(
        "Cold E2E ms:",
        round(cold_elapsed_ms, 3),
    )

    print(
        "Snapshot MiB:",
        round(
            entry.gdn_snapshot_bytes
            / 1024
            / 1024,
            3,
        ),
    )
    print(
        "Unique pinned KV blocks:",
        cache.num_unique_pinned_kv_blocks,
    )

    print(
        "KV block MiB:",
        round(
            cache.kv_block_bytes
            / 1024
            / 1024,
            3,
        ),
    )

    print(
        "Pinned KV capacity MiB:",
        round(
            cache.current_pinned_kv_capacity_bytes
            / 1024
            / 1024,
            3,
        ),
    )

    print(
        "Total Prefix Cache capacity MiB:",
        round(
            cache.current_prefix_cache_capacity_bytes
            / 1024
            / 1024,
            3,
        ),
    )

    assert_entry_owns_kv_blocks(
        llm,
        entry.kv_block_ids,
    )

    # =========================================
    # 连续 Hot 请求
    # =========================================

    hot_elapsed_values = []
    hot_comparisons = []
    first_hot_output_ids = None

    for run_index in range(args.hot_runs):
        print(
            f"\nRunning hot request "
            f"{run_index + 1}/{args.hot_runs}..."
        )

        (
            (
                hot_output_ids,
                hot_prefill_tokens,
                hot_decode_tokens,
            ),
            hot_elapsed_ms,
        ) = timed_request(
            llm,
            prompt,
            sampling_params,
        )

        assert (
            hot_prefill_tokens
            == expected_hot_prefill_tokens
        )

        assert (
            hot_decode_tokens
            == args.output_tokens - 1
        )

        comparison = compare_tokens(
            cold_output_ids,
            hot_output_ids,
        )

        hot_comparisons.append(comparison)
        hot_elapsed_values.append(
            hot_elapsed_ms
        )

        # 即使 BF16 Hot 与 Cold 有轻微差异，同一份
        # Snapshot 的多次恢复也应该是确定性的。
        if first_hot_output_ids is None:
            first_hot_output_ids = (
                hot_output_ids
            )
        else:
            assert (
                hot_output_ids
                == first_hot_output_ids
            )

        # 每次 Hot 请求结束后，Entry 仍应是 Prefix
        # Blocks 的唯一 owner，不能泄漏 request ref。
        assert_entry_owns_kv_blocks(
            llm,
            entry.kv_block_ids,
        )

        print(
            "Hot E2E ms:",
            round(hot_elapsed_ms, 3),
        )

        print(
            "Token comparison:",
            comparison,
        )

    # FP32 仍然要求严格一致。
    if args.snapshot_dtype == "float32":
        assert all(
            item["exact_match"]
            for item in hot_comparisons
        )

    assert (
        llm.scheduler.num_prefix_hit_requests
        == args.hot_runs
    )

    assert (
        llm.scheduler.num_prefix_hit_tokens
        == (
            args.hot_runs
            * args.checkpoint_tokens
        )
    )

    assert (
        cache.num_gdn_restores
        == args.hot_runs
    )

    hot_median_ms = median(
        hot_elapsed_values
    )

    e2e_reduction_percent = (
        (
            cold_elapsed_ms
            - hot_median_ms
        )
        / cold_elapsed_ms
        * 100.0
    )

    result = {
        "snapshot_dtype": (
            args.snapshot_dtype
        ),
        "prompt_tokens": (
            num_prompt_tokens
        ),
        "checkpoint_tokens": (
            args.checkpoint_tokens
        ),
        "suffix_tokens": (
            expected_hot_prefill_tokens
        ),
        "output_tokens": (
            args.output_tokens
        ),
        "hot_runs": (
            args.hot_runs
        ),
        "cold_prefill_tokens": (
            cold_prefill_tokens
        ),
        "hot_prefill_tokens": (
            expected_hot_prefill_tokens
        ),
        "skipped_prefill_tokens_per_hit": (
            args.checkpoint_tokens
        ),
        "snapshot_bytes": (
            entry.gdn_snapshot_bytes
        ),
        "snapshot_mib": (
            entry.gdn_snapshot_bytes
            / 1024
            / 1024
        ),
        "cold_elapsed_ms": (
            cold_elapsed_ms
        ),
        "hot_elapsed_ms": (
            hot_elapsed_values
        ),
        "hot_median_ms": (
            hot_median_ms
        ),
        "single_run_e2e_reduction_percent": (
            e2e_reduction_percent
        ),
        "token_comparisons": (
            hot_comparisons
        ),
        "prefix_hit_requests": (
            llm.scheduler
            .num_prefix_hit_requests
        ),
        "prefix_hit_tokens": (
            llm.scheduler
            .num_prefix_hit_tokens
        ),
        "gdn_restores": (
            cache.num_gdn_restores
        ),
        "unique_pinned_kv_blocks": (
            cache.num_unique_pinned_kv_blocks
        ),
        "kv_block_bytes": (
            cache.kv_block_bytes
        ),
        "pinned_kv_capacity_bytes": (
            cache.current_pinned_kv_capacity_bytes
        ),
        "total_prefix_cache_capacity_bytes": (
            cache.current_prefix_cache_capacity_bytes
        ),
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nHot median E2E ms:",
        round(hot_median_ms, 3),
    )

    print(
        "Observed E2E reduction:",
        round(
            e2e_reduction_percent,
            2,
        ),
        "%",
    )

    print(
        "Result saved to:",
        args.output,
    )

    cached_block_ids = entry.kv_block_ids

    assert cache.discard(key)

    block_manager = (
        llm.scheduler.block_manager
    )

    for block_id in cached_block_ids:
        block = block_manager.blocks[
            block_id
        ]

        assert block.ref_count == 0
        assert block.cache_ref_count == 0
        assert block.request_ref_count == 0

    print(
        "\nPrefix stress test passed."
    )


if __name__ == "__main__":
    main()