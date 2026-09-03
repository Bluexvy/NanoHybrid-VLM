import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoTokenizer

from nanovllm import (
    LLM,
    SamplingParams,
)

from test_prefix_hit import (
    MODEL_PATH,
    CHECKPOINT_INTERVAL_BLOCKS,
    CHECKPOINT_TOKENS,
    NUM_GENERATED_TOKENS,
    build_long_prompt,
    run_one_request,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--snapshot-dtype",
        choices=[
            "float32",
            "bfloat16",
        ],
        required=True,
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
    # CUDA Kernel 是异步提交的。计时前后同步，避免只测到
    # CPU 提交时间而没有包含实际 GPU 执行时间。
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


def first_mismatch_index(
    left: list[int],
    right: list[int],
) -> int | None:
    for index, (a, b) in enumerate(
        zip(left, right)
    ):
        if a != b:
            return index

    if len(left) != len(right):
        return min(
            len(left),
            len(right),
        )

    return None


def main():
    args = parse_args()

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_PATH
        )
    )

    prompt, num_prompt_tokens = (
        build_long_prompt(tokenizer)
    )

    assert (
        CHECKPOINT_TOKENS
        < num_prompt_tokens
        < 1536
    )

    llm = LLM(
        MODEL_PATH,
        enforce_eager=True,
        tensor_parallel_size=1,

        max_model_len=1536,

        max_num_batched_tokens=(
            CHECKPOINT_TOKENS
        ),

        max_num_seqs=1,
        num_state_slots=1,
        gpu_memory_utilization=0.9,

        hybrid_prefix_cache_mode=(
            "opportunistic"
        ),

        prefix_checkpoint_interval_blocks=(
            CHECKPOINT_INTERVAL_BLOCKS
        ),

        prefix_recurrent_snapshot_dtype=(
            args.snapshot_dtype
        ),

        max_new_prefix_snapshots_per_request=1,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=NUM_GENERATED_TOKENS,
        ignore_eos=True,
    )

    cache = llm.prefix_state_cache

    assert cache is not None

    # =========================================
    # Cold：完整 Prefill 并创建 Snapshot
    # =========================================

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

    assert cache.num_entries == 1

    key, entry = next(
        iter(cache.entries.items())
    )

    conv_snapshot_bytes = (
        entry.conv_state_snapshot.numel()
        * entry.conv_state_snapshot.element_size()
    )

    recurrent_snapshot_bytes = (
        entry.recurrent_state_snapshot.numel()
        * entry.recurrent_state_snapshot.element_size()
    )

    # =========================================
    # Hot：恢复 Snapshot，只计算 suffix
    # =========================================

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

    expected_hot_prefill_tokens = (
        num_prompt_tokens
        - CHECKPOINT_TOKENS
    )

    assert (
        cold_prefill_tokens
        == num_prompt_tokens
    )

    assert (
        hot_prefill_tokens
        == expected_hot_prefill_tokens
    )

    assert (
        cold_decode_tokens
        == NUM_GENERATED_TOKENS - 1
    )

    assert (
        hot_decode_tokens
        == NUM_GENERATED_TOKENS - 1
    )

    assert (
        llm.scheduler.num_prefix_hit_requests
        == 1
    )

    assert (
        llm.scheduler.num_prefix_hit_tokens
        == CHECKPOINT_TOKENS
    )

    assert cache.num_gdn_restores == 1

    matching_token_count = sum(
        cold_token == hot_token
        for cold_token, hot_token in zip(
            cold_output_ids,
            hot_output_ids,
        )
    )

    mismatch_index = first_mismatch_index(
        cold_output_ids,
        hot_output_ids,
    )

    exact_match = (
        cold_output_ids
        == hot_output_ids
    )

    result = {
        "snapshot_dtype": (
            args.snapshot_dtype
        ),
        "actual_recurrent_snapshot_dtype": (
            str(
                entry
                .recurrent_state_snapshot
                .dtype
            )
        ),
        "prompt_tokens": (
            num_prompt_tokens
        ),
        "checkpoint_tokens": (
            CHECKPOINT_TOKENS
        ),
        "cold_prefill_tokens": (
            cold_prefill_tokens
        ),
        "hot_prefill_tokens": (
            hot_prefill_tokens
        ),
        "skipped_prefill_tokens": (
            cold_prefill_tokens
            - hot_prefill_tokens
        ),
        "cold_decode_tokens": (
            cold_decode_tokens
        ),
        "hot_decode_tokens": (
            hot_decode_tokens
        ),
        "generated_tokens": (
            NUM_GENERATED_TOKENS
        ),
        "exact_token_match": (
            exact_match
        ),
        "matching_token_count": (
            matching_token_count
        ),
        "first_mismatch_index": (
            mismatch_index
        ),
        "conv_snapshot_bytes": (
            conv_snapshot_bytes
        ),
        "recurrent_snapshot_bytes": (
            recurrent_snapshot_bytes
        ),
        "total_snapshot_bytes": (
            entry.gdn_snapshot_bytes
        ),
        "cold_elapsed_ms": (
            cold_elapsed_ms
        ),
        "hot_elapsed_ms": (
            hot_elapsed_ms
        ),
        "cold_output_ids": (
            cold_output_ids
        ),
        "hot_output_ids": (
            hot_output_ids
        ),
    }

    # FP32 是 correctness baseline，必须逐 token 一致。
    if args.snapshot_dtype == "float32":
        assert exact_match

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {
                    "cold_output_ids",
                    "hot_output_ids",
                }
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "\nResult saved to:",
        args.output,
    )

    # 测试结束后释放 Entry 的缓存所有权。
    assert cache.discard(key)


if __name__ == "__main__":
    main()