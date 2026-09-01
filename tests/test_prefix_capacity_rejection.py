from transformers import AutoTokenizer

from nanovllm import (
    LLM,
    SamplingParams,
)

from test_prefix_commit import (
    MODEL_PATH,
    BLOCK_SIZE,
    build_long_prompt,
)

from test_prefix_hit import (
    run_one_request,
)


CHECKPOINT_TOKENS = 4096

CHECKPOINT_INTERVAL_BLOCKS = (
    CHECKPOINT_TOKENS
    // BLOCK_SIZE
)

# 一个BF16 4K Entry实际需要：
#
# 128 MiB KV
# 25.5 MiB GDN
# 总计153.5 MiB
#
# 因此128 MiB预算一定放不下。
CACHE_CAPACITY_MIB = 128

OUTPUT_TOKENS = 16


def main():
    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_PATH
        )
    )

    prompt, num_prompt_tokens = (
        build_long_prompt(
            tokenizer,
            target_tokens=4600,
        )
    )

    assert (
        CHECKPOINT_TOKENS
        < num_prompt_tokens
        < 5120
    )

    llm = LLM(
        MODEL_PATH,
        enforce_eager=True,
        tensor_parallel_size=1,

        max_model_len=(
            num_prompt_tokens
            + OUTPUT_TOKENS
            + BLOCK_SIZE
        ),

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
            "bfloat16"
        ),

        max_new_prefix_snapshots_per_request=1,

        hybrid_prefix_cache_capacity_mib=(
            CACHE_CAPACITY_MIB
        ),
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=OUTPUT_TOKENS,
        ignore_eos=True,
    )

    cache = llm.prefix_state_cache
    assert cache is not None

    expected_standalone_bytes = (
        cache.estimated_gdn_snapshot_bytes_per_entry
        + CHECKPOINT_INTERVAL_BLOCKS
        * cache.kv_block_bytes
    )

    print(
        "Entry requires MiB:",
        expected_standalone_bytes
        / 1024
        / 1024,
    )

    print(
        "Cache budget MiB:",
        cache.capacity_bytes
        / 1024
        / 1024,
    )

    assert (
        expected_standalone_bytes
        > cache.capacity_bytes
    )

    # =========================================
    # 第一次请求
    # =========================================

    print("\nRunning first request...")

    (
        first_output_ids,
        first_prefill_tokens,
        first_decode_tokens,
    ) = run_one_request(
        llm,
        prompt,
        sampling_params,
    )

    # 推理本身必须完整成功。
    assert (
        first_prefill_tokens
        == num_prompt_tokens
    )

    assert (
        first_decode_tokens
        == OUTPUT_TOKENS - 1
    )

    assert (
        len(first_output_ids)
        == OUTPUT_TOKENS
    )

    # Commit因容量不足被拒绝。
    assert cache.num_entries == 0
    assert cache.num_commits == 0
    assert cache.num_evictions == 0
    assert cache.num_capacity_rejections == 1

    assert cache.current_gdn_snapshot_bytes == 0

    assert (
        cache.current_pinned_kv_capacity_bytes
        == 0
    )

    assert (
        cache.current_prefix_cache_capacity_bytes
        == 0
    )

    # =========================================
    # 第二次相同请求
    # =========================================

    print("\nRunning second request...")

    (
        second_output_ids,
        second_prefill_tokens,
        second_decode_tokens,
    ) = run_one_request(
        llm,
        prompt,
        sampling_params,
    )

    # 第一次没有成功缓存，因此第二次仍必须处理完整Prompt。
    assert (
        second_prefill_tokens
        == num_prompt_tokens
    )

    assert (
        second_decode_tokens
        == OUTPUT_TOKENS - 1
    )

    # 缓存失败不能改变模型生成结果。
    assert (
        second_output_ids
        == first_output_ids
    )

    # 第二次也尝试提交，并再次被容量拒绝。
    assert cache.num_capacity_rejections == 2
    assert cache.num_entries == 0
    assert cache.num_commits == 0
    assert cache.num_evictions == 0

    # 两次Scheduler lookup都是miss。
    assert cache.num_lookups == 2
    assert cache.num_hits == 0
    assert cache.num_misses == 2

    assert (
        llm.scheduler.num_prefix_hit_requests
        == 0
    )

    assert (
        llm.scheduler.num_prefix_hit_tokens
        == 0
    )

    # 请求结束后，没有Entry pin KV，也没有活跃请求；
    # 所有物理KV Blocks都应释放。
    block_manager = (
        llm.scheduler.block_manager
    )

    assert not block_manager.used_block_ids

    assert (
        len(block_manager.free_block_ids)
        == len(block_manager.blocks)
    )

    print(
        "\nCapacity rejections:",
        cache.num_capacity_rejections,
    )

    print(
        "Both requests completed with identical "
        "outputs and full Prefill."
    )

    print(
        "\nPart 5B-4A passed: oversized Prefix "
        "Entries are rejected without affecting "
        "inference correctness."
    )


if __name__ == "__main__":
    main()