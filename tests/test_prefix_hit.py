from transformers import AutoTokenizer

from nanovllm import (
    LLM,
    SamplingParams,
)

NUM_GENERATED_TOKENS = 64

# 复用旧测试中的 Prompt 构造和 KV 所有权检查。
from test_prefix_commit import (
    MODEL_PATH,
    BLOCK_SIZE,
    CHECKPOINT_INTERVAL_BLOCKS,
    CHECKPOINT_TOKENS,
    build_long_prompt,
    assert_entry_owns_kv_blocks,
)


def run_one_request(
    llm: LLM,
    prompt: str,
    sampling_params: SamplingParams,
) -> tuple[list[int], int, int]:
    """
    手动驱动一次请求，从而累计真实被 Scheduler
    调度的 Prefill/Decode token 数。

    返回：
        completion_token_ids
        total_prefill_tokens
        total_decode_tokens
    """

    seq_id = llm.add_request(
        prompt,
        sampling_params,
    )

    completion_token_ids: list[int] | None = None

    total_prefill_tokens = 0
    total_decode_tokens = 0

    while not llm.is_finished():
        outputs, stats = llm.step()

        total_prefill_tokens += (
            stats.num_prefill_tokens
        )

        total_decode_tokens += (
            stats.num_decode_tokens
        )

        for (
            finished_seq_id,
            token_ids,
        ) in outputs:
            if finished_seq_id == seq_id:
                completion_token_ids = token_ids

    if completion_token_ids is None:
        raise RuntimeError(
            f"Sequence {seq_id} did not produce "
            "a completed output"
        )

    return (
        completion_token_ids,
        total_prefill_tokens,
        total_decode_tokens,
    )


def main() -> None:
    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_PATH
        )
    )

    long_prompt, num_prompt_tokens = (
        build_long_prompt(tokenizer)
    )

    print(
        "Prompt tokens:",
        num_prompt_tokens,
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

        # 强制第一次请求自然形成：
        #
        # chunk 1 = 1024 tokens
        # chunk 2 = 剩余 suffix
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

        # 先用 FP32 Snapshot 验证严格正确性。
        prefix_recurrent_snapshot_dtype=(
            "float32"
        ),

        max_new_prefix_snapshots_per_request=1,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=NUM_GENERATED_TOKENS,

        # 即使模型提前生成 EOS，也继续生成满 64 tokens，
        # 方便严格比较 Cold/Hot 完整序列。
        ignore_eos=True,
    )

    cache = llm.prefix_state_cache

    assert cache is not None

    scheduler = llm.scheduler

    # =========================================
    # 第一阶段：Cold request
    # =========================================

    print("\nRunning cold request...")

    (
        cold_output_ids,
        cold_prefill_tokens,
        cold_decode_tokens,
    ) = run_one_request(
        llm,
        long_prompt,
        sampling_params,
    )

    print(
        "Cold Prefill tokens:",
        cold_prefill_tokens,
    )

    print(
        "Cold Decode tokens:",
        cold_decode_tokens,
    )

    print(
        "Cold output token IDs:",
        cold_output_ids,
    )

    # 首次请求没有缓存可以复用，所以必须处理
    # 完整 Prompt。
    assert (
        cold_prefill_tokens
        == num_prompt_tokens
    )

    # Prefill Forward 产生第一个 completion token，
    # 后续 63 个 token 分别由 63 次 Decode 产生。
    assert (
        cold_decode_tokens
        == NUM_GENERATED_TOKENS - 1
    )

    assert (
        len(cold_output_ids)
        == NUM_GENERATED_TOKENS
    )

    assert cache.num_entries == 1
    assert cache.num_commits == 1
    assert cache.num_duplicate_commits == 0

    key, entry = next(
        iter(cache.entries.items())
    )

    assert (
        key.num_cached_tokens
        == CHECKPOINT_TOKENS
    )

    assert (
        len(entry.kv_block_ids)
        == CHECKPOINT_INTERVAL_BLOCKS
    )

    # 第一次请求进入 Scheduler 时执行了一次 lookup，
    # 但当时 Cache 为空，所以是 miss。
    assert cache.num_lookups == 1
    assert cache.num_hits == 0
    assert cache.num_misses == 1

    assert scheduler.num_prefix_hit_requests == 0
    assert scheduler.num_prefix_hit_tokens == 0
    assert cache.num_gdn_restores == 0

    # 请求已结束，Prefix Entry 是缓存块唯一 owner。
    assert_entry_owns_kv_blocks(
        llm,
        entry.kv_block_ids,
    )

    # =========================================
    # 第二阶段：Hot request
    # =========================================

    print("\nRunning hot request...")

    (
        hot_output_ids,
        hot_prefill_tokens,
        hot_decode_tokens,
    ) = run_one_request(
        llm,
        long_prompt,
        sampling_params,
    )

    expected_suffix_tokens = (
        num_prompt_tokens
        - CHECKPOINT_TOKENS
    )

    print(
        "Hot Prefill tokens:",
        hot_prefill_tokens,
    )

    print(
        "Expected suffix tokens:",
        expected_suffix_tokens,
    )

    print(
        "Hot Decode tokens:",
        hot_decode_tokens,
    )

    print(
        "Hot output token IDs:",
        hot_output_ids,
    )

    # 这是最重要的性能正确性断言：
    #
    # 第二个请求不能再处理完整 Prompt，
    # 只能处理 1024 边界之后的 suffix。
    assert (
        hot_prefill_tokens
        == expected_suffix_tokens
    )

    assert (
        hot_decode_tokens
        == NUM_GENERATED_TOKENS - 1
    )

    assert (
        len(hot_output_ids)
        == NUM_GENERATED_TOKENS
    )

    # temperature=0 且 Snapshot 使用 FP32，
    # Cold 完整 Prefill 与 Hot Prefix 恢复路径应生成
    # 完全相同的 token。
    assert hot_output_ids == cold_output_ids

    # 第二次请求应增加一次 Cache hit。
    assert cache.num_lookups == 2
    assert cache.num_hits == 1
    assert cache.num_misses == 1

    assert (
        scheduler.num_prefix_hit_requests
        == 1
    )

    assert (
        scheduler.num_prefix_hit_tokens
        == CHECKPOINT_TOKENS
    )

    # Engine 应恢复一次 GDN Snapshot。
    assert cache.num_gdn_restores == 1

    # Hot 请求从 1024 开始执行，不会再次自然经过
    # 1024 边界，所以不会发生 duplicate commit。
    assert cache.num_entries == 1
    assert cache.num_commits == 1
    assert cache.num_duplicate_commits == 0

    # Entry 对象和 Snapshot 都不能被 Hot 请求替换。
    assert cache.entries[key] is entry

    # Hot 请求结束后，请求引用应被释放，
    # Entry 的 cache owner 仍然保留。
    assert_entry_owns_kv_blocks(
        llm,
        entry.kv_block_ids,
    )

    # =========================================
    # 第三阶段：显式释放 Entry
    # =========================================

    cached_block_ids = entry.kv_block_ids

    assert cache.discard(key)

    assert cache.num_entries == 0
    assert cache.current_gdn_snapshot_bytes == 0

    block_manager = (
        scheduler.block_manager
    )

    for block_id in cached_block_ids:
        block = block_manager.blocks[block_id]

        assert block.ref_count == 0
        assert block.cache_ref_count == 0
        assert block.request_ref_count == 0

        assert (
            block_id
            not in block_manager.used_block_ids
        )

        assert (
            block_id
            in block_manager.free_block_ids
        )

    print(
        "\nPart 4D-1 passed: Prefix hit skipped "
        f"{CHECKPOINT_TOKENS} Prompt tokens and "
        f"preserved all {NUM_GENERATED_TOKENS} "
        "greedy completion tokens."
    )

if __name__ == "__main__":
    main()