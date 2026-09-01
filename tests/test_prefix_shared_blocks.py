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


FIRST_CHECKPOINT_TOKENS = 4096
SECOND_CHECKPOINT_TOKENS = 8192

CHECKPOINT_INTERVAL_BLOCKS = (
    FIRST_CHECKPOINT_TOKENS
    // BLOCK_SIZE
)

FIRST_CHECKPOINT_BLOCKS = (
    FIRST_CHECKPOINT_TOKENS
    // BLOCK_SIZE
)

SECOND_CHECKPOINT_BLOCKS = (
    SECOND_CHECKPOINT_TOKENS
    // BLOCK_SIZE
)

CACHE_CAPACITY_MIB = 320
OUTPUT_TOKENS = 16


def mib(num_bytes: int) -> float:
    return (
        num_bytes
        / 1024
        / 1024
    )


def main() -> None:
    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_PATH
        )
    )

    # 构造一个长度超过 8192 tokens 的 Prompt。
    #
    # 第一次请求会在 4096 创建 Entry A。
    # 第二次请求命中 A，然后在 8192 创建 Entry B。
    prompt, num_prompt_tokens = (
        build_long_prompt(
            tokenizer,
            target_tokens=8700,
        )
    )

    print(
        "Prompt tokens:",
        num_prompt_tokens,
    )

    assert (
        SECOND_CHECKPOINT_TOKENS
        < num_prompt_tokens
        < 9216
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

        # 每轮 Prefill 最多处理 4096 tokens。
        #
        # 冷请求：
        #   chunk 1 = 4096
        #   chunk 2 = 4096
        #   chunk 3 = 剩余部分
        #
        # 热请求：
        #   恢复 4096 Prefix
        #   chunk 1 = 4096，执行到 8192
        #   chunk 2 = 剩余部分
        max_num_batched_tokens=(
            FIRST_CHECKPOINT_TOKENS
        ),

        max_num_seqs=1,
        num_state_slots=1,
        gpu_memory_utilization=0.9,

        hybrid_prefix_cache_mode=(
            "opportunistic"
        ),

        # 每隔 4096 tokens 才允许创建检查点。
        prefix_checkpoint_interval_blocks=(
            CHECKPOINT_INTERVAL_BLOCKS
        ),

        prefix_recurrent_snapshot_dtype=(
            "bfloat16"
        ),

        # 每个请求只允许创建一个新 Entry。
        #
        # 这样第一次请求只创建 4096 Entry，
        # 不会同时创建 8192 Entry。
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

    block_manager = (
        llm.scheduler.block_manager
    )

    gdn_bytes = (
        cache
        .estimated_gdn_snapshot_bytes_per_entry
    )

    kv_block_bytes = (
        cache.kv_block_bytes
    )

    print(
        "One KV block MiB:",
        mib(kv_block_bytes),
    )

    print(
        "One GDN snapshot MiB:",
        mib(gdn_bytes),
    )

    # ==================================================
    # 1. 冷请求：创建 4096-token Entry A
    # ==================================================

    print(
        "\nRunning cold request and creating "
        "the 4096-token Entry..."
    )

    (
        cold_output_ids,
        cold_prefill_tokens,
        cold_decode_tokens,
    ) = run_one_request(
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
        == OUTPUT_TOKENS - 1
    )

    assert (
        len(cold_output_ids)
        == OUTPUT_TOKENS
    )

    assert cache.num_entries == 1
    assert cache.num_commits == 1

    key_a = (
        cache.entry_keys_lru_to_mru[0]
    )

    entry_a = cache.entries[key_a]

    assert (
        key_a.num_cached_tokens
        == FIRST_CHECKPOINT_TOKENS
    )

    assert (
        len(entry_a.kv_block_ids)
        == FIRST_CHECKPOINT_BLOCKS
    )

    entry_a_block_ids = (
        entry_a.kv_block_ids
    )

    # 请求已经结束，因此这些 Block 只有
    # Entry A 的 cache owner。
    for block_id in entry_a_block_ids:
        block = block_manager.blocks[
            block_id
        ]

        assert block.cache_ref_count == 1
        assert block.request_ref_count == 0
        assert block.ref_count == 1

    expected_a_capacity = (
        FIRST_CHECKPOINT_BLOCKS
        * kv_block_bytes
        + gdn_bytes
    )

    assert (
        cache.current_prefix_cache_capacity_bytes
        == expected_a_capacity
    )

    print(
        "Entry A boundary:",
        key_a.num_cached_tokens,
    )

    print(
        "Entry A KV blocks:",
        len(entry_a_block_ids),
    )

    print(
        "Entry A capacity MiB:",
        mib(expected_a_capacity),
    )

    # ==================================================
    # 2. 第二次请求：命中 A，并创建 8192 Entry B
    # ==================================================

    print(
        "\nRunning second request, restoring "
        "4096 tokens and creating the "
        "8192-token Entry..."
    )

    (
        second_output_ids,
        second_prefill_tokens,
        second_decode_tokens,
    ) = run_one_request(
        llm,
        prompt,
        sampling_params,
    )

    # 第二次请求应该跳过前 4096 tokens。
    assert (
        second_prefill_tokens
        == (
            num_prompt_tokens
            - FIRST_CHECKPOINT_TOKENS
        )
    )

    assert (
        second_decode_tokens
        == OUTPUT_TOKENS - 1
    )

    assert (
        second_output_ids
        == cold_output_ids
    )

    # 此时应该同时存在 A 和 B。
    assert cache.num_entries == 2
    assert cache.num_commits == 2

    keys = (
        cache.entry_keys_lru_to_mru
    )

    assert len(keys) == 2

    key_b = keys[-1]
    entry_b = cache.entries[key_b]

    assert key_b != key_a

    assert (
        key_b.num_cached_tokens
        == SECOND_CHECKPOINT_TOKENS
    )

    assert (
        len(entry_b.kv_block_ids)
        == SECOND_CHECKPOINT_BLOCKS
    )

    entry_b_block_ids = (
        entry_b.kv_block_ids
    )

    # 最关键的断言：
    #
    # Entry B 的前 16 个物理 KV block ID，
    # 必须与 Entry A 的 16 个 block ID 完全相同。
    assert (
        entry_b_block_ids[
            :FIRST_CHECKPOINT_BLOCKS
        ]
        == entry_a_block_ids
    )

    # B 后半部分是新增加的 16 个物理块。
    entry_b_new_block_ids = (
        entry_b_block_ids[
            FIRST_CHECKPOINT_BLOCKS:
        ]
    )

    assert (
        len(entry_b_new_block_ids)
        == FIRST_CHECKPOINT_BLOCKS
    )

    assert not (
        set(entry_a_block_ids)
        & set(entry_b_new_block_ids)
    )

    # A 和 B 共享的前 16 个 Block：
    #
    # cache_ref_count == 2
    #   一个引用来自 A
    #   一个引用来自 B
    #
    # 请求已经结束，所以 request_ref_count == 0。
    for block_id in entry_a_block_ids:
        block = block_manager.blocks[
            block_id
        ]

        assert block.cache_ref_count == 2
        assert block.request_ref_count == 0
        assert block.ref_count == 2

    # B 新增的后 16 个 Block 只有 B 一个 owner。
    for block_id in entry_b_new_block_ids:
        block = block_manager.blocks[
            block_id
        ]

        assert block.cache_ref_count == 1
        assert block.request_ref_count == 0
        assert block.ref_count == 1

    # ==================================================
    # 3. 验证唯一物理块容量统计
    # ==================================================

    # 两个 Entry 一共引用：
    #
    # A：16 个 blocks
    # B：32 个 blocks
    #
    # 如果错误地按 Entry 分别相加，会得到48个。
    #
    # 实际物理 Block：
    #
    # A 的16个 + B新增的16个 = 32个。
    assert (
        cache.num_unique_pinned_kv_blocks
        == SECOND_CHECKPOINT_BLOCKS
    )

    expected_shared_capacity = (
        SECOND_CHECKPOINT_BLOCKS
        * kv_block_bytes
        + 2 * gdn_bytes
    )

    assert (
        cache.current_prefix_cache_capacity_bytes
        == expected_shared_capacity
    )

    # 错误的重复计算方式：
    naive_capacity = (
        (
            FIRST_CHECKPOINT_BLOCKS
            * kv_block_bytes
            + gdn_bytes
        )
        +
        (
            SECOND_CHECKPOINT_BLOCKS
            * kv_block_bytes
            + gdn_bytes
        )
    )

    assert (
        expected_shared_capacity
        < naive_capacity
    )

    print(
        "Entry B boundary:",
        key_b.num_cached_tokens,
    )

    print(
        "Entry B KV blocks:",
        len(entry_b_block_ids),
    )

    print(
        "Unique pinned KV blocks:",
        cache.num_unique_pinned_kv_blocks,
    )

    print(
        "Correct shared capacity MiB:",
        mib(expected_shared_capacity),
    )

    print(
        "Naive duplicated capacity MiB:",
        mib(naive_capacity),
    )

    # ==================================================
    # 4. 删除 A：共享 KV 不能被释放
    # ==================================================

    # A 的所有 KV blocks 同时被 B 引用。
    #
    # 因此删除 A 真正能释放的只有 A 自己的
    # GDN Snapshot，不能释放任何 KV block 容量。
    reclaimable_a = (
        cache.reclaimable_capacity_bytes(
            entry_a
        )
    )

    assert reclaimable_a == gdn_bytes

    print(
        "\nDiscarding Entry A..."
    )

    assert cache.discard(key_a)

    assert cache.num_entries == 1
    assert key_a not in cache.entries
    assert key_b in cache.entries

    # 删除 A 后，B 仍然持有共享的前16个块。
    for block_id in entry_a_block_ids:
        block = block_manager.blocks[
            block_id
        ]

        assert block.cache_ref_count == 1
        assert block.request_ref_count == 0
        assert block.ref_count == 1

        assert (
            block_id
            in block_manager.used_block_ids
        )

    # B 仍拥有32个唯一物理块。
    assert (
        cache.num_unique_pinned_kv_blocks
        == SECOND_CHECKPOINT_BLOCKS
    )

    expected_b_capacity = (
        SECOND_CHECKPOINT_BLOCKS
        * kv_block_bytes
        + gdn_bytes
    )

    assert (
        cache.current_prefix_cache_capacity_bytes
        == expected_b_capacity
    )

    print(
        "Entry A reclaimable MiB:",
        mib(reclaimable_a),
    )

    print(
        "Capacity after discarding A MiB:",
        mib(
            cache
            .current_prefix_cache_capacity_bytes
        ),
    )

    # ==================================================
    # 5. 再次请求：应命中更长的 Entry B
    # ==================================================

    print(
        "\nRunning third request and hitting "
        "the 8192-token Entry..."
    )

    (
        third_output_ids,
        third_prefill_tokens,
        third_decode_tokens,
    ) = run_one_request(
        llm,
        prompt,
        sampling_params,
    )

    # 现在只执行8192之后的Prompt suffix。
    assert (
        third_prefill_tokens
        == (
            num_prompt_tokens
            - SECOND_CHECKPOINT_TOKENS
        )
    )

    assert (
        third_decode_tokens
        == OUTPUT_TOKENS - 1
    )

    assert (
        third_output_ids
        == cold_output_ids
    )

    assert (
        llm.scheduler.num_prefix_hit_requests
        == 2
    )

    assert (
        llm.scheduler.num_prefix_hit_tokens
        == (
            FIRST_CHECKPOINT_TOKENS
            + SECOND_CHECKPOINT_TOKENS
        )
    )

    # 第二次请求恢复 A，第三次请求恢复 B。
    assert cache.num_gdn_restores == 2

    # ==================================================
    # 6. 删除最后一个 Entry
    # ==================================================

    print(
        "\nDiscarding Entry B..."
    )

    all_cached_block_ids = (
        entry_b_block_ids
    )

    assert cache.discard(key_b)

    assert cache.num_entries == 0
    assert cache.current_gdn_snapshot_bytes == 0

    assert (
        cache.current_pinned_kv_capacity_bytes
        == 0
    )

    assert (
        cache.current_prefix_cache_capacity_bytes
        == 0
    )

    # 最后一个 cache owner 被删除后，
    # 这些物理块应全部回到 free list。
    for block_id in all_cached_block_ids:
        block = block_manager.blocks[
            block_id
        ]

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

    assert not block_manager.used_block_ids

    assert (
        len(block_manager.free_block_ids)
        == len(block_manager.blocks)
    )

    print(
        "\nPart 5B-4B passed: hierarchical "
        "Prefix Entries shared physical KV blocks, "
        "capacity counted unique blocks, and the "
        "8192-token longest prefix was restored "
        "correctly."
    )


if __name__ == "__main__":
    main()