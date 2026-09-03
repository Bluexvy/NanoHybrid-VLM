from transformers import AutoTokenizer

from nanovllm import (
    LLM,
    SamplingParams,
)

from test_prefix_commit import (
    MODEL_PATH,
    BLOCK_SIZE,
    render_chat_prompt,
    count_tokens,
)

from test_prefix_hit import (
    run_one_request,
)


CHECKPOINT_TOKENS = 4096

CHECKPOINT_INTERVAL_BLOCKS = (
    CHECKPOINT_TOKENS
    // BLOCK_SIZE
)

CACHE_CAPACITY_MIB = 320


def build_agent_prompt(
    tokenizer,
    agent_name: str,
    target_tokens: int = 4600,
) -> tuple[str, int]:
    """
    构造不同Agent的长Prompt。

    agent_name会反复出现在内容中，确保不同Agent的
    Prefix token blocks真正不同，而不是只改最后几个
    suffix tokens。
    """

    unit = (
        f"你是{agent_name}。"
        f"{agent_name}必须严格遵守工具调用规范、"
        "安全规则和结构化输出协议。"
    )

    def build(
        repeats: int,
    ) -> tuple[str, int]:
        prompt = render_chat_prompt(
            tokenizer,
            unit * repeats,
        )

        return (
            prompt,
            count_tokens(
                tokenizer,
                prompt,
            ),
        )

    low = 1
    high = 1

    while True:
        _, num_tokens = build(high)

        if num_tokens >= target_tokens:
            break

        low = high
        high *= 2

    while low < high:
        middle = (
            low + high
        ) // 2

        _, num_tokens = build(middle)

        if num_tokens < target_tokens:
            low = middle + 1
        else:
            high = middle

    return build(low)


def assert_capacity_valid(
    cache,
) -> None:
    assert (
        cache.current_prefix_cache_capacity_bytes
        <= cache.capacity_bytes
    )

    assert (
        cache.remaining_capacity_bytes
        >= 0
    )

    assert (
        0.0
        <= cache.capacity_utilization
        <= 1.0
    )


def main():
    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_PATH
        )
    )

    prompt_a, tokens_a = build_agent_prompt(
        tokenizer,
        "代码分析Agent-A",
    )

    prompt_b, tokens_b = build_agent_prompt(
        tokenizer,
        "数据库查询Agent-B",
    )

    prompt_c, tokens_c = build_agent_prompt(
        tokenizer,
        "文档检索Agent-C",
    )

    for num_tokens in (
        tokens_a,
        tokens_b,
        tokens_c,
    ):
        assert (
            CHECKPOINT_TOKENS
            < num_tokens
            < 5120
        )

    max_prompt_tokens = max(
        tokens_a,
        tokens_b,
        tokens_c,
    )

    llm = LLM(
        MODEL_PATH,
        enforce_eager=True,
        tensor_parallel_size=1,

        max_model_len=(
            max_prompt_tokens
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
        max_tokens=1,
        ignore_eos=True,
    )

    cache = llm.prefix_state_cache
    assert cache is not None

    print(
        "Capacity MiB:",
        cache.capacity_bytes / 1024 / 1024,
    )

    # =========================================
    # 1. 创建Entry A
    # =========================================

    print("\nCreating Agent A...")

    (
        _,
        prefill_a,
        _,
    ) = run_one_request(
        llm,
        prompt_a,
        sampling_params,
    )

    assert prefill_a == tokens_a
    assert cache.num_entries == 1

    key_a = (
        cache.entry_keys_lru_to_mru[-1]
    )

    assert_capacity_valid(cache)

    print(
        "LRU order:",
        [
            key.block_hash
            for key
            in cache.entry_keys_lru_to_mru
        ],
    )

    # =========================================
    # 2. 创建Entry B
    # =========================================

    print("\nCreating Agent B...")

    (
        _,
        prefill_b,
        _,
    ) = run_one_request(
        llm,
        prompt_b,
        sampling_params,
    )

    assert prefill_b == tokens_b
    assert cache.num_entries == 2

    key_b = (
        cache.entry_keys_lru_to_mru[-1]
    )

    assert key_a != key_b

    assert (
        cache.entry_keys_lru_to_mru
        == (
            key_a,
            key_b,
        )
    )

    # 两个4K Entry约占307 MiB，应能同时存在。
    assert (
        cache.current_prefix_cache_capacity_bytes
        == (
            2
            * (
                16 * cache.kv_block_bytes
                + cache
                .estimated_gdn_snapshot_bytes_per_entry
            )
        )
    )

    assert_capacity_valid(cache)

    # =========================================
    # 3. 命中A，将A移动到MRU
    # =========================================

    print("\nTouching Agent A...")

    (
        _,
        hot_prefill_a,
        _,
    ) = run_one_request(
        llm,
        prompt_a,
        sampling_params,
    )

    assert (
        hot_prefill_a
        == tokens_a - CHECKPOINT_TOKENS
    )

    # 原来[A, B]，命中A后变成[B, A]。
    assert (
        cache.entry_keys_lru_to_mru
        == (
            key_b,
            key_a,
        )
    )

    assert cache.num_lru_touches == 1

    # =========================================
    # 4. 创建C，应淘汰LRU的B
    # =========================================

    print("\nCreating Agent C...")

    (
        _,
        prefill_c,
        _,
    ) = run_one_request(
        llm,
        prompt_c,
        sampling_params,
    )

    assert prefill_c == tokens_c

    key_c = (
        cache.entry_keys_lru_to_mru[-1]
    )

    assert key_c not in {
        key_a,
        key_b,
    }

    # B是LRU，所以应该被淘汰；A因刚被命中而保留。
    assert key_b not in cache.entries
    assert key_a in cache.entries
    assert key_c in cache.entries

    assert (
        cache.entry_keys_lru_to_mru
        == (
            key_a,
            key_c,
        )
    )

    assert cache.num_entries == 2
    assert cache.num_evictions == 1
    assert cache.num_capacity_rejections == 0

    assert_capacity_valid(cache)

    print(
        "After C, LRU order:",
        [
            key.block_hash
            for key
            in cache.entry_keys_lru_to_mru
        ],
    )

    # =========================================
    # 5. 再次访问B
    # =========================================

    print("\nRevisiting evicted Agent B...")

    (
        _,
        replay_prefill_b,
        _,
    ) = run_one_request(
        llm,
        prompt_b,
        sampling_params,
    )

    # B已经被淘汰，必须重新执行完整Prompt。
    assert replay_prefill_b == tokens_b

    # 为重新创建B腾空间时，当前LRU为A：
    #
    # [A, C] → 淘汰A → [C, B]
    assert key_a not in cache.entries
    assert key_c in cache.entries
    assert key_b in cache.entries

    assert (
        cache.entry_keys_lru_to_mru
        == (
            key_c,
            key_b,
        )
    )

    assert cache.num_entries == 2
    assert cache.num_evictions == 2
    assert cache.num_capacity_rejections == 0

    assert_capacity_valid(cache)

    print(
        "Final LRU order:",
        [
            key.block_hash
            for key
            in cache.entry_keys_lru_to_mru
        ],
    )

    print(
        "Evictions:",
        cache.num_evictions,
    )

    print(
        "Current capacity MiB:",
        (
            cache
            .current_prefix_cache_capacity_bytes
            / 1024
            / 1024
        ),
    )

    print(
        "Utilization:",
        cache.capacity_utilization,
    )

    # =========================================
    # 6. 清理剩余Entries
    # =========================================

    remaining_keys = tuple(
        cache.entries.keys()
    )

    for key in remaining_keys:
        assert cache.discard(key)

    assert cache.num_entries == 0
    assert (
        cache.num_unique_pinned_kv_blocks
        == 0
    )
    assert (
        cache.current_gdn_snapshot_bytes
        == 0
    )
    assert (
        cache.current_prefix_cache_capacity_bytes
        == 0
    )

    print(
        "\nPart 5B-3 passed: capacity-aware "
        "multi-entry LRU eviction is correct."
    )


if __name__ == "__main__":
    main()