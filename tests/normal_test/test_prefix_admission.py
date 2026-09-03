from transformers import AutoTokenizer

from nanovllm import (
    LLM,
    SamplingParams,
)

from test_prefix_commit import (
    MODEL_PATH,
    BLOCK_SIZE,
    CHECKPOINT_TOKENS,
    CHECKPOINT_INTERVAL_BLOCKS,
    build_long_prompt,
)

from test_prefix_hit import (
    run_one_request,
)

from test_prefix_lru_eviction import (
    build_agent_prompt,
)


OUTPUT_TOKENS = 16


def main() -> None:
    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_PATH
        )
    )

    hot_prompt, hot_prompt_tokens = (
        build_long_prompt(
            tokenizer,
            target_tokens=1150,
        )
    )

    assert (
        CHECKPOINT_TOKENS
        < hot_prompt_tokens
        < 1536
    )

    llm = LLM(
        MODEL_PATH,
        enforce_eager=True,
        tensor_parallel_size=1,

        max_model_len=1536,
        max_num_batched_tokens=1024,

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

        hybrid_prefix_cache_capacity_mib=512,

        # 新功能配置。
        prefix_admission_policy=(
            "frequency"
        ),

        prefix_admission_min_observations=2,

        # 故意设得很小，用来验证CPU候选历史LRU。
        prefix_admission_max_candidates=2,
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

    # ==================================================
    # 1. 第一次：只记录热度，不缓存
    # ==================================================

    print(
        "\nFirst occurrence: observe only..."
    )

    (
        first_output_ids,
        first_prefill_tokens,
        first_decode_tokens,
    ) = run_one_request(
        llm,
        hot_prompt,
        sampling_params,
    )

    assert (
        first_prefill_tokens
        == hot_prompt_tokens
    )

    assert (
        first_decode_tokens
        == OUTPUT_TOKENS - 1
    )

    assert cache.num_entries == 0
    assert cache.num_commits == 0

    assert (
        cache.current_gdn_snapshot_bytes
        == 0
    )

    assert (
        cache.current_pinned_kv_capacity_bytes
        == 0
    )

    assert (
        cache.num_admission_observations
        == 1
    )

    assert (
        cache.num_admission_deferrals
        == 1
    )

    assert (
        cache.num_admission_accepts
        == 0
    )

    assert (
        cache.num_admission_candidates
        == 1
    )

    first_candidate_key = (
        cache.admission_keys_lru_to_mru[0]
    )

    assert (
        first_candidate_key.num_cached_tokens
        == CHECKPOINT_TOKENS
    )

    assert (
        cache.admission_observation_count(
            first_candidate_key
        )
        == 1
    )

    # 没有Entry owner，请求结束后所有KV应释放。
    assert not block_manager.used_block_ids

    # ==================================================
    # 2. 第二次：达到阈值，创建Entry
    # ==================================================

    print(
        "Second occurrence: admit and commit..."
    )

    (
        second_output_ids,
        second_prefill_tokens,
        second_decode_tokens,
    ) = run_one_request(
        llm,
        hot_prompt,
        sampling_params,
    )

    # 第二次开始时仍没有GPU Entry，
    # 因此仍然需要完整Prefill。
    assert (
        second_prefill_tokens
        == hot_prompt_tokens
    )

    assert (
        second_decode_tokens
        == OUTPUT_TOKENS - 1
    )

    assert (
        second_output_ids
        == first_output_ids
    )

    assert cache.num_entries == 1
    assert cache.num_commits == 1

    key, entry = next(
        iter(cache.entries.items())
    )

    assert key == first_candidate_key

    assert (
        cache.admission_observation_count(
            key
        )
        == 2
    )

    assert (
        cache.num_admission_observations
        == 2
    )

    assert (
        cache.num_admission_deferrals
        == 1
    )

    assert (
        cache.num_admission_accepts
        == 1
    )

    assert (
        cache.current_gdn_snapshot_bytes
        == entry.gdn_snapshot_bytes
    )

    # ==================================================
    # 3. 第三次：真正Prefix Hit
    # ==================================================

    print(
        "Third occurrence: restore hot Prefix..."
    )

    (
        third_output_ids,
        third_prefill_tokens,
        third_decode_tokens,
    ) = run_one_request(
        llm,
        hot_prompt,
        sampling_params,
    )

    assert (
        third_prefill_tokens
        == (
            hot_prompt_tokens
            - CHECKPOINT_TOKENS
        )
    )

    assert (
        third_decode_tokens
        == OUTPUT_TOKENS - 1
    )

    assert (
        third_output_ids
        == first_output_ids
    )

    assert (
        llm.scheduler
        .num_prefix_hit_requests
        == 1
    )

    assert (
        llm.scheduler
        .num_prefix_hit_tokens
        == CHECKPOINT_TOKENS
    )

    assert cache.num_gdn_restores == 1

    # Prefix Hit不属于新的checkpoint准入观察。
    assert (
        cache.num_admission_observations
        == 2
    )

    assert (
        cache.num_admission_hit_touches
        == 1
    )

    # ==================================================
    # 4. 一次性Prompt不能污染GPU Cache
    # ==================================================

    print(
        "Testing bounded cold-candidate history..."
    )

    cold_sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        ignore_eos=True,
    )

    cold_prompts = []

    for agent_name in (
        "一次性Agent-A",
        "一次性Agent-B",
        "一次性Agent-C",
    ):
        prompt, num_tokens = (
            build_agent_prompt(
                tokenizer,
                agent_name=agent_name,
                target_tokens=1150,
            )
        )

        assert (
            CHECKPOINT_TOKENS
            < num_tokens
            < 1536
        )

        cold_prompts.append(
            (
                prompt,
                num_tokens,
            )
        )

    for prompt, num_tokens in cold_prompts:
        (
            _,
            prefill_tokens,
            _,
        ) = run_one_request(
            llm,
            prompt,
            cold_sampling_params,
        )

        assert prefill_tokens == num_tokens

        # 这些Prompt只出现一次，不得进入GPU Cache。
        assert cache.num_entries == 1

        # CPU候选历史始终不能超过配置上限。
        assert (
            cache.num_admission_candidates
            <= 2
        )

    # 此时：
    #
    # H、A、B、C依次进入或刷新候选历史。
    # max_candidates=2，因此发生两次元数据淘汰。
    assert (
        cache.num_admission_candidate_evictions
        == 2
    )

    # Agent-A的首次记录已经被淘汰。
    #
    # 再运行一次Agent-A时，它会被重新视为第一次观察，
    # 仍然不能进入GPU Cache。
    agent_a_prompt, agent_a_tokens = (
        cold_prompts[0]
    )

    (
        _,
        agent_a_prefill_tokens,
        _,
    ) = run_one_request(
        llm,
        agent_a_prompt,
        cold_sampling_params,
    )

    assert (
        agent_a_prefill_tokens
        == agent_a_tokens
    )

    assert cache.num_entries == 1

    assert (
        cache.num_admission_candidate_evictions
        == 3
    )

    # 总观察：
    #
    # hot第1次
    # hot第2次
    # A、B、C
    # A再次出现
    #
    # 共6次。
    assert (
        cache.num_admission_observations
        == 6
    )

    assert (
        cache.num_admission_accepts
        == 1
    )

    assert (
        cache.num_admission_deferrals
        == 5
    )

    assert (
        cache.num_admission_candidates
        == 2
    )

    # ==================================================
    # 5. 释放GPU Entry
    # ==================================================

    assert cache.discard(key)

    assert cache.num_entries == 0

    assert (
        cache.current_prefix_cache_capacity_bytes
        == 0
    )

    assert not block_manager.used_block_ids

    print(
        "\nAdmission observations:",
        cache.num_admission_observations,
    )

    print(
        "Admission accepts:",
        cache.num_admission_accepts,
    )

    print(
        "Admission deferrals:",
        cache.num_admission_deferrals,
    )

    print(
        "Candidate metadata evictions:",
        cache
        .num_admission_candidate_evictions,
    )

    print(
        "\nPart 5C passed: two-observation "
        "admission prevented one-hit Prefixes from "
        "occupying GPU cache while preserving hot "
        "Prefix correctness."
    )


if __name__ == "__main__":
    main()