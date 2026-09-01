from test_prefix_hit import (
    MODEL_PATH,
    CHECKPOINT_TOKENS,
    build_long_prompt,
    run_one_request,
)

from transformers import AutoTokenizer

from nanovllm import (
    LLM,
    SamplingParams,
)


def main():
    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_PATH
        )
    )

    prompt, num_prompt_tokens = (
        build_long_prompt(tokenizer)
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

        prefix_checkpoint_interval_blocks=4,

        prefix_recurrent_snapshot_dtype=(
            "bfloat16"
        ),

        max_new_prefix_snapshots_per_request=1,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        ignore_eos=True,
    )

    cache = llm.prefix_state_cache
    assert cache is not None

    # Cold 请求创建一个新 Entry。
    run_one_request(
        llm,
        prompt,
        sampling_params,
    )

    assert cache.num_entries == 1

    key, entry = next(
        iter(cache.entries.items())
    )

    assert (
        cache.entry_keys_lru_to_mru
        == (key,)
    )

    assert cache.num_lru_touches == 0

    assert cache.peek_lru_entry() is entry

    # Hot 请求实际命中该 Entry，应产生一次 touch。
    run_one_request(
        llm,
        prompt,
        sampling_params,
    )

    assert cache.num_lru_touches == 1

    assert (
        cache.entry_keys_lru_to_mru
        == (key,)
    )

    assert cache.peek_lru_entry() is entry

    print(
        "LRU touches:",
        cache.num_lru_touches,
    )

    print(
        "LRU order:",
        [
            item.num_cached_tokens
            for item
            in cache.entry_keys_lru_to_mru
        ],
    )

    assert cache.discard(key)

    assert cache.peek_lru_entry() is None
    assert cache.entry_keys_lru_to_mru == ()

    print(
        "Part 5B-1 passed: Prefix Entry LRU "
        "access tracking is correct."
    )


if __name__ == "__main__":
    main()