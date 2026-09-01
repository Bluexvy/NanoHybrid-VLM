import torch
from transformers import AutoTokenizer

from nanovllm import (
    LLM,
    SamplingParams,
)


MODEL_PATH = "/workspace/models/Qwen3.5-9B"

BLOCK_SIZE = 256
CHECKPOINT_INTERVAL_BLOCKS = 4

CHECKPOINT_TOKENS = (
    BLOCK_SIZE
    * CHECKPOINT_INTERVAL_BLOCKS
)


def render_chat_prompt(
    tokenizer,
    content: str,
) -> str:
    return tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": content,
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def count_tokens(
    tokenizer,
    prompt: str,
) -> int:
    return len(
        tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )
    )


def build_long_prompt(
    tokenizer,
    target_tokens: int = 1150,
) -> tuple[str, int]:
    """
    构造一个长度略大于 1024 token 的 Prompt。

    使用二分搜索控制重复次数，避免手工猜测
    中文字符串最终会被 tokenizer 切成多少 token。
    """

    unit = (
        "这是用于验证混合状态前缀缓存的固定公共前缀，"
        "需要保持内容和顺序完全一致。"
    )

    def build(repeats: int) -> tuple[str, int]:
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

    # 先指数增长，找到足够长的右边界。
    low = 1
    high = 1

    while True:
        _, num_tokens = build(high)

        if num_tokens >= target_tokens:
            break

        low = high
        high *= 2

    # 再二分搜索满足目标长度的最小 repeats。
    while low < high:
        middle = (low + high) // 2

        _, num_tokens = build(middle)

        if num_tokens < target_tokens:
            low = middle + 1
        else:
            high = middle

    prompt, num_tokens = build(low)

    return prompt, num_tokens


def assert_entry_owns_kv_blocks(
    llm: LLM,
    kv_block_ids: tuple[int, ...],
) -> None:
    """
    请求结束后，Prefix Entry 应当是这些 KV blocks
    的唯一 owner。
    """

    block_manager = (
        llm.scheduler.block_manager
    )

    for block_id in kv_block_ids:
        block = block_manager.blocks[
            block_id
        ]

        assert (
            block_id
            in block_manager.used_block_ids
        )

        # 当前已经没有活跃请求持有该 Block。
        assert block.request_ref_count == 0

        # Prefix Entry 持有一个缓存引用。
        assert block.cache_ref_count == 1

        # 总引用数 = 请求引用 + 缓存引用。
        assert block.ref_count == 1


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
        "long prompt tokens:",
        num_prompt_tokens,
    )

    # 必须让第一轮 Prefill 在 1024 停下，
    # 同时 Prompt 后面仍然有 suffix。
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
            "float32"
        ),
        max_new_prefix_snapshots_per_request=1,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
    )

    cache = llm.prefix_state_cache

    assert cache is not None
    assert cache.num_entries == 0
    assert cache.num_commits == 0

    # =========================================
    # 第一阶段：首次请求创建 Prefix Entry
    # =========================================

    print("\nRunning first long request...")

    llm.generate(
        [long_prompt],
        sampling_params,
        use_tqdm=False,
    )

    assert cache.num_entries == 1
    assert cache.num_commits == 1
    assert cache.num_duplicate_commits == 0

    key, entry = next(
        iter(cache.entries.items())
    )
        
    prompt_token_ids = tokenizer.encode(
        long_prompt,
        add_special_tokens=False,
    )

    # 第一次 llm.generate() 进入 Scheduler 时，
    # Cache 尚为空，因此已经产生：
    #
    #   1次 lookup
    #   1次 miss
    #
    # 下面的手动 lookup_longest() 再产生：
    #
    #   1次 lookup
    #   1次 hit
    lookup_entry = cache.lookup_longest(
        prompt_token_ids
    )

    assert lookup_entry is entry

    assert cache.num_lookups == 2
    assert cache.num_hits == 1
    assert cache.num_misses == 1
    exact_checkpoint_prompt = (
        prompt_token_ids[
            :CHECKPOINT_TOKENS
        ]
    )

    exact_boundary_result = (
        cache.lookup_longest(
            exact_checkpoint_prompt
        )
    )

    different_token_ids = (
        prompt_token_ids.copy()
    )

    different_token_ids[0] = (
        different_token_ids[0] + 1
    ) % tokenizer.vocab_size

    different_result = cache.lookup_longest(
        different_token_ids
    )

    assert different_result is None

    print(
        "Different Prefix correctly missed."
    )

    assert exact_boundary_result is None

    print(
        "Exact-boundary Prompt correctly "
        "left at least one token uncached."
    )

    print(
        "Longest Prefix lookup hit:",
        lookup_entry.key.num_cached_tokens,
    )

    print(
        "cached boundary:",
        key.num_cached_tokens,
    )

    print(
        "cached KV blocks:",
        entry.kv_block_ids,
    )

    print(
        "GDN snapshot MiB:",
        (
            entry.gdn_snapshot_bytes
            / 1024
            / 1024
        ),
    )

    assert (
        key.num_cached_tokens
        == CHECKPOINT_TOKENS
    )

    assert (
        len(entry.kv_block_ids)
        == CHECKPOINT_INTERVAL_BLOCKS
    )

    assert (
        entry.recurrent_state_snapshot.dtype
        == torch.float32
    )

    assert (
        cache.current_gdn_snapshot_bytes
        == entry.gdn_snapshot_bytes
    )

    # 请求已经结束，但这些 Blocks 不能被释放，
    # 因为 Prefix Entry 仍然持有缓存引用。
    assert_entry_owns_kv_blocks(
        llm,
        entry.kv_block_ids,
    )

    # 保存一份 CPU 基准，用来确认后续状态池复用
    # 不会修改 Prefix Snapshot。
    conv_snapshot_before = (
        entry.conv_state_snapshot
        .detach()
        .cpu()
        .clone()
    )

    recurrent_snapshot_before = (
        entry.recurrent_state_snapshot
        .detach()
        .cpu()
        .clone()
    )

    state_manager = (
        llm.model_runner
        .hybrid_state_manager
    )

    assert state_manager is not None

    restore_slot = 0

    # 故意破坏 active slot，模拟它曾经属于其他请求。
    state_manager.conv_state_pool[
        restore_slot
    ].zero_()

    state_manager.recurrent_state_pool[
        restore_slot
    ].zero_()

    state_manager.reset_slot(
        restore_slot
    )

    assert not (
        state_manager.is_slot_initialized(
            restore_slot
        )
    )

    cache.restore_gdn_state(
        entry,
        state_slot=restore_slot,
    )

    assert (
        state_manager.is_slot_initialized(
            restore_slot
        )
    )

    assert torch.equal(
        state_manager.conv_state_pool[
            restore_slot
        ],
        entry.conv_state_snapshot,
    )

    assert torch.equal(
        state_manager.recurrent_state_pool[
            restore_slot
        ],
        entry.recurrent_state_snapshot.to(
            dtype=(
                state_manager.spec
                .recurrent_dtype
            )
        ),
    )

    assert cache.num_gdn_restores == 1

    print(
        "GDN Snapshot restored into active "
        "state slot."
    )

    # =========================================
    # 第二阶段：复用 active state slot
    # =========================================

    short_prompt = render_chat_prompt(
        tokenizer,
        "请只回答数字：一加一等于多少？",
    )

    print("\nRunning short request...")

    llm.generate(
        [short_prompt],
        sampling_params,
        use_tqdm=False,
    )

    # max_num_seqs=1、num_state_slots=1，
    # 所以短请求必然复用刚才使用过的 state slot。
    #
    # 如果 Prefix Snapshot 只是 active pool 的 view，
    # 这里就会被短请求覆盖。
    assert torch.equal(
        entry.conv_state_snapshot.cpu(),
        conv_snapshot_before,
    )

    assert torch.equal(
        entry.recurrent_state_snapshot.cpu(),
        recurrent_snapshot_before,
    )

    print(
        "Snapshot stayed unchanged after "
        "active state-slot reuse."
    )

    # =========================================
    # 第三阶段：相同长请求通过 Prefix Hit 复用 Entry
    # =========================================

    bytes_before_hot_reuse = (
        cache.current_gdn_snapshot_bytes
    )

    hits_before_hot_reuse = (
        cache.num_hits
    )

    restores_before_hot_reuse = (
        cache.num_gdn_restores
    )

    print("\nRunning identical hot request...")

    llm.generate(
        [long_prompt],
        sampling_params,
        use_tqdm=False,
    )

    assert cache.num_entries == 1
    assert cache.num_commits == 1

    # 当前请求会在Scheduler admission阶段命中Entry，
    # 从缓存边界之后继续执行，不会再次经过同一个
    # checkpoint并调用commit。
    assert cache.num_duplicate_commits == 0

    assert (
        cache.num_hits
        == hits_before_hot_reuse + 1
    )

    assert (
        cache.num_gdn_restores
        == restores_before_hot_reuse + 1
    )

    assert (
        cache.current_gdn_snapshot_bytes
        == bytes_before_hot_reuse
    )

    # 已存在的 Entry 不能被替换成第二条请求的
    # active state view。
    assert cache.entries[key] is entry

    assert torch.equal(
        entry.conv_state_snapshot.cpu(),
        conv_snapshot_before,
    )

    assert torch.equal(
        entry.recurrent_state_snapshot.cpu(),
        recurrent_snapshot_before,
    )
    print(
        "Identical hot request reused the existing "
        "Entry through Scheduler lookup without "
        "creating another Snapshot."
    )

    # =========================================
    # 第四阶段：discard 释放缓存所有权
    # =========================================

    cached_block_ids = (
        entry.kv_block_ids
    )

    discarded = cache.discard(key)

    assert discarded
    assert cache.num_entries == 0
    assert cache.current_gdn_snapshot_bytes == 0

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

        assert (
            block_id
            not in block_manager.used_block_ids
        )

        assert (
            block_id
            in block_manager.free_block_ids
        )

    print(
        "\nPart 3G passed: Prefix commit, "
        "snapshot independence, duplicate "
        "detection and KV ownership release "
        "are correct."
    )


if __name__ == "__main__":
    main()