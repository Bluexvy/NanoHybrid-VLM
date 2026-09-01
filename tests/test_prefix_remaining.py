import argparse
import subprocess
import sys
from pathlib import Path

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


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

THIS_FILE = Path(__file__).resolve()

OUTPUT_TOKENS = 16


def mib(num_bytes: int) -> float:
    return (
        num_bytes
        / 1024
        / 1024
    )


def record_outputs(
    outputs,
    completed: dict[int, list[int]],
) -> None:
    for seq_id, token_ids in outputs:
        completed[seq_id] = token_ids


# ============================================================
# Case 1：缓存释放时，活跃请求必须继续持有 KV blocks
# ============================================================

def test_active_request_ownership() -> None:
    """
    验证两个热请求并发共享同一 Prefix Entry 时：

        ref_count
        =
        cache_ref_count
        +
        request_ref_count

    删除 Prefix Entry 只能释放 cache owner，
    不能释放两个活跃请求仍在使用的 KV blocks。
    """

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_PATH
        )
    )

    prompt, num_prompt_tokens = (
        build_long_prompt(
            tokenizer,
            target_tokens=1150,
        )
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

        max_model_len=(
            num_prompt_tokens
            + OUTPUT_TOKENS
            + BLOCK_SIZE
        ),

        max_num_batched_tokens=(
            CHECKPOINT_TOKENS
        ),

        # 后面要同时运行两个热请求。
        max_num_seqs=2,
        num_state_slots=2,

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

    # --------------------------------------------------------
    # 1. 冷请求创建 Prefix Entry
    # --------------------------------------------------------

    print(
        "\n[Active ownership] "
        "Creating cold Prefix Entry..."
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

    assert cache.num_entries == 1

    key, entry = next(
        iter(cache.entries.items())
    )

    prefix_block_ids = (
        entry.kv_block_ids
    )

    assert (
        len(prefix_block_ids)
        == CHECKPOINT_INTERVAL_BLOCKS
    )

    # 冷请求结束后，只剩缓存引用。
    for block_id in prefix_block_ids:
        block = block_manager.blocks[
            block_id
        ]

        assert block.cache_ref_count == 1
        assert block.request_ref_count == 0
        assert block.ref_count == 1

    # --------------------------------------------------------
    # 2. 同时加入两个相同的热请求
    # --------------------------------------------------------

    seq_ids = [
        llm.add_request(
            prompt,
            sampling_params,
        ),
        llm.add_request(
            prompt,
            sampling_params,
        ),
    ]

    completed: dict[int, list[int]] = {}

    outputs, first_step_stats = (
        llm.step()
    )

    record_outputs(
        outputs,
        completed,
    )

    expected_suffix_tokens = (
        num_prompt_tokens
        - CHECKPOINT_TOKENS
    )

    # 两个热请求应在同一个 Variable-length
    # Batched Prefill 中处理两个 suffix。
    assert (
        first_step_stats.num_prefill_tokens
        == 2 * expected_suffix_tokens
    )

    assert (
        cache.num_gdn_restores
        == 2
    )

    assert (
        llm.scheduler
        .num_prefix_hit_requests
        == 2
    )

    assert (
        llm.scheduler
        .num_prefix_hit_tokens
        == 2 * CHECKPOINT_TOKENS
    )

    # 此刻的引用关系：
    #
    # 1 个 cache owner
    # 2 个 request owners
    #
    # ref_count = 3
    for block_id in prefix_block_ids:
        block = block_manager.blocks[
            block_id
        ]

        assert block.cache_ref_count == 1
        assert block.request_ref_count == 2
        assert block.ref_count == 3

        assert (
            block_id
            in block_manager.used_block_ids
        )

    # --------------------------------------------------------
    # 2.1 直接验证重复 commit
    # --------------------------------------------------------
    #
    # 两个热请求仍处于活跃状态，因此它们的
    # block_table 和 state_slot 都仍然有效。
    #
    # 选取其中一条请求，用相同 PrefixKey 再执行一次
    # commit。PrefixStateCache 必须返回原Entry，不能：
    #
    #   1. 创建第二份GDN Snapshot；
    #   2. 再增加KV cache_ref_count；
    #   3. 增加num_commits。

    active_seq = next(
        seq
        for seq in llm.scheduler.running
        if seq.seq_id == seq_ids[0]
    )

    assert active_seq.state_slot is not None

    duplicate_kv_block_ids = tuple(
        active_seq.block_table[
            :len(prefix_block_ids)
        ]
    )

    snapshot_bytes_before_duplicate = (
        cache.current_gdn_snapshot_bytes
    )

    duplicate_entry, created = cache.commit(
        key=key,
        kv_block_ids=duplicate_kv_block_ids,
        state_slot=active_seq.state_slot,
    )

    assert duplicate_entry is entry
    assert not created

    assert cache.num_entries == 1
    assert cache.num_commits == 1
    assert cache.num_duplicate_commits == 1

    assert (
        cache.current_gdn_snapshot_bytes
        == snapshot_bytes_before_duplicate
    )

    # 重复提交不能再次pin这些物理块。
    for block_id in prefix_block_ids:
        block = block_manager.blocks[
            block_id
        ]

        assert block.cache_ref_count == 1
        assert block.request_ref_count == 2
        assert block.ref_count == 3

    # --------------------------------------------------------
    # 3. 请求仍在运行时删除 Prefix Entry
    # --------------------------------------------------------

    print(
        "[Active ownership] "
        "Discarding cache Entry while two "
        "requests are still active..."
    )

    assert cache.discard(key)

    assert cache.num_entries == 0
    assert cache.current_gdn_snapshot_bytes == 0

    assert (
        cache.current_pinned_kv_capacity_bytes
        == 0
    )

    # 缓存引用消失，但两个请求引用仍然存在。
    #
    # 所以这些 Block 必须继续留在 used_block_ids。
    for block_id in prefix_block_ids:
        block = block_manager.blocks[
            block_id
        ]

        assert block.cache_ref_count == 0
        assert block.request_ref_count == 2
        assert block.ref_count == 2

        assert (
            block_id
            in block_manager.used_block_ids
        )

        assert (
            block_id
            not in block_manager.free_block_ids
        )

    # --------------------------------------------------------
    # 4. 继续 Decode，直到两个请求都结束
    # --------------------------------------------------------

    total_decode_tokens = 0

    while not llm.is_finished():
        outputs, stats = llm.step()

        total_decode_tokens += (
            stats.num_decode_tokens
        )

        record_outputs(
            outputs,
            completed,
        )

    assert (
        total_decode_tokens
        == 2 * (OUTPUT_TOKENS - 1)
    )

    assert set(completed) == set(seq_ids)

    # 两个热请求都必须与冷请求生成相同结果。
    for seq_id in seq_ids:
        assert (
            completed[seq_id]
            == cold_output_ids
        )

    # 所有请求和 Cache owner 都已释放。
    for block_id in prefix_block_ids:
        block = block_manager.blocks[
            block_id
        ]

        assert block.cache_ref_count == 0
        assert block.request_ref_count == 0
        assert block.ref_count == 0

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
        "[Active ownership] PASSED: cache eviction "
        "did not free KV blocks still owned by "
        "active requests."
    )


# ============================================================
# Case 2：共享层级 Entry 的自动 LRU 驱逐
# ============================================================

def test_shared_lru_eviction() -> None:
    """
    构造：

        Entry A：4096 tokens
        Entry B：8192 tokens，与A共享前16个KV blocks

    再创建独立的 Entry C。

    320 MiB 预算下：

        A + B = 307 MiB

    C 需要153.5 MiB，因此：

        先驱逐A，只释放25.5 MiB，不够
        再驱逐B，释放281.5 MiB，才够

    用来验证 _ensure_capacity_for() 每次驱逐后都会
    重新计算实际可回收容量。
    """

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_PATH
        )
    )

    long_prompt, long_prompt_tokens = (
        build_long_prompt(
            tokenizer,
            target_tokens=8700,
        )
    )

    independent_prompt, independent_tokens = (
        build_agent_prompt(
            tokenizer,
            agent_name="完全独立的Agent-C",
            target_tokens=4600,
        )
    )

    assert (
        8192
        < long_prompt_tokens
        < 9216
    )

    assert (
        4096
        < independent_tokens
        < 5120
    )

    llm = LLM(
        MODEL_PATH,
        enforce_eager=True,
        tensor_parallel_size=1,

        max_model_len=(
            max(
                long_prompt_tokens,
                independent_tokens,
            )
            + BLOCK_SIZE
        ),

        max_num_batched_tokens=4096,
        max_num_seqs=1,
        num_state_slots=1,

        gpu_memory_utilization=0.9,

        hybrid_prefix_cache_mode=(
            "opportunistic"
        ),

        prefix_checkpoint_interval_blocks=(
            4096 // BLOCK_SIZE
        ),

        prefix_recurrent_snapshot_dtype=(
            "bfloat16"
        ),

        max_new_prefix_snapshots_per_request=1,

        hybrid_prefix_cache_capacity_mib=320,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
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

    # --------------------------------------------------------
    # 1. 第一次长请求创建 4096 Entry A
    # --------------------------------------------------------

    print(
        "\n[Shared LRU] Creating Entry A..."
    )

    (
        first_output_ids,
        first_prefill_tokens,
        _,
    ) = run_one_request(
        llm,
        long_prompt,
        sampling_params,
    )

    assert (
        first_prefill_tokens
        == long_prompt_tokens
    )

    assert cache.num_entries == 1

    key_a = (
        cache.entry_keys_lru_to_mru[0]
    )

    assert key_a.num_cached_tokens == 4096

    # --------------------------------------------------------
    # 2. 第二次长请求命中A，并创建8192 Entry B
    # --------------------------------------------------------

    print(
        "[Shared LRU] Creating Entry B..."
    )

    (
        second_output_ids,
        second_prefill_tokens,
        _,
    ) = run_one_request(
        llm,
        long_prompt,
        sampling_params,
    )

    assert (
        second_prefill_tokens
        == long_prompt_tokens - 4096
    )

    assert (
        second_output_ids
        == first_output_ids
    )

    assert cache.num_entries == 2

    key_a_after, key_b = (
        cache.entry_keys_lru_to_mru
    )

    assert key_a_after == key_a
    assert key_b.num_cached_tokens == 8192

    entry_a = cache.entries[key_a]
    entry_b = cache.entries[key_b]

    assert (
        entry_b.kv_block_ids[:16]
        == entry_a.kv_block_ids
    )

    expected_hierarchical_capacity = (
        32 * kv_block_bytes
        + 2 * gdn_bytes
    )

    assert (
        cache.current_prefix_cache_capacity_bytes
        == expected_hierarchical_capacity
    )

    assert (
        mib(expected_hierarchical_capacity)
        == 307.0
    )

    # A全部KV都与B共享。
    assert (
        cache.reclaimable_capacity_bytes(
            entry_a
        )
        == gdn_bytes
    )

    print(
        "[Shared LRU] Capacity before C MiB:",
        mib(
            cache
            .current_prefix_cache_capacity_bytes
        ),
    )

    # --------------------------------------------------------
    # 3. 创建独立 Entry C，触发连续两次驱逐
    # --------------------------------------------------------

    print(
        "[Shared LRU] Creating independent Entry C..."
    )

    (
        _,
        third_prefill_tokens,
        _,
    ) = run_one_request(
        llm,
        independent_prompt,
        sampling_params,
    )

    assert (
        third_prefill_tokens
        == independent_tokens
    )

    # 驱逐顺序：
    #
    # 第一次驱逐 A：
    #   只能回收 A 的 GDN Snapshot。
    #
    # 第二次驱逐 B：
    #   才能回收全部32个共享/独占KV blocks。
    assert cache.num_evictions == 2

    assert (
        cache.total_evicted_capacity_bytes
        == expected_hierarchical_capacity
    )

    assert cache.num_capacity_rejections == 0

    assert cache.num_entries == 1

    key_c = (
        cache.entry_keys_lru_to_mru[0]
    )

    assert key_c not in {
        key_a,
        key_b,
    }

    assert key_c.num_cached_tokens == 4096

    expected_c_capacity = (
        16 * kv_block_bytes
        + gdn_bytes
    )

    assert (
        cache.current_prefix_cache_capacity_bytes
        == expected_c_capacity
    )

    assert (
        cache.num_unique_pinned_kv_blocks
        == 16
    )

    assert (
        cache.current_prefix_cache_capacity_bytes
        <= cache.capacity_bytes
    )

    print(
        "[Shared LRU] Evictions:",
        cache.num_evictions,
    )

    print(
        "[Shared LRU] Total evicted MiB:",
        mib(
            cache.total_evicted_capacity_bytes
        ),
    )

    print(
        "[Shared LRU] Final C capacity MiB:",
        mib(
            cache
            .current_prefix_cache_capacity_bytes
        ),
    )

    # --------------------------------------------------------
    # 4. 清理 C
    # --------------------------------------------------------

    assert cache.discard(key_c)

    assert cache.num_entries == 0
    assert not block_manager.used_block_ids

    assert (
        len(block_manager.free_block_ids)
        == len(block_manager.blocks)
    )

    print(
        "[Shared LRU] PASSED: shared hierarchy "
        "required two LRU evictions and capacity "
        "was recomputed after each eviction."
    )


# ============================================================
# Case 3：Hash碰撞必须由真实Token比较拦截
# ============================================================

def test_hash_collision_guard() -> None:
    """
    人工让另一个 Prompt 在 Hash 计算阶段得到相同
    block_hash。

    lookup_longest() 仍然必须逐 Block 比较真实
    token_ids，拒绝这个错误命中。
    """

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_PATH
        )
    )

    prompt, num_prompt_tokens = (
        build_long_prompt(
            tokenizer,
            target_tokens=1150,
        )
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

        prefix_checkpoint_interval_blocks=4,

        prefix_recurrent_snapshot_dtype=(
            "bfloat16"
        ),

        max_new_prefix_snapshots_per_request=1,

        hybrid_prefix_cache_capacity_mib=512,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        ignore_eos=True,
    )

    cache = llm.prefix_state_cache
    assert cache is not None

    block_manager = (
        llm.scheduler.block_manager
    )

    # --------------------------------------------------------
    # 1. 创建正常 Entry
    # --------------------------------------------------------

    print(
        "\n[Hash collision] Creating resident Entry..."
    )

    (
        _,
        prefill_tokens,
        _,
    ) = run_one_request(
        llm,
        prompt,
        sampling_params,
    )

    assert prefill_tokens == num_prompt_tokens
    assert cache.num_entries == 1

    key, _ = next(
        iter(cache.entries.items())
    )

    assert key.num_cached_tokens == 1024

    prompt_token_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
    )

    fake_prompt_token_ids = list(
        prompt_token_ids
    )

    # 修改第一个真实Token，确保Prompt内容不同。
    fake_prompt_token_ids[0] = (
        fake_prompt_token_ids[0] + 1
    ) % tokenizer.vocab_size

    assert (
        fake_prompt_token_ids
        != prompt_token_ids
    )

    # --------------------------------------------------------
    # 2. 人工伪造相同Hash
    # --------------------------------------------------------

    original_compute_hash = (
        block_manager.compute_hash
    )

    # lookup_longest() 最多会扫描多少个完整Prompt block。
    #
    # 当前Prompt约1150 tokens：
    #
    #   (1150 - 1) // 256 = 4
    #
    # 前4次compute_hash调用属于查询Prompt的链式Hash计算。
    # 后面的调用来自validate_prefix_blocks()，必须使用真实Hash。
    num_prompt_hash_calls = (
        (len(fake_prompt_token_ids) - 1)
        // BLOCK_SIZE
    )

    prompt_hash_call_count = 0

    def forged_compute_hash(
        token_ids: list[int],
        prefix: int = -1,
    ) -> int:
        nonlocal prompt_hash_call_count

        prompt_hash_call_count += 1

        # 前面的Prompt blocks仍然正常计算。
        #
        # 只在最后一个可缓存边界上强制返回Entry的Hash，
        # 模拟两个不同Token前缀最终发生xxhash64碰撞。
        if (
            prompt_hash_call_count
            == num_prompt_hash_calls
        ):
            return key.block_hash

        # 查询阶段结束后，validate_prefix_blocks()
        # 必须使用真实Hash检查resident KV blocks。
        return original_compute_hash(
            token_ids,
            prefix,
        )

    block_manager.compute_hash = (
        forged_compute_hash
    )
    collisions_before = (
        cache.num_hash_collisions
    )

    hits_before = cache.num_hits
    misses_before = cache.num_misses

    try:
        collision_result = (
            cache.lookup_longest(
                fake_prompt_token_ids
            )
        )
    finally:
        # 无论测试是否抛异常，都恢复真实Hash函数。
        block_manager.compute_hash = (
            original_compute_hash
        )

    # 即使PrefixKey中的Hash相同，也不能返回Entry。
    assert collision_result is None

    assert (
        cache.num_hash_collisions
        == collisions_before + 1
    )

    assert cache.num_hits == hits_before

    assert (
        cache.num_misses
        == misses_before + 1
    )

    # 错误候选不能更新LRU热度。
    assert cache.num_lru_touches == 0

    # Entry本身仍然正常resident。
    assert cache.num_entries == 1
    assert cache.entries[key].key == key

    print(
        "[Hash collision] Detected collisions:",
        cache.num_hash_collisions,
    )

    assert cache.discard(key)

    assert cache.num_entries == 0
    assert not block_manager.used_block_ids

    print(
        "[Hash collision] PASSED: matching Hash "
        "with different real tokens was rejected."
    )


CASE_FUNCTIONS = {
    "active_ownership": (
        test_active_request_ownership
    ),
    "shared_lru": (
        test_shared_lru_eviction
    ),
    "hash_collision": (
        test_hash_collision_guard
    ),
}


# 每一项都是传给Python解释器的一组完整参数。
#
# 普通测试只有脚本路径；
# 参数化测试同时包含其命令行参数。
EXISTING_TEST_COMMANDS = [
    [
        "tests/test_prefix_commit.py",
    ],
    [
        "tests/test_prefix_hit.py",
    ],
    [
    "tests/test_prefix_admission.py",
    ],

    # FP32 GDN Snapshot。
    [
        "tests/prefix_dtype_probe.py",
        "--snapshot-dtype",
        "float32",
        "--output",
        (
            "artifacts/prefix_regression/"
            "dtype_float32.json"
        ),
    ],

    # BF16 GDN Snapshot。
    [
        "tests/prefix_dtype_probe.py",
        "--snapshot-dtype",
        "bfloat16",
        "--output",
        (
            "artifacts/prefix_regression/"
            "dtype_bfloat16.json"
        ),
    ],

    # 4K Prefix压力测试。
    [
        "tests/prefix_stress.py",
        "--checkpoint-tokens",
        "4096",
        "--suffix-tokens",
        "512",
        "--output-tokens",
        "128",
        "--hot-runs",
        "3",
        "--snapshot-dtype",
        "bfloat16",
        "--output",
        (
            "artifacts/prefix_regression/"
            "stress_4k.json"
        ),
    ],

    # 8K Prefix压力测试。
    [
        "tests/prefix_stress.py",
        "--checkpoint-tokens",
        "8192",
        "--suffix-tokens",
        "512",
        "--output-tokens",
        "128",
        "--hot-runs",
        "3",
        "--snapshot-dtype",
        "bfloat16",
        "--output",
        (
            "artifacts/prefix_regression/"
            "stress_8k.json"
        ),
    ],

    # 16K Prefix压力测试。
    [
        "tests/prefix_stress.py",
        "--checkpoint-tokens",
        "16384",
        "--suffix-tokens",
        "512",
        "--output-tokens",
        "128",
        "--hot-runs",
        "3",
        "--snapshot-dtype",
        "bfloat16",
        "--output",
        (
            "artifacts/prefix_regression/"
            "stress_16k.json"
        ),
    ],

    [
        "tests/test_prefix_lru_order.py",
    ],
    [
        "tests/test_prefix_lru_eviction.py",
    ],
    [
        "tests/test_prefix_capacity_rejection.py",
    ],
    [
        "tests/test_prefix_shared_blocks.py",
    ],
]

def run_subprocess(
    command: list[str],
) -> None:
    print(
        "\n"
        + "=" * 72
    )

    print(
        "Running:",
        " ".join(command),
    )

    print(
        "=" * 72,
        flush=True,
    )

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


def run_remaining_cases() -> None:
    for case_name in CASE_FUNCTIONS:
        run_subprocess(
            [
                sys.executable,
                str(THIS_FILE),
                "--case",
                case_name,
            ]
        )


def run_full_regression() -> None:
    # 每组命令在独立Python进程中运行。
    #
    # 参数化测试会得到自己的参数和输出文件，
    # 每个子进程退出后也会释放该进程持有的模型显存。
    for test_command in EXISTING_TEST_COMMANDS:
        run_subprocess(
            [
                sys.executable,
                *test_command,
            ]
        )

    run_remaining_cases()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--case",
        choices=tuple(CASE_FUNCTIONS),
        default=None,
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Run all existing Prefix Cache tests "
            "before the remaining edge cases."
        ),
    )

    args = parser.parse_args()

    if args.case is not None:
        CASE_FUNCTIONS[args.case]()

        print(
            f"\nCase {args.case} passed."
        )

        return

    if args.full:
        run_full_regression()
    else:
        run_remaining_cases()

    print(
        "\n"
        + "=" * 72
    )

    print(
        "ALL CURRENT PREFIX CACHE TESTS PASSED"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()