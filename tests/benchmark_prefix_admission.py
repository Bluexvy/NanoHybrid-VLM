import argparse
import json
import subprocess
import sys

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
)

from test_prefix_lru_eviction import (
    build_agent_prompt,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

THIS_FILE = Path(__file__).resolve()

CHECKPOINT_TOKENS = 4096

CHECKPOINT_INTERVAL_BLOCKS = (
    CHECKPOINT_TOKENS
    // BLOCK_SIZE
)

CACHE_CAPACITY_MIB = 320


def mib(num_bytes: int) -> float:
    return (
        num_bytes
        / 1024
        / 1024
    )


def run_one_token_request(
    llm: LLM,
    prompt: str,
    sampling_params: SamplingParams,
) -> dict:
    """
    运行只生成一个token的请求。

    因为completion长度为1：

        E2E latency
        ≈
        first-token latency

    同时统计Scheduler实际执行的Prefill tokens。
    """

    torch.cuda.synchronize()

    start_time = perf_counter()

    seq_id = llm.add_request(
        prompt,
        sampling_params,
    )

    completion_token_ids = None

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
                completion_token_ids = (
                    token_ids
                )

    torch.cuda.synchronize()

    elapsed_ms = (
        perf_counter() - start_time
    ) * 1000.0

    if completion_token_ids is None:
        raise RuntimeError(
            f"Sequence {seq_id} did not finish"
        )

    assert len(completion_token_ids) == 1

    # 第一个token由最终Prefill Forward产生，
    # 因此不应该执行额外Decode轮。
    assert total_decode_tokens == 0

    return {
        "output_token_ids": (
            completion_token_ids
        ),
        "prefill_tokens": (
            total_prefill_tokens
        ),
        "single_token_latency_ms": (
            elapsed_ms
        ),
    }


def build_workload_prompts(
    tokenizer,
    repeat_index: int,
    pollution_count: int,
) -> tuple[
    str,
    int,
    list[tuple[str, int]],
]:
    """
    两种Policy在同一个repeat_index下使用完全相同的
    Hot Agent Prompt和污染Prompt。
    """

    hot_prompt, hot_prompt_tokens = (
        build_agent_prompt(
            tokenizer,
            agent_name=(
                f"固定HotAgent-{repeat_index}"
            ),
            target_tokens=4600,
        )
    )

    pollution_prompts = []

    for pollution_index in range(
        pollution_count
    ):
        prompt, num_tokens = (
            build_agent_prompt(
                tokenizer,
                agent_name=(
                    "一次性污染Agent-"
                    f"{repeat_index}-"
                    f"{pollution_index}"
                ),
                target_tokens=4600,
            )
        )

        pollution_prompts.append(
            (
                prompt,
                num_tokens,
            )
        )

    all_token_counts = [
        hot_prompt_tokens,
        *[
            num_tokens
            for _, num_tokens
            in pollution_prompts
        ],
    ]

    for num_tokens in all_token_counts:
        assert (
            CHECKPOINT_TOKENS
            < num_tokens
            < 5120
        )

    return (
        hot_prompt,
        hot_prompt_tokens,
        pollution_prompts,
    )


def run_policy(
    policy: str,
    repeat_index: int,
    pollution_count: int,
    output_path: Path,
) -> dict:
    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_PATH
        )
    )

    (
        hot_prompt,
        hot_prompt_tokens,
        pollution_prompts,
    ) = build_workload_prompts(
        tokenizer,
        repeat_index=repeat_index,
        pollution_count=pollution_count,
    )

    max_prompt_tokens = max(
        hot_prompt_tokens,
        *[
            num_tokens
            for _, num_tokens
            in pollution_prompts
        ],
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

        # 320 MiB最多容纳两个独立4K Entry：
        #
        # 2 × 153.5 MiB = 307 MiB
        hybrid_prefix_cache_capacity_mib=(
            CACHE_CAPACITY_MIB
        ),

        prefix_admission_policy=policy,

        prefix_admission_min_observations=2,

        prefix_admission_max_candidates=64,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        ignore_eos=True,
    )

    cache = llm.prefix_state_cache
    assert cache is not None

    # ==================================================
    # 1. Policy相关的预热
    # ==================================================

    # always：
    #     第一次即创建Entry。
    #
    # frequency：
    #     第一次只观察，第二次才创建Entry。
    warmup_runs = (
        1
        if policy == "always"
        else 2
    )

    warmup_results = []

    print(
        f"\n[{policy}] Warming hot Agent "
        f"with {warmup_runs} request(s)..."
    )

    for _ in range(warmup_runs):
        warmup_results.append(
            run_one_token_request(
                llm,
                hot_prompt,
                sampling_params,
            )
        )

    assert cache.num_entries == 1

    hot_key = next(
        iter(cache.entries)
    )

    assert (
        hot_key.num_cached_tokens
        == CHECKPOINT_TOKENS
    )

    # ==================================================
    # 2. 污染前测一次Hot请求
    # ==================================================

    print(
        f"[{policy}] Measuring hot request "
        "before pollution..."
    )

    pre_pollution_hot = (
        run_one_token_request(
            llm,
            hot_prompt,
            sampling_params,
        )
    )

    expected_hot_suffix = (
        hot_prompt_tokens
        - CHECKPOINT_TOKENS
    )

    assert (
        pre_pollution_hot[
            "prefill_tokens"
        ]
        == expected_hot_suffix
    )

    counters_before_pollution = {
        "commits": cache.num_commits,
        "evictions": cache.num_evictions,
        "capacity_rejections": (
            cache.num_capacity_rejections
        ),
        "admission_observations": (
            cache.num_admission_observations
        ),
        "admission_accepts": (
            cache.num_admission_accepts
        ),
        "admission_deferrals": (
            cache.num_admission_deferrals
        ),
    }

    # ==================================================
    # 3. 注入一次性长Prompt
    # ==================================================

    pollution_results = []

    print(
        f"[{policy}] Injecting "
        f"{pollution_count} one-hit Prefixes..."
    )

    for (
        pollution_prompt,
        pollution_tokens,
    ) in pollution_prompts:
        request_result = (
            run_one_token_request(
                llm,
                pollution_prompt,
                sampling_params,
            )
        )

        # 所有污染Prompt只出现一次，所以都必须完整Prefill。
        assert (
            request_result["prefill_tokens"]
            == pollution_tokens
        )

        pollution_results.append(
            request_result
        )

    hot_resident_after_pollution = (
        hot_key in cache.entries
    )

    counters_after_pollution = {
        "commits": cache.num_commits,
        "evictions": cache.num_evictions,
        "capacity_rejections": (
            cache.num_capacity_rejections
        ),
        "admission_observations": (
            cache.num_admission_observations
        ),
        "admission_accepts": (
            cache.num_admission_accepts
        ),
        "admission_deferrals": (
            cache.num_admission_deferrals
        ),
    }

    # 策略的核心结构性断言。
    if policy == "always":
        # 一次性Prompt进入GPU Cache后，
        # 热Agent应被LRU污染驱逐。
        assert not hot_resident_after_pollution

        assert (
            counters_after_pollution[
                "evictions"
            ]
            > counters_before_pollution[
                "evictions"
            ]
        )

    else:
        # frequency策略下，一次性Prompt只记录
        # CPU observation，不创建GPU Entry。
        assert hot_resident_after_pollution

        assert (
            counters_after_pollution[
                "evictions"
            ]
            == counters_before_pollution[
                "evictions"
            ]
        )

        assert (
            counters_after_pollution[
                "commits"
            ]
            == counters_before_pollution[
                "commits"
            ]
        )

    # ==================================================
    # 4. 污染后再次运行Hot请求
    # ==================================================

    print(
        f"[{policy}] Measuring hot request "
        "after pollution..."
    )

    post_pollution_hot = (
        run_one_token_request(
            llm,
            hot_prompt,
            sampling_params,
        )
    )

    # 相同Prompt必须生成相同首token。
    assert (
        post_pollution_hot[
            "output_token_ids"
        ]
        == pre_pollution_hot[
            "output_token_ids"
        ]
    )

    if policy == "always":
        # Hot Entry已被污染驱逐，所以重新完整Prefill。
        assert (
            post_pollution_hot[
                "prefill_tokens"
            ]
            == hot_prompt_tokens
        )

    else:
        # Hot Entry仍然resident，只处理suffix。
        assert (
            post_pollution_hot[
                "prefill_tokens"
            ]
            == expected_hot_suffix
        )

    counters_after_post_hot = {
        "commits": cache.num_commits,
        "evictions": cache.num_evictions,
        "capacity_rejections": (
            cache.num_capacity_rejections
        ),
        "admission_observations": (
            cache.num_admission_observations
        ),
        "admission_accepts": (
            cache.num_admission_accepts
        ),
        "admission_deferrals": (
            cache.num_admission_deferrals
        ),
    }

    measured_hot_hits = sum(
        result["prefill_tokens"]
        < hot_prompt_tokens
        for result in (
            pre_pollution_hot,
            post_pollution_hot,
        )
    )

    measured_hot_hit_rate = (
        measured_hot_hits / 2
    )

    result = {
        "policy": policy,
        "repeat_index": repeat_index,
        "pollution_count": pollution_count,

        "hot_prompt_tokens": (
            hot_prompt_tokens
        ),
        "checkpoint_tokens": (
            CHECKPOINT_TOKENS
        ),
        "expected_hot_suffix_tokens": (
            expected_hot_suffix
        ),

        "warmup_runs": warmup_runs,
        "warmup_results": warmup_results,

        "pre_pollution_hot": (
            pre_pollution_hot
        ),

        "post_pollution_hot": (
            post_pollution_hot
        ),

        "hot_resident_after_pollution": (
            hot_resident_after_pollution
        ),

        "measured_hot_hits": (
            measured_hot_hits
        ),

        "measured_hot_hit_rate": (
            measured_hot_hit_rate
        ),

        "pollution_total_prefill_tokens": sum(
            item["prefill_tokens"]
            for item in pollution_results
        ),

        "pollution_total_latency_ms": sum(
            item["single_token_latency_ms"]
            for item in pollution_results
        ),

        "pollution_request_latencies_ms": [
            item["single_token_latency_ms"]
            for item in pollution_results
        ],

        "counters_before_pollution": (
            counters_before_pollution
        ),

        "counters_after_pollution": (
            counters_after_pollution
        ),

        "counters_after_post_hot": (
            counters_after_post_hot
        ),

        "commits_during_pollution": (
            counters_after_pollution["commits"]
            - counters_before_pollution["commits"]
        ),

        "evictions_during_pollution": (
            counters_after_pollution["evictions"]
            - counters_before_pollution["evictions"]
        ),

        "deferrals_during_pollution": (
            counters_after_pollution[
                "admission_deferrals"
            ]
            - counters_before_pollution[
                "admission_deferrals"
            ]
        ),

        "commits_pollution_and_post": (
            counters_after_post_hot["commits"]
            - counters_before_pollution["commits"]
        ),

        "evictions_pollution_and_post": (
            counters_after_post_hot["evictions"]
            - counters_before_pollution["evictions"]
        ),

        "cache_entries_after_post": (
            cache.num_entries
        ),

        "cache_capacity_mib_after_post": mib(
            cache
            .current_prefix_cache_capacity_bytes
        ),

        "unique_pinned_kv_blocks_after_post": (
            cache.num_unique_pinned_kv_blocks
        ),

        "admission_candidate_count": (
            cache.num_admission_candidates
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"[{policy}] Hot resident after pollution:",
        hot_resident_after_pollution,
    )

    print(
        f"[{policy}] Post-pollution Prefill:",
        post_pollution_hot["prefill_tokens"],
    )

    print(
        f"[{policy}] Post-pollution latency ms:",
        round(
            post_pollution_hot[
                "single_token_latency_ms"
            ],
            3,
        ),
    )

    print(
        f"[{policy}] Evictions during pollution:",
        result["evictions_during_pollution"],
    )

    print(
        f"[{policy}] Commits during pollution:",
        result["commits_during_pollution"],
    )

    print(
        f"[{policy}] Result saved to:",
        output_path,
    )

    # 保存完统计后清理所有GPU Prefix Entries。
    for resident_key in tuple(
        cache.entries.keys()
    ):
        assert cache.discard(
            resident_key
        )

    assert cache.num_entries == 0
    assert not (
        llm.scheduler
        .block_manager
        .used_block_ids
    )

    return result


def run_child(
    policy: str,
    repeat_index: int,
    pollution_count: int,
    output_path: Path,
) -> None:
    command = [
        sys.executable,
        str(THIS_FILE),
        "--policy",
        policy,
        "--repeat-index",
        str(repeat_index),
        "--pollution-count",
        str(pollution_count),
        "--output",
        str(output_path),
    ]

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


def aggregate_results(
    result_paths: dict[
        str,
        list[Path],
    ],
    output_directory: Path,
) -> dict:
    results_by_policy = {}

    for policy, paths in result_paths.items():
        results_by_policy[policy] = [
            json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
            for path in paths
        ]

    def values(
        policy: str,
        field: str,
    ) -> list[float]:
        return [
            float(result[field])
            for result
            in results_by_policy[policy]
        ]

    always_post_latencies = [
        result[
            "post_pollution_hot"
        ][
            "single_token_latency_ms"
        ]
        for result
        in results_by_policy["always"]
    ]

    frequency_post_latencies = [
        result[
            "post_pollution_hot"
        ][
            "single_token_latency_ms"
        ]
        for result
        in results_by_policy["frequency"]
    ]

    always_median_post_ms = median(
        always_post_latencies
    )

    frequency_median_post_ms = median(
        frequency_post_latencies
    )

    latency_reduction_percent = (
        (
            always_median_post_ms
            - frequency_median_post_ms
        )
        / always_median_post_ms
        * 100.0
    )

    summary = {
        "num_repetitions": len(
            results_by_policy["always"]
        ),

        "always": {
            "median_post_pollution_latency_ms": (
                always_median_post_ms
            ),

            "median_hot_hit_rate": median(
                values(
                    "always",
                    "measured_hot_hit_rate",
                )
            ),

            "median_evictions_during_pollution": (
                median(
                    values(
                        "always",
                        "evictions_during_pollution",
                    )
                )
            ),

            "median_commits_during_pollution": (
                median(
                    values(
                        "always",
                        "commits_during_pollution",
                    )
                )
            ),
        },

        "frequency": {
            "median_post_pollution_latency_ms": (
                frequency_median_post_ms
            ),

            "median_hot_hit_rate": median(
                values(
                    "frequency",
                    "measured_hot_hit_rate",
                )
            ),

            "median_evictions_during_pollution": (
                median(
                    values(
                        "frequency",
                        "evictions_during_pollution",
                    )
                )
            ),

            "median_commits_during_pollution": (
                median(
                    values(
                        "frequency",
                        "commits_during_pollution",
                    )
                )
            ),
        },

        "post_pollution_latency_reduction_percent": (
            latency_reduction_percent
        ),
    }

    summary_path = (
        output_directory
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "PREFIX ADMISSION BENCHMARK SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        "Always median post-pollution latency ms:",
        round(
            always_median_post_ms,
            3,
        ),
    )

    print(
        "Frequency median post-pollution latency ms:",
        round(
            frequency_median_post_ms,
            3,
        ),
    )

    print(
        "Observed latency reduction:",
        round(
            latency_reduction_percent,
            2,
        ),
        "%",
    )

    print(
        "Always median hot hit rate:",
        summary["always"][
            "median_hot_hit_rate"
        ],
    )

    print(
        "Frequency median hot hit rate:",
        summary["frequency"][
            "median_hot_hit_rate"
        ],
    )

    print(
        "Always median pollution evictions:",
        summary["always"][
            "median_evictions_during_pollution"
        ],
    )

    print(
        "Frequency median pollution evictions:",
        summary["frequency"][
            "median_evictions_during_pollution"
        ],
    )

    print(
        "Summary saved to:",
        summary_path,
    )

    return summary


def run_parent(
    repetitions: int,
    pollution_count: int,
    output_directory: Path,
) -> None:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_paths = {
        "always": [],
        "frequency": [],
    }

    # 交替Policy顺序，减少固定运行顺序造成的温度、
    # 功耗或系统负载偏差。
    for repeat_index in range(
        repetitions
    ):
        policy_order = (
            ["always", "frequency"]
            if repeat_index % 2 == 0
            else ["frequency", "always"]
        )

        for policy in policy_order:
            output_path = (
                output_directory
                / (
                    f"{policy}_"
                    f"{repeat_index}.json"
                )
            )

            run_child(
                policy=policy,
                repeat_index=repeat_index,
                pollution_count=(
                    pollution_count
                ),
                output_path=output_path,
            )

            result_paths[policy].append(
                output_path
            )

    aggregate_results(
        result_paths=result_paths,
        output_directory=output_directory,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--policy",
        choices=[
            "always",
            "frequency",
        ],
        default=None,
    )

    parser.add_argument(
        "--repeat-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--pollution-count",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(
            "artifacts/prefix_admission_benchmark"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.pollution_count < 2:
        raise ValueError(
            "pollution-count must be at least 2"
        )

    if args.repetitions <= 0:
        raise ValueError(
            "repetitions must be positive"
        )

    if args.policy is not None:
        if args.output is None:
            raise ValueError(
                "--output is required with --policy"
            )

        run_policy(
            policy=args.policy,
            repeat_index=args.repeat_index,
            pollution_count=(
                args.pollution_count
            ),
            output_path=args.output,
        )

        return

    run_parent(
        repetitions=args.repetitions,
        pollution_count=args.pollution_count,
        output_directory=(
            args.output_directory
        ),
    )


if __name__ == "__main__":
    main()