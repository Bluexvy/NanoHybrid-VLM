import argparse
import json
from math import floor, ceil
from pathlib import Path
from time import perf_counter

from PIL import Image, ImageDraw

from nanovllm import LLM, SamplingParams


MODEL_PATH = "/workspace/models/Qwen3.5-9B"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--policy",
        choices=(
            "decode_first",
            "prefill_first",
        ),
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    
    parser.add_argument(
        "--token-budget",
        type=int,
        default=512,
    )

    return parser.parse_args()


def percentile(
    values: list[float],
    quantile: float,
) -> float:
    """
    使用线性插值计算百分位数。
    """

    if not values:
        raise ValueError(
            "Cannot calculate percentile "
            "of an empty list"
        )

    ordered = sorted(values)

    position = (
        len(ordered) - 1
    ) * quantile

    lower_index = floor(position)
    upper_index = ceil(position)

    if lower_index == upper_index:
        return ordered[lower_index]

    fraction = (
        position - lower_index
    )

    return (
        ordered[lower_index]
        * (1.0 - fraction)
        + ordered[upper_index]
        * fraction
    )


def summarize(
    values: list[float],
) -> dict[str, float]:
    return {
        "p50": percentile(
            values,
            0.50,
        ),
        "p95": percentile(
            values,
            0.95,
        ),
        "p99": percentile(
            values,
            0.99,
        ),
        "max": max(values),
    }


def create_text_tokens(
    llm: LLM,
    length: int,
) -> list[int]:
    """
    创建长度严格等于 length 的文本 token。
    """

    candidate_ids = (
        llm.tokenizer.encode(
            "性能测试",
            add_special_tokens=False,
        )
    )

    if not candidate_ids:
        raise RuntimeError(
            "Tokenizer returned no token"
        )

    token_id = candidate_ids[0]

    return [
        token_id
        for _ in range(length)
    ]


def create_image() -> Image.Image:
    """
    创建一张1024×1024测试图片。

    经过 Processor 后大约产生1024个
    visual tokens。
    """

    image = Image.new(
        mode="RGB",
        size=(1024, 1024),
        color="white",
    )

    draw = ImageDraw.Draw(image)

    draw.rectangle(
        xy=(128, 320, 448, 704),
        fill="red",
    )

    draw.ellipse(
        xy=(576, 320, 896, 704),
        fill="blue",
    )

    return image


def main() -> None:
    args = parse_args()

    llm = LLM(
        MODEL_PATH,
        enforce_eager=True,
        tensor_parallel_size=1,

        # 本次实验的唯一自变量。
        scheduler_policy=args.policy,

        max_model_len=8192,

        # 每轮最多执行512个 token。
        max_num_batched_tokens=(
            args.token_budget
        ),

        # 4条已有请求 + 2条迟到请求。
        max_num_seqs=6,
        num_state_slots=6,

        max_prefill_wait_ms=50.0,
        gpu_memory_utilization=0.8,
    )

    # step() 只会返回已经完成的请求。
    # 这里按照 seq_id 保存最终 token。
    completed_outputs: dict[
        int,
        list[int],
    ] = {}

    def execute_one_step() -> None:
        outputs, _ = llm.step()

        for seq_id, token_ids in outputs:
            completed_outputs[
                seq_id
            ] = token_ids

    # =====================================
    # 第一阶段：建立4条正在Decode的请求
    # =====================================

    incumbent_ids = []

    for _ in range(4):
        seq_id = llm.add_request(
            create_text_tokens(
                llm,
                128,
            ),
            SamplingParams(
                temperature=0.0,
                max_tokens=128,

                # 固定输出长度，避免提前EOS。
                ignore_eos=True,
            ),
        )

        incumbent_ids.append(
            seq_id
        )

    # 第一次 step 通常就能完成四条请求的
    # Batched Prefill。
    #
    # 使用 while 是为了兼容将来 prompt 更长、
    # 需要 Chunked Prefill 的情况。
    while any(
        llm.request_metrics[
            seq_id
        ].first_token_time is None
        for seq_id in incumbent_ids
    ):
        execute_one_step()

    # 再运行8轮 Decode，使它们已经稳定处于
    # 连续生成阶段。
    for _ in range(8):
        execute_one_step()

    # =====================================
    # 第二阶段：注入两个迟到的长请求
    # =====================================

    # 这个时间点之后的 incumbent token 间隔，
    # 都算作受到长 Prefill 干扰的阶段。
    interference_start = (
        perf_counter()
    )

    image_seq_id = llm.add_request(
        {
            "prompt": (
                "请简要描述图片中的内容。"
            ),
            "multi_modal_data": {
                "image": create_image(),
            },
        },
        SamplingParams(
            temperature=0.0,
            max_tokens=16,
            ignore_eos=True,
        ),
    )

    long_text_seq_id = (
        llm.add_request(
            create_text_tokens(
                llm,
                4096,
            ),
            SamplingParams(
                temperature=0.0,
                max_tokens=16,
                ignore_eos=True,
            ),
        )
    )

    late_request_ids = [
        image_seq_id,
        long_text_seq_id,
    ]

    # 继续执行，直到6条请求全部完成。
    while not llm.is_finished():
        execute_one_step()

    benchmark_finish = (
        perf_counter()
    )

    # 两条迟到请求都产生第一个 token，
    # 说明它们的 Prefill 都已经完成。
    late_first_token_times = []

    for seq_id in late_request_ids:
        first_token_time = (
            llm.request_metrics[
                seq_id
            ].first_token_time
        )

        if first_token_time is None:
            raise RuntimeError(
                f"Late Sequence {seq_id} "
                "has no first-token time"
            )

        late_first_token_times.append(
            first_token_time
        )

    interference_end = max(
        late_first_token_times
    )

    # =====================================
    # 第三阶段：检查 token 时间戳完整性
    # =====================================

    for seq_id in incumbent_ids:
        metrics = llm.request_metrics[
            seq_id
        ]

        if (
            len(metrics.token_timestamps)
            != metrics.num_completion_tokens
        ):
            raise AssertionError(
                f"Sequence {seq_id} has "
                "incomplete token timestamps"
            )

    # =====================================
    # 第四阶段：计算已有请求的逐token TPOT
    # =====================================

    incumbent_intervals_ms = []

    for seq_id in incumbent_ids:
        timestamps = (
            llm.request_metrics[
                seq_id
            ].token_timestamps
        )

        for previous, current in zip(
            timestamps,
            timestamps[1:],
        ):
            # token interval 表示时间区间：
            #
            # [previous, current]
            #
            # 只要它与：
            #
            # [interference_start, interference_end]
            #
            # 存在重叠，就属于 Prefill 干扰区间。
            interval_overlaps_interference = (
                current >= interference_start
                and previous <= interference_end
            )

            if interval_overlaps_interference:
                interval_ms = (
                    current - previous
                ) * 1000.0

                incumbent_intervals_ms.append(
                    interval_ms
                )

    if not incumbent_intervals_ms:
        raise RuntimeError(
            "No incumbent token intervals "
            "were recorded"
        )

    # =====================================
    # 第五阶段：统计迟到请求 TTFT
    # =====================================

    late_ttft_ms = []

    for seq_id in late_request_ids:
        ttft_ms = llm.request_metrics[
            seq_id
        ].ttft_ms

        if ttft_ms is None:
            raise RuntimeError(
                f"Late Sequence {seq_id} "
                "has no TTFT"
            )

        late_ttft_ms.append(
            ttft_ms
        )

    incumbent_tpot_summary = (
        summarize(
            incumbent_intervals_ms
        )
    )

    late_ttft_summary = (
        summarize(
            late_ttft_ms
        )
    )

    # 本实验预留了6个 slot，
    # 不应该触发资源抢占。
    if (
        llm.scheduler.num_preemptions
        != 0
    ):
        raise AssertionError(
            "Dynamic scheduler benchmark "
            "unexpectedly triggered preemption"
        )

    if (
        llm.model_runner
        .visual_cache_bytes
        != 0
    ):
        raise AssertionError(
            "Visual cache leaked"
        )

    # =====================================
    # 第六阶段：保存结果
    # =====================================

    result = {
        "policy": args.policy,

        "workload": {
            "incumbent_requests": 4,
            "incumbent_prompt_tokens": 128,
            "incumbent_output_tokens": 128,
            "warmup_decode_steps": 8,

            "late_image_size": 1024,
            "late_text_tokens": 4096,
            "late_output_tokens": 16,

            "token_budget": (
            args.token_budget
            ),
            "max_num_seqs": 6,
        },

        "incumbent_token_intervals": (
            len(incumbent_intervals_ms)
        ),

        "incumbent_tpot_ms": (
            incumbent_tpot_summary
        ),

        "late_request_ttft_ms": (
            late_ttft_summary
        ),

        "interference_elapsed_seconds": (
            benchmark_finish
            - interference_start
        ),
        "prefill_interference_seconds": (
            interference_end
            - interference_start
        ),

        "scheduler": {
            "num_preemptions": (
                llm.scheduler.num_preemptions
            ),
            "num_recomputed_tokens": (
                llm.scheduler
                .num_recomputed_tokens
            ),
        },

        "vision": {
            "forwards": (
                llm.model_runner
                .num_vision_forwards
            ),
            "cache_hits": (
                llm.model_runner
                .num_visual_cache_hits
            ),
            "cache_misses": (
                llm.model_runner
                .num_visual_cache_misses
            ),
            "final_cache_bytes": (
                llm.model_runner
                .visual_cache_bytes
            ),
        },

        # 下一步用来检查两种策略是否改变模型输出。
        "completed_outputs": {
            str(seq_id): token_ids
            for seq_id, token_ids
            in completed_outputs.items()
        },
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nPolicy:", args.policy)

    print(
        "Incumbent token intervals:",
        len(incumbent_intervals_ms),
    )

    print(
        "Incumbent TPOT:",
        incumbent_tpot_summary,
    )

    print(
        "Late-request TTFT:",
        late_ttft_summary,
    )

    print(
        "Interference elapsed:",
        (
            benchmark_finish
            - interference_start
        ),
        "seconds",
    )

    print(
        "Vision forwards:",
        llm.model_runner
        .num_vision_forwards,
    )

    print(
        "Vision cache hits:",
        llm.model_runner
        .num_visual_cache_hits,
    )

    print(
        "Result saved to:",
        args.output,
    )
    print(
        "Prefill interference window:",
        (
            interference_end
            - interference_start
        ),
        "seconds",
    )


if __name__ == "__main__":
    main()