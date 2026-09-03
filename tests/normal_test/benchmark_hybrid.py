import argparse
import json
from math import floor, ceil
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image, ImageDraw

from nanovllm import LLM, SamplingParams


MODEL_PATH = "/workspace/models/Qwen3.5-9B"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workload",
        choices=(
            "text",
            "image",
            "mixed",
        ),
        required=True,
    )

    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--output-tokens",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--token-budget",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def percentile(
    values: list[float],
    quantile: float,
) -> float:
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

    fraction = position - lower_index

    return (
        ordered[lower_index]
        * (1.0 - fraction)
        + ordered[upper_index]
        * fraction
    )


def summarize_latency(
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
    }


def create_test_image(
    size: int,
) -> Image.Image:
    image = Image.new(
        mode="RGB",
        size=(size, size),
        color="white",
    )

    draw = ImageDraw.Draw(image)

    margin = size // 8

    draw.rectangle(
        xy=(
            margin,
            size // 3,
            size // 2 - margin,
            2 * size // 3,
        ),
        fill="red",
    )

    draw.ellipse(
        xy=(
            size // 2 + margin,
            size // 3,
            size - margin,
            2 * size // 3,
        ),
        fill="blue",
    )

    return image


def create_text_token_ids(
    llm: LLM,
    num_tokens: int,
) -> list[int]:
    """
    构造长度严格等于 num_tokens 的输入。

    Benchmark 关注 Runtime，不评估文本质量，
    因此使用重复的合法词元。
    """

    candidate_ids = (
        llm.tokenizer.encode(
            "性能测试",
            add_special_tokens=False,
        )
    )

    if not candidate_ids:
        raise RuntimeError(
            "Tokenizer returned no tokens"
        )

    token_id = candidate_ids[0]

    return [
        token_id
        for _ in range(num_tokens)
    ]


def build_prompts(
    llm: LLM,
    workload: str,
    concurrency: int,
    prompt_tokens: int,
    image_size: int,
):
    text_prompt = create_text_token_ids(
        llm,
        prompt_tokens,
    )

    image = create_test_image(
        image_size
    )

    image_prompt = {
        "prompt": (
            "请简要描述图片中的颜色和形状。"
        ),
        "multi_modal_data": {
            "image": image,
        },
    }

    prompts = []

    for index in range(concurrency):
        if workload == "text":
            prompts.append(
                text_prompt.copy()
            )

        elif workload == "image":
            prompts.append(
                {
                    "prompt": (
                        image_prompt["prompt"]
                    ),
                    "multi_modal_data": {
                        "image": image.copy(),
                    },
                }
            )

        else:
            # Mixed workload：
            # 偶数请求为文本，奇数请求为图片。
            if index % 2 == 0:
                prompts.append(
                    text_prompt.copy()
                )
            else:
                prompts.append(
                    {
                        "prompt": (
                            image_prompt["prompt"]
                        ),
                        "multi_modal_data": {
                            "image": image.copy(),
                        },
                    }
                )

    return prompts


def bytes_to_gib(
    num_bytes: int,
) -> float:
    return (
        num_bytes
        / 1024**3
    )


def main() -> None:
    args = parse_args()

    if args.prompt_tokens <= 0:
        raise ValueError(
            "prompt-tokens must be positive"
        )

    if args.output_tokens <= 0:
        raise ValueError(
            "output-tokens must be positive"
        )

    if args.concurrency <= 0:
        raise ValueError(
            "concurrency must be positive"
        )

    llm = LLM(
        MODEL_PATH,
        enforce_eager=True,
        tensor_parallel_size=1,

        # 足够覆盖8K文本和输出。
        max_model_len=16384,

        max_num_batched_tokens=(
            args.token_budget
        ),

        max_num_seqs=args.concurrency,
        num_state_slots=args.concurrency,

        max_prefill_wait_ms=50.0,

        gpu_memory_utilization=0.8,
    )

    prompts = build_prompts(
        llm=llm,
        workload=args.workload,
        concurrency=args.concurrency,
        prompt_tokens=args.prompt_tokens,
        image_size=args.image_size,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.output_tokens,

        # Benchmark 固定输出长度，
        # 防止请求提前遇到 EOS。
        ignore_eos=True,
    )

    # Benchmark 只统计模型运行阶段的
    # CUDA peak，不包含模型加载峰值。
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    benchmark_start = perf_counter()

    for prompt in prompts:
        llm.add_request(
            prompt,
            sampling_params,
        )

    total_prefill_tokens = 0
    total_decode_tokens = 0

    total_prefill_elapsed = 0.0
    total_decode_elapsed = 0.0

    while not llm.is_finished():
        _, step_stats = llm.step()

        total_prefill_tokens += (
            step_stats.num_prefill_tokens
        )

        total_decode_tokens += (
            step_stats.num_decode_tokens
        )

        total_prefill_elapsed += (
            step_stats.prefill_elapsed
        )

        total_decode_elapsed += (
            step_stats.decode_elapsed
        )

    torch.cuda.synchronize()

    benchmark_elapsed = (
        perf_counter()
        - benchmark_start
    )

    request_metrics = (
        llm.get_completed_request_metrics()
    )

    if (
        len(request_metrics)
        != args.concurrency
    ):
        raise RuntimeError(
            "Not all requests completed"
        )

    ttft_values = [
        metrics.ttft_ms
        for metrics in request_metrics
    ]

    tpot_values = [
        metrics.tpot_ms
        for metrics in request_metrics
    ]

    e2e_values = [
        metrics.e2e_ms
        for metrics in request_metrics
    ]

    queue_values = [
        metrics.queue_ms
        for metrics in request_metrics
    ]

    preprocessing_values = [
        metrics.preprocessing_ms
        for metrics in request_metrics
    ]

    memory_bytes = (
        llm.model_runner.get_memory_stats()
    )

    memory_gib = {
        name.replace(
            "_bytes",
            "_gib",
        ): bytes_to_gib(value)
        for name, value
        in memory_bytes.items()
    }

    prefill_tokens_per_second = (
        total_prefill_tokens
        / total_prefill_elapsed
        if total_prefill_elapsed > 0
        else 0.0
    )

    decode_tokens_per_second = (
        total_decode_tokens
        / total_decode_elapsed
        if total_decode_elapsed > 0
        else 0.0
    )

    result = {
        "config": {
            "model": MODEL_PATH,
            "workload": args.workload,
            "prompt_tokens": (
                args.prompt_tokens
            ),
            "image_size": (
                args.image_size
            ),
            "output_tokens": (
                args.output_tokens
            ),
            "concurrency": (
                args.concurrency
            ),
            "token_budget": (
                args.token_budget
            ),
        },
        "throughput": {
            "requests_per_second": (
                args.concurrency
                / benchmark_elapsed
            ),
            "prefill_tokens_per_second": (
                prefill_tokens_per_second
            ),
            "decode_tokens_per_second": (
                decode_tokens_per_second
            ),
            "benchmark_elapsed_seconds": (
                benchmark_elapsed
            ),
            "executed_prefill_tokens": (
                total_prefill_tokens
            ),
            "executed_decode_tokens": (
                total_decode_tokens
            ),
        },
        "latency_ms": {
            "preprocessing": (
                summarize_latency(
                    preprocessing_values
                )
            ),
            "queue": summarize_latency(
                queue_values
            ),
            "ttft": summarize_latency(
                ttft_values
            ),
            "tpot": summarize_latency(
                tpot_values
            ),
            "e2e": summarize_latency(
                e2e_values
            ),
        },
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
            "num_vision_forwards": (
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
        },
        "memory_bytes": memory_bytes,
        "memory_gib": memory_gib,
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

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()