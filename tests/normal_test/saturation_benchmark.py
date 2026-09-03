import argparse
import json
from collections import Counter
from math import ceil, floor
from pathlib import Path
from statistics import fmean
from threading import Event, Thread
from time import perf_counter

import torch
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
        "--requests",
        type=int,
        default=96,
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--token-budget",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=0.2,
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
    ordered = sorted(values)

    position = (
        len(ordered) - 1
    ) * quantile

    lower = floor(position)
    upper = ceil(position)

    if lower == upper:
        return ordered[lower]

    fraction = position - lower

    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def summarize(
    values: list[float],
) -> dict[str, float] | None:
    if not values:
        return None

    return {
        "count": len(values),
        "mean": fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


class GpuMonitor:
    """
    使用 NVML 后台采样 GPU 利用率、功耗和显存。

    需要：
        pip install nvidia-ml-py

    如果没有安装，Benchmark 仍然可以运行，
    只是 GPU 监控字段为空。
    """

    def __init__(
        self,
        gpu_index: int,
        interval: float,
    ):
        self.gpu_index = gpu_index
        self.interval = interval

        self.enabled = False
        self.error = None

        self.stop_event = Event()
        self.thread = None

        self.gpu_utilization = []
        self.memory_utilization = []
        self.memory_used_gib = []
        self.power_watts = []

        self.power_limit_watts = None

        try:
            import pynvml

            self.pynvml = pynvml
            pynvml.nvmlInit()

            self.handle = (
                pynvml.nvmlDeviceGetHandleByIndex(
                    gpu_index
                )
            )

            try:
                self.power_limit_watts = (
                    pynvml
                    .nvmlDeviceGetPowerManagementLimit(
                        self.handle
                    )
                    / 1000.0
                )
            except Exception:
                self.power_limit_watts = None

            self.enabled = True

        except Exception as exception:
            self.error = repr(exception)

    def _sample_once(self) -> bool:
        try:
            utilization = (
                self.pynvml
                .nvmlDeviceGetUtilizationRates(
                    self.handle
                )
            )

            memory = (
                self.pynvml
                .nvmlDeviceGetMemoryInfo(
                    self.handle
                )
            )

            power_watts = (
                self.pynvml
                .nvmlDeviceGetPowerUsage(
                    self.handle
                )
                / 1000.0
            )

            self.gpu_utilization.append(
                float(utilization.gpu)
            )

            self.memory_utilization.append(
                float(utilization.memory)
            )

            self.memory_used_gib.append(
                memory.used / 1024**3
            )

            self.power_watts.append(
                power_watts
            )

            return True

        except Exception as exception:
            self.error = repr(exception)
            return False

    def _run(self):
        if not self._sample_once():
            return

        while not self.stop_event.wait(
            self.interval
        ):
            if not self._sample_once():
                return

    def start(self):
        if not self.enabled:
            return

        self.thread = Thread(
            target=self._run,
            daemon=True,
        )

        self.thread.start()

    def stop(self):
        if not self.enabled:
            return

        self.stop_event.set()

        if self.thread is not None:
            self.thread.join()

        try:
            self.pynvml.nvmlShutdown()
        except Exception:
            pass

    def result(self):
        return {
            "enabled": self.enabled,
            "error": self.error,
            "sample_count": (
                len(self.gpu_utilization)
            ),
            "power_limit_watts": (
                self.power_limit_watts
            ),
            "gpu_utilization_percent": (
                summarize(
                    self.gpu_utilization
                )
            ),
            "memory_controller_percent": (
                summarize(
                    self.memory_utilization
                )
            ),
            "memory_used_gib": (
                summarize(
                    self.memory_used_gib
                )
            ),
            "power_watts": (
                summarize(
                    self.power_watts
                )
            ),
        }


def create_text_tokens(
    llm: LLM,
    length: int,
) -> list[int]:
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
        for _ in range(length)
    ]


def create_image() -> Image.Image:
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


def build_request(
    llm: LLM,
    request_index: int,
    image: Image.Image,
):
    """
    每6条请求构成一个固定周期：

    0～3：
        Decode-heavy
        128 Prompt + 256 Decode

    4：
        Long Prefill
        4096 Prompt + 32 Decode

    5：
        Vision Prefill
        图片 Prompt + 32 Decode
    """

    workload_position = (
        request_index % 6
    )

    if workload_position <= 3:
        request_type = "decode_heavy"

        prompt = create_text_tokens(
            llm,
            128,
        )

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=256,
            ignore_eos=True,
        )

    elif workload_position == 4:
        request_type = "long_prefill"

        prompt = create_text_tokens(
            llm,
            4096,
        )

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=32,
            ignore_eos=True,
        )

    else:
        request_type = "vision_prefill"

        prompt = {
            "prompt": (
                "请简要描述图片中的颜色和形状。"
            ),
            "multi_modal_data": {
                "image": image.copy(),
            },
        }

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=32,
            ignore_eos=True,
        )

    return (
        request_type,
        prompt,
        sampling_params,
    )


def collect_metric_values(
    metrics,
    attribute: str,
) -> list[float]:
    values = []

    for metric in metrics:
        value = getattr(
            metric,
            attribute,
        )

        if value is not None:
            values.append(float(value))

    return values


def main():
    args = parse_args()

    if args.requests <= 0:
        raise ValueError(
            "requests must be positive"
        )

    if args.concurrency <= 0:
        raise ValueError(
            "concurrency must be positive"
        )

    if args.token_budget <= 0:
        raise ValueError(
            "token-budget must be positive"
        )

    llm = LLM(
        MODEL_PATH,
        enforce_eager=True,
        tensor_parallel_size=1,

        scheduler_policy=args.policy,

        max_model_len=8192,

        max_num_batched_tokens=(
            args.token_budget
        ),

        max_num_seqs=args.concurrency,
        num_state_slots=args.concurrency,

        max_prefill_wait_ms=50.0,

        gpu_memory_utilization=(
            args.gpu_memory_utilization
        ),
    )

    # =====================================
    # 第一阶段：运行小规模热身
    # =====================================

    warmup_requests = min(
        4,
        args.concurrency,
    )

    warmup_sampling = SamplingParams(
        temperature=0.0,
        max_tokens=8,
        ignore_eos=True,
    )

    warmup_prompt = create_text_tokens(
        llm,
        128,
    )

    for _ in range(warmup_requests):
        llm.add_request(
            warmup_prompt.copy(),
            warmup_sampling,
        )

    while not llm.is_finished():
        llm.step()

    torch.cuda.synchronize()

    # 热身结果不进入正式统计。
    torch.cuda.reset_peak_memory_stats()

    scheduler_preemptions_start = (
        llm.scheduler.num_preemptions
    )

    scheduler_recomputed_start = (
        llm.scheduler.num_recomputed_tokens
    )

    vision_forwards_start = (
        llm.model_runner.num_vision_forwards
    )

    vision_hits_start = (
        llm.model_runner
        .num_visual_cache_hits
    )

    vision_misses_start = (
        llm.model_runner
        .num_visual_cache_misses
    )

    # =====================================
    # 第二阶段：构造持续闭环负载
    # =====================================

    image = create_image()

    # seq_id -> 请求信息
    request_information = {}

    measured_seq_ids = []

    completed_outputs = {}

    submitted_requests = 0
    completed_requests = 0

    total_prefill_tokens = 0
    total_decode_tokens = 0

    total_prefill_elapsed = 0.0
    total_decode_elapsed = 0.0

    num_prefill_microbatches = 0
    num_decode_microbatches = 0

    waiting_depths = []
    running_depths = []
    outstanding_depths = []

    gpu_monitor = GpuMonitor(
        gpu_index=args.gpu_index,
        interval=args.monitor_interval,
    )

    def submit_one_request():
        nonlocal submitted_requests

        request_index = submitted_requests

        (
            request_type,
            prompt,
            sampling_params,
        ) = build_request(
            llm=llm,
            request_index=request_index,
            image=image,
        )

        seq_id = llm.add_request(
            prompt,
            sampling_params,
        )

        request_information[seq_id] = {
            "request_index": request_index,
            "request_type": request_type,
        }

        measured_seq_ids.append(seq_id)

        submitted_requests += 1

    benchmark_start = perf_counter()

    gpu_monitor.start()

    # 首先填满目标并发。
    while (
        submitted_requests < args.requests
        and submitted_requests
        < args.concurrency
    ):
        submit_one_request()

    progress_interval = max(
        args.requests // 10,
        1,
    )

    next_progress = progress_interval

    # 每完成一个请求，就立即补充一个请求。
    while completed_requests < args.requests:
        waiting_depths.append(
            len(llm.scheduler.waiting)
        )

        running_depths.append(
            len(llm.scheduler.running)
        )

        outstanding_depths.append(
            submitted_requests
            - completed_requests
        )

        outputs, step_stats = llm.step()

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

        if step_stats.num_prefill_tokens > 0:
            num_prefill_microbatches += 1

        if step_stats.num_decode_tokens > 0:
            num_decode_microbatches += 1

        for seq_id, token_ids in outputs:
            information = (
                request_information.get(
                    seq_id
                )
            )

            if information is None:
                raise RuntimeError(
                    f"Unknown completed seq_id: "
                    f"{seq_id}"
                )

            request_index = information[
                "request_index"
            ]

            completed_outputs[
                str(request_index)
            ] = token_ids

            completed_requests += 1

        # 恢复到目标并发。
        while (
            submitted_requests < args.requests
            and (
                submitted_requests
                - completed_requests
            ) < args.concurrency
        ):
            submit_one_request()

        if completed_requests >= next_progress:
            print(
                f"Progress: "
                f"{completed_requests}/"
                f"{args.requests}"
            )

            while (
                next_progress
                <= completed_requests
            ):
                next_progress += (
                    progress_interval
                )

    torch.cuda.synchronize()

    benchmark_finish = perf_counter()

    gpu_monitor.stop()

    benchmark_elapsed = (
        benchmark_finish
        - benchmark_start
    )

    # =====================================
    # 第三阶段：请求级延迟
    # =====================================

    request_metrics = [
        llm.request_metrics[seq_id]
        for seq_id in measured_seq_ids
    ]

    if any(
        metric.finish_time is None
        for metric in request_metrics
    ):
        raise RuntimeError(
            "Some requests did not finish"
        )

    request_details = []

    metrics_by_type = {
        "decode_heavy": [],
        "long_prefill": [],
        "vision_prefill": [],
    }

    token_intervals_ms = []

    for seq_id in measured_seq_ids:
        metric = llm.request_metrics[
            seq_id
        ]

        information = (
            request_information[seq_id]
        )

        request_type = information[
            "request_type"
        ]

        metrics_by_type[
            request_type
        ].append(metric)

        timestamps = (
            metric.token_timestamps
        )

        for previous, current in zip(
            timestamps,
            timestamps[1:],
        ):
            token_intervals_ms.append(
                (
                    current - previous
                ) * 1000.0
            )

        request_details.append({
            "request_index": (
                information[
                    "request_index"
                ]
            ),
            "seq_id": seq_id,
            "request_type": request_type,
            "prompt_tokens": (
                metric.num_prompt_tokens
            ),
            "completion_tokens": (
                metric.num_completion_tokens
            ),
            "preprocessing_ms": (
                metric.preprocessing_ms
            ),
            "queue_ms": metric.queue_ms,
            "ttft_ms": metric.ttft_ms,
            "tpot_ms": metric.tpot_ms,
            "e2e_ms": metric.e2e_ms,
        })

    # =====================================
    # 第四阶段：按请求类型汇总
    # =====================================

    latency_by_type = {}

    for request_type, metrics in (
        metrics_by_type.items()
    ):
        latency_by_type[request_type] = {
            "requests": len(metrics),
            "queue_ms": summarize(
                collect_metric_values(
                    metrics,
                    "queue_ms",
                )
            ),
            "ttft_ms": summarize(
                collect_metric_values(
                    metrics,
                    "ttft_ms",
                )
            ),
            "request_tpot_ms": summarize(
                collect_metric_values(
                    metrics,
                    "tpot_ms",
                )
            ),
            "e2e_ms": summarize(
                collect_metric_values(
                    metrics,
                    "e2e_ms",
                )
            ),
        }

    total_prompt_tokens = sum(
        metric.num_prompt_tokens
        for metric in request_metrics
    )

    total_completion_tokens = sum(
        metric.num_completion_tokens
        for metric in request_metrics
    )

    memory_stats = (
        llm.model_runner.get_memory_stats()
    )

    result = {
        "config": {
            "model": MODEL_PATH,
            "policy": args.policy,
            "requests": args.requests,
            "concurrency": args.concurrency,
            "token_budget": (
                args.token_budget
            ),
            "gpu_memory_utilization": (
                args.gpu_memory_utilization
            ),
            "workload_cycle": [
                "decode_heavy",
                "decode_heavy",
                "decode_heavy",
                "decode_heavy",
                "long_prefill",
                "vision_prefill",
            ],
        },

        "workload": {
            "request_counts": dict(
                Counter(
                    information[
                        "request_type"
                    ]
                    for information
                    in request_information.values()
                )
            ),
            "total_prompt_tokens": (
                total_prompt_tokens
            ),
            "total_completion_tokens": (
                total_completion_tokens
            ),
        },

        "throughput": {
            "elapsed_seconds": (
                benchmark_elapsed
            ),
            "requests_per_second": (
                args.requests
                / benchmark_elapsed
            ),
            "output_tokens_per_second": (
                total_completion_tokens
                / benchmark_elapsed
            ),
            "all_tokens_per_second": (
                (
                    total_prompt_tokens
                    + total_completion_tokens
                )
                / benchmark_elapsed
            ),
            "scheduled_prefill_tokens": (
                total_prefill_tokens
            ),
            "scheduled_decode_tokens": (
                total_decode_tokens
            ),
            "prefill_compute_tokens_per_second": (
                total_prefill_tokens
                / total_prefill_elapsed
                if total_prefill_elapsed > 0
                else 0.0
            ),
            "decode_compute_tokens_per_second": (
                total_decode_tokens
                / total_decode_elapsed
                if total_decode_elapsed > 0
                else 0.0
            ),
        },

        "latency": {
            "all_requests": {
                "queue_ms": summarize(
                    collect_metric_values(
                        request_metrics,
                        "queue_ms",
                    )
                ),
                "ttft_ms": summarize(
                    collect_metric_values(
                        request_metrics,
                        "ttft_ms",
                    )
                ),
                "request_tpot_ms": summarize(
                    collect_metric_values(
                        request_metrics,
                        "tpot_ms",
                    )
                ),
                "e2e_ms": summarize(
                    collect_metric_values(
                        request_metrics,
                        "e2e_ms",
                    )
                ),
            },

            # 所有请求的逐 token 时间间隔，
            # 样本量远大于之前的4个样本。
            "token_interval_ms": (
                summarize(
                    token_intervals_ms
                )
            ),

            "by_request_type": (
                latency_by_type
            ),
        },

        "scheduler": {
            "prefill_microbatches": (
                num_prefill_microbatches
            ),
            "decode_microbatches": (
                num_decode_microbatches
            ),
            "waiting_depth": (
                summarize(
                    [
                        float(value)
                        for value in waiting_depths
                    ]
                )
            ),
            "running_depth": (
                summarize(
                    [
                        float(value)
                        for value in running_depths
                    ]
                )
            ),
            "outstanding_depth": (
                summarize(
                    [
                        float(value)
                        for value
                        in outstanding_depths
                    ]
                )
            ),
            "preemptions": (
                llm.scheduler.num_preemptions
                - scheduler_preemptions_start
            ),
            "recomputed_tokens": (
                llm.scheduler
                .num_recomputed_tokens
                - scheduler_recomputed_start
            ),
        },

        "vision": {
            "forwards": (
                llm.model_runner
                .num_vision_forwards
                - vision_forwards_start
            ),
            "cache_hits": (
                llm.model_runner
                .num_visual_cache_hits
                - vision_hits_start
            ),
            "cache_misses": (
                llm.model_runner
                .num_visual_cache_misses
                - vision_misses_start
            ),
            "final_cache_bytes": (
                llm.model_runner
                .visual_cache_bytes
            ),
        },

        "gpu": gpu_monitor.result(),

        "memory": {
            "torch_peak_allocated_gib": (
                torch.cuda
                .max_memory_allocated()
                / 1024**3
            ),
            "torch_peak_reserved_gib": (
                torch.cuda
                .max_memory_reserved()
                / 1024**3
            ),
            "engine": {
                name: int(value)
                for name, value
                in memory_stats.items()
            },
        },

        "request_details": (
            request_details
        ),

        # 使用 request_index 而不是 seq_id，
        # 方便比较两个独立进程的生成结果。
        "completed_outputs": (
            completed_outputs
        ),
    }

    if (
        llm.model_runner.visual_cache_bytes
        != 0
    ):
        raise AssertionError(
            "Visual cache leaked after benchmark"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Benchmark completed")
    print("Policy:", args.policy)
    print(
        "Elapsed:",
        round(benchmark_elapsed, 3),
        "seconds",
    )
    print(
        "Requests/s:",
        round(
            result["throughput"][
                "requests_per_second"
            ],
            3,
        ),
    )
    print(
        "Output tokens/s:",
        round(
            result["throughput"][
                "output_tokens_per_second"
            ],
            3,
        ),
    )
    print(
        "Token interval:",
        result["latency"][
            "token_interval_ms"
        ],
    )
    print(
        "All request TTFT:",
        result["latency"][
            "all_requests"
        ]["ttft_ms"],
    )
    print(
        "GPU:",
        result["gpu"],
    )
    print(
        "Preemptions:",
        result["scheduler"][
            "preemptions"
        ],
    )
    print(
        "Recomputed tokens:",
        result["scheduler"][
            "recomputed_tokens"
        ],
    )
    print(
        "Result saved to:",
        args.output,
    )


if __name__ == "__main__":
    main()