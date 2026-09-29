from __future__ import annotations

import argparse
import functools
import statistics
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams
from nanovllm.engine.hybrid_state import HybridStateManager
import nanovllm.layers.gated_delta_net as gdn_module
from nanovllm.layers.gated_delta_net import Qwen3_5GatedDeltaNet


TRACE_ENABLED = False
EVENT_ENABLED = False

EVENT_PAIRS: dict[
    str,
    list[tuple[torch.cuda.Event, torch.cuda.Event]],
] = defaultdict(list)


@contextmanager
def instrument_range(name: str):
    """
    Trace 模式：
        生成 PyTorch Profiler 和 NVTX 标签。

    Event 模式：
        使用 CUDA Event 记录这个区间提交的 GPU 工作。
    """

    start_event = None
    stop_event = None

    if EVENT_ENABLED:
        start_event = torch.cuda.Event(
            enable_timing=True
        )
        stop_event = torch.cuda.Event(
            enable_timing=True
        )
        start_event.record()

    try:
        if TRACE_ENABLED:
            with torch.profiler.record_function(name):
                torch.cuda.nvtx.range_push(name)
                try:
                    yield
                finally:
                    torch.cuda.nvtx.range_pop()
        else:
            yield
    finally:
        if EVENT_ENABLED:
            assert start_event is not None
            assert stop_event is not None

            stop_event.record()

            EVENT_PAIRS[name].append(
                (start_event, stop_event)
            )


def patch_method(
    cls,
    method_name: str,
    label: str,
) -> None:
    """
    在当前Python进程中给方法增加计时标签。

    这是monkey patch，不会修改项目源码。
    """

    if not hasattr(cls, method_name):
        return

    original = getattr(cls, method_name)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        with instrument_range(label):
            return original(*args, **kwargs)

    setattr(cls, method_name, wrapped)


def patch_module_function(
    module,
    function_name: str,
    label: str,
) -> None:
    """
    对模块里的Python函数增加计时标签。
    """

    if not hasattr(module, function_name):
        return

    original = getattr(module, function_name)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        with instrument_range(label):
            return original(*args, **kwargs)

    setattr(module, function_name, wrapped)


def install_instrumentation() -> None:
    # FLA路径：
    # 一次Gather/Scatter覆盖所有GDN层。
    patch_method(
        HybridStateManager,
        "read_batched_states",
        "nano::gdn_state_gather",
    )

    patch_method(
        HybridStateManager,
        "write_batched_states",
        "nano::gdn_state_scatter",
    )

    # Triton路径只Gather/Scatter Conv State。
    patch_method(
        HybridStateManager,
        "read_batched_conv_states",
        "nano::conv_state_gather",
    )

    patch_method(
        HybridStateManager,
        "write_batched_conv_states",
        "nano::conv_state_scatter",
    )

    # 每个GDN层的整体Forward。
    patch_method(
        Qwen3_5GatedDeltaNet,
        "forward",
        "nano::gdn_layer_forward",
    )

    # 每个GDN层的Conv部分。
    patch_method(
        Qwen3_5GatedDeltaNet,
        "_run_causal_conv1d",
        "nano::gdn_conv_method",
    )

    # FLA Delta Rule路径。
    patch_method(
        Qwen3_5GatedDeltaNet,
        "_run_gated_delta_rule",
        "nano::fla_delta_rule_method",
    )

    # 自研CUDA Conv Kernel调用。
    patch_module_function(
        gdn_module,
        "state_aware_causal_conv1d_cuda",
        "nano::state_aware_conv_cuda",
    )

    # 自研CUDA Recurrent Kernel调用。
    patch_module_function(
        gdn_module,
        "state_aware_gdn_decode_cuda",
        "nano::state_aware_recurrent_cuda",
    )


def make_llm(args) -> LLM:
    return LLM(
        args.model,

        enforce_eager=(
            args.mode == "eager"
        ),

        tensor_parallel_size=1,

        gdn_decode_backend=args.backend,

        max_model_len=512,
        max_num_batched_tokens=2048,
        max_num_seqs=args.batch_size,
        num_state_slots=args.batch_size,

        gpu_memory_utilization=0.78,

        hybrid_cuda_graph_batch_sizes=(
            args.batch_size,
        ),

        enable_prefix_cache=False,
        hybrid_prefix_cache_mode="disabled",
    )


def make_prompt(tokenizer, index: int = 0) -> str:
    return tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": (
                    "请用一段话解释线性注意力的"
                    f"基本原理和主要优点。请求编号{index}"
                ),
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def prepare_decode_requests(
    llm: LLM,
    tokenizer,
    args,
    required_decode_steps: int,
) -> None:
    """
    加入固定请求并完成Prefill和外部预热，
    最终让所有请求稳定处于Decode阶段。
    """

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=(
            1
            + args.warmup
            + required_decode_steps
            + 8
        ),
        ignore_eos=True,
    )

    for index in range(args.batch_size):
        llm.add_request(
            make_prompt(tokenizer, index),
            sampling_params,
        )

    # 第一轮应该执行Prefill。
    _, stats = llm.step()

    if stats.num_prefill_tokens <= 0:
        raise RuntimeError(
            "第一轮没有执行Prefill"
        )

    print(
        "Prefill完成：",
        f"{stats.num_prefill_tokens} tokens",
    )

    # 排除首次编译、Allocator、Lazy Loading和Graph Replay。
    for step in range(args.warmup):
        _, stats = llm.step()

        if (
            stats.num_decode_tokens
            != args.batch_size
        ):
            raise RuntimeError(
                f"预热第{step}轮Decode Batch异常："
                f"{stats.num_decode_tokens}"
            )

    torch.cuda.synchronize()

    print(
        f"完成{args.warmup}轮Decode预热"
    )


def run_trace(
    llm: LLM,
    args,
    output_dir: Path,
) -> None:
    """
    PyTorch Profiler负责看：
        CPU调度
        CUDA Kernel
        调用关系
        Timeline
    """

    global TRACE_ENABLED

    trace_path = output_dir / (
        f"trace_{args.mode}_{args.backend}"
        f"_b{args.batch_size}.json"
    )

    table_path = output_dir / (
        f"table_{args.mode}_{args.backend}"
        f"_b{args.batch_size}.txt"
    )

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]

    TRACE_ENABLED = True

    with torch.profiler.profile(
        activities=activities,

        # 第一轮先不要打开Stack和Memory，
        # 否则Trace会非常大。
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
    ) as profiler:

        for step in range(args.steps):
            with instrument_range(
                "nano::decode_step"
            ):
                _, stats = llm.step()

            if (
                stats.num_decode_tokens
                != args.batch_size
            ):
                raise RuntimeError(
                    f"Trace第{step}轮Decode Batch异常"
                )

        torch.cuda.synchronize()

    TRACE_ENABLED = False

    profiler.export_chrome_trace(
        str(trace_path)
    )

    table = (
        profiler
        .key_averages()
        .table(
            sort_by="self_device_time_total",
            row_limit=80,
        )
    )

    table_path.write_text(
        table,
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("Top CUDA operators")
    print("=" * 80)
    print(table)

    print()
    print("语义区间：")

    for event in profiler.key_averages():
        if not event.key.startswith("nano::"):
            continue

        calls = int(event.count)
        device_total_us = float(
            getattr(
                event,
                "device_time_total",
                0.0,
            )
        )

        per_call_us = (
            device_total_us / calls
            if calls > 0
            else 0.0
        )

        print(
            f"{event.key:<38}"
            f"calls={calls:<5}"
            f"total={device_total_us:>12.3f} us  "
            f"per_call={per_call_us:>10.3f} us"
        )

    print()
    print("Trace:", trace_path)
    print("Table:", table_path)


def run_event_measurement(
    llm: LLM,
    args,
) -> None:
    """
    CUDA Event负责回答：
        Gather每个Decode Step花多久？
        FLA部分花多久？
        Scatter花多久？
        整个Decode Step的GPU时间是多少？
    """

    global EVENT_ENABLED

    EVENT_PAIRS.clear()
    EVENT_ENABLED = True

    for step in range(args.steps):
        with instrument_range(
            "nano::decode_step"
        ):
            _, stats = llm.step()

        if (
            stats.num_decode_tokens
            != args.batch_size
        ):
            raise RuntimeError(
                f"Event第{step}轮Decode Batch异常"
            )

    torch.cuda.synchronize()
    EVENT_ENABLED = False

    print()
    print("=" * 100)
    print(
        f"CUDA Event结果："
        f"mode={args.mode}, "
        f"backend={args.backend}, "
        f"B={args.batch_size}"
    )
    print("=" * 100)

    for name, pairs in EVENT_PAIRS.items():
        elapsed_us = [
            start.elapsed_time(stop) * 1000.0
            for start, stop in pairs
        ]

        total_us = sum(elapsed_us)

        calls_per_step = (
            len(elapsed_us) / args.steps
        )

        gpu_us_per_step = (
            total_us / args.steps
        )

        median_call_us = statistics.median(
            elapsed_us
        )

        print(
            f"{name:<38}"
            f"calls/step={calls_per_step:>6.2f}  "
            f"GPU/step={gpu_us_per_step:>11.3f} us  "
            f"median/call={median_call_us:>10.3f} us"
        )


def run_clean_benchmark(
    llm: LLM,
    args,
) -> None:
    """
    不对内部函数插桩。

    GPU时间：
        CUDA Event测得的GPU执行时间。

    Wall时间：
        CPU调度、同步和GPU执行共同形成的真实步延迟。
    """

    gpu_measurements = []
    wall_measurements = []

    for repeat in range(args.repeats):
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(
            enable_timing=True
        )
        stop_event = torch.cuda.Event(
            enable_timing=True
        )

        wall_start = time.perf_counter()
        start_event.record()

        for step in range(args.steps):
            _, stats = llm.step()

            if (
                stats.num_decode_tokens
                != args.batch_size
            ):
                raise RuntimeError(
                    "Benchmark Decode Batch异常"
                )

        stop_event.record()
        stop_event.synchronize()
        wall_stop = time.perf_counter()

        gpu_ms_per_step = (
            start_event.elapsed_time(stop_event)
            / args.steps
        )

        wall_ms_per_step = (
            (wall_stop - wall_start)
            * 1000.0
            / args.steps
        )

        gpu_measurements.append(
            gpu_ms_per_step
        )

        wall_measurements.append(
            wall_ms_per_step
        )

        print(
            f"repeat={repeat}: "
            f"GPU={gpu_ms_per_step:.4f} ms/step, "
            f"Wall={wall_ms_per_step:.4f} ms/step"
        )

    print()
    print("=" * 80)
    print(
        f"Clean Benchmark："
        f"mode={args.mode}, "
        f"backend={args.backend}, "
        f"B={args.batch_size}"
    )
    print("=" * 80)

    print(
        "Median GPU step:",
        f"{statistics.median(gpu_measurements):.4f} ms",
    )

    print(
        "Median Wall step:",
        f"{statistics.median(wall_measurements):.4f} ms",
    )

    median_gpu_ms = statistics.median(
        gpu_measurements
    )

    decode_tokens_per_second = (
        args.batch_size
        / (median_gpu_ms / 1000.0)
    )

    print(
        "Decode throughput:",
        f"{decode_tokens_per_second:.2f} tok/s",
    )


def run_e2e(
    llm: LLM,
    tokenizer,
    args,
) -> None:
    """
    从add_request开始，测到所有请求完成。

    包含：
        Processor
        排队
        Prefill
        Decode
        Sampling
        Scheduler
    """

    # 先用同样的Batch预热一次完整请求。
    warmup_params = SamplingParams(
        temperature=0,
        max_tokens=4,
        ignore_eos=True,
    )

    for index in range(args.batch_size):
        llm.add_request(
            make_prompt(tokenizer, 1000 + index),
            warmup_params,
        )

    while not llm.is_finished():
        llm.step()

    torch.cuda.synchronize()

    # 正式测量。
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.output_tokens,
        ignore_eos=True,
    )

    measured_seq_ids = []

    torch.cuda.synchronize()
    wall_start = time.perf_counter()

    for index in range(args.batch_size):
        seq_id = llm.add_request(
            make_prompt(tokenizer, index),
            sampling_params,
        )
        measured_seq_ids.append(seq_id)

    while not llm.is_finished():
        llm.step()

    torch.cuda.synchronize()
    wall_end = time.perf_counter()

    measured_id_set = set(measured_seq_ids)

    metrics = [
        item
        for item in llm.get_completed_request_metrics()
        if item.seq_id in measured_id_set
    ]

    print()
    print("=" * 90)
    print(
        f"端到端结果："
        f"mode={args.mode}, "
        f"backend={args.backend}, "
        f"B={args.batch_size}"
    )
    print("=" * 90)

    for item in metrics:
        print(
            f"seq={item.seq_id:<4}"
            f"prompt={item.num_prompt_tokens:<5}"
            f"output={item.num_completion_tokens:<5}"
            f"preprocess={item.preprocessing_ms:>8.3f} ms  "
            f"queue={item.queue_ms:>8.3f} ms  "
            f"TTFT={item.ttft_ms:>9.3f} ms  "
            f"TPOT={item.tpot_ms:>9.3f} ms  "
            f"E2E={item.e2e_ms:>10.3f} ms"
        )

    wall_seconds = wall_end - wall_start

    total_output_tokens = sum(
        item.num_completion_tokens
        for item in metrics
    )

    print()
    print(
        "Batch wall time:",
        f"{wall_seconds * 1000.0:.3f} ms",
    )

    print(
        "Output throughput:",
        f"{total_output_tokens / wall_seconds:.2f} tok/s",
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="/workspace/models/Qwen3.5-9B",
    )

    parser.add_argument(
        "--phase",
        choices=(
            "trace",
            "event",
            "benchmark",
            "e2e",
        ),
        required=True,
    )

    parser.add_argument(
        "--backend",
        choices=(
            "fla",
            "state_aware_triton",
            "state_aware_cuda",
        ),
        required=True,
    )

    parser.add_argument(
        "--mode",
        choices=("eager", "graph"),
        default="eager",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--output-tokens",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "/workspace/nano-vllm/artifacts/"
            "profile_walkthrough"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch-size必须大于0")

    if args.steps <= 0:
        raise ValueError("steps必须大于0")

    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    install_instrumentation()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model
    )

    llm = make_llm(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.phase == "e2e":
        run_e2e(
            llm,
            tokenizer,
            args,
        )
        return

    if args.phase == "benchmark":
        required_decode_steps = (
            args.steps * args.repeats
        )
    else:
        required_decode_steps = args.steps

    prepare_decode_requests(
        llm,
        tokenizer,
        args,
        required_decode_steps,
    )

    if args.phase == "trace":
        run_trace(
            llm,
            args,
            output_dir,
        )

    elif args.phase == "event":
        run_event_measurement(
            llm,
            args,
        )

    elif args.phase == "benchmark":
        run_clean_benchmark(
            llm,
            args,
        )


if __name__ == "__main__":
    main()