from __future__ import annotations

import argparse
import functools
import json
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams
from nanovllm.engine.hybrid_state import (
    HybridStateManager,
)
from nanovllm.layers.gated_delta_net import (
    Qwen3_5GatedDeltaNet,
)


REPO_ROOT = Path("/workspace/nano-vllm")

MODEL_PATH = Path(
    "/workspace/models/Qwen3.5-9B"
)

OUTPUT_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "kernels"
    / "gdn_decode_profile"
)


@contextmanager
def profile_range(name: str):
    """
    同时创建两种标记：

    1. record_function：
       供 torch.profiler 识别。

    2. NVTX range：
       后续使用 Nsight Systems 时也能看到。
    """

    with torch.profiler.record_function(name):
        torch.cuda.nvtx.range_push(name)

        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()


def wrap_method(
    cls,
    method_name: str,
    range_name: str,
) -> None:
    """
    给现有类方法增加 Profile Range。

    这只是当前 Python 进程中的 monkey patch，
    不会修改项目源码文件。
    """

    original_method = getattr(
        cls,
        method_name,
    )

    @functools.wraps(original_method)
    def wrapped(*args, **kwargs):
        with profile_range(range_name):
            return original_method(
                *args,
                **kwargs,
            )

    setattr(
        cls,
        method_name,
        wrapped,
    )


def install_eager_profile_ranges() -> None:
    """
    给 Eager Decode 的关键边界添加标记。

    CUDA Graph replay 不会重新执行这些 Python
    函数，所以这些细粒度标记主要用于 Eager。
    """

    wrap_method(
        HybridStateManager,
        "read_batched_states",
        "nano::gdn_state_gather",
    )

    wrap_method(
        HybridStateManager,
        "write_batched_states",
        "nano::gdn_state_scatter",
    )

    wrap_method(
        Qwen3_5GatedDeltaNet,
        "forward",
        "nano::gdn_layer_forward",
    )

    wrap_method(
        Qwen3_5GatedDeltaNet,
        "_run_causal_conv1d",
        "nano::gdn_causal_conv",
    )

    wrap_method(
        Qwen3_5GatedDeltaNet,
        "_run_gated_delta_rule",
        "nano::gdn_delta_rule",
    )


def make_prompt(
    tokenizer,
) -> str:
    """
    构造固定 Prompt，保证不同 Profile Case
    使用完全相同的输入。
    """

    return tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": (
                    "请用一段话解释线性注意力的"
                    "基本原理和主要优点。"
                ),
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def event_to_dict(event) -> dict[str, object]:
    """
    将 PyTorch Profiler 的事件转换成 JSON。

    PyTorch 2.8 使用：
        self_device_time_total
        device_time_total

    对 CUDA 而言，这两个字段就是 CUDA 时间，
    单位是微秒。
    """

    return {
        "name": event.key,
        "calls": int(event.count),

        # 当前 Op 自身的 CPU 时间。
        "self_cpu_us": float(
            event.self_cpu_time_total
        ),

        # 包括子 Op 的 CPU 时间。
        "cpu_total_us": float(
            event.cpu_time_total
        ),

        # 当前 Op 自身关联的 CUDA Kernel 时间。
        "self_cuda_us": float(
            getattr(
                event,
                "self_device_time_total",
                0.0,
            )
        ),

        # 包括嵌套子 Op 的 CUDA 时间。
        "cuda_total_us": float(
            getattr(
                event,
                "device_time_total",
                0.0,
            )
        ),

        "self_cpu_memory_bytes": int(
            getattr(
                event,
                "self_cpu_memory_usage",
                0,
            )
        ),

        "self_cuda_memory_bytes": int(
            getattr(
                event,
                "self_device_memory_usage",
                0,
            )
        ),

        "input_shapes": str(
            getattr(
                event,
                "input_shapes",
                [],
            )
        ),
    }


def create_llm(
    *,
    mode: str,
    batch_size: int,
) -> LLM:
    """
    每个脚本进程只构造一个 Profile Case。

    不把不同 Batch 放进同一进程，避免前一个
    CUDA Graph private pool 和缓存影响后一个 Case。
    """

    return LLM(
        str(MODEL_PATH),

        # eager：
        #   不使用 CUDA Graph，便于观察细粒度 Op。
        #
        # graph：
        #   捕获当前 batch_size 对应的 Decode Graph。
        enforce_eager=(
            mode == "eager"
        ),

        tensor_parallel_size=1,

        max_model_len=512,

        max_num_batched_tokens=2048,

        max_num_seqs=batch_size,

        # 每条活跃请求需要一个独立 GDN state slot。
        num_state_slots=batch_size,

        gpu_memory_utilization=0.78,

        # 每个进程只捕获一个目标 Bucket，
        # 避免其他 Bucket 干扰显存和 Profile。
        hybrid_cuda_graph_batch_sizes=(
            batch_size,
        ),

        # 关闭 Prefix Cache，保证本实验只研究
        # 普通 Prefill + Decode。
        enable_prefix_cache=False,

        hybrid_prefix_cache_mode=(
            "disabled"
        ),
    )


def run_profile(
    *,
    mode: str,
    batch_size: int,
    external_warmup_steps: int,
    wait_steps: int,
    profiler_warmup_steps: int,
    active_steps: int,
) -> None:

    if mode == "eager":
        install_eager_profile_ranges()

    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    tokenizer = (
        AutoTokenizer.from_pretrained(
            str(MODEL_PATH)
        )
    )

    prompt = make_prompt(tokenizer)

    llm = create_llm(
        mode=mode,
        batch_size=batch_size,
    )

    total_profiler_steps = (
        wait_steps
        + profiler_warmup_steps
        + active_steps
    )

    # Prefill 会产生第一个 completion token。
    #
    # 后面还需要：
    # 1. 外部预热 Decode；
    # 2. Profiler 的 wait/warmup/active；
    # 3. 额外安全余量。
    max_tokens = (
        1
        + external_warmup_steps
        + total_profiler_steps
        + 4
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=max_tokens,
        ignore_eos=True,
    )

    for _ in range(batch_size):
        llm.add_request(
            prompt,
            sampling_params,
        )

    # 第一轮只完成 Prefill。
    _, prefill_stats = llm.step()

    if prefill_stats.num_prefill_tokens <= 0:
        raise RuntimeError(
            "The first step did not execute Prefill"
        )

    # -------------------------------
    # 模型外部预热
    # -------------------------------
    #
    # 用于排除：
    # - Triton/FLA 首次编译；
    # - CUDA lazy module loading；
    # - allocator 首次申请；
    # - cache 尚未稳定；
    # - CUDA Graph 首次 replay。
    #
    # 这些步骤完全不进入 Profiler。
    for warmup_index in range(
        external_warmup_steps
    ):
        _, stats = llm.step()

        if (
            stats.num_decode_tokens
            != batch_size
        ):
            raise RuntimeError(
                "Unexpected Decode batch during "
                f"external warmup {warmup_index}: "
                f"expected {batch_size}, "
                f"got {stats.num_decode_tokens}"
            )

    torch.cuda.synchronize()

    output_dir = (
        OUTPUT_ROOT
        / f"{mode}_b{batch_size}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trace_path = (
        output_dir
        / "trace.json"
    )

    table_path = (
        output_dir
        / "operator_table.txt"
    )

    summary_path = (
        output_dir
        / "summary.json"
    )

    captured_result: dict[
        str,
        object
    ] = {}

    def on_trace_ready(profiler) -> None:
        """
        schedule 完成 active 阶段后调用一次。
        """

        averages = list(
            profiler.key_averages(
                group_by_input_shape=True
            )
        )

        # self_device_time_total：
        # 只统计当前事件自身的 CUDA 时间。
        #
        # 查看最慢 CUDA Kernel/Op 时必须使用 self，
        # 否则父范围和子范围会重复统计。
        top_cuda_events = sorted(
            averages,
            key=lambda event: getattr(
                event,
                "self_device_time_total",
                0.0,
            ),
            reverse=True,
        )

        top_cuda_rows = [
            event_to_dict(event)
            for event in top_cuda_events[:100]
        ]

        # nano:: 开头的是我们手工添加的语义区间。
        #
        # 对这些父范围主要观察 cuda_total_us，
        # 因为真正的 Kernel 是它们的子事件。
        semantic_rows = [
            event_to_dict(event)
            for event in averages
            if event.key.startswith("nano::")
        ]

        table = (
            profiler
            .key_averages()
            .table(
                sort_by=(
                    "self_device_time_total"
                ),
                row_limit=100,
            )
        )

        profiler.export_chrome_trace(
            str(trace_path)
        )

        table_path.write_text(
            table,
            encoding="utf-8",
        )

        payload = {
            "mode": mode,
            "batch_size": batch_size,
            "external_warmup_steps": (
                external_warmup_steps
            ),
            "wait_steps": wait_steps,
            "profiler_warmup_steps": (
                profiler_warmup_steps
            ),
            "active_steps": active_steps,

            "semantic_ranges": (
                semantic_rows
            ),

            "top_cuda_events": (
                top_cuda_rows
            ),
        }

        summary_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        captured_result["table"] = table
        captured_result["semantic_ranges"] = (
            semantic_rows
        )

    schedule = torch.profiler.schedule(
        wait=wait_steps,
        warmup=profiler_warmup_steps,
        active=active_steps,
        repeat=1,
    )

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]

    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=on_trace_ready,

        # 记录输入 shape，方便区分
        # B=1、B=4、B=16 的算子。
        record_shapes=True,

        # 记录算子分配和释放的内存。
        profile_memory=True,

        # Stack 会显著增加开销，第一轮先关闭。
        with_stack=False,

        with_flops=False,
    ) as profiler:

        for profile_step_index in range(
            total_profiler_steps
        ):
            with profile_range(
                "nano::decode_step"
            ):
                _, stats = llm.step()

                if (
                    stats.num_decode_tokens
                    != batch_size
                ):
                    raise RuntimeError(
                        "Unexpected Decode batch "
                        f"at profiler step "
                        f"{profile_step_index}: "
                        f"expected {batch_size}, "
                        f"got "
                        f"{stats.num_decode_tokens}"
                    )

                # 保证当前 Profiler step 对应的
                # CUDA 工作已经完成，避免异步 Kernel
                # 跨越 schedule 边界。
                torch.cuda.synchronize()

            # 告诉 Profiler 当前逻辑 step 已结束。
            profiler.step()

    if "table" not in captured_result:
        raise RuntimeError(
            "Profiler did not produce an active trace"
        )

    print()
    print("=" * 80)
    print(
        f"Profile finished: "
        f"mode={mode}, B={batch_size}"
    )
    print("=" * 80)
    print(captured_result["table"])

    print()
    print("Semantic ranges:")

    for row in captured_result[
        "semantic_ranges"
    ]:
        calls = int(row["calls"])
        total_cuda_us = float(
            row["cuda_total_us"]
        )

        per_call_cuda_us = (
            total_cuda_us / calls
            if calls > 0
            else 0.0
        )

        print(
            f"{row['name']}: "
            f"calls={calls}, "
            f"cuda_total={total_cuda_us:.3f} us, "
            f"cuda_per_call={per_call_cuda_us:.3f} us"
        )

    print()
    print(f"Trace:   {trace_path}")
    print(f"Table:   {table_path}")
    print(f"Summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=(
            "eager",
            "graph",
        ),
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--external-warmup-steps",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--wait-steps",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--profiler-warmup-steps",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--active-steps",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error(
            "--batch-size must be positive"
        )

    return args


if __name__ == "__main__":
    run_profile(
        **vars(parse_args())
    )