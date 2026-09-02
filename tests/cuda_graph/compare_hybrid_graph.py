from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


REPO_ROOT = Path(
    "/workspace/nano-vllm"
)

MODEL_PATH = Path(
    "/workspace/models/Qwen3.5-9B"
)

ARTIFACT_DIR = (
    REPO_ROOT
    / "artifacts"
    / "cuda_graph"
)

EAGER_DUMP_PATH = (
    ARTIFACT_DIR
    / "hybrid_eager_dump.pt"
)

GRAPH_DUMP_PATH = (
    ARTIFACT_DIR
    / "hybrid_graph_dump.pt"
)


def snapshot_logical_kv(
    runner,
    seq,
) -> torch.Tensor:
    """
    将请求分散在物理 Paged KV Blocks 中的 K/V，
    按逻辑 token 顺序重新拼接。

    原始 KV Cache：
        [2, Lfa, Nblocks, block_size, Hkv, D]

    返回：
        [2, Lfa, context_len, Hkv, D]
    """

    block_ids = tuple(
        int(block_id)
        for block_id in seq.block_table
    )

    if not block_ids:
        raise RuntimeError(
            "Sequence does not own KV blocks"
        )

    block_index = torch.tensor(
        block_ids,
        dtype=torch.long,
        device=runner.kv_cache.device,
    )

    # 选择这条请求实际使用的物理块。
    #
    # [2, Lfa, Nblocks, block_size, Hkv, D]
    #                  ↓ index_select
    # [2, Lfa, request_blocks, block_size, Hkv, D]
    physical_blocks = (
        runner.kv_cache.index_select(
            dim=2,
            index=block_index,
        )
    )

    # 将：
    #     request_blocks × block_size
    #
    # 展平为逻辑 token 轴。
    #
    # [2, Lfa, request_blocks, block_size, Hkv, D]
    #                         ↓
    # [2, Lfa, request_blocks * block_size, Hkv, D]
    logical_kv = physical_blocks.flatten(
        start_dim=2,
        end_dim=3,
    )

    context_len = len(seq)

    # 最后一个物理块通常没有填满，
    # 因此只保留真实上下文对应的 token。
    logical_kv = logical_kv[
        :,
        :,
        :context_len,
    ]

    return (
        logical_kv
        .detach()
        .cpu()
        .clone()
    )


def snapshot_final_runtime_state(
    runner,
    seq,
) -> dict[str, object]:
    """
    保存请求最后一次 Decode 后的完整运行时状态。
    """

    state_manager = (
        runner.hybrid_state_manager
    )

    if state_manager is None:
        raise RuntimeError(
            "HybridStateManager is missing"
        )

    if seq.state_slot is None:
        raise RuntimeError(
            "Sequence does not own a state slot"
        )

    state_slot = int(
        seq.state_slot
    )

    if not (
        state_manager
        .is_slot_initialized(state_slot)
    ):
        raise RuntimeError(
            "Sequence state slot is not initialized"
        )

    # active conv state：
    #
    # [num_gdn_layers, conv_dim, kernel_size]
    conv_state = (
        state_manager
        .conv_state_pool[state_slot]
        .detach()
        .cpu()
        .clone()
    )

    # active recurrent state：
    #
    # [num_gdn_layers, value_heads, Dk, Dv]
    recurrent_state = (
        state_manager
        .recurrent_state_pool[state_slot]
        .detach()
        .cpu()
        .clone()
    )

    logical_kv = snapshot_logical_kv(
        runner,
        seq,
    )

    return {
        "state_slot": state_slot,
        "context_len": len(seq),
        "block_table": tuple(
            int(block_id)
            for block_id in seq.block_table
        ),
        "conv_state": conv_state,
        "recurrent_state": recurrent_state,
        "logical_kv": logical_kv,
    }


def run_child(
    mode: str,
    output_path: Path,
) -> None:
    """
    子进程执行单个模式：

        mode=eager
        mode=graph
    """

    if mode not in {
        "eager",
        "graph",
    }:
        raise ValueError(
            f"Unsupported mode: {mode}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available"
        )

    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_PATH
        )
    )

    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": (
                    "请只用一句话解释线性注意力。"
                ),
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    enforce_eager = (
        mode == "eager"
    )

    print(
        f"\nLoading {mode} runtime..."
    )

    llm = LLM(
        str(MODEL_PATH),
        enforce_eager=enforce_eager,
        tensor_parallel_size=1,
        max_num_seqs=1,
        num_state_slots=1,
        max_model_len=512,
        max_num_batched_tokens=512,
        gpu_memory_utilization=0.80,
        hybrid_cuda_graph_batch_sizes=(
            1,
        ),
    )

    runner = llm.model_runner

    # 保存原始 bound method。
    original_run_hybrid_decode = (
        runner.run_hybrid_decode
    )

    # 最后一次 Decode 的状态会保存到这里。
    final_runtime_state: dict[
        str,
        object,
    ] = {}

    def traced_run_hybrid_decode(
        seqs,
    ):
        """
        包装 run_hybrid_decode()。

        原模型执行完成后、Engine append 新 token 前，
        请求仍持有 state slot 和 KV blocks，
        因此可以在这里保存运行时状态。
        """

        token_ids = (
            original_run_hybrid_decode(
                seqs
            )
        )

        if len(seqs) != 1:
            raise RuntimeError(
                "This comparison expects B=1"
            )

        seq = seqs[0]

        # 当前 token_ids 还没有被 Engine append。
        #
        # 如果当前已有7个completion token，
        # 本轮返回的是第8个，也就是最后一轮Decode。
        is_final_decode = (
            seq.num_completion_tokens + 1
            == seq.max_tokens
        )

        if is_final_decode:
            final_runtime_state.clear()

            final_runtime_state.update(
                snapshot_final_runtime_state(
                    runner,
                    seq,
                )
            )

        return token_ids

    # 实例属性优先于类方法，因此 ModelRunner.run_hybrid()
    # 会调用这个包装后的函数。
    runner.run_hybrid_decode = (
        traced_run_hybrid_decode
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=8,

        # 保证一定执行到8个生成token，
        # 这样可以准确识别最后一次Decode。
        ignore_eos=True,
    )

    outputs = llm.generate(
        [prompt],
        sampling_params,
    )

    if not final_runtime_state:
        raise RuntimeError(
            "The final Decode state was not captured"
        )

    output_token_ids = outputs[0][
        "token_ids"
    ]

    payload = {
        "mode": mode,
        "prompt": prompt,
        "token_ids": torch.tensor(
            output_token_ids,
            dtype=torch.long,
        ),
        "context_len": (
            final_runtime_state[
                "context_len"
            ]
        ),
        "block_table": (
            final_runtime_state[
                "block_table"
            ]
        ),
        "conv_state": (
            final_runtime_state[
                "conv_state"
            ]
        ),
        "recurrent_state": (
            final_runtime_state[
                "recurrent_state"
            ]
        ),
        "logical_kv": (
            final_runtime_state[
                "logical_kv"
            ]
        ),
        "graph_replays": (
            runner.num_hybrid_graph_replays
        ),
        "eager_fallbacks": (
            runner
            .num_hybrid_graph_eager_fallbacks
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        payload,
        output_path,
    )

    print(
        f"\n{mode} result saved to:",
        output_path,
    )

    print(
        "generated token IDs:",
        output_token_ids,
    )

    print(
        "context length:",
        payload["context_len"],
    )

    print(
        "conv state shape:",
        tuple(
            payload[
                "conv_state"
            ].shape
        ),
    )

    print(
        "recurrent state shape:",
        tuple(
            payload[
                "recurrent_state"
            ].shape
        ),
    )

    print(
        "logical KV shape:",
        tuple(
            payload[
                "logical_kv"
            ].shape
        ),
    )

    print(
        "graph replays:",
        payload["graph_replays"],
    )


def compare_tensor_exact(
    name: str,
    eager_tensor: torch.Tensor,
    graph_tensor: torch.Tensor,
) -> None:
    """
    比较两个 Tensor，并打印误差统计。
    """

    if (
        eager_tensor.shape
        != graph_tensor.shape
    ):
        raise AssertionError(
            f"{name} shape mismatch: "
            f"{tuple(eager_tensor.shape)} vs "
            f"{tuple(graph_tensor.shape)}"
        )

    if (
        eager_tensor.dtype
        != graph_tensor.dtype
    ):
        raise AssertionError(
            f"{name} dtype mismatch: "
            f"{eager_tensor.dtype} vs "
            f"{graph_tensor.dtype}"
        )

    if eager_tensor.numel() == 0:
        max_error = 0.0
        mean_error = 0.0
    else:
        error = (
            eager_tensor.float()
            - graph_tensor.float()
        ).abs()

        max_error = (
            error.max().item()
        )

        mean_error = (
            error.mean().item()
        )

    exact = torch.equal(
        eager_tensor,
        graph_tensor,
    )

    print(
        f"{name}: "
        f"exact={exact}, "
        f"max_error={max_error:.8g}, "
        f"mean_error={mean_error:.8g}"
    )

    if not exact:
        raise AssertionError(
            f"{name} is not exactly equal"
        )


def compare_dumps() -> None:
    eager = torch.load(
        EAGER_DUMP_PATH,
        map_location="cpu",
        weights_only=False,
    )

    graph = torch.load(
        GRAPH_DUMP_PATH,
        map_location="cpu",
        weights_only=False,
    )

    print(
        "\n"
        + "=" * 72
    )
    print(
        "Comparing Eager and CUDA Graph"
    )
    print(
        "=" * 72
    )

    if (
        eager["context_len"]
        != graph["context_len"]
    ):
        raise AssertionError(
            "Context lengths differ"
        )

    compare_tensor_exact(
        "generated token IDs",
        eager["token_ids"],
        graph["token_ids"],
    )

    compare_tensor_exact(
        "GDN conv_state",
        eager["conv_state"],
        graph["conv_state"],
    )

    compare_tensor_exact(
        "GDN recurrent_state",
        eager["recurrent_state"],
        graph["recurrent_state"],
    )

    compare_tensor_exact(
        "Full Attention logical KV",
        eager["logical_kv"],
        graph["logical_kv"],
    )

    if graph["graph_replays"] <= 0:
        raise AssertionError(
            "Graph process did not replay "
            "a CUDA Graph"
        )

    if eager["graph_replays"] != 0:
        raise AssertionError(
            "Eager process unexpectedly "
            "replayed a CUDA Graph"
        )

    print(
        "\nPart 5A passed:"
    )

    print(
        "- generated token IDs are exact"
    )
    print(
        "- all GDN conv states are exact"
    )
    print(
        "- all FP32 recurrent states are exact"
    )
    print(
        "- all Full Attention logical KV "
        "values are exact"
    )


def run_all() -> None:
    script_path = Path(
        __file__
    ).resolve()

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    commands = [
        (
            "eager",
            EAGER_DUMP_PATH,
        ),
        (
            "graph",
            GRAPH_DUMP_PATH,
        ),
    ]

    for mode, output_path in commands:
        command = [
            sys.executable,
            str(script_path),
            "--mode",
            mode,
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
            "=" * 72
        )

        subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
        )

    compare_dumps()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "all",
            "eager",
            "graph",
        ],
        default="all",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "all":
        run_all()
        return

    if args.output is None:
        raise ValueError(
            "--output is required for child mode"
        )

    run_child(
        args.mode,
        args.output,
    )


if __name__ == "__main__":
    main()