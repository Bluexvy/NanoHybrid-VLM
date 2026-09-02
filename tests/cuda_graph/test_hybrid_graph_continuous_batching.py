from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


REPO_ROOT = Path(
    "/workspace/nano-vllm"
)

MODEL_PATH = (
    "/workspace/models/Qwen3.5-9B"
)

ARTIFACT_DIR = (
    REPO_ROOT
    / "artifacts"
    / "cuda_graph"
)

EAGER_PATH = (
    ARTIFACT_DIR
    / "continuous_batching_eager.pt"
)

GRAPH_PATH = (
    ARTIFACT_DIR
    / "continuous_batching_graph.pt"
)


def tensor_digest(
    tensor: torch.Tensor,
) -> str:
    """
    对 Tensor 原始字节计算 SHA256。

    digest 完全相同意味着：
    - shape 相同；
    - dtype 相同；
    - 每一个元素的二进制表示相同。
    """

    cpu_bytes = (
        tensor
        .detach()
        .contiguous()
        .view(torch.uint8)
        .cpu()
        .numpy()
        .tobytes()
    )

    return hashlib.sha256(
        cpu_bytes
    ).hexdigest()


def logical_kv_for_sequence(
    runner,
    seq,
) -> torch.Tensor:
    """
    根据 block_table 将物理 KV Block
    重新排列成逻辑 token 顺序。

    返回：
        [2, full_attention_layers,
         context_len, kv_heads, head_dim]
    """

    block_ids = tuple(
        int(block_id)
        for block_id in seq.block_table
    )

    if not block_ids:
        raise RuntimeError(
            "Sequence has no KV blocks"
        )

    block_indices = torch.tensor(
        block_ids,
        dtype=torch.long,
        device=runner.kv_cache.device,
    )

    # [2, Lfa, num_blocks, block_size, H, D]
    physical_kv = (
        runner.kv_cache.index_select(
            dim=2,
            index=block_indices,
        )
    )

    # [2, Lfa, logical_capacity, H, D]
    logical_kv = physical_kv.flatten(
        start_dim=2,
        end_dim=3,
    )

    return logical_kv[
        :,
        :,
        :len(seq),
    ]


def sequence_state_fingerprint(
    runner,
    seq,
) -> dict[str, object]:
    """
    保存一条请求最终 KV/GDN 状态的精确摘要。
    """

    manager = (
        runner.hybrid_state_manager
    )

    if manager is None:
        raise RuntimeError(
            "HybridStateManager is missing"
        )

    if seq.state_slot is None:
        raise RuntimeError(
            "Sequence has no state slot"
        )

    slot = int(seq.state_slot)

    conv_state = (
        manager.conv_state_pool[slot]
    )

    recurrent_state = (
        manager.recurrent_state_pool[slot]
    )

    logical_kv = (
        logical_kv_for_sequence(
            runner,
            seq,
        )
    )

    return {
        "context_len": len(seq),
        "conv_shape": tuple(
            conv_state.shape
        ),
        "recurrent_shape": tuple(
            recurrent_state.shape
        ),
        "kv_shape": tuple(
            logical_kv.shape
        ),
        "conv_sha256": tensor_digest(
            conv_state
        ),
        "recurrent_sha256": tensor_digest(
            recurrent_state
        ),
        "kv_sha256": tensor_digest(
            logical_kv
        ),
    }


def run_child(
    mode: str,
    output_path: Path,
) -> None:
    if mode not in {
        "eager",
        "graph",
    }:
        raise ValueError(mode)

    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_PATH
        )
    )

    raw_prompts = [
        "2加3等于多少？只回答结果。",
        "请列出三个质数，只回答结果。",
        "请用一句话解释什么是线性注意力。",
    ]

    prompts = [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in raw_prompts
    ]

    # 三条请求分别生成4、6、8个token。
    #
    # 它们不会同时结束，所以Decode batch会：
    #     B=3 → B=2 → B=1
    sampling_params = [
        SamplingParams(
            temperature=0,
            max_tokens=4,
            ignore_eos=True,
        ),
        SamplingParams(
            temperature=0,
            max_tokens=6,
            ignore_eos=True,
        ),
        SamplingParams(
            temperature=0,
            max_tokens=8,
            ignore_eos=True,
        ),
    ]

    llm = LLM(
        MODEL_PATH,
        enforce_eager=(
            mode == "eager"
        ),
        tensor_parallel_size=1,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=3,
        num_state_slots=3,

        # B=2 Workspace约198MiB。
        gpu_memory_utilization=0.78,

        # 捕获Graph 1和Graph 2。
        hybrid_cuda_graph_batch_sizes=(
            1,
            2,
        ),
    )

    runner = llm.model_runner

    original_decode = (
        runner.run_hybrid_decode
    )

    route_trace: list[
        dict[str, object]
    ] = []

    final_fingerprints: dict[
        int,
        dict[str, object],
    ] = {}

    def traced_decode(seqs):
        replays_before = (
            runner.num_hybrid_graph_replays
        )

        fallbacks_before = (
            runner
            .num_hybrid_graph_eager_fallbacks
        )

        token_ids = original_decode(
            seqs
        )

        replays_after = (
            runner.num_hybrid_graph_replays
        )

        fallbacks_after = (
            runner
            .num_hybrid_graph_eager_fallbacks
        )

        if replays_after > replays_before:
            execution_path = "graph"

        elif fallbacks_after > fallbacks_before:
            execution_path = "eager"

        else:
            raise RuntimeError(
                "Decode did not record Graph "
                "or Eager execution"
            )

        route_trace.append(
            {
                "batch_size": len(seqs),
                "execution_path": (
                    execution_path
                ),
                "seq_ids": tuple(
                    int(seq.seq_id)
                    for seq in seqs
                ),
                "completion_tokens_before": (
                    tuple(
                        seq.num_completion_tokens
                        for seq in seqs
                    )
                ),
            }
        )

        # run_hybrid_decode() 返回后，
        # Scheduler 还没有 append 新生成的 token。
        #
        # 因此 +1 后达到 max_tokens 的请求
        # 将在本轮完成。此时保存它的最终运行时状态。
        for seq in seqs:
            will_finish = (
                seq.num_completion_tokens + 1
                == seq.max_tokens
            )

            if will_finish:
                final_fingerprints[
                    int(seq.seq_id)
                ] = (
                    sequence_state_fingerprint(
                        runner,
                        seq,
                    )
                )

        return token_ids

    runner.run_hybrid_decode = (
        traced_decode
    )

    outputs = llm.generate(
        prompts,
        sampling_params,
    )

    if len(final_fingerprints) != 3:
        raise AssertionError(
            "Did not capture all three "
            "final request states"
        )

    scheduler = llm.scheduler

    state_allocator = (
        scheduler.state_slot_allocator
    )

    if state_allocator is None:
        raise RuntimeError(
            "StateSlotAllocator is missing"
        )

    payload = {
        "mode": mode,
        "token_ids": [
            output["token_ids"]
            for output in outputs
        ],
        "route_trace": route_trace,
        "final_fingerprints": (
            final_fingerprints
        ),
        "graph_replays": (
            runner.num_hybrid_graph_replays
        ),
        "eager_fallbacks": (
            runner
            .num_hybrid_graph_eager_fallbacks
        ),
        "fallback_reasons": dict(
            runner
            .hybrid_graph_fallback_reasons
        ),
        "captured_buckets": tuple(
            sorted(
                runner.hybrid_graphs.keys()
            )
        ),
        # 请求结束后的资源回收状态。
        "running_requests": len(
            scheduler.running
        ),
        "waiting_requests": len(
            scheduler.waiting
        ),
        "used_state_slots": (
            state_allocator.num_used_slots
        ),
        "used_kv_blocks": len(
            scheduler
            .block_manager
            .used_block_ids
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
        f"\n{mode} route trace:"
    )

    for item in route_trace:
        print(item)

    print(
        "graph replays:",
        payload["graph_replays"],
    )

    print(
        "eager fallbacks:",
        payload["eager_fallbacks"],
    )

    print(
        "used state slots after finish:",
        payload["used_state_slots"],
    )

    print(
        "used KV blocks after finish:",
        payload["used_kv_blocks"],
    )


def compare_results() -> None:
    eager = torch.load(
        EAGER_PATH,
        map_location="cpu",
        weights_only=False,
    )

    graph = torch.load(
        GRAPH_PATH,
        map_location="cpu",
        weights_only=False,
    )

    print(
        "\n"
        + "=" * 72
    )
    print(
        "Comparing Continuous Batching"
    )
    print(
        "=" * 72
    )

    # 三条请求的最终生成必须完全一致。
    if (
        eager["token_ids"]
        != graph["token_ids"]
    ):
        raise AssertionError(
            "Generated token IDs differ"
        )

    print(
        "All request token IDs: exact"
    )

    # 每条请求最终状态的二进制摘要必须一致。
    if (
        eager["final_fingerprints"]
        != graph["final_fingerprints"]
    ):
        raise AssertionError(
            "Final per-request KV/GDN "
            "fingerprints differ"
        )

    print(
        "All per-request KV/GDN states: exact"
    )

    eager_batches = [
        item["batch_size"]
        for item in eager["route_trace"]
    ]

    graph_batches = [
        item["batch_size"]
        for item in graph["route_trace"]
    ]

    expected_batches = [
        3,
        3,
        3,
        2,
        2,
        1,
        1,
    ]

    if eager_batches != expected_batches:
        raise AssertionError(
            "Unexpected Eager batch transition: "
            f"{eager_batches}"
        )

    if graph_batches != expected_batches:
        raise AssertionError(
            "Unexpected Graph batch transition: "
            f"{graph_batches}"
        )

    graph_paths = [
        item["execution_path"]
        for item in graph["route_trace"]
    ]

    expected_graph_paths = [
        "eager",
        "eager",
        "eager",
        "graph",
        "graph",
        "graph",
        "graph",
    ]

    if graph_paths != expected_graph_paths:
        raise AssertionError(
            "Unexpected execution paths: "
            f"{graph_paths}"
        )

    if graph["graph_replays"] != 4:
        raise AssertionError(
            "Expected four Graph replays"
        )

    if graph["eager_fallbacks"] != 3:
        raise AssertionError(
            "Expected three Eager fallbacks"
        )

    # 三条请求结束后，运行时资源必须全部释放。
    for result in (
        eager,
        graph,
    ):
        if result["running_requests"] != 0:
            raise AssertionError(
                "Running queue leaked requests"
            )

        if result["waiting_requests"] != 0:
            raise AssertionError(
                "Waiting queue leaked requests"
            )

        if result["used_state_slots"] != 0:
            raise AssertionError(
                "GDN state slots were not released"
            )

        if result["used_kv_blocks"] != 0:
            raise AssertionError(
                "KV blocks were not released"
            )

    print(
        "Batch transition:",
        expected_batches,
    )

    print(
        "Graph execution paths:",
        expected_graph_paths,
    )

    print(
        "Resource cleanup: passed"
    )

    print(
        "\nPart 5B passed: Continuous Batching "
        "safely switched between Eager, Graph 2 "
        "and Graph 1 without mixing per-request "
        "KV or GDN states."
    )


def run_all() -> None:
    script = Path(
        __file__
    ).resolve()

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for mode, output_path in [
        ("eager", EAGER_PATH),
        ("graph", GRAPH_PATH),
    ]:
        command = [
            sys.executable,
            str(script),
            "--mode",
            mode,
            "--output",
            str(output_path),
        ]

        print(
            "\nRunning:",
            " ".join(command),
        )

        subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
        )

    compare_results()


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
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "all":
        run_all()
        return

    if args.output is None:
        raise ValueError(
            "--output is required "
            "for child mode"
        )

    run_child(
        args.mode,
        args.output,
    )


if __name__ == "__main__":
    main()