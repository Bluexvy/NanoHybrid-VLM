from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


MODEL_PATH = (
    "/workspace/models/Qwen3.5-9B"
)


def main() -> None:
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

    llm = LLM(
        MODEL_PATH,

        # 关闭强制 Eager，启用 Graph。
        enforce_eager=False,

        tensor_parallel_size=1,

        # 首次测试只允许 B=1。
        max_num_seqs=1,
        num_state_slots=1,

        max_model_len=512,
        max_num_batched_tokens=512,

        # 给首次 Graph capture 留出较宽松显存。
        gpu_memory_utilization=0.80,

        hybrid_cuda_graph_batch_sizes=(
            1,
        ),
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=8,
    )

    outputs = llm.generate(
        [prompt],
        sampling_params,
    )

    runner = llm.model_runner
    workspace = (
        runner.hybrid_graph_workspace
    )

    if workspace is None:
        raise AssertionError(
            "Hybrid Graph Workspace was not created"
        )

    if (
        runner.num_hybrid_graph_replays
        <= 0
    ):
        raise AssertionError(
            "Hybrid CUDA Graph was never replayed"
        )

    print("\nGenerated text:")
    print(outputs[0]["text"])

    print("\nCUDA Graph statistics:")
    print(
        "captured batch sizes:",
        tuple(
            sorted(
                runner.hybrid_graphs.keys()
            )
        ),
    )
    print(
        "graph replays:",
        runner.num_hybrid_graph_replays,
    )
    print(
        "eager fallbacks:",
        (
            runner
            .num_hybrid_graph_eager_fallbacks
        ),
    )
    print(
        "fallback reasons:",
        runner.hybrid_graph_fallback_reasons,
    )
    print(
        "workspace MiB:",
        workspace.allocated_bytes / 1024**2,
    )
    print(
        "capture allocated delta MiB:",
        (
            runner
            .hybrid_graph_capture_allocated_bytes
            / 1024**2
        ),
    )

    print(
        "\nPart 4 smoke passed: "
        "Qwen3.5 Hybrid Decode captured and "
        "replayed a B=1 CUDA Graph."
    )


if __name__ == "__main__":
    main()