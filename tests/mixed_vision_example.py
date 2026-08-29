from PIL import Image, ImageDraw

from nanovllm import LLM, SamplingParams


MODEL_PATH = "/workspace/models/Qwen3.5-9B"


def create_test_image() -> Image.Image:
    image = Image.new(
        mode="RGB",
        size=(320, 320),
        color="white",
    )

    draw = ImageDraw.Draw(image)

    draw.rectangle(
        xy=(40, 100, 140, 200),
        fill="red",
    )

    draw.ellipse(
        xy=(180, 100, 280, 200),
        fill="blue",
    )

    return image


def format_text_prompt(
    llm: LLM,
    text: str,
) -> str:
    """
    为纯文本请求套用模型的 chat template。

    图文请求不调用这个函数，因为 InputProcessor
    会使用 AutoProcessor 自动构造图文模板。
    """

    return llm.tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": text,
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> None:
    llm = LLM(
        MODEL_PATH,
        enforce_eager=True,
        tensor_parallel_size=1,

        max_model_len=1024,

        # 三条 prompt 总长度应小于512，
        # 使它们能够进入同一个 Prefill microbatch。
        max_num_batched_tokens=512,

        max_num_seqs=3,
        num_state_slots=3,

        gpu_memory_utilization=0.8,
    )

    image = create_test_image()

    text_prompt_1 = format_text_prompt(
        llm,
        "2加3等于多少？只回答结果。",
    )

    text_prompt_2 = format_text_prompt(
        llm,
        "用一句话解释什么是线性注意力。",
    )

    prompts = [
        text_prompt_1,

        {
            "prompt": (
                "请描述图片中的颜色和形状。"
            ),
            "multi_modal_data": {
                "image": image,
            },
        },

        text_prompt_2,
    ]

    sampling_params = SamplingParams(
        temperature=0.6,

        # 本轮重点是混合批处理，不要求模型完成
        # 很长的思考过程。
        max_tokens=64,
    )

    outputs = llm.generate(
        prompts,
        sampling_params,
    )

    for index, output in enumerate(outputs):
        print(
            f"\n请求 {index} 输出："
        )
        print(output["text"])

        print(
            f"请求 {index} token IDs："
        )
        print(output["token_ids"])

    runner = llm.model_runner

    print("\nPrefill 批处理统计：")

    print(
        "prefill microbatches:",
        runner.num_prefill_microbatches,
    )

    print(
        "max prefill batch size:",
        runner.max_observed_prefill_batch_size,
    )

    print(
        "mixed prefill microbatches:",
        runner.num_mixed_prefill_microbatches,
    )

    print("\n视觉缓存统计：")

    print(
        "current bytes:",
        runner.visual_cache_bytes,
    )

    print(
        "peak bytes:",
        runner.peak_visual_cache_bytes,
    )

    print(
        "vision forwards:",
        runner.num_vision_forwards,
    )

    print(
        "cache misses:",
        runner.num_visual_cache_misses,
    )

    print(
        "cache hits:",
        runner.num_visual_cache_hits,
    )
    
    print("\n请求级延迟：")

    for metrics in (
        llm.get_completed_request_metrics()
    ):
        print(
            {
                "seq_id": metrics.seq_id,
                "prompt_tokens": (
                    metrics.num_prompt_tokens
                ),
                "completion_tokens": (
                    metrics.num_completion_tokens
                ),
                "preprocessing_ms": round(
                    metrics.preprocessing_ms,
                    3,
                ),
                "queue_ms": round(
                    metrics.queue_ms,
                    3,
                ),
                "ttft_ms": round(
                    metrics.ttft_ms,
                    3,
                ),
                "tpot_ms": round(
                    metrics.tpot_ms,
                    3,
                ),
                "e2e_ms": round(
                    metrics.e2e_ms,
                    3,
                ),
            }
        )


if __name__ == "__main__":
    main()