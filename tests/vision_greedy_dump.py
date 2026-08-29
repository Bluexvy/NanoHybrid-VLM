import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from nanovllm import LLM, SamplingParams


MODEL_PATH = "/workspace/models/Qwen3.5-9B"

PROMPT = (
    "请描述图片中的颜色和形状，"
    "只用一句话简洁回答。"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--budget",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    return parser.parse_args()


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


def main() -> None:
    args = parse_args()

    llm = LLM(
        MODEL_PATH,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=1024,

        # 两次运行唯一需要改变的执行参数。
        max_num_batched_tokens=args.budget,

        max_num_seqs=1,
        num_state_slots=1,
        gpu_memory_utilization=0.8,
    )

    prompt = {
        "prompt": PROMPT,
        "multi_modal_data": {
            "image": create_test_image(),
        },
    }

    sampling_params = SamplingParams(
        # 必须是真正的 greedy。
        temperature=0.0,

        # 比较前64个生成 token。
        max_tokens=64,
    )

    processed_prompt = (
        llm.input_processor.process(
            prompt
        )
    )
    
    output = llm.generate(
        [prompt],
        sampling_params,
    )[0]

    runner = llm.model_runner

    result = {
        "model": MODEL_PATH,
        "prompt": PROMPT,
        "prompt_token_ids": (
            processed_prompt.token_ids
        ),
        "max_num_batched_tokens": args.budget,
        "max_tokens": 64,
        "temperature": 0.0,
        "generated_token_ids": (
            output["token_ids"]
        ),
        "generated_text": output["text"],
        "prefill_microbatches": (
            runner.num_prefill_microbatches
        ),
        "vision_forwards": (
            runner.num_vision_forwards
        ),
        "visual_cache_hits": (
            runner.num_visual_cache_hits
        ),
        "visual_cache_misses": (
            runner.num_visual_cache_misses
        ),
        "current_visual_cache_bytes": (
            runner.visual_cache_bytes
        ),
        "peak_visual_cache_bytes": (
            runner.peak_visual_cache_bytes
        ),
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

    print("\n结果已保存：", args.output)
    print("token IDs:", output["token_ids"])
    print("prefill microbatches:",
          runner.num_prefill_microbatches)
    print("vision forwards:",
          runner.num_vision_forwards)
    print("cache misses:",
          runner.num_visual_cache_misses)
    print("cache hits:",
          runner.num_visual_cache_hits)


if __name__ == "__main__":
    main()