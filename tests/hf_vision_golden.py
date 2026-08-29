import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
)


MODEL_PATH = "/workspace/models/Qwen3.5-9B"

OUTPUT_PATH = Path(
    "artifacts/golden/"
    "qwen35_9b_vision_hf.json"
)

PROMPT = (
    "请描述图片中的颜色和形状，"
    "只用一句话简洁回答。"
)

MAX_NEW_TOKENS = 64


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
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required"
        )

    # 虽然 greedy 不使用随机采样，
    # 仍然固定 seed，保证实验环境明确。
    torch.manual_seed(0)

    image = create_test_image()

    processor = (
        AutoProcessor.from_pretrained(
            MODEL_PATH,
        )
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": PROMPT,
                },
            ],
        },
    ]

    # 这里故意与 nano InputProcessor 使用
    # 完全相同的两阶段处理方式。
    formatted_prompt = (
        processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )

    batch = processor(
        text=[formatted_prompt],
        images=[image],
        padding=False,
        return_tensors="pt",
    )

    prompt_token_ids = (
        batch["input_ids"][0].tolist()
    )

    print(
        "prompt length:",
        len(prompt_token_ids),
    )

    print(
        "image grid:",
        batch["image_grid_thw"].tolist(),
    )

    print(
        "image token count:",
        int(
            (
                batch["mm_token_type_ids"]
                == 1
            ).sum().item()
        ),
    )

    model = (
        AutoModelForMultimodalLM
        .from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16,

            # Full Attention 使用官方 eager 路径，
            # 优先保证排障清晰。
            attn_implementation="eager",

            low_cpu_mem_usage=True,
        )
    )

    model = model.eval().to("cuda")

    model_inputs = {}

    for name, tensor in batch.items():
        if name == "pixel_values":
            model_inputs[name] = tensor.to(
                device="cuda",
                dtype=torch.bfloat16,
            )
        else:
            model_inputs[name] = tensor.to(
                device="cuda",
            )

    with torch.inference_mode():
        generation = model.generate(
            **model_inputs,

            max_new_tokens=(
                MAX_NEW_TOKENS
            ),

            # Hugging Face 的 greedy 模式。
            do_sample=False,

            use_cache=True,

            return_dict_in_generate=True,

            pad_token_id=(
                processor.tokenizer.eos_token_id
            ),
        )

    all_token_ids = (
        generation.sequences[0]
    )

    prompt_length = len(
        prompt_token_ids
    )

    generated_token_ids = (
        all_token_ids[
            prompt_length:
        ].tolist()
    )

    generated_text = processor.decode(
        generated_token_ids,
        skip_special_tokens=False,
    )

    result = {
        "model": MODEL_PATH,
        "dtype": "bfloat16",
        "attention_implementation": (
            "eager"
        ),
        "prompt": PROMPT,
        "prompt_token_ids": (
            prompt_token_ids
        ),
        "generated_token_ids": (
            generated_token_ids
        ),
        "generated_text": (
            generated_text
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nHF generated token IDs:")
    print(generated_token_ids)

    print("\nHF generated text:")
    print(generated_text)

    print("\nHF golden saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()