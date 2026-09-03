import argparse
import json
from statistics import median
from time import perf_counter

import torch
from transformers import (
    AutoModelForMultimodalLM,
    AutoTokenizer,
)


MODEL_PATH = "/workspace/models/Qwen3.5-9B"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt-tokens",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--output-tokens",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--attention",
        choices=(
            "eager",
            "flash_attention_2",
        ),
        default="eager",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
    )

    candidate_ids = tokenizer.encode(
        "性能测试",
        add_special_tokens=False,
    )

    if not candidate_ids:
        raise RuntimeError(
            "Tokenizer returned no tokens"
        )

    token_id = candidate_ids[0]

    input_ids = torch.full(
        (
            args.batch_size,
            args.prompt_tokens,
        ),
        fill_value=token_id,
        dtype=torch.long,
        device="cuda",
    )

    attention_mask = torch.ones_like(
        input_ids,
    )

    model = (
        AutoModelForMultimodalLM
        .from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16,
            attn_implementation=args.attention,
            low_cpu_mem_usage=True,
        )
        .eval()
        .to("cuda")
    )

    def generate_once():
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                use_cache=True,
                min_new_tokens=args.output_tokens,
                max_new_tokens=args.output_tokens,
                pad_token_id=(
                    tokenizer.eos_token_id
                ),
            )

        return output_ids

    # 使用正式 shape 热身一次。
    warmup_output = generate_once()
    del warmup_output

    torch.cuda.synchronize()

    memory_before = (
        torch.cuda.memory_allocated()
    )

    torch.cuda.reset_peak_memory_stats()

    elapsed_values = []
    generated_tokens_values = []

    for _ in range(args.repeats):
        torch.cuda.synchronize()

        start = perf_counter()

        output_ids = generate_once()

        torch.cuda.synchronize()

        elapsed = perf_counter() - start

        generated_tokens = (
            output_ids.shape[0]
            * (
                output_ids.shape[1]
                - args.prompt_tokens
            )
        )

        elapsed_values.append(elapsed)
        generated_tokens_values.append(
            generated_tokens
        )

        del output_ids

    median_elapsed = median(
        elapsed_values
    )

    generated_tokens = int(
        median(generated_tokens_values)
    )

    requests_per_second = (
        args.batch_size
        / median_elapsed
    )

    output_tokens_per_second = (
        generated_tokens
        / median_elapsed
    )

    result = {
        "backend": "huggingface",
        "model": MODEL_PATH,
        "dtype": "bfloat16",
        "attention": args.attention,
        "prompt_tokens": args.prompt_tokens,
        "output_tokens": args.output_tokens,
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "elapsed_seconds": elapsed_values,
        "median_elapsed_seconds": (
            median_elapsed
        ),
        "requests_per_second": (
            requests_per_second
        ),
        "output_tokens_per_second": (
            output_tokens_per_second
        ),
        "memory_before_gib": (
            memory_before / 1024**3
        ),
        "peak_memory_gib": (
            torch.cuda
            .max_memory_allocated()
            / 1024**3
        ),
    }

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()