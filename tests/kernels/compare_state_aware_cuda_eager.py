from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


REPO_ROOT = Path("/workspace/nano-vllm")
MODEL_PATH = Path("/workspace/models/Qwen3.5-9B")
OUTPUT_DIR = REPO_ROOT / "artifacts" / "kernels" / "state_aware_cuda_eager"

BATCH_SIZES = (1, 4, 8)
OUTPUT_TOKENS = 64


def make_prompt(tokenizer) -> str:
    return tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "请用一段话解释线性注意力的基本原理和主要优点。",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def run_generation_case(
    llm: LLM,
    prompt: str,
    batch_size: int,
    output_tokens: int,
) -> dict[str, object]:
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=output_tokens,
        ignore_eos=True,
    )

    seq_ids = [
        llm.add_request(prompt, sampling_params)
        for _ in range(batch_size)
    ]

    completed_outputs: dict[int, list[int]] = {}

    total_prefill_tokens = 0
    total_decode_tokens = 0
    total_prefill_seconds = 0.0
    total_decode_seconds = 0.0

    start_time = perf_counter()

    while not llm.is_finished():
        outputs, stats = llm.step()

        total_prefill_tokens += stats.num_prefill_tokens
        total_decode_tokens += stats.num_decode_tokens
        total_prefill_seconds += stats.prefill_elapsed
        total_decode_seconds += stats.decode_elapsed

        for seq_id, token_ids in outputs:
            completed_outputs[seq_id] = token_ids

    torch.cuda.synchronize()

    elapsed_seconds = perf_counter() - start_time

    if set(completed_outputs) != set(seq_ids):
        raise RuntimeError(
            f"Completed Sequence IDs do not match submitted IDs: "
            f"submitted={seq_ids}, completed={sorted(completed_outputs)}"
        )

    token_ids = [
        completed_outputs[seq_id]
        for seq_id in seq_ids
    ]

    request_metrics = {
        metric.seq_id: metric
        for metric in llm.get_completed_request_metrics()
        if metric.seq_id in seq_ids
    }

    mean_ttft_ms = sum(request_metrics[seq_id].ttft_ms for seq_id in seq_ids) / batch_size
    mean_tpot_ms = sum(request_metrics[seq_id].tpot_ms for seq_id in seq_ids) / batch_size
    mean_e2e_ms = sum(request_metrics[seq_id].e2e_ms for seq_id in seq_ids) / batch_size

    prefill_throughput = total_prefill_tokens / total_prefill_seconds if total_prefill_seconds > 0 else 0.0
    decode_throughput = total_decode_tokens / total_decode_seconds if total_decode_seconds > 0 else 0.0

    return {
        "batch_size": batch_size,
        "output_tokens": output_tokens,
        "token_ids": token_ids,
        "prefill_tokens": total_prefill_tokens,
        "decode_tokens": total_decode_tokens,
        "prefill_seconds": total_prefill_seconds,
        "decode_seconds": total_decode_seconds,
        "elapsed_seconds": elapsed_seconds,
        "prefill_throughput": prefill_throughput,
        "decode_throughput": decode_throughput,
        "mean_ttft_ms": mean_ttft_ms,
        "mean_tpot_ms": mean_tpot_ms,
        "mean_e2e_ms": mean_e2e_ms,
    }


def run_child(
    backend: str,
    output_path: Path,
) -> None:
    torch.manual_seed(20260907)
    torch.cuda.manual_seed_all(20260907)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    prompt = make_prompt(tokenizer)

    initialization_start = perf_counter()

    llm = LLM(
        str(MODEL_PATH),
        enforce_eager=True,
        tensor_parallel_size=1,
        gdn_decode_backend=backend,
        max_model_len=512,
        max_num_batched_tokens=1024,
        max_num_seqs=max(BATCH_SIZES),
        num_state_slots=max(BATCH_SIZES),
        gpu_memory_utilization=0.78,
        enable_prefix_cache=False,
        hybrid_prefix_cache_mode="disabled",
    )

    torch.cuda.synchronize()

    initialization_seconds = perf_counter() - initialization_start

    print(f"\nBackend: {backend}")
    print(f"Initialization: {initialization_seconds:.3f} seconds")

    print("\nRunning warmup...")

    run_generation_case(
        llm=llm,
        prompt=prompt,
        batch_size=1,
        output_tokens=8,
    )

    case_results: dict[str, object] = {}

    for batch_size in BATCH_SIZES:
        print(f"\nRunning B={batch_size}...")

        result = run_generation_case(
            llm=llm,
            prompt=prompt,
            batch_size=batch_size,
            output_tokens=OUTPUT_TOKENS,
        )

        case_results[str(batch_size)] = result

        print(f"Decode throughput: {result['decode_throughput']:.3f} token/s")
        print(f"Mean TTFT: {result['mean_ttft_ms']:.3f} ms")
        print(f"Mean TPOT: {result['mean_tpot_ms']:.3f} ms")
        print(f"Mean E2E: {result['mean_e2e_ms']:.3f} ms")
        print(f"First output token IDs: {result['token_ids'][0]}")

    payload = {
        "backend": backend,
        "model_path": str(MODEL_PATH),
        "initialization_seconds": initialization_seconds,
        "batch_sizes": list(BATCH_SIZES),
        "output_tokens": OUTPUT_TOKENS,
        "cases": case_results,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved result to: {output_path}")


def find_first_token_difference(
    left: list[int],
    right: list[int],
) -> int | None:
    common_length = min(
        len(left),
        len(right),
    )

    for index in range(common_length):
        if left[index] != right[index]:
            return index

    if len(left) != len(right):
        return common_length

    return None


def compare_results() -> None:
    triton_path = OUTPUT_DIR / "triton.json"
    cuda_path = OUTPUT_DIR / "cuda.json"

    triton_result = json.loads(
        triton_path.read_text(
            encoding="utf-8",
        )
    )

    cuda_result = json.loads(
        cuda_path.read_text(
            encoding="utf-8",
        )
    )

    print("\n" + "=" * 88)
    print("Comparing state_aware_triton and state_aware_cuda")
    print("=" * 88)

    all_tokens_match = True

    for batch_size in BATCH_SIZES:
        case_key = str(batch_size)

        triton_case = triton_result["cases"][case_key]
        cuda_case = cuda_result["cases"][case_key]

        triton_outputs = triton_case["token_ids"]
        cuda_outputs = cuda_case["token_ids"]

        if len(triton_outputs) != len(cuda_outputs):
            raise AssertionError(
                f"B={batch_size}: output batch sizes differ"
            )

        case_matches = True

        for request_index, (triton_tokens, cuda_tokens) in enumerate(zip(triton_outputs, cuda_outputs)):
            difference_index = find_first_token_difference(
                triton_tokens,
                cuda_tokens,
            )

            if difference_index is not None:
                case_matches = False
                all_tokens_match = False

                triton_token = triton_tokens[difference_index] if difference_index < len(triton_tokens) else None
                cuda_token = cuda_tokens[difference_index] if difference_index < len(cuda_tokens) else None

                print(
                    f"B={batch_size}, request={request_index}: first difference at "
                    f"token {difference_index}, Triton={triton_token}, CUDA={cuda_token}"
                )

        triton_decode = float(triton_case["decode_throughput"])
        cuda_decode = float(cuda_case["decode_throughput"])
        speedup = cuda_decode / triton_decode

        triton_tpot = float(triton_case["mean_tpot_ms"])
        cuda_tpot = float(cuda_case["mean_tpot_ms"])
        tpot_change = (cuda_tpot / triton_tpot - 1.0) * 100.0

        print(
            f"B={batch_size}: "
            f"tokens_match={case_matches}, "
            f"Triton={triton_decode:.3f} tok/s, "
            f"CUDA={cuda_decode:.3f} tok/s, "
            f"speedup={speedup:.3f}x, "
            f"TPOT change={tpot_change:+.2f}%"
        )

    if not all_tokens_match:
        raise AssertionError(
            "CUDA and Triton generated different greedy token IDs"
        )

    print(
        "\nPart CUDA Runtime Eager PASSED: "
        "the Triton and CUDA backends generated identical greedy tokens."
    )


def run_all() -> None:
    script_path = Path(__file__).resolve()

    commands = [
        (
            "state_aware_triton",
            OUTPUT_DIR / "triton.json",
        ),
        (
            "state_aware_cuda",
            OUTPUT_DIR / "cuda.json",
        ),
    ]

    for backend, output_path in commands:
        command = [
            sys.executable,
            str(script_path),
            "--backend",
            backend,
            "--output",
            str(output_path),
        ]

        print("\n" + "=" * 88)
        print("Running:", " ".join(command))
        print("=" * 88)

        subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
        )

    compare_results()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backend",
        choices=(
            "state_aware_triton",
            "state_aware_cuda",
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.backend is None:
        run_all()
        return

    if args.output is None:
        raise ValueError(
            "--output is required when --backend is specified"
        )

    run_child(
        backend=args.backend,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()