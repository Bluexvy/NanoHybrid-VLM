from __future__ import annotations

import atexit
import json
import math
import statistics
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from PIL import Image, ImageDraw

from nanovllm import LLM, SamplingParams


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    position = (len(ordered) - 1) * ratio
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }

    return {
        "mean": statistics.mean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def bytes_to_mib(value: int) -> float:
    return value / 1024**2


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_exact_token_prompt(llm: LLM, length: int) -> list[int]:
    if length <= 0:
        raise ValueError("Prompt length must be positive")

    seed_ids = llm.tokenizer.encode(
        "Tensor parallel inference validates cache and state consistency. ",
        add_special_tokens=False,
    )

    if not seed_ids:
        raise RuntimeError("Tokenizer returned an empty seed")

    repeats = math.ceil(length / len(seed_ids))
    return (seed_ids * repeats)[:length]


def build_synthetic_image(size: int = 448) -> Image.Image:
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    margin = max(size // 10, 1)
    middle = size // 2
    draw.rectangle(
        (margin, margin, middle - margin // 2, size - margin),
        fill="red",
    )
    draw.ellipse(
        (middle + margin // 2, margin, size - margin, size - margin),
        fill="blue",
    )
    return image


def close_llm(llm: LLM | None) -> None:
    if llm is None:
        return

    # LLMEngine registers exit() with atexit. Explicit cleanup is useful for
    # a multi-process TP test, but it must be unregistered to avoid a second
    # call after model_runner has already been deleted.
    atexit.unregister(llm.exit)
    llm.exit()


def run_requests(
    llm: LLM,
    prompts: list[Any],
    sampling_params: SamplingParams | list[SamplingParams],
) -> dict[str, Any]:
    if isinstance(sampling_params, SamplingParams):
        params = [sampling_params] * len(prompts)
    else:
        params = sampling_params

    if len(params) != len(prompts):
        raise ValueError("Sampling params must align with prompts")

    seq_ids = [
        llm.add_request(prompt, request_params)
        for prompt, request_params in zip(prompts, params)
    ]

    completed: dict[int, list[int]] = {}
    prefill_step_ms: list[float] = []
    decode_step_ms: list[float] = []
    total_prefill_tokens = 0
    total_decode_tokens = 0
    torch.cuda.synchronize()
    total_start = perf_counter()

    while not llm.is_finished():
        outputs, stats = llm.step()

        if stats.num_prefill_tokens:
            total_prefill_tokens += stats.num_prefill_tokens
            prefill_step_ms.append(stats.prefill_elapsed * 1000.0)

        if stats.num_decode_tokens:
            total_decode_tokens += stats.num_decode_tokens
            decode_step_ms.append(stats.decode_elapsed * 1000.0)

        for seq_id, token_ids in outputs:
            completed[seq_id] = list(token_ids)

    torch.cuda.synchronize()
    total_seconds = perf_counter() - total_start

    missing = [seq_id for seq_id in seq_ids if seq_id not in completed]
    if missing:
        raise RuntimeError(f"Requests did not complete: {missing}")

    metrics = [llm.request_metrics[seq_id] for seq_id in seq_ids]
    decode_seconds = sum(decode_step_ms) / 1000.0
    prefill_seconds = sum(prefill_step_ms) / 1000.0

    return {
        "seq_ids": seq_ids,
        "outputs": [completed[seq_id] for seq_id in seq_ids],
        "prompt_tokens": [metric.num_prompt_tokens for metric in metrics],
        "completion_tokens": [metric.num_completion_tokens for metric in metrics],
        "total_prefill_tokens": total_prefill_tokens,
        "total_decode_tokens": total_decode_tokens,
        "prefill_tokens_per_second": (
            total_prefill_tokens / prefill_seconds
            if prefill_seconds > 0
            else 0.0
        ),
        "decode_tokens_per_second": (
            total_decode_tokens / decode_seconds
            if decode_seconds > 0
            else 0.0
        ),
        "prefill_step_ms": summarize(prefill_step_ms),
        "decode_step_ms": summarize(decode_step_ms),
        "ttft_ms": summarize(
            [float(metric.ttft_ms) for metric in metrics if metric.ttft_ms is not None]
        ),
        "tpot_ms": summarize(
            [float(metric.tpot_ms) for metric in metrics if metric.tpot_ms is not None]
        ),
        "e2e_ms": summarize(
            [float(metric.e2e_ms) for metric in metrics if metric.e2e_ms is not None]
        ),
        "total_seconds": total_seconds,
    }


def memory_stats_mib(llm: LLM) -> dict[str, float]:
    raw = llm.model_runner.get_memory_stats()
    return {
        key.replace("_bytes", "_mib"): bytes_to_mib(value)
        for key, value in raw.items()
    }
