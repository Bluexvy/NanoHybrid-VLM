from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from transformers import AutoConfig


FULL_CASES = (
    ("long_16k_b4", "long_context", 16_384, 4, 128, 16_384),
    ("long_32k_b2", "long_context", 32_768, 2, 128, 16_384),
    ("long_64k_b1", "long_context", 65_536, 1, 128, 16_384),
    ("long_128k_b1", "long_context", 131_072, 1, 64, 16_384),
    ("batch_2k_b32", "large_batch", 2_048, 32, 256, 16_384),
    ("batch_512_b64", "large_batch", 512, 64, 256, 16_384),
    ("batch_512_b128", "large_batch", 512, 128, 256, 32_768),
    ("batch_128_b256", "large_batch", 128, 256, 256, 32_768),
    ("decode_128_b128_o1024", "long_decode", 128, 128, 1_024, 32_768),
)

EXTREME_CASES = FULL_CASES + (
    ("long_128k_b2", "extreme_context", 131_072, 2, 64, 32_768),
    ("long_64k_b4", "extreme_context", 65_536, 4, 64, 32_768),
    ("long_32k_b8", "extreme_context", 32_768, 8, 128, 32_768),
    ("mixed_8k_b32", "extreme_batch", 8_192, 32, 128, 32_768),
    ("batch_1k_b128", "extreme_batch", 1_024, 128, 512, 32_768),
    ("batch_256_b256", "extreme_batch", 256, 256, 512, 32_768),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("full", "extreme"), default="full")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--backend", default="state_aware_cuda")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--continue-after-group-failure", action="store_true")
    return parser.parse_args()


def model_context_limit(model: str) -> int | None:
    root = AutoConfig.from_pretrained(model)
    text = getattr(root, "text_config", root)
    value = getattr(text, "max_position_embeddings", None)
    return int(value) if isinstance(value, int) else None


def classify_failure(text: str, returncode: int | None, timed_out: bool) -> str:
    lower = text.lower()
    if timed_out:
        return "timeout"
    if "out of memory" in lower or "cannot hold" in lower or "no available memory" in lower:
        return "oom"
    if returncode is not None and returncode < 0:
        return f"signal_{-returncode}"
    return "failed"


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=15)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_case(args: argparse.Namespace, case: tuple[Any, ...]) -> dict[str, Any]:
    name, group, prompt, batch, output_tokens, token_budget = case
    json_path = args.output_dir / f"{name}.json"
    log_path = args.output_dir / f"{name}.log"
    runner = Path(__file__).with_name("run_generation.py")
    command = [
        args.python_bin,
        str(runner),
        "--model",
        args.model,
        "--output",
        str(json_path),
        "--label",
        name,
        "--tp-size",
        str(args.tp_size),
        "--mode",
        "eager",
        "--backend",
        args.backend,
        "--cases",
        f"{prompt}:{batch}",
        "--output-tokens",
        str(output_tokens),
        "--repeats",
        "1",
        "--warmup-output-tokens",
        "8",
        "--token-budget",
        str(token_budget),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--graph-buckets",
        "1",
    ]
    started = time.time()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        console, _ = process.communicate(timeout=args.timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        terminate_process_group(process)
        console = "capacity case timed out\n"
        if process.stdout is not None:
            console += process.stdout.read()
    log_path.write_text(console, encoding="utf-8")
    status = "passed" if process.returncode == 0 and json_path.exists() else classify_failure(
        console, process.returncode, timed_out
    )
    return {
        "name": name,
        "group": group,
        "prompt_tokens": prompt,
        "batch_size": batch,
        "output_tokens": output_tokens,
        "total_prompt_tokens": prompt * batch,
        "token_budget": token_budget,
        "status": status,
        "returncode": process.returncode,
        "elapsed_seconds": time.time() - started,
        "result": str(json_path),
        "log": str(log_path),
        "command": command,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    context_limit = model_context_limit(args.model)
    cases = FULL_CASES if args.profile == "full" else EXTREME_CASES
    failed_groups: set[str] = set()
    records: list[dict[str, Any]] = []

    for case in cases:
        name, group, prompt, _, output_tokens, _ = case
        if context_limit is not None and prompt + output_tokens > context_limit:
            records.append(
                {
                    "name": name,
                    "group": group,
                    "status": "skipped_model_context_limit",
                    "requested_tokens": prompt + output_tokens,
                    "model_context_limit": context_limit,
                }
            )
            continue
        if group in failed_groups and not args.continue_after_group_failure:
            records.append(
                {"name": name, "group": group, "status": "skipped_after_group_failure"}
            )
            continue
        print(f"\n========== capacity: {name} ==========", flush=True)
        record = run_case(args, case)
        records.append(record)
        print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
        if record["status"] != "passed":
            failed_groups.add(group)

    payload = {
        "profile": args.profile,
        "model": args.model,
        "model_context_limit": context_limit,
        "cases": records,
        "passed_cases": sum(item["status"] == "passed" for item in records),
        "capacity_failures_are_expected": True,
    }
    manifest = args.output_dir / "capacity_sweep.json"
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
