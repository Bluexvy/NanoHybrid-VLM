from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from nanovllm.kernels.state_aware_gdn_cuda import (
    load_state_aware_gdn_cuda_extension,
    state_aware_causal_conv1d_cuda,
    state_aware_gdn_decode_cuda,
)


NUM_GDN_LAYERS = 24
NUM_HEADS = 32
HEAD_DIM = 128
NUM_CHANNELS = 8192


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=("recurrent", "conv"), required=True)
    parser.add_argument("--batch-size", type=int, choices=(16, 64, 128), required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--conv-block-size", type=int, default=256)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def recurrent_callable(batch_size: int):
    query = torch.randn(batch_size, 1, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    g = -torch.nn.functional.softplus(
        torch.randn(batch_size, 1, NUM_HEADS, device="cuda", dtype=torch.float32)
    )
    beta = torch.sigmoid(
        torch.randn(batch_size, 1, NUM_HEADS, device="cuda", dtype=torch.bfloat16)
    )
    state_pool = torch.zeros(
        batch_size,
        NUM_GDN_LAYERS,
        NUM_HEADS,
        HEAD_DIM,
        HEAD_DIM,
        device="cuda",
        dtype=torch.float32,
    )
    slot_ids = torch.arange(batch_size - 1, -1, -1, device="cuda", dtype=torch.long)
    output = torch.empty_like(value)

    def launch() -> None:
        state_aware_gdn_decode_cuda(
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            recurrent_state_pool=state_pool,
            state_slot_ids=slot_ids,
            gdn_index=7,
            output=output,
        )

    return launch


def conv_callable(batch_size: int, block_size: int):
    inputs = torch.randn(batch_size, NUM_CHANNELS, 1, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(NUM_CHANNELS, 4, device="cuda", dtype=torch.bfloat16)
    state_pool = torch.zeros(
        batch_size,
        NUM_GDN_LAYERS,
        NUM_CHANNELS,
        4,
        device="cuda",
        dtype=torch.bfloat16,
    )
    slot_ids = torch.arange(batch_size - 1, -1, -1, device="cuda", dtype=torch.long)
    output = torch.empty_like(inputs)

    def launch() -> None:
        state_aware_causal_conv1d_cuda(
            x=inputs,
            weight=weights,
            conv_state_pool=state_pool,
            state_slot_ids=slot_ids,
            gdn_index=7,
            output=output,
            block_size=block_size,
        )

    return launch


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")

    torch.manual_seed(20260930 + args.batch_size)
    torch.cuda.manual_seed_all(20260930 + args.batch_size)
    load_state_aware_gdn_cuda_extension()
    launch = (
        recurrent_callable(args.batch_size)
        if args.kernel == "recurrent"
        else conv_callable(args.batch_size, args.conv_block_size)
    )

    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()

    measurements_us: list[float] = []
    torch.cuda.nvtx.range_push(f"a800_ncu_{args.kernel}_b{args.batch_size}")
    for _ in range(args.iterations):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        stop.record()
        stop.synchronize()
        measurements_us.append(start.elapsed_time(stop) * 1000.0)
    torch.cuda.nvtx.range_pop()

    payload = {
        "kernel": args.kernel,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "mean_us": statistics.mean(measurements_us),
        "median_us": statistics.median(measurements_us),
        "min_us": min(measurements_us),
        "max_us": max(measurements_us),
        "samples_us": measurements_us,
        "gpu": torch.cuda.get_device_name(0),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
