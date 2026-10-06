from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


DEFAULT_PAYLOADS = (
    8 * 1024,
    128 * 1024,
    1024 * 1024,
    8 * 1024**2,
    32 * 1024**2,
    128 * 1024**2,
    256 * 1024**2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def iterations_for(payload_bytes: int, maximum: int) -> int:
    if payload_bytes >= 128 * 1024**2:
        return min(maximum, 10)
    if payload_bytes >= 32 * 1024**2:
        return min(maximum, 20)
    if payload_bytes >= 8 * 1024**2:
        return min(maximum, 40)
    return maximum


def worker(
    rank: int,
    world_size: int,
    port: int,
    warmup: int,
    max_iterations: int,
    output: str,
) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    check = torch.tensor([rank + 1.0], device=rank)
    dist.all_reduce(check)
    expected = world_size * (world_size + 1) / 2
    if check.item() != expected:
        raise RuntimeError(f"AllReduce correctness failure: {check.item()} != {expected}")

    results: dict[str, object] = {
        "world_size": world_size,
        "correctness_sum": check.item(),
        "cases": {},
    }
    for payload_bytes in DEFAULT_PAYLOADS:
        tensor = torch.ones(payload_bytes // 2, device=rank, dtype=torch.bfloat16)
        count = iterations_for(payload_bytes, max_iterations)
        for _ in range(min(warmup, count)):
            dist.all_reduce(tensor)
        torch.cuda.synchronize(rank)
        dist.barrier(device_ids=[rank])

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(count):
            dist.all_reduce(tensor)
        end.record()
        end.synchronize()

        mean_ms = start.elapsed_time(end) / count
        algorithm_gbps = payload_bytes / (mean_ms / 1000.0) / 1e9
        bus_factor = 2.0 * (world_size - 1) / world_size
        if rank == 0:
            results["cases"][str(payload_bytes)] = {
                "payload_bytes": payload_bytes,
                "iterations": count,
                "mean_allreduce_us": mean_ms * 1000.0,
                "algorithm_bandwidth_gbps": algorithm_gbps,
                "bus_bandwidth_gbps": algorithm_gbps * bus_factor,
                "estimated_64_allreduces_ms": mean_ms * 64.0,
            }
        del tensor

    dist.barrier(device_ids=[rank])
    if rank == 0:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(results, ensure_ascii=False, indent=2))
    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    if torch.cuda.device_count() < args.world_size:
        raise RuntimeError(
            f"Need {args.world_size} GPUs, found {torch.cuda.device_count()}"
        )
    mp.spawn(
        worker,
        args=(
            args.world_size,
            free_port(),
            args.warmup,
            args.iterations,
            str(args.output),
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
