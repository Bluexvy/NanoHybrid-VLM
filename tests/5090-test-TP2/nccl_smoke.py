from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def worker(
    rank: int,
    world_size: int,
    port: int,
    warmup: int,
    iterations: int,
    output: str,
) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )

    correctness = torch.tensor(
        [float(rank + 1)], device=rank, dtype=torch.float32
    )
    dist.all_reduce(correctness)
    expected = world_size * (world_size + 1) / 2
    if correctness.item() != expected:
        raise RuntimeError(
            f"NCCL AllReduce mismatch on rank {rank}: "
            f"{correctness.item()} != {expected}"
        )

    results: dict[str, object] = {
        "world_size": world_size,
        "correctness_sum": correctness.item(),
        "cases": {},
    }

    # These payloads match hidden-state AllReduce sizes for hidden=4096:
    # B=1 -> 8 KiB BF16, B=16 -> 128 KiB BF16.
    for payload_bytes in (8 * 1024, 128 * 1024, 1024 * 1024):
        numel = payload_bytes // 2
        tensor = torch.ones(numel, device=rank, dtype=torch.bfloat16)

        for _ in range(warmup):
            dist.all_reduce(tensor)
        torch.cuda.synchronize(rank)
        dist.barrier(device_ids=[rank])

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            dist.all_reduce(tensor)
        end.record()
        end.synchronize()

        mean_ms = start.elapsed_time(end) / iterations
        if rank == 0:
            results["cases"][str(payload_bytes)] = {
                "payload_bytes": payload_bytes,
                "mean_allreduce_us": mean_ms * 1000.0,
                "estimated_64_allreduces_ms": mean_ms * 64.0,
            }

    dist.barrier(device_ids=[rank])
    if rank == 0:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
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
            find_free_port(),
            args.warmup,
            args.iterations,
            str(args.output),
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
