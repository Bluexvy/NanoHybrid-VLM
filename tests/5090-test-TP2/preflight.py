from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
from pathlib import Path

import torch
from torch.utils import cpp_extension
from transformers import AutoConfig

from suite_common import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, default=2)
    return parser.parse_args()


def read_model_geometry(model_path: str, tp_size: int) -> dict[str, object]:
    root = AutoConfig.from_pretrained(model_path)
    text = getattr(root, "text_config", root)

    field_names = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "vocab_size",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
    )

    geometry: dict[str, object] = {}
    divisibility: dict[str, object] = {}

    for name in field_names:
        value = getattr(text, name, None)
        geometry[name] = value
        if isinstance(value, int) and name in {
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "vocab_size",
            "linear_num_key_heads",
            "linear_num_value_heads",
        }:
            divisibility[name] = {
                "value": value,
                "divisible": value % tp_size == 0,
            }

    layer_types = list(getattr(text, "layer_types", ()))
    geometry["full_attention_layers"] = layer_types.count("full_attention")
    geometry["linear_attention_layers"] = layer_types.count("linear_attention")
    geometry["tp_divisibility"] = divisibility

    invalid = [
        name
        for name, result in divisibility.items()
        if not bool(result["divisible"])
    ]
    if invalid:
        raise RuntimeError(
            f"Model dimensions are not divisible by TP={tp_size}: {invalid}"
        )

    return geometry


def engine_port_available() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", 2333))
        except OSError:
            return False
    return True


def main() -> None:
    args = parse_args()

    if not cpp_extension.is_ninja_available():
        raise RuntimeError(
            "Ninja is unavailable; state_aware_cuda "
            "cannot build its PyTorch CUDA Extension"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() < args.tp_size:
        raise RuntimeError(
            f"Need {args.tp_size} visible GPUs, found {torch.cuda.device_count()}"
        )
    if not torch.distributed.is_nccl_available():
        raise RuntimeError("PyTorch NCCL backend is unavailable")
    if not engine_port_available():
        raise RuntimeError(
            "TCP port 2333 is occupied; ModelRunner uses this fixed port"
        )

    stale_shared_memory = Path("/dev/shm/nanovllm")
    if os.name == "posix" and stale_shared_memory.exists():
        raise RuntimeError(
            "Found stale /dev/shm/nanovllm. Ensure no NanoHybrid-VLM "
            "process is running, then remove that stale shared-memory file."
        )

    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        with torch.cuda.device(index):
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "total_memory_gib": total_bytes / 1024**3,
                "free_memory_gib": free_bytes / 1024**3,
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )

    peer_access = {}
    for source in range(args.tp_size):
        for destination in range(args.tp_size):
            if source == destination:
                continue
            peer_access[f"{source}->{destination}"] = bool(
                torch.cuda.can_device_access_peer(source, destination)
            )

    payload = {
        "ninja_available": True,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nccl_available": torch.distributed.is_nccl_available(),
        "nccl_version": (
            torch.cuda.nccl.version()
            if torch.distributed.is_nccl_available()
            else None
        ),
        "visible_device_count": torch.cuda.device_count(),
        "devices": devices,
        "peer_access": peer_access,
        "engine_port_2333_available": True,
        "stale_nanovllm_shared_memory": False,
        "model": args.model,
        "tp_size": args.tp_size,
        "model_geometry": read_model_geometry(args.model, args.tp_size),
    }

    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
