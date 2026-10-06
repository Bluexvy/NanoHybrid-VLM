from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import subprocess
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
    parser.add_argument("--allow-non-a800", action="store_true")
    parser.add_argument("--allow-no-nvlink", action="store_true")
    return parser.parse_args()


def command_output(*command: str) -> str:
    try:
        return subprocess.check_output(
            command, stderr=subprocess.STDOUT, text=True, timeout=20
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"


def read_model_geometry(model_path: str, tp_size: int) -> dict[str, object]:
    root = AutoConfig.from_pretrained(model_path)
    text = getattr(root, "text_config", root)
    names = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "vocab_size",
        "max_position_embeddings",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
    )
    geometry = {name: getattr(text, name, None) for name in names}
    divisible_names = {
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "vocab_size",
        "linear_num_key_heads",
        "linear_num_value_heads",
    }
    divisibility = {
        name: {"value": value, "divisible": value % tp_size == 0}
        for name, value in geometry.items()
        if name in divisible_names and isinstance(value, int)
    }
    invalid = [name for name, item in divisibility.items() if not item["divisible"]]
    if invalid:
        raise RuntimeError(f"Model dimensions are not divisible by TP={tp_size}: {invalid}")
    layer_types = list(getattr(text, "layer_types", ()))
    geometry.update(
        full_attention_layers=layer_types.count("full_attention"),
        linear_attention_layers=layer_types.count("linear_attention"),
        tp_divisibility=divisibility,
    )
    return geometry


def engine_port_available() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", 2333))
        except OSError:
            return False
    return True


def topology_has_nvlink(topology: str) -> bool:
    for line in topology.splitlines():
        cells = re.split(r"\s+", line.strip())
        # Skip the header and match the actual row: GPU0  X  NV8  ...
        if len(cells) >= 3 and cells[0] == "GPU0" and cells[1] == "X":
            return cells[2].startswith("NV")
    return False


def main() -> None:
    args = parse_args()
    if not cpp_extension.is_ninja_available():
        raise RuntimeError("Ninja is unavailable; the CUDA extension cannot be built")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; leave no-card mode before running")
    if torch.cuda.device_count() < args.tp_size:
        raise RuntimeError(
            f"Need {args.tp_size} visible GPUs, found {torch.cuda.device_count()}"
        )
    if not torch.distributed.is_nccl_available():
        raise RuntimeError("PyTorch NCCL backend is unavailable")
    if not engine_port_available():
        raise RuntimeError("TCP port 2333 is occupied; ModelRunner uses this fixed port")

    stale_shared_memory = Path("/dev/shm/nanovllm")
    if os.name == "posix" and stale_shared_memory.exists():
        raise RuntimeError(
            "Found stale /dev/shm/nanovllm. Stop old NanoHybrid-VLM processes "
            "and remove only that stale shared-memory file before retrying."
        )

    devices: list[dict[str, object]] = []
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

    tested_devices = devices[: args.tp_size]
    if not args.allow_non_a800:
        wrong = [device for device in tested_devices if "A800" not in str(device["name"])]
        small = [device for device in tested_devices if float(device["total_memory_gib"]) < 70]
        if wrong or small:
            raise RuntimeError(
                "Strict A800-80G check failed. Use --allow-non-a800 only for an "
                f"intentional portability run. devices={tested_devices}"
            )

    peer_access = {
        f"{source}->{destination}": bool(
            torch.cuda.can_device_access_peer(source, destination)
        )
        for source in range(args.tp_size)
        for destination in range(args.tp_size)
        if source != destination
    }
    topology = command_output("nvidia-smi", "topo", "-m")
    nvlink_detected = topology_has_nvlink(topology)
    if not args.allow_no_nvlink and not nvlink_detected:
        raise RuntimeError(
            "GPU0/GPU1 NVLink was not detected by `nvidia-smi topo -m`. "
            "Do not label this run as NVLink, or use --allow-no-nvlink for diagnostics.\n"
            + topology
        )

    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nccl_version": torch.cuda.nccl.version(),
        "visible_device_count": torch.cuda.device_count(),
        "devices": devices,
        "peer_access": peer_access,
        "nvidia_smi_topology": topology,
        "nvlink_pair_detected": nvlink_detected,
        "nvidia_smi_nvlink_status": command_output("nvidia-smi", "nvlink", "--status"),
        "nvcc_version": command_output("nvcc", "--version"),
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
