"""Check local weights, dependencies and GPU/compiler before loading models."""
import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import hashlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmark_utils import chat_ids, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--backend", choices=("auto", "cuda", "reference"), default="auto")
    parser.add_argument("--gdn-decode-backend", choices=("fla", "state_aware_triton", "state_aware_cuda"),
                        default="state_aware_cuda")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = {"passed": False, "python": sys.version, "executable": sys.executable,
        "model": str(Path(args.model).resolve()), "draft": str(Path(args.draft).resolve()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "packages": {},
        "gdn_decode_backend": args.gdn_decode_backend, "gdn_verification_backend": args.backend,
        "gdn_replay_backend": args.backend}
    try:
        for package in ("torch", "transformers", "triton", "flash-attn", "flash-linear-attention", "causal-conv1d", "ninja"):
            try: report["packages"][package] = metadata.version(package)
            except metadata.PackageNotFoundError: report["packages"][package] = None
        revision = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, capture_output=True)
        report["git_commit"] = revision.stdout.strip() if revision.returncode == 0 else None
        report["config_sha256"] = {name: hashlib.sha256((Path(path) / "config.json").read_bytes()).hexdigest()
                                   for name, path in (("target", args.model), ("draft", args.draft))}
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() < args.tp:
            raise RuntimeError(f"Need {args.tp} visible CUDA GPUs; got {torch.cuda.device_count()}")
        report["devices"] = [{"rank": i, "name": torch.cuda.get_device_name(i),
            "memory_bytes": torch.cuda.get_device_properties(i).total_memory,
            "compute_capability": torch.cuda.get_device_capability(i)} for i in range(args.tp)]
        if any(device["compute_capability"][0] < 8 for device in report["devices"]):
            raise RuntimeError("BF16 suite requires Ampere or newer GPUs")
        from nanovllm.config import Config
        from transformers import AutoTokenizer
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        from causal_conv1d import causal_conv1d_fn
        config = Config(args.model, speculative_method="dflash", dflash_model=args.draft,
            dflash_num_speculative_tokens=3, tensor_parallel_size=args.tp,
            max_num_seqs=2, hybrid_cuda_graph_batch_sizes=(1, 2), enforce_eager=True,
            gdn_decode_backend=args.gdn_decode_backend)
        if args.gdn_decode_backend == "state_aware_cuda":
            text = config.text_config
            if (text.linear_key_head_dim, text.linear_value_head_dim, text.linear_conv_kernel_dim) != (128, 128, 4):
                raise ValueError("state_aware_cuda requires GDN Dk/Dv=128 and convolution width=4")
        if not list(Path(args.model).glob("*.safetensors")):
            raise RuntimeError("Target model directory contains no safetensors weights")
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        ids = chat_ids(tokenizer, "Hello", thinking=False)
        report["chat_template_token_count"] = len(ids)
        report["target"] = {"hidden_size": config.text_config.hidden_size,
                            "vocab_size": config.text_config.vocab_size,
                            "num_hidden_layers": config.text_config.num_hidden_layers,
                            "dtype": str(getattr(config.text_config, "dtype", None))}
        from nanovllm.models.dflash import read_dflash_config
        draft = read_dflash_config(args.draft, config.text_config, args.tp)
        report["draft"] = {"architecture": draft.architectures[0], "layers": draft.num_hidden_layers,
            "target_layer_ids": draft.target_layer_ids, "block_size": draft.block_size,
            "mask_token_id": draft.mask_token_id, "is_dflash2": draft.is_dflash2}
        if draft.is_dflash2:
            from download_qwen38 import describe_pair
            report["layout"] = describe_pair(json.loads((Path(args.model) / "config.json").read_text()),
                json.loads((Path(args.draft) / "config.json").read_text()), args.tp)
        if args.backend != "reference" or args.gdn_decode_backend == "state_aware_cuda":
            from torch.utils.cpp_extension import CUDA_HOME
            nvcc = shutil.which("nvcc") or (str(Path(CUDA_HOME) / "bin/nvcc") if CUDA_HOME else None)
            if not nvcc or not Path(nvcc).is_file() or not shutil.which("c++"):
                raise RuntimeError("Selected CUDA backends need nvcc and a C++ compiler; reference verification does not disable state_aware_cuda Decode")
            compiler = subprocess.run([nvcc, "--version"], text=True, capture_output=True, check=True)
            report["nvcc"] = compiler.stdout
        report["passed"] = True
    except Exception:
        report["error"] = traceback.format_exc()
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]: raise SystemExit(1)


if __name__ == "__main__":
    main()
