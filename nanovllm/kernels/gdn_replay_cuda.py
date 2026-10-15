"""Lazy PyTorch C++/CUDA Extension for state-only GDN replay."""
from pathlib import Path
from threading import Lock
import os
import torch

_extension = None
_load_lock = Lock()


def load_gdn_replay_cuda_extension():
    global _extension
    with _load_lock:
        if _extension is None:
            from torch.utils.cpp_extension import load
            source = Path(__file__).resolve().parent / "cuda"
            build = source.parents[2] / ".cache" / "torch_extensions" / "gdn_replay_cuda"
            build.mkdir(parents=True, exist_ok=True)
            _extension = load(
                name="nanovllm_gdn_replay_cuda_ext",
                sources=[str(source / "gdn_replay_binding.cpp"), str(source / "gdn_replay_kernel.cu")],
                extra_cflags=["-O2"] if os.name != "nt" else ["/O2"],
                extra_cuda_cflags=["-O3", "-lineinfo", "--fmad=false"],
                build_directory=str(build), verbose=False,
            )
    return _extension


def cuda_replay_supported(inputs, conv, state, num_tokens):
    types = (torch.float32, torch.float16, torch.bfloat16)
    return (state.is_cuda and state.dtype == torch.float32
            and 1 <= inputs.key.shape[-1] <= 128 and 1 <= num_tokens <= 16
            and 1 <= inputs.key.shape[0] <= 65535 and 1 <= inputs.key.shape[2] <= 65535
            and inputs.key.dtype in types and inputs.value.dtype == inputs.key.dtype
            and inputs.beta.dtype in types and inputs.g.dtype == torch.float32
            and conv.dtype in types and inputs.conv_input.dtype == conv.dtype)


@torch.inference_mode()
def replay_gdn_state_cuda(inputs, conv, state, num_tokens):
    if not cuda_replay_supported(inputs, conv, state, num_tokens):
        raise ValueError("CUDA replay requires CUDA FP32 state, supported dtypes, Dk<=128 and prefix<=16")
    result = load_gdn_replay_cuda_extension().gdn_replay(
        inputs.key, inputs.value, inputs.g, inputs.beta, inputs.conv_input,
        conv, state, num_tokens,
    )
    return result[0], result[1]
