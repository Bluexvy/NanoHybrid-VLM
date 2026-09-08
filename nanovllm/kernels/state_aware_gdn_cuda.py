from __future__ import annotations

import os
from pathlib import Path

import torch


_extension = None


def _load_extension():
    """
    第一次调用 CUDA 算子时编译 Extension。

    后续调用直接复用已经加载的动态库，不会重复编译。
    """

    global _extension

    if _extension is not None:
        return _extension

    project_root = Path(__file__).resolve().parents[2]
    cuda_source_dir = project_root / "nanovllm" / "kernels" / "cuda"
    build_directory = project_root / ".cache" / "torch_extensions" / "state_aware_gdn_cuda"

    build_directory.mkdir(parents=True, exist_ok=True)

    cuda_home = "/workspace/cuda-12.8"

    os.environ["CUDA_HOME"] = cuda_home
    os.environ["PATH"] = f"{cuda_home}/bin:{os.environ['PATH']}"
    os.environ["MAX_JOBS"] = "1"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

    from torch.utils import cpp_extension

    cpp_extension.CUDA_HOME = cuda_home

    _extension = cpp_extension.load(
        name="nanovllm_state_aware_gdn_cuda_ext",
        sources=[
            str(cuda_source_dir / "state_aware_gdn_binding.cpp"),
            str(cuda_source_dir / "state_aware_gdn_kernel.cu"),
        ],
        extra_cflags=[
            "-O2",
        ],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
        ],
        build_directory=str(build_directory),
        verbose=True,
    )

    return _extension

def load_state_aware_gdn_cuda_extension():
    """
    提前编译并加载 CUDA Extension。

    ModelRunner 会在 CUDA Graph Capture 之前调用，
    避免首次 JIT 编译发生在 Graph Capture 内。
    """

    return _load_extension()

@torch.inference_mode()
def state_aware_gdn_decode_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state_pool: torch.Tensor,
    state_slot_ids: torch.Tensor,
    gdn_index: int,
    *,
    output: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """
    CUDA C++ Extension 的 Python 接口。

    输入：
        query/key:
            [B, 1, 32, 128], BF16

        value:
            [B, 1, 32, 128], BF16

        g:
            [B, 1, 32], FP32

        beta:
            [B, 1, 32], BF16

        recurrent_state_pool:
            [num_slots, num_gdn_layers, 32, 128, 128],
            FP32

        state_slot_ids:
            [B], INT64

    副作用：
        根据 state_slot_ids 和 gdn_index，
        原地更新 recurrent_state_pool。

    返回：
        [B, 1, 32, 128], BF16
    """

    if output is None:
        output = torch.empty(
            value.shape,
            dtype=value.dtype,
            device=value.device,
        )
    if scale is None:
        scale = query.shape[-1] ** -0.5

    extension = _load_extension()

    return extension.state_aware_gdn(
        query,
        key,
        value,
        g,
        beta,
        recurrent_state_pool,
        state_slot_ids,
        gdn_index,
        output,
        scale,
    )
    
@torch.inference_mode()
def state_aware_causal_conv1d_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state_pool: torch.Tensor,
    state_slot_ids: torch.Tensor,
    gdn_index: int,
    *,
    output: torch.Tensor | None = None,
    block_size: int = 256,
) -> torch.Tensor:
    """
    State-aware Causal Conv Decode。

    x:
        [B, C, 1], BF16

    weight:
        [C, 4], BF16

    conv_state_pool:
        [num_slots, num_gdn_layers, C, 4], BF16

    state_slot_ids:
        [B], INT64
    """

    if output is None:
        output = torch.empty(
            x.shape,
            dtype=x.dtype,
            device=x.device,
        )

    extension = _load_extension()

    return extension.state_aware_causal_conv1d(
        x,
        weight,
        conv_state_pool,
        state_slot_ids,
        gdn_index,
        output,
        block_size,
    )