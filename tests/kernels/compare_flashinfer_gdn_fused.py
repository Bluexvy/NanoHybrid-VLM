from __future__ import annotations

"""
Compare NanoHybrid-VLM's state-aware GDN decode composition with
FlashInfer's serving-native fused GDN decode step without modifying engine
source code.

The timed boundary is deliberately the same for both implementations:

    hidden_states + raw mixed_qkv + weights + state pools
        -> b/a projection
        -> causal conv update + SiLU
        -> q/k/v split + q/k L2 normalization
        -> recurrent gated-delta-rule update
        -> core output + in-place state updates

One-time weight packing, state-layout conversion, extension/JIT loading and
CUDA graph capture are outside the measured region.

Examples:

    python tests/kernels/compare_flashinfer_gdn_fused.py --probe-only
    python tests/kernels/compare_flashinfer_gdn_fused.py
    python tests/kernels/compare_flashinfer_gdn_fused.py \
        --batch-sizes 1 2 4 8 16 --allow-composable-fallback
    python tests/kernels/compare_flashinfer_gdn_fused.py --with-cuda-graph
"""

import argparse
import importlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from nanovllm.kernels.state_aware_gdn_cuda import (
    load_state_aware_gdn_cuda_extension,
    state_aware_causal_conv1d_cuda,
    state_aware_gdn_decode_cuda,
)


DEFAULT_BATCH_SIZES = (1, 2, 4, 8)
DEFAULT_OUTPUT_PATH = Path(
    "/workspace/nano-vllm/artifacts/kernels/flashinfer_gdn_comparison.json"
)


@dataclass(frozen=True)
class GDNGeometry:
    """Qwen3.5-9B GDN geometry used by the local CUDA kernels."""

    hidden_size: int = 4096
    num_qk_heads: int = 16
    num_v_heads: int = 32
    head_dim: int = 128
    conv_width: int = 4
    num_gdn_layers: int = 2
    gdn_index: int = 1

    @property
    def qkv_dim(self) -> int:
        return (
            2 * self.num_qk_heads + self.num_v_heads
        ) * self.head_dim

    @property
    def n_ba(self) -> int:
        return 2 * self.num_v_heads

    @property
    def conv_state_len(self) -> int:
        return self.conv_width - 1

    @property
    def qk_repeat(self) -> int:
        if self.num_v_heads % self.num_qk_heads != 0:
            raise ValueError(
                "num_v_heads must be divisible by num_qk_heads"
            )
        return self.num_v_heads // self.num_qk_heads


@dataclass
class CommonInputs:
    hidden_states: torch.Tensor
    w_ba: torch.Tensor
    mixed_qkv: torch.Tensor
    conv_weight: torch.Tensor
    conv_bias: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    slot_ids_i64: torch.Tensor
    slot_ids_i32: torch.Tensor
    initial_conv_history: torch.Tensor
    initial_ssm_kv: torch.Tensor


@dataclass
class NanoBuffers:
    conv_pool: torch.Tensor
    ssm_pool: torch.Tensor
    conv_output: torch.Tensor
    recurrent_output: torch.Tensor


@dataclass
class FlashInferBuffers:
    conv_pool: torch.Tensor
    ssm_pool: torch.Tensor
    output: torch.Tensor


def import_flashinfer():
    try:
        return importlib.import_module("flashinfer")
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "FlashInfer is unavailable in this Python environment. "
            "Install a version compatible with this PyTorch/CUDA stack, "
            "then rerun this script. Original import error: "
            f"{error}"
        ) from error


def make_common_inputs(
    batch_size: int,
    geometry: GDNGeometry,
    seed: int,
) -> CommonInputs:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_slots = batch_size + 4

    hidden_states = (
        torch.randn(
            batch_size,
            geometry.hidden_size,
            device=device,
            dtype=dtype,
        )
        * 0.02
    )

    # FlashInfer consumes [hidden, 2*HV]. The local PyTorch linears are
    # mathematically equivalent to hidden_states @ w_ba. Weight packing is a
    # model-load-time operation and is intentionally outside the timer.
    w_ba = (
        torch.randn(
            geometry.hidden_size,
            geometry.n_ba,
            device=device,
            dtype=dtype,
        )
        * 0.02
    )

    mixed_qkv = (
        torch.randn(
            batch_size,
            geometry.qkv_dim,
            device=device,
            dtype=dtype,
        )
        * 0.02
    )

    conv_weight = (
        torch.randn(
            geometry.qkv_dim,
            geometry.conv_width,
            device=device,
            dtype=dtype,
        )
        * 0.02
    )

    # The local Qwen3.5 layer uses bias=False. FlashInfer's fused API accepts
    # a bias tensor, so zeros preserve the same equation.
    conv_bias = torch.zeros(
        geometry.qkv_dim,
        device=device,
        dtype=dtype,
    )

    initial_a = torch.empty(
        geometry.num_v_heads,
        device=device,
        dtype=torch.float32,
    ).uniform_(0.01, 1.0)
    a_log = torch.log(initial_a)

    dt_bias = torch.ones(
        geometry.num_v_heads,
        device=device,
        dtype=dtype,
    )

    # Deliberately non-identity slot order to exercise dynamic pool lookup.
    slot_ids_i64 = (
        torch.arange(
            batch_size - 1,
            -1,
            -1,
            device=device,
            dtype=torch.int64,
        )
        + 2
    ).contiguous()
    slot_ids_i32 = slot_ids_i64.to(torch.int32)

    # Common logical history: the last width-1 raw conv inputs.
    initial_conv_history = (
        torch.randn(
            num_slots,
            geometry.num_gdn_layers,
            geometry.qkv_dim,
            geometry.conv_state_len,
            device=device,
            dtype=dtype,
        )
        * 0.02
    )

    # Common logical recurrent state in the local K-major [K,V] layout.
    initial_ssm_kv = (
        torch.randn(
            num_slots,
            geometry.num_gdn_layers,
            geometry.num_v_heads,
            geometry.head_dim,
            geometry.head_dim,
            device=device,
            dtype=torch.float32,
        )
        * 0.001
    )

    return CommonInputs(
        hidden_states=hidden_states.contiguous(),
        w_ba=w_ba.contiguous(),
        mixed_qkv=mixed_qkv.contiguous(),
        conv_weight=conv_weight.contiguous(),
        conv_bias=conv_bias,
        a_log=a_log.contiguous(),
        dt_bias=dt_bias.contiguous(),
        slot_ids_i64=slot_ids_i64,
        slot_ids_i32=slot_ids_i32,
        initial_conv_history=initial_conv_history.contiguous(),
        initial_ssm_kv=initial_ssm_kv.contiguous(),
    )


def make_nano_buffers(
    common: CommonInputs,
    geometry: GDNGeometry,
) -> NanoBuffers:
    batch_size = common.hidden_states.shape[0]

    # The local kernel keeps four entries and, on every decode step, changes
    # [old0, h0, h1, h2] into [h0, h1, h2, current]. FlashInfer stores only
    # [h0, h1, h2]. The leading value is therefore irrelevant to the first
    # update; zero makes this representation explicit.
    conv_pool = torch.zeros(
        *common.initial_conv_history.shape[:-1],
        geometry.conv_width,
        device="cuda",
        dtype=torch.bfloat16,
    )
    conv_pool[..., 1:].copy_(common.initial_conv_history)

    return NanoBuffers(
        conv_pool=conv_pool.contiguous(),
        ssm_pool=common.initial_ssm_kv.clone(),
        conv_output=torch.empty(
            batch_size,
            geometry.qkv_dim,
            1,
            device="cuda",
            dtype=torch.bfloat16,
        ),
        recurrent_output=torch.empty(
            batch_size,
            1,
            geometry.num_v_heads,
            geometry.head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        ),
    )


def make_flashinfer_buffers(
    common: CommonInputs,
    geometry: GDNGeometry,
) -> FlashInferBuffers:
    batch_size = common.hidden_states.shape[0]

    # FlashInfer's DS conv pool is [P,C,width-1] for one layer.
    conv_pool = common.initial_conv_history[
        :, geometry.gdn_index
    ].clone()

    # FlashInfer's fused API uses V-major/K-last [P,H,V,K]. Convert once at
    # backend initialization, never inside the timed decode step.
    ssm_pool = (
        common.initial_ssm_kv[:, geometry.gdn_index]
        .transpose(-1, -2)
        .contiguous()
    )

    return FlashInferBuffers(
        conv_pool=conv_pool,
        ssm_pool=ssm_pool,
        output=torch.empty(
            batch_size,
            1,
            geometry.num_v_heads,
            geometry.head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        ),
    )


@torch.inference_mode()
def run_nano_step(
    common: CommonInputs,
    buffers: NanoBuffers,
    geometry: GDNGeometry,
) -> torch.Tensor:
    # Same b/a projection boundary consumed by FlashInfer fused decode.
    ba = torch.matmul(
        common.hidden_states,
        common.w_ba,
    )
    b, a = ba.split(
        geometry.num_v_heads,
        dim=-1,
    )

    beta = torch.sigmoid(b).unsqueeze(1)
    g = (
        -common.a_log.float().exp()
        * torch.nn.functional.softplus(
            a.float() + common.dt_bias.float()
        )
    ).unsqueeze(1)

    conv_output = state_aware_causal_conv1d_cuda(
        x=common.mixed_qkv.unsqueeze(-1),
        weight=common.conv_weight,
        conv_state_pool=buffers.conv_pool,
        state_slot_ids=common.slot_ids_i64,
        gdn_index=geometry.gdn_index,
        output=buffers.conv_output,
        block_size=256,
    )

    q_width = geometry.num_qk_heads * geometry.head_dim
    v_width = geometry.num_v_heads * geometry.head_dim

    query_flat, key_flat, value_flat = torch.split(
        conv_output.squeeze(-1),
        (q_width, q_width, v_width),
        dim=-1,
    )

    query = query_flat.reshape(
        common.hidden_states.shape[0],
        1,
        geometry.num_qk_heads,
        geometry.head_dim,
    )
    key = key_flat.reshape_as(query)
    value = value_flat.reshape(
        common.hidden_states.shape[0],
        1,
        geometry.num_v_heads,
        geometry.head_dim,
    )

    if geometry.qk_repeat > 1:
        query = query.repeat_interleave(
            geometry.qk_repeat,
            dim=2,
        )
        key = key.repeat_interleave(
            geometry.qk_repeat,
            dim=2,
        )

    return state_aware_gdn_decode_cuda(
        query=query,
        key=key,
        value=value,
        g=g.contiguous(),
        beta=beta.contiguous(),
        recurrent_state_pool=buffers.ssm_pool,
        state_slot_ids=common.slot_ids_i64,
        gdn_index=geometry.gdn_index,
        output=buffers.recurrent_output,
        scale=geometry.head_dim**-0.5,
    )


@torch.inference_mode()
def run_flashinfer_step(
    flashinfer,
    common: CommonInputs,
    buffers: FlashInferBuffers,
    geometry: GDNGeometry,
) -> torch.Tensor:
    output, _, _ = flashinfer.gdn_fused_decode_step(
        common.hidden_states,
        common.w_ba,
        common.mixed_qkv,
        common.conv_weight,
        common.conv_bias,
        buffers.conv_pool,
        common.a_log,
        common.dt_bias,
        geometry.head_dim**-0.5,
        buffers.ssm_pool,
        common.slot_ids_i32,
        use_qk_l2norm=True,
        out=buffers.output,
    )
    return output


def flashinfer_specialized_supported(
    flashinfer,
    batch_size: int,
    geometry: GDNGeometry,
) -> bool:
    return bool(
        flashinfer.gdn_fused_decode_step_supported(
            batch_size=batch_size,
            hidden_size=geometry.hidden_size,
            n_ba=geometry.n_ba,
            qkv_dim=geometry.qkv_dim,
            num_qk_heads=geometry.num_qk_heads,
            num_v_heads=geometry.num_v_heads,
            head_dim=geometry.head_dim,
            conv_width=geometry.conv_width,
            conv_state_len=geometry.conv_state_len,
            device=torch.device("cuda"),
            conv_state_layout="DS",
        )
    )


def tensor_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, float]:
    difference = actual.float() - expected.float()
    absolute = difference.abs()
    denominator = expected.float().abs().clamp_min(1e-6)
    return {
        "max_abs": float(absolute.max().item()),
        "mean_abs": float(absolute.mean().item()),
        "max_rel": float((absolute / denominator).max().item()),
    }


def assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, float]:
    error = tensor_error(actual, expected)
    try:
        torch.testing.assert_close(
            actual.float(),
            expected.float(),
            atol=atol,
            rtol=rtol,
        )
    except AssertionError as exception:
        raise AssertionError(
            f"{name} mismatch: {error}; atol={atol}, rtol={rtol}"
        ) from exception
    return error


@torch.inference_mode()
def run_correctness_case(
    flashinfer,
    batch_size: int,
    geometry: GDNGeometry,
    decode_steps: int,
) -> dict[str, object]:
    common = make_common_inputs(
        batch_size=batch_size,
        geometry=geometry,
        seed=20260914 + batch_size,
    )
    nano = make_nano_buffers(common, geometry)
    flash = make_flashinfer_buffers(common, geometry)

    nano_output = None
    flash_output = None

    for _ in range(decode_steps):
        nano_output = run_nano_step(
            common,
            nano,
            geometry,
        )
        flash_output = run_flashinfer_step(
            flashinfer,
            common,
            flash,
            geometry,
        )

    torch.cuda.synchronize()
    assert nano_output is not None
    assert flash_output is not None

    selected = common.slot_ids_i64

    nano_ssm_selected = nano.ssm_pool[
        selected,
        geometry.gdn_index,
    ]
    flash_ssm_selected_kv = flash.ssm_pool[
        common.slot_ids_i32.long()
    ].transpose(-1, -2)

    nano_conv_history = nano.conv_pool[
        selected,
        geometry.gdn_index,
        :,
        1:,
    ]
    flash_conv_history = flash.conv_pool[
        common.slot_ids_i32.long()
    ]

    output_error = assert_close(
        "core output",
        nano_output,
        flash_output,
        atol=2e-2,
        rtol=2e-2,
    )
    ssm_error = assert_close(
        "recurrent state",
        nano_ssm_selected,
        flash_ssm_selected_kv,
        atol=5e-4,
        rtol=5e-3,
    )
    conv_error = assert_close(
        "conv state",
        nano_conv_history,
        flash_conv_history,
        atol=2e-2,
        rtol=2e-2,
    )

    return {
        "decode_steps": decode_steps,
        "output_error": output_error,
        "ssm_state_error": ssm_error,
        "conv_state_error": conv_error,
    }


def measure_cuda_us(
    function: Callable[[], object],
    *,
    warmup_steps: int,
    measured_steps: int,
    repeats: int,
) -> dict[str, object]:
    for _ in range(warmup_steps):
        function()
    torch.cuda.synchronize()

    samples_us: list[float] = []

    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(measured_steps):
            function()
        end.record()
        end.synchronize()

        samples_us.append(
            start.elapsed_time(end)
            * 1000.0
            / measured_steps
        )

    ordered = sorted(samples_us)
    p99_index = min(
        len(ordered) - 1,
        max(0, int(0.99 * len(ordered))),
    )
    return {
        "median_us": statistics.median(samples_us),
        "min_us": min(samples_us),
        "p99_repeat_us": ordered[p99_index],
        "samples_us": samples_us,
    }


def capture_cuda_graph(
    function: Callable[[], object],
) -> tuple[torch.cuda.CUDAGraph, Callable[[], None]]:
    # All lazy module/extension specialization must happen before capture.
    for _ in range(3):
        function()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        function()

    return graph, graph.replay


@torch.inference_mode()
def run_benchmark_case(
    flashinfer,
    batch_size: int,
    geometry: GDNGeometry,
    warmup_steps: int,
    measured_steps: int,
    repeats: int,
    with_cuda_graph: bool,
) -> dict[str, object]:
    common = make_common_inputs(
        batch_size=batch_size,
        geometry=geometry,
        seed=20261000 + batch_size,
    )
    nano = make_nano_buffers(common, geometry)
    flash = make_flashinfer_buffers(common, geometry)

    nano_call = lambda: run_nano_step(
        common,
        nano,
        geometry,
    )
    flash_call = lambda: run_flashinfer_step(
        flashinfer,
        common,
        flash,
        geometry,
    )

    # Warm both implementations before either is measured. This excludes
    # PyTorch extension loading and FlashInfer lazy specialization/JIT.
    for _ in range(warmup_steps):
        nano_call()
        flash_call()
    torch.cuda.synchronize()

    eager_nano = measure_cuda_us(
        nano_call,
        warmup_steps=0,
        measured_steps=measured_steps,
        repeats=repeats,
    )
    eager_flash = measure_cuda_us(
        flash_call,
        warmup_steps=0,
        measured_steps=measured_steps,
        repeats=repeats,
    )

    result: dict[str, object] = {
        "eager": {
            "nano": eager_nano,
            "flashinfer": eager_flash,
            "nano_over_flashinfer": (
                eager_nano["median_us"]
                / eager_flash["median_us"]
            ),
        }
    }

    if with_cuda_graph:
        _, nano_replay = capture_cuda_graph(nano_call)
        _, flash_replay = capture_cuda_graph(flash_call)

        graph_nano = measure_cuda_us(
            nano_replay,
            warmup_steps=3,
            measured_steps=measured_steps,
            repeats=repeats,
        )
        graph_flash = measure_cuda_us(
            flash_replay,
            warmup_steps=3,
            measured_steps=measured_steps,
            repeats=repeats,
        )

        result["cuda_graph"] = {
            "nano": graph_nano,
            "flashinfer": graph_flash,
            "nano_over_flashinfer": (
                graph_nano["median_us"]
                / graph_flash["median_us"]
            ),
        }

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Correctness and equal-boundary performance comparison between "
            "NanoHybrid-VLM state-aware GDN and FlashInfer fused GDN decode."
        )
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
    )
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--measured-steps",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--with-cuda-graph",
        action="store_true",
    )
    parser.add_argument(
        "--allow-composable-fallback",
        action="store_true",
        help=(
            "Benchmark FlashInfer even when its support probe says the "
            "specialized fully-fused kernel is unavailable. Results are "
            "then explicitly labeled composable_fallback."
        ),
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This comparison requires a CUDA GPU")

    flashinfer = import_flashinfer()
    geometry = GDNGeometry()

    # Compile/load the local CUDA extension before all measurements.
    load_state_aware_gdn_cuda_extension()

    flashinfer_version = getattr(
        flashinfer,
        "__version__",
        "unknown",
    )
    gpu_name = torch.cuda.get_device_name()

    print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA: {torch.version.cuda}")
    print(f"FlashInfer: {flashinfer_version}")
    print(
        "Geometry: "
        f"hidden={geometry.hidden_size}, "
        f"Hqk={geometry.num_qk_heads}, "
        f"Hv={geometry.num_v_heads}, "
        f"D={geometry.head_dim}, "
        f"qkv_dim={geometry.qkv_dim}, "
        f"conv_width={geometry.conv_width}"
    )

    payload: dict[str, object] = {
        "environment": {
            "gpu": gpu_name,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "flashinfer": flashinfer_version,
        },
        "geometry": {
            "hidden_size": geometry.hidden_size,
            "num_qk_heads": geometry.num_qk_heads,
            "num_v_heads": geometry.num_v_heads,
            "head_dim": geometry.head_dim,
            "qkv_dim": geometry.qkv_dim,
            "n_ba": geometry.n_ba,
            "conv_width": geometry.conv_width,
        },
        "cases": {},
    }

    for batch_size in args.batch_sizes:
        supported = flashinfer_specialized_supported(
            flashinfer,
            batch_size,
            geometry,
        )
        route = (
            "specialized_fused"
            if supported
            else "composable_fallback"
        )

        print(
            f"\nB={batch_size}: FlashInfer route={route}"
        )

        case: dict[str, object] = {
            "flashinfer_route": route,
        }

        if not args.probe_only:
            correctness = run_correctness_case(
                flashinfer,
                batch_size,
                geometry,
                args.decode_steps,
            )
            case["correctness"] = correctness
            print(
                "  correctness passed: "
                f"output max_abs="
                f"{correctness['output_error']['max_abs']:.6e}, "
                f"state max_abs="
                f"{correctness['ssm_state_error']['max_abs']:.6e}, "
                f"conv max_abs="
                f"{correctness['conv_state_error']['max_abs']:.6e}"
            )

            if supported or args.allow_composable_fallback:
                benchmark = run_benchmark_case(
                    flashinfer,
                    batch_size,
                    geometry,
                    args.warmup_steps,
                    args.measured_steps,
                    args.repeats,
                    args.with_cuda_graph,
                )
                case["benchmark"] = benchmark

                eager = benchmark["eager"]
                print(
                    "  eager median: "
                    f"nano={eager['nano']['median_us']:.3f} us, "
                    f"flashinfer={eager['flashinfer']['median_us']:.3f} us, "
                    f"nano/flashinfer={eager['nano_over_flashinfer']:.3f}x"
                )

                if "cuda_graph" in benchmark:
                    graph = benchmark["cuda_graph"]
                    print(
                        "  graph median: "
                        f"nano={graph['nano']['median_us']:.3f} us, "
                        f"flashinfer={graph['flashinfer']['median_us']:.3f} us, "
                        f"nano/flashinfer={graph['nano_over_flashinfer']:.3f}x"
                    )
            else:
                case["benchmark_skipped"] = (
                    "FlashInfer specialized fused kernel is unsupported for "
                    "this geometry. Pass --allow-composable-fallback to time "
                    "the fallback with an explicit label."
                )
                print("  performance skipped: specialized kernel unsupported")

        payload["cases"][str(batch_size)] = case

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
