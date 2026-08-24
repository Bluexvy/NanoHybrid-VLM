import torch
import torch.nn.functional as F

from transformers import AutoModelForMultimodalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    torch_chunk_gated_delta_rule,
)

from nanovllm.layers.gated_delta_net import (
    Qwen3_5GatedDeltaNet,
)


MODEL_ID = "Qwen/Qwen3.5-0.8B"
DEVICE = "cuda"
DTYPE = torch.bfloat16


LAYER_IDX = 0
BATCH_SIZE = 1
SEQUENCE_LENGTH = 7
SEED = 2026

def compare_outputs(
    name: str,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    max_abs_tolerance: float,
):
    reference_fp32 = reference.float()
    candidate_fp32 = candidate.float()

    difference = (
        reference_fp32
        - candidate_fp32
    ).abs()

    max_abs_error = difference.max().item()
    mean_abs_error = difference.mean().item()

    rmse = (
        difference
        .pow(2)
        .mean()
        .sqrt()
        .item()
    )

    cosine_similarity = F.cosine_similarity(
        reference_fp32.flatten(),
        candidate_fp32.flatten(),
        dim=0,
    ).item()

    print(f"\n=== {name} ===")
    print("shape:", tuple(reference.shape))
    print("max absolute error:", max_abs_error)
    print("mean absolute error:", mean_abs_error)
    print("RMSE:", rmse)
    print("cosine similarity:", cosine_similarity)

    if max_abs_error > max_abs_tolerance:
        raise AssertionError(
            f"{name}: max absolute error "
            f"{max_abs_error} exceeds "
            f"{max_abs_tolerance}"
        )

    if cosine_similarity < 0.999:
        raise AssertionError(
            f"{name}: cosine similarity "
            f"{cosine_similarity} is too low"
        )
        
def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This probe requires a CUDA GPU"
        )

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print("loading official model...")

    official_model = (
        AutoModelForMultimodalLM
        .from_pretrained(
            MODEL_ID,
            dtype=DTYPE,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        .eval()
        .to(DEVICE)
    )

    text_config = (
        official_model.config.text_config
    )

    print(
        "layer type:",
        text_config.layer_types[LAYER_IDX],
    )

    if (
        text_config.layer_types[LAYER_IDX]
        != "linear_attention"
    ):
        raise ValueError(
            f"Layer {LAYER_IDX} is not a GDN layer"
        )
    official_layer = (
        official_model
        .model
        .language_model
        .layers[LAYER_IDX]
        .linear_attn
    )

    print(
        "official layer class:",
        type(official_layer).__name__,
    )

    print(
        "official causal conv function:",
        official_layer.causal_conv1d_fn,
    )

    print(
        "official chunk delta function:",
        official_layer.chunk_gated_delta_rule,
    )
    nano_layer = Qwen3_5GatedDeltaNet(
        text_config,
        layer_idx=LAYER_IDX,
        backend="fla"
    )

    nano_layer = (
        nano_layer
        .eval()
        .to(
            device=DEVICE,
            dtype=DTYPE,
        )
    )

    load_result = nano_layer.load_state_dict(
        official_layer.state_dict(),
        strict=True,
    )

    print(
        "missing parameter names:",
        load_result.missing_keys,
    )

    print(
        "unexpected parameter names:",
        load_result.unexpected_keys,
    )
    hidden_states = torch.randn(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        text_config.hidden_size,
        device=DEVICE,
        dtype=DTYPE,
    )

    print(
        "\nhidden states:",
        tuple(hidden_states.shape),
        hidden_states.dtype,
        hidden_states.device,
    )
    with torch.inference_mode():
        official_fast_output = (
            official_layer(hidden_states)
        )

        (
            nano_output,
            nano_conv_state,
            nano_recurrent_state,
        ) = nano_layer(hidden_states)

    torch.cuda.synchronize()
    
    official_layer.causal_conv1d_fn = None

    official_layer.chunk_gated_delta_rule = (
        torch_chunk_gated_delta_rule
    )

    with torch.inference_mode():
        official_torch_output = (
            official_layer(hidden_states)
        )

    torch.cuda.synchronize()
    
    print("\n=== Nano states ===")

    print(
        "conv state:",
        tuple(nano_conv_state.shape),
        nano_conv_state.dtype,
    )

    print(
        "recurrent state:",
        tuple(nano_recurrent_state.shape),
        nano_recurrent_state.dtype,
    )

    expected_conv_shape = (
        BATCH_SIZE,
        nano_layer.conv_dim,
        nano_layer.conv_kernel_size,
    )

    expected_recurrent_shape = (
        BATCH_SIZE,
        nano_layer.num_v_heads,
        nano_layer.head_k_dim,
        nano_layer.head_v_dim,
    )

    if (
        nano_conv_state.shape
        != expected_conv_shape
    ):
        raise AssertionError(
            "Unexpected conv state shape"
        )

    if (
        nano_recurrent_state.shape
        != expected_recurrent_shape
    ):
        raise AssertionError(
            "Unexpected recurrent state shape"
        )

    if (
        nano_recurrent_state.dtype
        != torch.float32
    ):
        raise AssertionError(
            "Recurrent state must be FP32"
        )

    compare_outputs(
        name="HF torch chunk vs nano recurrent",
        reference=official_torch_output,
        candidate=nano_output,
        max_abs_tolerance=0.002,
    )

    compare_outputs(
        name="HF fast FLA vs nano recurrent",
        reference=official_fast_output,
        candidate=nano_output,
        max_abs_tolerance=0.005,
    )

    compare_outputs(
        name="HF fast FLA vs HF torch chunk",
        reference=official_fast_output,
        candidate=official_torch_output,
        max_abs_tolerance=0.005,
    )

    print("\nGDN layer alignment passed.")
    
    
if __name__ == "__main__":
    main()