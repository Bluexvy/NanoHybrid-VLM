"""DFlash v1/v2 adapter, with explicit context-only KV and TP projections.

Checkpoint layout and math follow z-lab/dflash's DFlashDraftModel and
DFlash2DraftModel (MIT; copyright 2026 Z Lab, see third_party/dflash-LICENSE).
No remote Python code is executed. Target embeddings/LM head remain shared.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from nanovllm.models.qwen3_5_mtp import MTPLinear, MTPMLP, _tp_context


def read_dflash_config(path, target_config=None, tp_size=1):
    raw = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
    required = {"hidden_size", "intermediate_size", "num_hidden_layers", "num_target_layers",
                "num_attention_heads", "num_key_value_heads", "rms_norm_eps", "hidden_act",
                "vocab_size", "max_position_embeddings"}
    if required - raw.keys():
        raise ValueError("DFlash configuration is missing: " + ", ".join(sorted(required - raw.keys())))
    dimensions = required - {"rms_norm_eps", "hidden_act"}
    if (type(tp_size) is not int or tp_size < 1
            or any(type(raw[name]) is not int or raw[name] < 1 for name in dimensions)):
        raise ValueError("DFlash dimensions and TP size must be positive integers")
    cfg = SimpleNamespace(**raw)
    extra = raw.get("dflash_config", {})
    architecture = raw.get("architectures")
    if architecture not in (["DFlashDraftModel"], ["DFlash2DraftModel"]) or raw.get("model_type") != "qwen3":
        raise ValueError("Expected a DFlash v1/v2 Qwen3 draft checkpoint, not MTP/EAGLE")
    cfg.is_dflash2 = architecture == ["DFlash2DraftModel"]
    cfg.target_layer_ids = tuple(extra.get("target_layer_ids", raw.get("target_layer_ids", ())))
    cfg.block_size = extra.get("block_size", raw.get("block_size", 16))
    cfg.mask_token_id = extra.get("mask_token_id", raw.get("mask_token_id"))
    cfg.input_embedding_scale = float(extra.get("input_embedding_scale", raw.get("input_embedding_scale", 1.0)))
    cfg.output_multiplier = float(extra.get("output_multiplier", raw.get("output_multiplier", 1.0)))
    cfg.final_logit_softcapping = extra.get("final_logit_softcapping", raw.get("final_logit_softcapping"))
    rope = raw.get("rope_parameters") or raw.get("rope_scaling") or {}
    if rope.get("rope_type", rope.get("type", "default")) != "default":
        raise ValueError("DFlash currently supports default full-head RoPE only")
    cfg.rope_theta = float(rope.get("rope_theta", raw.get("rope_theta", 10000.0)))
    cfg.head_dim = raw.get("head_dim", cfg.hidden_size // cfg.num_attention_heads)
    cfg.attention_bias = raw.get("attention_bias", False)
    cfg.is_causal = raw.get("is_causal")
    cfg.sliding_window = raw.get("sliding_window")
    cfg.layer_types = raw.get("layer_types", ["full_attention"] * cfg.num_hidden_layers)
    if cfg.is_dflash2:
        for name in ("conv_kernel_size", "conv_group_size", "selector_rank", "selector_top_k"):
            value = extra.get(name, raw.get(name))
            if type(value) is not int or value <= 0:
                raise ValueError(f"DFlash2 requires a positive integer {name}")
            setattr(cfg, name, value)
        if (cfg.conv_kernel_size != 2 or cfg.hidden_size % cfg.conv_group_size
                or cfg.selector_top_k > cfg.vocab_size):
            raise ValueError("Unsupported DFlash2 convolution or selector layout")
    if (not cfg.target_layer_ids or len(set(cfg.target_layer_ids)) != len(cfg.target_layer_ids)
            or any(type(i) is not int or not 0 <= i < cfg.num_target_layers for i in cfg.target_layer_ids)
            or len(cfg.layer_types) != cfg.num_hidden_layers
            or any(t not in {"full_attention", "sliding_attention"} for t in cfg.layer_types)
            or cfg.hidden_act != "silu" or cfg.head_dim <= 0 or cfg.head_dim % 2
            or cfg.rope_theta <= 0 or cfg.rms_norm_eps <= 0
            or type(cfg.block_size) is not int or cfg.block_size < 2
            or type(cfg.mask_token_id) is not int or not 0 <= cfg.mask_token_id < cfg.vocab_size
            or cfg.num_attention_heads % cfg.num_key_value_heads
            or any(v % tp_size for v in (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.intermediate_size))):
        raise ValueError("Unsupported or invalid DFlash configuration/head layout")
    if any(t == "sliding_attention" for t in cfg.layer_types) and (
            type(cfg.sliding_window) is not int or cfg.sliding_window <= 0):
        raise ValueError("Sliding DFlash layers require a positive sliding_window")
    if target_config is not None and any(getattr(cfg, name) != getattr(target_config, target_name)
            for name, target_name in (("hidden_size", "hidden_size"), ("vocab_size", "vocab_size"),
                                      ("num_target_layers", "num_hidden_layers"))):
        raise ValueError("DFlash target hidden size, vocabulary or layer count does not match")
    return cfg


def dflash_checkpoint_shapes(config):
    """Global (unsharded) checkpoint shapes, including v2's raw codebooks."""
    cfg = config
    required = {"fc.weight": (cfg.hidden_size, cfg.hidden_size * len(cfg.target_layer_ids)),
                "hidden_norm.weight": (cfg.hidden_size,), "norm.weight": (cfg.hidden_size,)}
    for index in range(cfg.num_hidden_layers):
        prefix = f"layers.{index}."
        attention = {
            "q_proj": (cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size),
            "k_proj": (cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size),
            "v_proj": (cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size),
            "o_proj": (cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim)}
        for name, shape in attention.items():
            required[prefix + f"self_attn.{name}.weight"] = shape
            if cfg.attention_bias:
                required[prefix + f"self_attn.{name}.bias"] = (shape[0],)
        for name in ("q_norm", "k_norm"):
            required[prefix + f"self_attn.{name}.weight"] = (cfg.head_dim,)
        for name in ("input_layernorm", "post_attention_layernorm"):
            required[prefix + f"{name}.weight"] = (cfg.hidden_size,)
        for name in ("gate_proj", "up_proj", "down_proj"):
            required[prefix + f"mlp.{name}.weight"] = ((cfg.hidden_size, cfg.intermediate_size)
                if name == "down_proj" else (cfg.intermediate_size, cfg.hidden_size))
        if getattr(cfg, "is_dflash2", False):
            for name in ("attention_conv", "mlp_conv"):
                required[prefix + name + ".base_kernel"] = (2, cfg.conv_kernel_size, cfg.hidden_size)
                required[prefix + name + ".kernel_projection.weight"] = (
                    2 * cfg.conv_kernel_size * (cfg.hidden_size // cfg.conv_group_size), cfg.hidden_size)
    if getattr(cfg, "is_dflash2", False):
        for name in ("predecessor_codebook", "successor_codebook"):
            required["candidate_selector." + name] = (cfg.vocab_size, cfg.selector_rank)
        required["candidate_selector.hidden_projection.weight"] = (cfg.selector_rank, cfg.hidden_size)
    return required


def validate_dflash_checkpoint(path, config):
    """Inspect local safetensors headers before allocating the target on GPU."""
    from safetensors import safe_open
    required = dflash_checkpoint_shapes(config)
    actual = {}
    unsupported_dtypes = []
    for file in sorted(Path(path).glob("*.safetensors")):
        with safe_open(str(file), framework="pt", device="cpu") as weights:
            for name in weights.keys():
                if name in actual:
                    raise ValueError(f"Duplicate DFlash checkpoint tensor: {name}")
                view = weights.get_slice(name)
                actual[name] = tuple(view.get_shape())
                if view.get_dtype() not in {"F32", "F16", "BF16"}:
                    unsupported_dtypes.append(name)
    missing = sorted(required.keys() - actual.keys())
    extra = sorted(actual.keys() - required.keys())
    wrong = sorted(name for name in required.keys() & actual.keys() if required[name] != actual[name])
    if missing or extra or wrong or unsupported_dtypes:
        raise ValueError(f"Invalid DFlash checkpoint: missing={missing}, unexpected={extra}, "
                         f"wrong_shape={wrong}, unsupported_dtype={unsupported_dtypes}")


class DFlashRMSNorm(nn.Module):
    # Qwen3's standard norm uses weight, unlike Qwen3.5's (1 + weight).
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return self.weight * value.to(x.dtype)


def rotate(x, positions, theta):
    if positions.ndim != 1 or positions.numel() != x.shape[1]:
        raise ValueError("DFlash text positions must match the token axis")
    dim = x.shape[-1]
    freq = 1.0 / theta ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    phase = positions.float()[:, None] * freq[None, :]
    phase = torch.cat((phase, phase), -1)
    first, second = x.chunk(2, -1)
    rotated = torch.cat((-second, first), -1)
    return x * phase.cos().to(x.dtype)[None] + rotated * phase.sin().to(x.dtype)[None]


class DFlashAttention(nn.Module):
    def __init__(self, config, layer, rank, size):
        super().__init__()
        self.heads, self.kv_heads = config.num_attention_heads // size, config.num_key_value_heads // size
        self.dim, self.theta = config.head_dim, config.rope_theta
        args = dict(bias=config.attention_bias, tp_rank=rank, tp_size=size)
        self.q_proj = MTPLinear(config.hidden_size, config.num_attention_heads * self.dim, mode="column", **args)
        self.k_proj = MTPLinear(config.hidden_size, config.num_key_value_heads * self.dim, mode="column", **args)
        self.v_proj = MTPLinear(config.hidden_size, config.num_key_value_heads * self.dim, mode="column", **args)
        self.o_proj = MTPLinear(config.num_attention_heads * self.dim, config.hidden_size, mode="row", **args)
        self.q_norm = DFlashRMSNorm(self.dim, config.rms_norm_eps)
        self.k_norm = DFlashRMSNorm(self.dim, config.rms_norm_eps)
        sliding = config.layer_types[layer] == "sliding_attention"
        self.is_causal = sliding if config.is_causal is None else bool(config.is_causal)
        self.window = config.sliding_window if sliding else None

    def project_kv(self, x, positions):
        k = self.k_norm(self.k_proj(x).view(-1, self.kv_heads, self.dim)).transpose(0, 1)
        v = self.v_proj(x).view(-1, self.kv_heads, self.dim).transpose(0, 1)
        return rotate(k, positions, self.theta), v

    def forward(self, x, positions, context):
        q = self.q_norm(self.q_proj(x).view(-1, self.heads, self.dim)).transpose(0, 1)
        q = rotate(q, positions, self.theta)
        noise_k, noise_v = self.project_kv(x, positions)
        k, v = (torch.cat((old, noise), 1) for old, noise in zip(context, (noise_k, noise_v)))
        # Position-independent masking matches the official cache/query offsets.
        query = context[0].shape[1] + torch.arange(x.shape[0], device=x.device)
        key = torch.arange(k.shape[1], device=x.device)
        mask = torch.ones((x.shape[0], k.shape[1]), dtype=torch.bool, device=x.device)
        if self.is_causal:
            mask &= key[None] <= query[:, None]
        if self.window is not None:
            mask &= query[:, None] - key[None] < self.window
            if not self.is_causal:
                mask &= key[None] - query[:, None] < self.window
        groups = self.heads // self.kv_heads
        # Keep an explicit batch dimension so CUDA can select fused SDPA kernels.
        out = F.scaled_dot_product_attention(q.unsqueeze(0), k.repeat_interleave(groups, 0).unsqueeze(0),
            v.repeat_interleave(groups, 0).unsqueeze(0), attn_mask=mask, scale=self.dim ** -0.5)[0]
        return self.o_proj(out.transpose(0, 1).reshape(x.shape[0], -1))


class DFlashLayer(nn.Module):
    def __init__(self, cfg, index, rank, size):
        super().__init__()
        self.self_attn = DFlashAttention(cfg, index, rank, size)
        self.input_layernorm = DFlashRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = DFlashRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MTPMLP(cfg, rank, size)
        self.attention_conv = self.mlp_conv = None
        if getattr(cfg, "is_dflash2", False):
            self.attention_conv = GroupedDynamicCausalConv(cfg.hidden_size, cfg.conv_kernel_size, cfg.conv_group_size)
            self.mlp_conv = GroupedDynamicCausalConv(cfg.hidden_size, cfg.conv_kernel_size, cfg.conv_group_size)

    def forward(self, x, positions, context):
        residual = x
        x = self.input_layernorm(x)
        if self.attention_conv is not None:
            x, kernel = self.attention_conv.prepare(x)
        x = self.self_attn(x, positions, context)
        if self.attention_conv is not None:
            x = self.attention_conv.finish(x, kernel)
        x = residual + x
        residual = x
        x = self.post_attention_layernorm(x)
        if self.mlp_conv is not None:
            x, kernel = self.mlp_conv.prepare(x)
        x = self.mlp(x)
        if self.mlp_conv is not None:
            x = self.mlp_conv.finish(x, kernel)
        return residual + x


class GroupedDynamicCausalConv(nn.Module):
    """Block-local two-tap convolution; no persistent GDN/conv state.

    Kept replicated across TP ranks: input and row-parallel outputs are full
    hidden vectors. The dynamic post kernel is computed from the PRE input.
    """
    def __init__(self, hidden_size, kernel_size, group_size):
        super().__init__()
        self.kernel_size, self.group_size = kernel_size, group_size
        self.base_kernel = nn.Parameter(torch.zeros(2, kernel_size, hidden_size))
        self.kernel_projection = nn.Linear(hidden_size, 2 * kernel_size * (hidden_size // group_size), bias=False)

    def convolve(self, hidden, dynamic, base):
        length, width = hidden.shape
        groups = width // self.group_size
        blocks = hidden.reshape(length, groups, self.group_size)
        dynamic = dynamic.reshape(length, self.kernel_size, groups, 1)
        result = torch.zeros_like(blocks)
        for offset in range(self.kernel_size):
            values = blocks if offset == 0 else F.pad(blocks[:-offset], (0, 0, 0, 0, offset, 0))
            result = result + base[offset].reshape(1, groups, self.group_size).to(hidden.dtype) * values
            result = torch.addcmul(result, dynamic[:, offset], values)
        return result.reshape_as(hidden)

    def prepare(self, hidden):
        dynamic = self.kernel_projection(hidden).reshape(hidden.shape[0], 2, self.kernel_size, -1)
        return self.convolve(hidden, dynamic[:, 0], self.base_kernel[0]), dynamic[:, 1]

    def finish(self, hidden, dynamic):
        return self.convolve(hidden, dynamic, self.base_kernel[1])


class CandidateSelector(nn.Module):
    """Rank-0 conditional path selection with exact q rows for rejection.

    Raw Parameter names match released checkpoints (no '.weight' suffix).
    """
    def __init__(self, cfg):
        super().__init__()
        self.top_k = cfg.selector_top_k
        self.predecessor_codebook = nn.Parameter(torch.empty(cfg.vocab_size, cfg.selector_rank))
        self.successor_codebook = nn.Parameter(torch.empty(cfg.vocab_size, cfg.selector_rank))
        nn.init.normal_(self.predecessor_codebook, std=.02)
        nn.init.normal_(self.successor_codebook, std=.02)
        self.hidden_projection = nn.Linear(cfg.hidden_size, cfg.selector_rank, bias=False)

    def select(self, hidden, logits, anchor, temperature):
        if temperature < 0 or hidden.shape[0] != logits.shape[0] or logits.shape[-1] < self.top_k:
            raise ValueError("Invalid DFlash2 selector input")
        unary, candidates = torch.topk(logits, self.top_k, dim=-1, sorted=False)
        gates = self.hidden_projection(hidden)
        previous = torch.as_tensor(anchor, dtype=torch.long, device=hidden.device).reshape(())
        tokens, distributions = [], []
        for position in range(hidden.shape[0]):
            predecessor = F.embedding(previous, self.predecessor_codebook) * gates[position]
            successor = F.embedding(candidates[position], self.successor_codebook)
            scores = unary[position] + torch.einsum("r,kr->k", predecessor, successor)
            if temperature == 0:
                selected = scores.argmax(-1)
            else:
                probabilities = torch.softmax(scores.float() / temperature, -1)
                selected = torch.multinomial(probabilities, 1).reshape(())
                # q depends on the selected predecessor. Outside Top-K q=0,
                # which is required by the common verifier's (p-q)+ residual.
                distributions.append(torch.zeros(logits.shape[-1], device=logits.device,
                    dtype=torch.float32).scatter(0, candidates[position], probabilities))
            previous = candidates[position, selected]
            tokens.append(previous)
        return torch.stack(tokens).tolist(), distributions if temperature else [None] * len(tokens)


class DFlashDraftModel(nn.Module):
    packed_modules_mapping = {"gate_proj": ("gate_up_proj", 0), "up_proj": ("gate_up_proj", 1)}

    def __init__(self, config, tp_rank=None, tp_size=None):
        super().__init__()
        self.config = config
        rank, size = _tp_context(tp_rank, tp_size)
        self.fc = nn.Linear(len(config.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = DFlashRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm = DFlashRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.layers = nn.ModuleList([DFlashLayer(config, i, rank, size) for i in range(config.num_hidden_layers)])
        if getattr(config, "is_dflash2", False):
            self.candidate_selector = CandidateSelector(config)

    def append_context(self, features, positions, past=None):
        """Commit VERIFIED target features only. Draft/noise KV never persists."""
        if features.shape != (positions.numel(), self.fc.in_features):
            raise ValueError("DFlash target feature shape does not match selected layers")
        context = self.hidden_norm(self.fc(features))
        result = []
        for index, layer in enumerate(self.layers):
            kv = layer.self_attn.project_kv(context, positions)
            if past is not None:
                kv = tuple(torch.cat((old, new), 1) for old, new in zip(past[index], kv))
            result.append(kv)
        return tuple(result)

    def forward(self, embeddings, positions, context):
        if len(context) != len(self.layers) or positions.numel() != embeddings.shape[0]:
            raise ValueError("Invalid DFlash block or context layer count")
        x = embeddings
        for layer, kv in zip(self.layers, context):
            x = layer(x, positions, kv)
        return self.norm(x)

    def transform_logits(self, logits):
        logits = logits * self.config.output_multiplier
        cap = self.config.final_logit_softcapping
        return torch.tanh(logits / float(cap)) * float(cap) if cap is not None and float(cap) > 0 else logits
