"""Dense Qwen3.5 MTP head with tensor-parallel projections.

Embedding and LM head are shared with the target, not duplicated here.
Architecture reference: vLLM qwen3_5_mtp.py; this backend deliberately uses
PyTorch SDPA instead of the target's paged attention and CUDA graphs.
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def _tp_context(rank=None, size=None):
    if rank is None:
        rank = dist.get_rank() if dist.is_initialized() else 0
    if size is None:
        size = dist.get_world_size() if dist.is_initialized() else 1
    if size <= 0 or not 0 <= rank < size:
        raise ValueError("Invalid MTP tensor-parallel rank")
    return rank, size


class MTPLinear(nn.Linear):
    """Shard column outputs / row inputs, retaining checkpoint parameter names."""
    def __init__(self, in_features, out_features, bias=False, *, mode="replicated",
                 tp_rank=0, tp_size=1):
        if mode not in {"replicated", "column", "row"}:
            raise ValueError("Invalid MTP projection mode")
        dimension = out_features if mode == "column" else in_features
        if mode != "replicated" and dimension % tp_size:
            raise ValueError("MTP projection dimension must divide TP size")
        super().__init__(in_features // tp_size if mode == "row" else in_features,
                         out_features // tp_size if mode == "column" else out_features, bias)
        self.mode, self.tp_rank, self.tp_size = mode, tp_rank, tp_size
        self.weight.weight_loader = self.load_weight
        if self.bias is not None:
            self.bias.weight_loader = self.load_weight

    def load_weight(self, parameter, value):
        if self.mode == "column":
            value = value.chunk(self.tp_size, dim=0)[self.tp_rank]
        elif self.mode == "row" and value.ndim == 2:
            value = value.chunk(self.tp_size, dim=1)[self.tp_rank]
        if value.shape != parameter.shape:
            raise ValueError("MTP checkpoint projection shape does not match shard")
        parameter.data.copy_(value)

    def forward(self, x):
        bias = self.bias if self.mode != "row" or self.tp_rank == 0 else None
        result = F.linear(x, self.weight, bias)
        if self.mode == "row" and self.tp_size > 1:
            dist.all_reduce(result)
        return result


class MTPRMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return (value * (1 + self.weight.float())).to(x.dtype)


class MTPMLP(nn.Module):
    def __init__(self, config, tp_rank=0, tp_size=1):
        super().__init__()
        size = config.intermediate_size
        if size % tp_size:
            raise ValueError("MTP intermediate_size must divide TP size")
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * size // tp_size, bias=False)
        self.down_proj = MTPLinear(size, config.hidden_size, mode="row",
                                  tp_rank=tp_rank, tp_size=tp_size)

        def load_shard(param, weight, shard_id):
            if shard_id not in (0, 1) or weight.shape != (size, config.hidden_size):
                raise ValueError("Invalid MTP gate/up weight shard")
            local_size = size // tp_size
            param.data[shard_id * local_size:(shard_id + 1) * local_size].copy_(
                weight.chunk(tp_size, dim=0)[tp_rank])

        self.gate_up_proj.weight.weight_loader = load_shard

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, -1)
        return self.down_proj(F.silu(gate) * up)


class MTPAttention(nn.Module):
    def __init__(self, config, tp_rank=0, tp_size=1):
        super().__init__()
        total_heads, total_kv_heads = config.num_attention_heads, config.num_key_value_heads
        if total_heads % tp_size or total_kv_heads % tp_size:
            raise ValueError("MTP Q/KV heads must divide TP size")
        self.heads, self.kv_heads = total_heads // tp_size, total_kv_heads // tp_size
        self.dim = getattr(config, "head_dim", config.hidden_size // total_heads)
        if self.heads % self.kv_heads:
            raise ValueError("MTP query heads must be divisible by KV heads")
        bias = getattr(config, "attention_bias", False)
        arguments = dict(tp_rank=tp_rank, tp_size=tp_size)
        self.q_proj = MTPLinear(config.hidden_size, 2 * total_heads * self.dim,
                                bias=bias, mode="column", **arguments)
        self.k_proj = MTPLinear(config.hidden_size, total_kv_heads * self.dim,
                                bias=bias, mode="column", **arguments)
        self.v_proj = MTPLinear(config.hidden_size, total_kv_heads * self.dim,
                                bias=bias, mode="column", **arguments)
        self.o_proj = MTPLinear(total_heads * self.dim, config.hidden_size,
                                bias=bias, mode="row", **arguments)
        self.q_norm = MTPRMSNorm(self.dim, config.rms_norm_eps)
        self.k_norm = MTPRMSNorm(self.dim, config.rms_norm_eps)

    def forward(self, positions, x, past, rotary_emb):
        count = x.shape[0]
        q, gate = self.q_proj(x).view(count, self.heads, 2 * self.dim).chunk(2, -1)
        k = self.k_proj(x).view(count, self.kv_heads, self.dim)
        v = self.v_proj(x).view(count, self.kv_heads, self.dim)
        q, k = rotary_emb(positions, self.q_norm(q), self.k_norm(k))
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        past_length = 0 if past is None else past[0].shape[1]
        if past is not None:
            k = torch.cat((past[0], k), dim=1)
            v = torch.cat((past[1], v), dim=1)
        updated = (k, v)
        groups = self.heads // self.kv_heads
        keys = k.repeat_interleave(groups, dim=0)
        values = v.repeat_interleave(groups, dim=0)
        # Explicit offset mask: is_causal=True alone uses upper-left alignment
        # for unequal Q/K lengths, which is wrong for a suffix with cached KV.
        key_positions = torch.arange(k.shape[1], device=x.device)
        query_positions = past_length + torch.arange(count, device=x.device)
        mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        out = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask)
        out = out.transpose(0, 1).reshape(count, -1)
        out = out * torch.sigmoid(gate.reshape(count, -1))
        return self.o_proj(out), updated

    def forward_step(self, positions, x, cache, past_length, rotary_emb):
        """Graph-safe single step into private fixed-capacity KV buffers."""
        if x.shape[0] != 1 or cache[0].shape != cache[1].shape:
            raise ValueError("MTP graph attention requires one token and matching KV buffers")
        q, gate = self.q_proj(x).view(1, self.heads, 2 * self.dim).chunk(2, -1)
        k = self.k_proj(x).view(1, self.kv_heads, self.dim)
        v = self.v_proj(x).view(1, self.kv_heads, self.dim)
        q, k = rotary_emb(positions, self.q_norm(q), self.k_norm(k))
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        index = past_length.reshape(1)
        cache[0].index_copy_(1, index, k)
        cache[1].index_copy_(1, index, v)
        groups = self.heads // self.kv_heads
        keys, values = cache[0].repeat_interleave(groups, 0), cache[1].repeat_interleave(groups, 0)
        valid = torch.arange(cache[0].shape[1], device=x.device).unsqueeze(0) <= past_length
        out = F.scaled_dot_product_attention(q, keys, values, attn_mask=valid)
        out = out.transpose(0, 1).reshape(1, -1) * torch.sigmoid(gate.reshape(1, -1))
        return self.o_proj(out)


class MTPDecoderLayer(nn.Module):
    def __init__(self, config, tp_rank=0, tp_size=1):
        super().__init__()
        self.self_attn = MTPAttention(config, tp_rank, tp_size)
        self.input_layernorm = MTPRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = MTPRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MTPMLP(config, tp_rank, tp_size)

    def forward(self, positions, x, past, rotary_emb):
        mixed, updated = self.self_attn(positions, self.input_layernorm(x), past, rotary_emb)
        x = x + mixed
        return x + self.mlp(self.post_attention_layernorm(x)), updated

    def forward_step(self, positions, x, cache, past_length, rotary_emb):
        mixed = self.self_attn.forward_step(positions, self.input_layernorm(x), cache, past_length, rotary_emb)
        x = x + mixed
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen3_5MTP(nn.Module):
    def __init__(self, config, tp_rank=None, tp_size=None):
        super().__init__()
        if getattr(config, "num_experts", 0):
            raise ValueError("This MTP backend supports dense Qwen3.5 only")
        self.tp_rank, self.tp_size = _tp_context(tp_rank, tp_size)
        self.pre_fc_norm_embedding = MTPRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_fc_norm_hidden = MTPRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.layers = nn.ModuleList([MTPDecoderLayer(config, self.tp_rank, self.tp_size)])
        self.norm = MTPRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, positions, inputs_embeds, hidden_states, past, rotary_emb):
        if inputs_embeds.shape != hidden_states.shape or inputs_embeds.ndim != 2:
            raise ValueError("MTP inputs must align as [tokens, hidden_size]")
        # Checkpoint FC expects embedding first, target hidden state second.
        x = self.fc(torch.cat((self.pre_fc_norm_embedding(inputs_embeds),
                               self.pre_fc_norm_hidden(hidden_states)), dim=-1))
        x, updated = self.layers[0](positions, x, past, rotary_emb)
        return self.norm(x), updated

    def forward_step(self, positions, inputs_embeds, hidden_states, cache, past_length, rotary_emb):
        if inputs_embeds.shape != hidden_states.shape or inputs_embeds.shape[0] != 1:
            raise ValueError("MTP graph step requires aligned single-token features")
        x = self.fc(torch.cat((self.pre_fc_norm_embedding(inputs_embeds),
                               self.pre_fc_norm_hidden(hidden_states)), -1))
        x = self.layers[0].forward_step(positions, x, cache, past_length, rotary_emb)
        return self.norm(x)
