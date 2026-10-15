"""Functional short-prefix GDN verification, suitable for CUDA Graph capture."""
import torch


def _use_cuda(backend, supported):
    if backend not in {"auto", "reference", "cuda"}:
        raise ValueError("Verification backend must be auto/reference/cuda")
    if backend == "cuda" and not supported:
        raise ValueError("Unsupported fused CUDA verification shape/dtype")
    return backend != "reference" and supported


@torch.inference_mode()
def verification_conv(raw, weight, bias, initial, backend="reference", activation="silu"):
    batch, channels, length = raw.shape
    if (initial.shape != (batch, channels, weight.shape[-1]) or weight.shape[0] != channels
            or length < 1 or activation not in {None, "silu"}):
        raise ValueError("Invalid verification convolution shapes")
    supported = (raw.is_cuda and 1 <= length <= 16 and initial.shape[-1] == 4
                 and raw.dtype in {torch.float32, torch.float16, torch.bfloat16}
                 and weight.dtype == raw.dtype and initial.dtype == raw.dtype)
    if _use_cuda(backend, supported):
        from nanovllm.kernels.gdn_replay_cuda import load_gdn_replay_cuda_extension
        output, state = load_gdn_replay_cuda_extension().gdn_conv_verify(
            raw, weight.contiguous(), bias, initial, activation == "silu")
        return output, state
    combined = torch.cat((initial, raw), -1)
    result = torch.zeros_like(raw, dtype=torch.float32)
    if bias is not None:
        result = result + bias.float()[None, :, None]
    for tap in range(weight.shape[-1]):
        result = result + combined[:, :, tap + 1:tap + 1 + length].float() * weight[:, tap].float()[None, :, None]
    if activation == "silu":
        result = result / (1 + torch.exp(-result))
    return result.to(raw.dtype), combined[:, :, -initial.shape[-1]:].contiguous()


@torch.inference_mode()
def verification_recurrent(query, key, value, g, beta, initial, backend="reference"):
    batch, length, heads, key_dim = key.shape
    value_dim = value.shape[-1]
    if (query.shape != key.shape or value.shape[:3] != key.shape[:3]
            or g.shape != (batch, length, heads) or beta.shape != g.shape
            or initial.shape != (batch, heads, key_dim, value_dim) or initial.dtype != torch.float32):
        raise ValueError("Invalid verification recurrent shapes")
    supported = (initial.is_cuda and 1 <= key_dim <= 128 and 1 <= length <= 16
                 and key.dtype in {torch.float32, torch.float16, torch.bfloat16}
                 and query.dtype == key.dtype and value.dtype == key.dtype
                 and g.dtype == torch.float32 and beta.dtype in {torch.float32, torch.float16, torch.bfloat16})
    if _use_cuda(backend, supported):
        from nanovllm.kernels.gdn_replay_cuda import load_gdn_replay_cuda_extension
        output, state = load_gdn_replay_cuda_extension().gdn_recurrent_verify(query, key, value, g, beta, initial)
        return output.to(query.dtype), state
    q, k = query.float(), key.float()
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * (key_dim ** -.5)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    state, outputs = initial, []
    for token in range(length):
        state = state * g[:, token].float().exp()[..., None, None]
        remembered = (state * k[:, token].unsqueeze(-1)).sum(-2)
        delta = (value[:, token].float() - remembered) * beta[:, token].float().unsqueeze(-1)
        state = state + k[:, token].unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((state * q[:, token].unsqueeze(-1)).sum(-2))
    return torch.stack(outputs, 1).to(query.dtype), state
