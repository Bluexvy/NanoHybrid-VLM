"""Compact-input GDN rollback, inspired by ReplaySSM's input caching.

State-only replay has a functional FP32 reference and an optional fused CUDA
backend. This is not lazy checkpoint / ring-buffer output-only verification.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GDNReplayInputs:
    # Pre-convolution mixed QKV. Conv history contains raw inputs, not SiLU output.
    conv_input: torch.Tensor       # [B, channels, replay_tokens]
    key: torch.Tensor              # [B, replay_tokens, value_heads, Dk], pre-L2
    value: torch.Tensor            # [B, replay_tokens, value_heads, Dv]
    g: torch.Tensor                # [B, replay_tokens, value_heads], log decay
    beta: torch.Tensor             # [B, replay_tokens, value_heads], update gate

    @property
    def nbytes(self) -> int:
        return sum(x.numel() * x.element_size() for x in
                   (self.conv_input, self.key, self.value, self.g, self.beta))


def capture_packed_gdn_inputs(conv_input, key, value, g, beta, seqlens, capture_tokens):
    """Split packed inputs at REQUEST boundaries and clone only each prefix."""
    lengths = tuple(seqlens or (key.shape[1],))
    if (key.shape[0] != 1 or sum(lengths) != key.shape[1]
            or capture_tokens <= 0 or any(length <= 0 for length in lengths)):
        raise ValueError("Invalid packed GDN replay capture boundaries")
    records, start = [], 0
    for length in lengths:
        end = start + min(capture_tokens, length)
        records.append(GDNReplayInputs(
            conv_input[:, :, start:end].detach().clone(),
            key[:, start:end].detach().clone(), value[:, start:end].detach().clone(),
            g[:, start:end].detach().clone(), beta[:, start:end].detach().clone()))
        start += length
    return records[0] if len(records) == 1 else tuple(records)


@torch.inference_mode()
def replay_gdn_state(inputs: GDNReplayInputs, conv_state: torch.Tensor,
                     recurrent_state: torch.Tensor, num_tokens: int,
                     backend: str = "reference"):
    """Return a new accepted-prefix state; never mutate the checkpoint views."""
    if inputs.key.ndim != 4 or inputs.value.ndim != 4:
        raise ValueError("Replay K/V must have shape [batch, tokens, heads, dim]")
    batch, length, heads, key_dim = inputs.key.shape
    if type(num_tokens) is not int or not 1 <= num_tokens <= length:
        raise ValueError("Replay boundary must be within the captured token prefix")
    if inputs.value.shape[:3] != (batch, length, heads):
        raise ValueError("Replay K/V token and head dimensions must align")
    if inputs.g.shape != (batch, length, heads) or inputs.beta.shape != inputs.g.shape:
        raise ValueError("Replay g/beta must align with K/V")
    if (inputs.conv_input.ndim != 3 or inputs.conv_input.shape[0] != batch
            or inputs.conv_input.shape[2] != length):
        raise ValueError("Raw convolution inputs must align with replay tokens")
    if (conv_state.ndim != 3 or conv_state.shape[:2] != inputs.conv_input.shape[:2]
            or conv_state.shape[-1] <= 0):
        raise ValueError("Invalid convolution checkpoint")
    if recurrent_state.shape != (batch, heads, key_dim, inputs.value.shape[-1]):
        raise ValueError("Invalid recurrent checkpoint")
    if recurrent_state.dtype != torch.float32:
        raise TypeError("Active recurrent checkpoint must be FP32")
    if any(x.device != recurrent_state.device for x in
           (conv_state, inputs.conv_input, inputs.key, inputs.value, inputs.g, inputs.beta)):
        raise ValueError("Replay inputs and checkpoints must share a device")

    if backend not in {"auto", "reference", "cuda"}:
        raise ValueError("Replay backend must be 'auto', 'reference' or 'cuda'")
    if backend == "cuda" and not recurrent_state.is_cuda:
        raise ValueError("CUDA replay requires CUDA checkpoints")
    if backend == "cuda" or (backend == "auto" and recurrent_state.is_cuda):
        from nanovllm.kernels.gdn_replay_cuda import cuda_replay_supported, replay_gdn_state_cuda
        if cuda_replay_supported(inputs, conv_state, recurrent_state, num_tokens):
            return replay_gdn_state_cuda(inputs, conv_state, recurrent_state, num_tokens)
        if backend == "cuda":
            raise ValueError("CUDA replay requires CUDA FP32 state, Dk<=128, prefix<=16 and supported dtypes")
        # Shapes/dtypes outside the CUDA specialization use the reference.
        # Compilation/launch failures are intentionally surfaced, not hidden.

    # Same mathematical recurrence as the model's torch GDN reference.
    # Q is needed to read the output, but not to update state. Main-model
    # hidden states were already produced by verification and can be reused.
    key = inputs.key[:, :num_tokens].float()
    key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
    state = recurrent_state  # read-only view; all updates below are out-of-place
    for token in range(num_tokens):
        k = key[:, token]
        decayed = state * inputs.g[:, token].float().exp()[..., None, None]
        remembered = (decayed * k.unsqueeze(-1)).sum(-2)
        delta = (inputs.value[:, token].float() - remembered)
        delta = delta * inputs.beta[:, token].float().unsqueeze(-1)
        state = decayed + k.unsqueeze(-1) * delta.unsqueeze(-2)

    # Conv update needs only its sliding raw-input history, not weights or SiLU.
    combined = torch.cat((conv_state, inputs.conv_input[:, :, :num_tokens].to(conv_state.dtype)), -1)
    conv = combined[:, :, -conv_state.shape[-1]:].contiguous()
    return conv, state
