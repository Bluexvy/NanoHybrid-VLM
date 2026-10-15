"""Speculative acceptance and checkpoint preflight (no CUDA imports)."""

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class GreedyVerification:
    token_ids: tuple[int, ...]
    accepted_draft_tokens: int

    @property
    def num_computed_tokens(self) -> int:
        # The last emitted token is still pending. The current pending token
        # plus the accepted drafts preceding it have been computed by target.
        return len(self.token_ids)


def verify_greedy(
    draft_token_ids: list[int],
    target_token_ids: list[int],
    *,
    remaining_tokens: int,
    eos_token_id: int,
    ignore_eos: bool = False,
) -> GreedyVerification:
    """Target predictions correspond to [pending_token, *draft_tokens]."""
    if remaining_tokens <= 0:
        raise ValueError("remaining_tokens must be positive")
    if len(target_token_ids) != len(draft_token_ids) + 1:
        raise ValueError("Target must provide one prediction per input token")
    output = []
    accepted = 0
    for i, draft in enumerate(draft_token_ids):
        target = target_token_ids[i]
        output.append(target)
        matched = draft == target
        accepted += int(matched)
        if (not matched or len(output) == remaining_tokens
                or (not ignore_eos and target == eos_token_id)):
            return GreedyVerification(tuple(output), accepted)
    output.append(target_token_ids[-1])
    return GreedyVerification(tuple(output), accepted)


def token_probabilities(logits, temperature: float):
    """Use the same temperature transform for draft and target distributions."""
    import torch
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if temperature == 0:
        return torch.nn.functional.one_hot(logits.argmax(-1), logits.shape[-1]).float()
    return torch.softmax(logits.float() / temperature, dim=-1)


def verify_sampling(draft_token_ids, draft_probabilities, target_probabilities, *,
                    remaining_tokens, eos_token_id, ignore_eos=False,
                    generator=None, uniforms=None):
    """Exact rejection sampling: accept min(1,p/q), reject from (p-q)+.

    All probabilities must already include the caller's sampling transforms.
    Only the coordinator calls this function; it broadcasts the accepted result.
    """
    import torch
    count = len(draft_token_ids)
    if remaining_tokens <= 0 or len(draft_probabilities) != count:
        raise ValueError("Invalid speculative sampling budget or draft distributions")
    if target_probabilities.ndim != 2 or target_probabilities.shape[0] != count + 1:
        raise ValueError("Target must provide one distribution per verification input")
    def checked(prob):
        prob = prob.float()
        if (not bool(torch.isfinite(prob).all()) or bool((prob < 0).any())
                or not bool(prob.sum() > 0)):
            raise ValueError("Sampling probabilities must be finite and non-negative")
        return prob / prob.sum()
    output, accepted = [], 0
    for i, token in enumerate(draft_token_ids):
        p, q = checked(target_probabilities[i]), checked(draft_probabilities[i])
        if p.shape != q.shape or not 0 <= token < p.numel() or not bool(q[token] > 0):
            raise ValueError("Draft token must have positive probability in its distribution")
        u = (float(uniforms[i]) if uniforms is not None else
             float(torch.rand((), device=p.device, generator=generator)))
        if not 0 <= u < 1:
            raise ValueError("Acceptance uniform must be in [0,1)")
        matched = u < min(1.0, float(p[token] / q[token]))
        if matched:
            chosen = token
            accepted += 1
        else:
            residual = (p - q).clamp_min(0)
            # A zero residual is possible only through finite-precision noise.
            distribution = residual / residual.sum() if bool(residual.sum() > 0) else p
            chosen = int(torch.multinomial(distribution, 1, generator=generator))
        output.append(chosen)
        if (not matched or len(output) == remaining_tokens
                or (not ignore_eos and chosen == eos_token_id)):
            return GreedyVerification(tuple(output), accepted)
    bonus = int(torch.multinomial(checked(target_probabilities[-1]), 1, generator=generator))
    output.append(bonus)
    return GreedyVerification(tuple(output), accepted)


def validate_mtp_checkpoint(model_path: str) -> None:
    """Inspect names, not weight payloads; fail before allocating the model."""
    path = Path(model_path)
    index = path / "model.safetensors.index.json"
    if index.exists():
        names = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"])
    else:
        from safetensors import safe_open
        names = set()
        for file in path.glob("*.safetensors"):
            with safe_open(str(file), framework="pt", device="cpu") as weights:
                names.update(weights.keys())
    required = {
        "mtp.fc.weight", "mtp.norm.weight",
        "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        *{f"mtp.layers.0.self_attn.{name}.weight" for name in
          ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")},
        *{f"mtp.layers.0.mlp.{name}.weight" for name in
          ("gate_proj", "up_proj", "down_proj")},
    }
    missing = sorted(required - names)
    if missing:
        raise ValueError("Checkpoint is missing trained MTP weights: "
                         + ", ".join(missing))
    layers = {name.split(".")[2] for name in names
              if name.startswith("mtp.layers.")}
    if layers != {"0"}:
        raise ValueError("The initial MTP backend supports exactly one MTP layer")
