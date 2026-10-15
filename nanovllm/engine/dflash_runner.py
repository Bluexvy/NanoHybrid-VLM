"""Parallel DFlash proposals over the existing speculative verifier/rollback."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DFlashPrefixSnapshot:
    num_cached_tokens: int
    layers: tuple

    def __post_init__(self):
        if (self.num_cached_tokens <= 0 or not self.layers
                or any(k.shape != v.shape or k.ndim != 3 or k.shape[1] != self.num_cached_tokens
                       for k, v in self.layers)):
            raise ValueError("DFlash snapshot must contain context KV at the target boundary")

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for kv in self.layers for t in kv)

    def clone(self):
        return type(self)(self.num_cached_tokens,
                          tuple(tuple(t.detach().clone() for t in kv) for kv in self.layers))


class DFlashRunnerMixin:
    def _uses_dflash(self):
        return getattr(self.config, "speculative_method", "mtp") == "dflash"

    def _mtp_verification_features(self, hidden):
        if self._uses_dflash():
            return self.model.model.language_model.last_dflash_hidden_states
        return super()._mtp_verification_features(hidden)

    @torch.inference_mode()
    def _advance_mtp(self, seq, next_token_ids, target_features, start):
        if not self._uses_dflash():
            return super()._advance_mtp(seq, next_token_ids, target_features, start)
        past = self.dflash_caches.get(seq.seq_id)
        length = 0 if past is None else past[0][0].shape[1]
        if start != length or target_features.shape[0] != len(next_token_ids):
            raise RuntimeError("DFlash context cache diverged from accepted target boundary")
        positions = torch.arange(start, start + len(next_token_ids), device=target_features.device)
        self.dflash_caches[seq.seq_id] = self.dflash.append_context(target_features, positions, past)
        # Common verifier's readiness marker, not a precomputed proposal.
        self.mtp_proposals[seq.seq_id] = True

    def update_mtp_after_prefill(self, seqs, token_ids):
        if not self._uses_dflash():
            return super().update_mtp_after_prefill(seqs, token_ids)
        features = self.model.model.language_model.last_dflash_hidden_states
        ids = self._mtp_sync_ids(token_ids, len(seqs), features.device)
        offset = 0
        for seq in seqs:
            if seq.is_multimodal:
                raise ValueError("DFlash currently supports text requests only")
            if seq.num_cached_tokens == 0:
                self.release_mtp_requests([seq.seq_id])
            count = seq.num_scheduled_tokens
            self._advance_mtp(seq, [0] * count, features[offset:offset + count], seq.num_cached_tokens)
            offset += count
        if offset != features.shape[0]:
            raise RuntimeError("Packed DFlash features do not match prefill lengths")
        return ids if self.rank == 0 else None

    @torch.inference_mode()
    def _draft_mtp(self, seq, count):
        if not self._uses_dflash():
            return super()._draft_mtp(seq, count)
        if count == 0:
            return [], []
        cache = self.dflash_caches[seq.seq_id]
        device = cache[0][0].device
        pending = seq[seq.num_cached_tokens]
        ids = torch.full((count + 1,), self.dflash.config.mask_token_id, dtype=torch.long, device=device)
        ids[0] = pending
        positions = torch.arange(seq.num_cached_tokens, seq.num_cached_tokens + count + 1, device=device)
        embedding = self.model.embed_input_ids(ids) * self.dflash.config.input_embedding_scale
        hidden = self.dflash(embedding, positions, cache)
        # Position j's mask hidden predicts the token at j, unlike the target's
        # next-token logits. Drop anchor hidden, not the last hidden.
        logits = self._mtp_logits(hidden[1:])
        probabilities, tokens = [], None
        if self.rank == 0:
            logits = self.dflash.transform_logits(logits)
            if seq.temperature == 0:
                tokens = logits.argmax(-1).tolist()
                probabilities = [None] * count
            else:
                from nanovllm.engine.speculative import token_probabilities
                distribution = token_probabilities(logits, seq.temperature)
                tokens = torch.multinomial(distribution, 1).flatten().tolist()
                probabilities = list(distribution.unbind(0))
        tokens = self._mtp_sync_ids(tokens, count, device)
        self.mtp_stats["dflash_draft_forwards"] += 1
        # Forward uses temporary noise KV; committed context is untouched.
        return tokens, probabilities if self.rank == 0 else [None] * count

    def release_mtp_requests(self, seq_ids):
        for seq_id in seq_ids:
            getattr(self, "dflash_caches", {}).pop(seq_id, None)
        return super().release_mtp_requests(seq_ids)

    def mtp_prefix_snapshot(self, seq_id, boundary):
        if not self._uses_dflash():
            return super().mtp_prefix_snapshot(seq_id, boundary)
        return DFlashPrefixSnapshot(boundary, self.dflash_caches[seq_id])

    def restore_mtp_prefix(self, key, seq_id):
        if not self._uses_dflash():
            return super().restore_mtp_prefix(key, seq_id)
        self.release_mtp_requests([seq_id])
        snapshot = self.mtp_prefix_snapshots[key].clone()
        self.dflash_caches[seq_id] = snapshot.layers
        self.mtp_proposals[seq_id] = True
        self.mtp_stats["prefix_restores"] += 1
        if self.world_size > 1:
            torch.distributed.barrier()
