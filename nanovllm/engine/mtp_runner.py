"""Request-isolated native MTP drafting, packed verification and state commit.

All ranks execute the same projections/collectives. Rank 0 alone samples and
verifies, then broadcasts decisions before any rank commits its local states.
"""
import torch
import torch.nn.functional as F
import torch.distributed as dist


class MTPRunnerMixin:
    def initialize_mtp_runtime(self):
        self.mtp_caches = {}
        self.dflash_caches = {}
        self.mtp_proposals = {}
        self.mtp_proposal_probs = {}
        self.mtp_draft_features = {}
        self.mtp_target_features = {}
        self.mtp_prefix_snapshots = {}
        self.mtp_pending_prefix_features = {}
        self.mtp_verification_graphs = {}
        self.mtp_draft_graphs = {}
        self.mtp_graph_memory_bytes = 0
        self.mtp_stats = {
            "proposed_tokens": 0, "accepted_tokens": 0,
            "verification_tokens": 0, "replay_tokens": 0,
            "gdn_replay_tokens": 0, "gdn_replay_layers": 0,
            "gdn_replay_buffer_bytes_peak": 0,
            "verification_graph_replays": 0, "verification_graph_captures": 0,
            "verification_graph_fallbacks": 0, "prefix_restores": 0,
            "draft_graph_replays": 0, "draft_graph_captures": 0, "draft_graph_fallbacks": 0,
            "dflash_draft_forwards": 0,
            "verification_request_rounds": 0, "verification_batch_rounds": 0,
            "emitted_tokens": 0, "requests_with_drafts": 0,
        }

    def _mtp_rotary(self):
        for layer in self.model.model.language_model.layers:
            if layer.block_type == "full_attention":
                return layer.self_attn.rotary_emb
        raise RuntimeError("MTP requires a full-attention RoPE implementation")

    def _mtp_logits(self, hidden):
        # Do not call ParallelLMHead.forward here: prefill selects only last rows.
        logits = F.linear(hidden, self.model.lm_head.weight)
        if self.world_size > 1:
            gathered = [torch.empty_like(logits) for _ in range(self.world_size)] if self.rank == 0 else None
            dist.gather(logits, gathered, dst=0)
            return torch.cat(gathered, -1) if self.rank == 0 else None
        return logits

    def _mtp_sync_ids(self, values, count, device):
        if self.world_size == 1:
            return values
        ids = (torch.tensor(values, dtype=torch.long, device=device) if self.rank == 0
               else torch.empty(count, dtype=torch.long, device=device))
        dist.broadcast(ids, src=0)
        return ids.tolist()

    def _mtp_select(self, logits, temperature):
        from nanovllm.engine.speculative import token_probabilities
        if logits is None:
            return None, None
        probability = token_probabilities(logits[-1], temperature)
        token = (int(logits[-1].argmax()) if temperature == 0 else
                 int(torch.multinomial(probability, 1)))
        return token, probability

    def _mtp_positions(self, seq, start, count, device):
        end = start + count
        full = getattr(seq, "mrope_position_ids", None)
        delta = getattr(seq, "mrope_position_delta", None) or 0
        positions = torch.arange(start, end, dtype=torch.long, device=device) + delta
        if not getattr(self, "uses_multimodal_rope", False):
            return positions
        positions = positions.unsqueeze(0).expand(3, -1).clone()
        if full is not None and start < full.shape[1]:
            overlap = min(end, full.shape[1]) - start
            positions[:, :overlap].copy_(full[:, start:start + overlap].to(device))
        return positions

    def _mtp_embeddings(self, seq, ids, start):
        """Shifted image tokens use visual embeddings, not image-pad lookup."""
        embeddings = self.model.embed_input_ids(ids)
        types = getattr(seq, "mm_token_type_ids", None)
        if types is None:
            return embeddings
        covered = types[start:start + ids.numel()]
        locations = [i for i, kind in enumerate(covered) if kind == self.IMAGE_TOKEN_TYPE]
        if locations:
            visual = self.get_or_create_visual_embeddings(seq)
            first = sum(kind == self.IMAGE_TOKEN_TYPE for kind in types[:start])
            indices = torch.tensor(locations, device=ids.device, dtype=torch.long)
            embeddings = embeddings.clone()
            embeddings.index_copy_(0, indices, visual[first:first + len(locations)].to(embeddings.dtype))
        return embeddings

    @torch.inference_mode()
    def _advance_mtp(self, seq, next_token_ids, target_features, start):
        past = self.mtp_caches.get(seq.seq_id)
        past_length = 0 if past is None else past[0].shape[1]
        if past_length != start:
            raise RuntimeError("MTP cache and target computed-token boundary diverged")
        if target_features.shape[0] != len(next_token_ids):
            raise RuntimeError("Shifted MTP tokens must align with target hidden states")
        ids = torch.tensor(next_token_ids, dtype=torch.long, device=target_features.device)
        positions = self._mtp_positions(seq, start + 1, len(next_token_ids), ids.device)
        embeddings = self._mtp_embeddings(seq, ids, start + 1)
        forward = getattr(self, "mtp_head_with_graph", None)
        hidden, updated = (forward(positions, embeddings, target_features, past) if forward is not None else
            self.model.mtp(positions, embeddings, target_features, past, self._mtp_rotary()))
        logits = self._mtp_logits(hidden[-1:])
        token, probabilities = self._mtp_select(logits, seq.temperature)
        token = self._mtp_sync_ids([token] if token is not None else None, 1, ids.device)[0]
        self.mtp_caches[seq.seq_id] = updated
        self.mtp_proposals[seq.seq_id] = token
        self.mtp_proposal_probs[seq.seq_id] = probabilities
        self.mtp_draft_features[seq.seq_id] = hidden[-1:].detach().clone()
        self.mtp_target_features[seq.seq_id] = target_features[-1:].detach().clone()

    def update_mtp_after_prefill(self, seqs, token_ids):
        features = self.model.model.language_model.last_mtp_hidden_states
        ids = self._mtp_sync_ids(token_ids, len(seqs), features.device)
        packed = 0
        for row, seq in enumerate(seqs):
            start = seq.num_cached_tokens
            end = start + seq.num_scheduled_tokens
            if start == 0:
                self.release_mtp_requests([seq.seq_id])
            elif seq.seq_id in self.mtp_pending_prefix_features:
                # A reusable snapshot stops BEFORE the lookahead embedding.
                checkpoint_feature = self.mtp_pending_prefix_features.pop(seq.seq_id)
                self._advance_mtp(seq, [seq[start]], checkpoint_feature, start - 1)
            next_id = seq[end] if end < seq.num_tokens else ids[row]
            shifted = list(seq.token_ids[start + 1:end]) + [next_id]
            count = end - start
            self._advance_mtp(seq, shifted, features[packed:packed + count], start)
            packed += count
        if packed != features.shape[0]:
            raise RuntimeError("Packed MTP features do not match prefill request boundaries")
        return ids if getattr(self, "rank", 0) == 0 else None

    @staticmethod
    def _mtp_verify_view(seq, drafts):
        # Do not pickle-copy Sequence: decode serialization omits ordinary IDs.
        view = type(seq).__new__(type(seq))
        view.__dict__.update(seq.__dict__)
        view.token_ids = list(seq.token_ids)
        types = getattr(seq, "mm_token_type_ids", None)
        if types is not None:
            view.mm_token_type_ids = list(types)
        for token in drafts:
            view.append_token(token)
        view.num_scheduled_tokens = 1 + len(drafts)
        view.is_prefill = True
        return view

    def _mtp_verification_features(self, hidden):
        return hidden

    def _verify_mtp_batch(self, seqs, drafts):
        from nanovllm.utils.context import reset_context
        views = [self._mtp_verify_view(seq, proposal) for seq, proposal in zip(seqs, drafts)]
        input_ids, positions = self.prepare_prefill(views)
        try:
            records = {}
            from nanovllm.utils.context import get_context
            context = get_context()
            context.gdn_mtp_verification = True
            context.gdn_mtp_verification_backend = getattr(self.config, "mtp_gdn_verification_backend", "reference")
            if any(drafts) and getattr(self.config, "mtp_gdn_rollback", "recompute") == "replay_inputs":
                context.gdn_replay_records = records
                # Capture all possible accepted boundaries, not just pending.
                context.gdn_replay_capture_tokens = max(view.num_scheduled_tokens for view in views)
            states = self.hybrid_state_manager.read_batched_states([seq.state_slot for seq in seqs])
            hidden, updated = self.model(input_ids=input_ids, positions=positions, gdn_states=states)
            logits = self._mtp_logits(hidden)
            captured_bytes = sum(record.nbytes if not isinstance(record, tuple) else
                                 sum(part.nbytes for part in record) for record in records.values())
            self.mtp_stats["gdn_replay_buffer_bytes_peak"] = max(
                self.mtp_stats["gdn_replay_buffer_bytes_peak"], captured_bytes)
            return logits, self._mtp_verification_features(hidden), updated, records
        finally:
            reset_context()

    def _verify_mtp_inputs(self, seq, drafts):
        return self._verify_mtp_batch([seq], [drafts])

    @torch.inference_mode()
    def _replay_gdn_inputs(self, seq, records, num_tokens):
        from nanovllm.engine.gdn_replay import replay_gdn_state
        manager = self.hybrid_state_manager
        expected = set(manager.spec.gdn_layer_indices)
        if set(records) != expected:
            raise RuntimeError("Verification did not capture every GDN layer exactly once")
        states = manager.read_states(seq.state_slot)
        for layer in sorted(expected):
            initial = states[layer]
            if initial is None or initial.conv_state is None or initial.recurrent_state is None:
                raise RuntimeError("GDN replay requires an initialized checkpoint")
            conv, recurrent = replay_gdn_state(records[layer], initial.conv_state,
                initial.recurrent_state, num_tokens,
                backend=getattr(self.config, "mtp_gdn_replay_backend", "auto"))
            states[layer] = type(initial)(conv_state=conv, recurrent_state=recurrent)
        self.mtp_stats["gdn_replay_tokens"] += num_tokens
        self.mtp_stats["gdn_replay_layers"] += len(expected)
        return states

    @staticmethod
    def _mtp_state_row(states, row):
        return [None if state is None else type(state)(
            conv_state=state.conv_state[row:row + 1],
            recurrent_state=state.recurrent_state[row:row + 1]) for state in states]

    @torch.inference_mode()
    def _draft_mtp(self, seq, count):
        if count == 0:
            return [], []
        tokens = [self.mtp_proposals[seq.seq_id]]
        distributions = [self.mtp_proposal_probs.get(seq.seq_id)]
        if count == 1:
            return tokens, distributions
        past = self.mtp_caches[seq.seq_id]
        hidden = self.mtp_draft_features[seq.seq_id]
        for index in range(1, count):
            ids = torch.tensor([tokens[-1]], dtype=torch.long, device=hidden.device)
            positions = self._mtp_positions(seq, seq.num_cached_tokens + index + 1, 1, ids.device)
            embeddings = self.model.embed_input_ids(ids)
            forward = getattr(self, "mtp_head_with_graph", None)
            hidden, past = (forward(positions, embeddings, hidden, past) if forward is not None else
                           self.model.mtp(positions, embeddings, hidden, past, self._mtp_rotary()))
            token, probabilities = self._mtp_select(self._mtp_logits(hidden), seq.temperature)
            token = self._mtp_sync_ids([token] if token is not None else None, 1, ids.device)[0]
            tokens.append(token)
            distributions.append(probabilities)
        # Speculative drafter cache is never installed as the committed cache.
        return tokens, distributions

    @torch.inference_mode()
    def run_mtp_batch(self, seqs, draft_counts=None):
        if not seqs or len({seq.state_slot for seq in seqs}) != len(seqs):
            raise ValueError("MTP batch must contain unique request state slots")
        from nanovllm.engine.speculative import verify_greedy, verify_sampling, token_probabilities
        configured = getattr(self.config, "mtp_num_speculative_tokens", 1)
        if draft_counts is None:
            draft_counts = [configured] * len(seqs)
        if len(draft_counts) != len(seqs):
            raise ValueError("MTP draft counts must align with requests")
        proposals, distributions = [], []
        for seq, count in zip(seqs, draft_counts):
            if seq.num_cached_tokens != seq.num_tokens - 1:
                raise RuntimeError("Decode must have exactly one pending token")
            if seq.seq_id not in self.mtp_proposals:
                raise RuntimeError("MTP cache was not initialized by prefill or prefix restore")
            remaining = seq.max_tokens - seq.num_completion_tokens
            count = min(count, configured, remaining - 1,
                        getattr(self.config, "max_model_len", seq.num_tokens + configured) - seq.num_tokens)
            if count < 0:
                raise ValueError("MTP request exceeded its generation/context budget")
            draft, probs = self._draft_mtp(seq, count)
            proposals.append(draft)
            distributions.append(probs)
        verify = getattr(self, "verify_mtp_with_graph", None)
        if verify is not None:
            logits, hidden, updated, records = verify(seqs, proposals)
        elif len(seqs) == 1:
            logits, hidden, updated, records = self._verify_mtp_inputs(seqs[0], proposals[0])
        else:
            logits, hidden, updated, records = self._verify_mtp_batch(seqs, proposals)
        lengths = [1 + len(tokens) for tokens in proposals]
        results, accepted_counts, offset = [], [], 0
        coordinator = getattr(self, "rank", 0) == 0
        for row, seq in enumerate(seqs):
            if coordinator:
                scores = logits[offset:offset + lengths[row]]
                arguments = dict(remaining_tokens=seq.max_tokens - seq.num_completion_tokens,
                    eos_token_id=self.config.eos, ignore_eos=seq.ignore_eos)
                if seq.temperature == 0:
                    result = verify_greedy(proposals[row], scores.argmax(-1).tolist(), **arguments)
                else:
                    result = verify_sampling(proposals[row], distributions[row],
                        token_probabilities(scores, seq.temperature), **arguments)
                results.append(list(result.token_ids))
                accepted_counts.append(result.accepted_draft_tokens)
            offset += lengths[row]
        if self.world_size > 1:
            width = configured + 1
            packet = torch.full((len(seqs), width + 2), -1, dtype=torch.long, device=hidden.device)
            if coordinator:
                for row, (tokens, accepted) in enumerate(zip(results, accepted_counts)):
                    packet[row, 0], packet[row, 1] = len(tokens), accepted
                    packet[row, 2:2 + len(tokens)] = torch.tensor(tokens, device=hidden.device)
            dist.broadcast(packet, src=0)
            if not coordinator:
                results = [row[2:2 + int(row[0])].tolist() for row in packet]
                accepted_counts = [int(row[1]) for row in packet]
        # Prepare every boundary before writing any request's active state.
        commits, offset = [], 0
        for row, (seq, tokens) in enumerate(zip(seqs, results)):
            computed = len(tokens)
            if computed == lengths[row]:
                states = self._mtp_state_row(updated, row) if len(seqs) > 1 else updated
            elif getattr(self.config, "mtp_gdn_rollback", "recompute") == "replay_inputs":
                record = {layer: parts[row] if isinstance(parts, tuple) else parts
                          for layer, parts in records.items()}
                states = self._replay_gdn_inputs(seq, record, computed)
            else:
                # Debug recompute can include multiple accepted draft tokens.
                _, _, states, _ = self._verify_mtp_inputs(seq, proposals[row][:computed - 1])
                self.mtp_stats["replay_tokens"] += computed
            commits.append((seq, tokens, states, hidden[offset:offset + computed]))
            offset += lengths[row]
        for seq, tokens, states, target_features in commits:
            self.hybrid_state_manager.write_batched_states([seq.state_slot], states)
            self._advance_mtp(seq, tokens, target_features, seq.num_cached_tokens)
        self.mtp_stats["proposed_tokens"] += sum(map(len, proposals))
        self.mtp_stats["verification_tokens"] += sum(lengths)
        self.mtp_stats["accepted_tokens"] += sum(accepted_counts)
        self.mtp_stats["verification_request_rounds"] += len(seqs)
        self.mtp_stats["verification_batch_rounds"] += 1
        self.mtp_stats["emitted_tokens"] += sum(map(len, results))
        self.mtp_stats["requests_with_drafts"] += sum(bool(tokens) for tokens in proposals)
        return results if coordinator else None

    def run_mtp_decode(self, seq, allow_draft=True):
        count = getattr(self.config, "mtp_num_speculative_tokens", 1) if allow_draft else 0
        results = self.run_mtp_batch([seq], [count])
        return results[0] if results is not None else None

    def release_mtp_requests(self, seq_ids):
        for seq_id in seq_ids:
            for mapping in (self.mtp_caches, self.mtp_proposals, self.mtp_proposal_probs,
                            self.mtp_draft_features, self.mtp_target_features,
                            self.mtp_pending_prefix_features):
                mapping.pop(seq_id, None)

    def mtp_prefix_snapshot(self, seq_id, boundary):
        from nanovllm.engine.mtp_cache import MTPPrefixSnapshot
        past = self.mtp_caches.get(seq_id)
        if past is None or past[0].shape[1] != boundary:
            raise RuntimeError("MTP prefix snapshot must match the computed target boundary")
        return MTPPrefixSnapshot(boundary, past[0][:, :boundary - 1],
            past[1][:, :boundary - 1], self.mtp_target_features[seq_id])

    def cache_mtp_prefix(self, key, seq_id):
        if key not in self.mtp_prefix_snapshots:
            self.mtp_prefix_snapshots[key] = self.mtp_prefix_snapshot(seq_id, key.num_cached_tokens).clone()
        if self.world_size > 1:
            dist.barrier()

    def restore_mtp_prefix(self, key, seq_id):
        snapshot = self.mtp_prefix_snapshots[key]
        self.release_mtp_requests([seq_id])
        self.mtp_caches[seq_id] = (snapshot.key.clone(), snapshot.value.clone())
        self.mtp_pending_prefix_features[seq_id] = snapshot.target_feature.clone()
        self.mtp_stats["prefix_restores"] += 1
        if self.world_size > 1:
            dist.barrier()

    def discard_mtp_prefixes(self, keys):
        for key in keys:
            self.mtp_prefix_snapshots.pop(key, None)
        if self.world_size > 1:
            dist.barrier()
