"""Packed/ragged MTP verification with real small GDN math on CPU."""
import pickle
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys
import torch
import test_mtp_cpu as base


class PackedGDN(base.ToyGDN):
    def _run_causal_conv1d(self, mixed_qkv, conv_state, prefill_seqlens):
        pieces, states, offset = [], [], 0
        for row, length in enumerate(prefill_seqlens):
            output, state = base._conv_reference(mixed_qkv[:, :, offset:offset + length],
                self.conv_weight, initial_state=conv_state[row:row + 1])
            pieces.append(output)
            states.append(state)
            offset += length
        return torch.cat(pieces, 2), torch.cat(states, 0)

    def _run_gated_delta_rule(self, query, key, value, g, beta, recurrent_state, **kwargs):
        pieces, states, offset = [], [], 0
        for row, length in enumerate(kwargs["prefill_seqlens"]):
            part = slice(offset, offset + length)
            output, state = base._recurrent_reference(query[:, part], key[:, part], value[:, part],
                g[:, part], beta[:, part], initial_state=recurrent_state[row:row + 1],
                output_final_state=True, use_qk_l2norm_in_kernel=True)
            pieces.append(output)
            states.append(state)
            offset += length
        return torch.cat(pieces, 1), torch.cat(states, 0)


class Pool:
    def __init__(self):
        self.spec = SimpleNamespace(gdn_layer_indices=(1, 4))
        self.requests = {slot: base.ReplayPool().states for slot in (3, 7, 9)}

    def read_states(self, slot):
        return self.requests[slot].copy()

    def read_batched_states(self, slots):
        result = [None] * 5
        for layer in (1, 4):
            result[layer] = SimpleNamespace(
                conv_state=torch.cat([self.requests[slot][layer].conv_state for slot in slots], 0),
                recurrent_state=torch.cat([self.requests[slot][layer].recurrent_state for slot in slots], 0))
        return result

    def write_batched_states(self, slots, states):
        for row, slot in enumerate(slots):
            for layer in (1, 4):
                self.requests[slot][layer].conv_state.copy_(states[layer].conv_state[row:row + 1])
                self.requests[slot][layer].recurrent_state.copy_(states[layer].recurrent_state[row:row + 1])


class Target:
    def __init__(self):
        self.layers = [PackedGDN(1), PackedGDN(4)]
        self.lm_head = SimpleNamespace(weight=torch.eye(16))
        self.forward_count = 0

    def __call__(self, input_ids, positions, gdn_states):
        self.forward_count += 1
        x = torch.nn.functional.one_hot(input_ids % 8, 8).float().unsqueeze(0)
        updated = [None] * 5
        for layer in self.layers:
            old = gdn_states[layer.layer_idx]
            x, conv, state = layer(x, old.conv_state, old.recurrent_state,
                                  prefill_seqlens=base._capture_context.prefill_seqlens)
            updated[layer.layer_idx] = SimpleNamespace(conv_state=conv, recurrent_state=state)
        return torch.nn.functional.one_hot(input_ids + 1, 16).float(), updated


class Runner(base.runner.MTPRunnerMixin):
    def __init__(self):
        self.initialize_mtp_runtime()
        self.world_size, self.rank = 1, 0
        self.config = SimpleNamespace(eos=-1, mtp_num_speculative_tokens=2,
            max_model_len=32, mtp_gdn_rollback="replay_inputs", mtp_gdn_replay_backend="reference")
        self.hybrid_state_manager, self.model = Pool(), Target()
        self.proposals = {1: [4, 8], 2: [8], 3: []}
        self.mtp_proposals = {key: values[0] if values else 0 for key, values in self.proposals.items()}
        self.advances = {}

    def _draft_mtp(self, seq, count):
        return self.proposals[seq.seq_id][:count], [None] * count

    def prepare_prefill(self, seqs):
        lengths = tuple(seq.num_scheduled_tokens for seq in seqs)
        base._capture_context.prefill_seqlens = lengths
        ids = [token for seq in seqs for token in seq[
            seq.num_cached_tokens:seq.num_cached_tokens + seq.num_scheduled_tokens]]
        return torch.tensor(ids), torch.arange(len(ids))

    def _advance_mtp(self, seq, token_ids, features, start):
        self.advances[seq.seq_id] = (token_ids, features.clone(), start)


def requests():
    result = []
    for seq_id, slot, pending in ((1, 3, 3), (2, 7, 7), (3, 9, 4)):
        seq = base.FakeSequence()
        seq.seq_id, seq.state_slot = seq_id, slot
        seq.token_ids[-1] = pending
        result.append(seq)
    return result


class BatchTests(unittest.TestCase):
    def test_ragged_accept_reject_and_state_isolation(self):
        torch.manual_seed(47)
        runner, seqs = Runner(), requests()
        pool = runner.hybrid_state_manager
        expected = {}
        base.reset_capture()
        for seq, inputs in zip(seqs, ([3, 4], [7, 8], [4])):
            base._capture_context.prefill_seqlens = (len(inputs),)
            _, expected[seq.seq_id] = runner.model(torch.tensor(inputs), None,
                                                  pool.read_batched_states([seq.state_slot]))
        runner.model.forward_count = 0
        modules = {"nanovllm.engine.speculative": base.acceptance,
                   "nanovllm.engine.gdn_replay": base.replay,
                   "nanovllm.utils.context": SimpleNamespace(get_context=lambda: base._capture_context,
                                                              reset_context=base.reset_capture)}
        with patch.dict(sys.modules, modules):
            result = runner.run_mtp_batch(seqs, [2, 1, 0])
        self.assertEqual(result, [[4, 5], [8, 9], [5]])
        self.assertEqual(runner.model.forward_count, 1)
        for seq in seqs:
            for layer in (1, 4):
                actual = pool.requests[seq.state_slot][layer]
                wanted = expected[seq.seq_id][layer]
                torch.testing.assert_close(actual.conv_state, wanted.conv_state)
                torch.testing.assert_close(actual.recurrent_state, wanted.recurrent_state,
                                           atol=2e-6, rtol=2e-6)
            self.assertEqual(runner.advances[seq.seq_id][1].shape[0], len(result[seq.seq_id - 1]))
        self.assertEqual(seqs[0].token_ids, [1, 2, 3])

    def test_duplicate_slots_are_rejected_before_forward(self):
        runner, seqs = Runner(), requests()
        seqs[1].state_slot = seqs[0].state_slot
        with self.assertRaisesRegex(ValueError, "unique"):
            runner.run_mtp_batch(seqs)
        self.assertEqual(runner.model.forward_count, 0)

    def test_capture_does_not_mix_requests(self):
        conv = torch.arange(18.).reshape(1, 3, 6)
        key = torch.arange(12.).reshape(1, 6, 1, 2)
        records = base.replay.capture_packed_gdn_inputs(conv, key, key, key[..., 0],
                                                       key[..., 0], (3, 2, 1), 2)
        self.assertEqual([record.key.shape[1] for record in records], [2, 2, 1])
        torch.testing.assert_close(records[1].key, key[:, 3:5])
        torch.testing.assert_close(records[2].conv_input, conv[:, :, 5:6])


if __name__ == "__main__":
    unittest.main()
