from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import helpers
import test_mtp_batch as batch
from nanovllm.engine.dflash_runner import DFlashRunnerMixin
from nanovllm.engine.prefix_cache import PrefixKey
from nanovllm.models.dflash import DFlashDraftModel


class Target(batch.Target):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 16)
        self.model = SimpleNamespace(language_model=SimpleNamespace())

    def embed_input_ids(self, ids):
        return self.embedding(ids)

    def __call__(self, **kwargs):
        hidden, updated = super().__call__(**kwargs)
        self.model.language_model.last_dflash_hidden_states = torch.cat((hidden, hidden * .5), -1)
        return hidden, updated


class Runner(DFlashRunnerMixin, batch.Runner):
    def __init__(self):
        super().__init__()
        self.config.speculative_method = "dflash"
        self.model = Target()
        self.dflash = DFlashDraftModel(helpers.tiny_config(), tp_rank=0, tp_size=1).eval()
        # Target fixture has 16 logits; draft fixture mask embedding has 32 IDs.
        self.config.max_model_len = 32
        self.mtp_proposals.clear()


def modules():
    return {"nanovllm.engine.speculative": batch.base.acceptance,
            "nanovllm.engine.gdn_replay": batch.base.replay,
            "nanovllm.utils.context": SimpleNamespace(get_context=lambda: batch.base._capture_context,
                                                     reset_context=batch.base.reset_capture)}


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(63)

    def initialize(self, runner, seqs):
        for seq in seqs:
            runner._advance_mtp(seq, [0] * seq.num_cached_tokens,
                torch.randn(seq.num_cached_tokens, 32), 0)

    def test_parallel_draft_uses_mask_positions_and_does_not_commit_noise(self):
        runner, seq = Runner(), batch.requests()[0]
        self.initialize(runner, [seq])
        saved = runner.mtp_prefix_snapshot(seq.seq_id, seq.num_cached_tokens).clone()
        observed = []
        hook = runner.dflash.register_forward_pre_hook(lambda module, args: observed.append(args[0].clone()))
        try:
            with torch.inference_mode(), patch.dict(sys.modules, modules()):
                for temperature in (0, .8):
                    seq.temperature = temperature
                    tokens, probabilities = runner._draft_mtp(seq, 2)
                    ids = torch.tensor([seq[seq.num_cached_tokens], 31, 31])
                    expected_embedding = runner.model.embed_input_ids(ids)
                    torch.testing.assert_close(observed[-1], expected_embedding)
                    expected_hidden = runner.dflash(expected_embedding, torch.arange(2, 5), saved.layers)
                    scores = runner._mtp_logits(expected_hidden[1:])
                    self.assertEqual(len(tokens), 2)
                    if temperature == 0:
                        self.assertEqual(tokens, scores.argmax(-1).tolist())
                    else:
                        expected = torch.softmax(scores.float() / temperature, -1)
                        torch.testing.assert_close(torch.stack(probabilities), expected)
        finally:
            hook.remove()
        for original, actual in zip(saved.layers, runner.dflash_caches[seq.seq_id]):
            for a, b in zip(original, actual):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(runner.mtp_stats["dflash_draft_forwards"], 2)

    def test_ragged_verification_commits_only_accepted_features_and_gdn(self):
        runner, seqs = Runner(), batch.requests()
        self.initialize(runner, seqs)
        expected, checkpoints = {}, {}
        with torch.inference_mode(), patch.dict(sys.modules, modules()):
            for seq, inputs in zip(seqs, ([3, 4], [7, 8], [4])):
                batch.base._capture_context.prefill_seqlens = (len(inputs),)
                hidden, states = runner.model(input_ids=torch.tensor(inputs), positions=None,
                    gdn_states=runner.hybrid_state_manager.read_batched_states([seq.state_slot]))
                features = runner.model.model.language_model.last_dflash_hidden_states
                cache = runner.dflash.append_context(features,
                    torch.arange(seq.num_cached_tokens, seq.num_cached_tokens + len(inputs)),
                    runner.dflash_caches[seq.seq_id])
                expected[seq.seq_id] = states, cache
                checkpoints[seq.seq_id] = runner.mtp_prefix_snapshot(seq.seq_id, seq.num_cached_tokens).clone()
            proposals = {1: [4, 8], 2: [8], 3: []}
            with patch.object(runner, "_draft_mtp", side_effect=lambda seq, count:
                              (proposals[seq.seq_id][:count], [None] * count)):
                result = runner.run_mtp_batch(seqs, [2, 1, 0])
        self.assertEqual(result, [[4, 5], [8, 9], [5]])
        self.assertEqual(runner.mtp_stats["verification_request_rounds"], 3)
        self.assertEqual(runner.mtp_stats["verification_batch_rounds"], 1)
        self.assertEqual(runner.mtp_stats["emitted_tokens"], 5)
        for seq in seqs:
            states, cache = expected[seq.seq_id]
            for layer in (1, 4):
                actual = runner.hybrid_state_manager.requests[seq.state_slot][layer]
                torch.testing.assert_close(actual.conv_state, states[layer].conv_state)
                torch.testing.assert_close(actual.recurrent_state, states[layer].recurrent_state,
                                           atol=2e-6, rtol=2e-6)
            for expected_kv, actual_kv in zip(cache, runner.dflash_caches[seq.seq_id]):
                for a, b in zip(expected_kv, actual_kv):
                    torch.testing.assert_close(a, b)
            snapshot = checkpoints[seq.seq_id]
            for saved, actual in zip(snapshot.layers, runner.dflash_caches[seq.seq_id]):
                torch.testing.assert_close(saved[0], actual[0][:, :snapshot.num_cached_tokens])

    def test_prefix_restore_is_independent_and_next_suffix_is_not_cached(self):
        runner, seq = Runner(), batch.requests()[0]
        self.initialize(runner, [seq])
        key = PrefixKey("dflash-test", 123, seq.num_cached_tokens)
        runner.cache_mtp_prefix(key, seq.seq_id)
        saved = runner.mtp_prefix_snapshots[key].clone()
        runner.restore_mtp_prefix(key, 99)
        runner.dflash_caches[99][0][0].add_(100)
        for actual, expected in zip(runner.mtp_prefix_snapshots[key].layers, saved.layers):
            for x, y in zip(actual, expected):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
        runner.release_mtp_requests([99])
        self.assertNotIn(99, runner.dflash_caches)
        self.assertNotIn(99, runner.mtp_proposals)

    def test_prefill_chunks_install_aligned_context(self):
        runner, seq = Runner(), batch.requests()[0]
        seq.is_multimodal = False
        seq.num_cached_tokens, seq.num_scheduled_tokens = 0, 2
        features = torch.randn(3, 32)
        runner.model.model.language_model.last_dflash_hidden_states = features[:2]
        runner.update_mtp_after_prefill([seq], [9])
        seq.num_cached_tokens, seq.num_scheduled_tokens = 2, 1
        runner.model.model.language_model.last_dflash_hidden_states = features[2:]
        runner.update_mtp_after_prefill([seq], [10])
        expected = runner.dflash.append_context(features, torch.arange(3))
        for a, b in zip(expected, runner.dflash_caches[seq.seq_id]):
            for x, y in zip(a, b):
                torch.testing.assert_close(x, y)
        seq.is_multimodal = True
        with self.assertRaisesRegex(ValueError, "text requests"):
            runner.update_mtp_after_prefill([seq], [10])

    def test_actual_text_forward_extracts_requested_decoder_outputs(self):
        # Exercise the actual target forward without importing CUDA-only layers.
        class Layer:
            block_type = "full_attention"
            def __call__(self, **kwargs):
                return kwargs["hidden_states"] + 1, None, None
        forward = batch.base.method("nanovllm/models/qwen3_5.py", "Qwen3_5TextModel", "forward",
                                    {"torch": torch})
        target = SimpleNamespace(hidden_size=2, num_hidden_layers=4,
            layers=[Layer() for _ in range(4)], embed_tokens=lambda ids: ids[:, None].expand(-1, 2).float(),
            norm=lambda x: x * 2, dflash_target_layer_ids=(1, 3))
        final, _ = forward(target, torch.tensor([1, 5]), torch.arange(2))
        torch.testing.assert_close(final, torch.tensor([[10., 10.], [18., 18.]]))
        torch.testing.assert_close(target.last_dflash_hidden_states,
            torch.tensor([[3., 3., 10., 10.], [7., 7., 18., 18.]]))


if __name__ == "__main__":
    unittest.main()
