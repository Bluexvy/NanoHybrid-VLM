"""Real CUDA Graph/accepted-state checks with a tiny hybrid target."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import helpers
import test_mtp_gpu as gpu
from nanovllm.engine.dflash_runner import DFlashRunnerMixin
from nanovllm.models.dflash import DFlashDraftModel


class Target(gpu.Target):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(language_model=SimpleNamespace())

    def forward(self, **kwargs):
        hidden, updated = super().forward(**kwargs)
        self.model.language_model.last_dflash_hidden_states = torch.cat((hidden, hidden * .5), -1)
        return hidden, updated


class Runner(DFlashRunnerMixin, gpu.Runner):
    def __init__(self):
        super().__init__()
        self.config.speculative_method = "dflash"
        self.model = Target().cuda().eval()
        config = helpers.tiny_config()
        config.hidden_size, config.intermediate_size = 8, 12
        config.num_attention_heads, config.num_key_value_heads = 2, 1
        config.num_target_layers, config.target_layer_ids = 5, (1, 4)
        config.vocab_size, config.mask_token_id = 16, 15
        self.dflash = DFlashDraftModel(config, tp_rank=0, tp_size=1).cuda().eval()


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA PyTorch")
class GPUIntegrationTests(unittest.TestCase):
    def test_verification_graph_keeps_its_own_live_multilayer_features(self):
        torch.manual_seed(157)
        runner, seqs = Runner(), gpu.requests()
        for iteration in range(2):
            if iteration:
                seqs.reverse()
                seqs[0].token_ids[-1] = 5
            drafts = [[4, 5], [8, 9]]
            with torch.inference_mode():
                expected_logits, expected_features, expected_states, _ = runner._verify_mtp_batch(seqs, drafts)
                expected_features = expected_features.clone()
                actual_logits, actual_features, actual_states, _ = runner.verify_mtp_with_graph(seqs, drafts)
            torch.testing.assert_close(actual_logits, expected_logits, rtol=3e-5, atol=3e-6)
            torch.testing.assert_close(actual_features, expected_features, rtol=3e-5, atol=3e-6)
            for index in (1, 4):
                torch.testing.assert_close(actual_states[index].recurrent_state,
                    expected_states[index].recurrent_state, rtol=3e-5, atol=3e-6)
        self.assertEqual(runner.mtp_stats["verification_graph_captures"], 1)
        self.assertEqual(runner.mtp_stats["verification_graph_replays"], 2)

    def test_partial_acceptance_commits_only_valid_gdn_and_draft_context(self):
        torch.manual_seed(159)
        runner, seqs = Runner(), gpu.requests()
        expected, proposals = {}, {}
        with torch.inference_mode():
            for row, seq in enumerate(seqs):
                runner._advance_mtp(seq, [0, 0], torch.randn(2, 16, device="cuda"), 0)
                scores, _, _, _ = runner._verify_mtp_inputs(seq, [])
                prediction = int(scores[0].argmax())
                proposals[seq.seq_id] = prediction if row == 0 else (prediction + 1) % 16
                used = [proposals[seq.seq_id]] if row == 0 else []
                _, features, states, _ = runner._verify_mtp_inputs(seq, used)
                positions = torch.arange(2, 2 + features.shape[0], device="cuda")
                cache = runner.dflash.append_context(features, positions, runner.dflash_caches[seq.seq_id])
                expected[seq.seq_id] = states, cache
            with patch.object(runner, "_draft_mtp", side_effect=lambda seq, count:
                              ([proposals[seq.seq_id]], [None])):
                result = runner.run_mtp_batch(seqs, [1, 1])
        self.assertEqual([len(tokens) for tokens in result], [2, 1])
        self.assertEqual(runner.mtp_stats["verification_graph_replays"], 1)
        for seq in seqs:
            state, cache = expected[seq.seq_id]
            active = runner.hybrid_state_manager.read_states(seq.state_slot)
            for layer in (1, 4):
                torch.testing.assert_close(active[layer].recurrent_state, state[layer].recurrent_state,
                                           rtol=3e-5, atol=3e-6)
            for a, b in zip(cache, runner.dflash_caches[seq.seq_id]):
                for x, y in zip(a, b):
                    torch.testing.assert_close(x, y, rtol=3e-5, atol=3e-6)


if __name__ == "__main__":
    unittest.main()
