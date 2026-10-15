from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import cpu_runtime
from nanovllm.config import Config


class ConfigTests(unittest.TestCase):
    def config(self, directory, **kwargs):
        text = SimpleNamespace(layer_types=("linear_attention", "full_attention"),
                               max_position_embeddings=1024, num_experts=0)
        root = SimpleNamespace(model_type="qwen3_5", get_text_config=lambda decoder: text)
        with patch("nanovllm.config.AutoConfig.from_pretrained", return_value=root), \
             patch("nanovllm.engine.speculative.validate_mtp_checkpoint"):
            return Config(directory, **kwargs)

    def test_tp_batch_graph_prefix_and_multidraft_are_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory, tensor_parallel_size=2, max_num_seqs=4,
                hybrid_cuda_graph_batch_sizes=(1, 2, 4), mtp_num_speculative_tokens=3,
                max_num_batched_tokens=32, hybrid_prefix_cache_mode="opportunistic")
            self.assertFalse(config.enforce_eager)
            self.assertEqual(config.mtp_num_speculative_tokens, 3)

    def test_token_budget_and_invalid_draft_count_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for count in (-1, 16, True):
                with self.assertRaises(ValueError):
                    self.config(directory, mtp_num_speculative_tokens=count)
            with self.assertRaisesRegex(ValueError, "budget"):
                self.config(directory, mtp_num_speculative_tokens=3, max_num_batched_tokens=2)

    def test_bad_verification_backend_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "verification_backend"):
                self.config(directory, mtp_gdn_verification_backend="unknown")


if __name__ == "__main__":
    unittest.main()
