import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from helpers import tiny_raw
from nanovllm.config import Config


class ConfigTests(unittest.TestCase):
    def make(self, directory, **kwargs):
        text = SimpleNamespace(hidden_size=16, vocab_size=32, num_hidden_layers=6,
            layer_types=("linear_attention", "full_attention"), max_position_embeddings=256, num_experts=0)
        root = SimpleNamespace(model_type="qwen3_5", get_text_config=lambda decoder: text)
        with patch("nanovllm.config.AutoConfig.from_pretrained", return_value=root), \
             patch("nanovllm.models.dflash.validate_dflash_checkpoint"), \
             patch("nanovllm.engine.speculative.validate_mtp_checkpoint") as native:
            result = Config(directory, **kwargs)
            native.assert_not_called()
            return result

    def test_dflash_does_not_require_native_mtp_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text(json.dumps(tiny_raw()))
            cfg = self.make(directory, speculative_method="dflash", dflash_model=directory,
                dflash_num_speculative_tokens=3, tensor_parallel_size=2, max_num_seqs=4,
                hybrid_cuda_graph_batch_sizes=(1, 2, 4), hybrid_prefix_cache_mode="opportunistic")
            self.assertEqual(cfg.mtp_num_speculative_tokens, 3)
            self.assertLessEqual(cfg.max_model_len, 128)

    def test_invalid_modes_budget_and_block_size_fail_early(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text(json.dumps(tiny_raw()))
            for kwargs in ({"speculative_method": "unknown"}, {"dflash_model": directory},
                           {"speculative_method": "dflash"},
                           {"speculative_method": "dflash", "dflash_model": directory,
                            "mtp_num_speculative_tokens": 1},
                           {"speculative_method": "dflash", "dflash_model": directory,
                            "dflash_num_speculative_tokens": 4},
                           {"speculative_method": "dflash", "dflash_model": directory,
                            "dflash_num_speculative_tokens": 2, "max_num_batched_tokens": 2}):
                with self.assertRaises(ValueError):
                    self.make(directory, **kwargs)


if __name__ == "__main__":
    unittest.main()
