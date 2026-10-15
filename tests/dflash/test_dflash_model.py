import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from helpers import DFlashDraftModel, read_dflash_config, tiny_config, tiny_raw
from nanovllm.models.dflash import validate_dflash_checkpoint
from safetensors.torch import save_file


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(52)
        self.model = DFlashDraftModel(tiny_config(), tp_rank=0, tp_size=1).eval()

    def test_context_chunks_match_full_and_draft_never_mutates_cache(self):
        feature = torch.randn(7, 32)
        full = self.model.append_context(feature, torch.arange(7))
        first = self.model.append_context(feature[:4], torch.arange(4))
        chunked = self.model.append_context(feature[4:], torch.arange(4, 7), first)
        saved = tuple(tuple(t.clone() for t in kv) for kv in chunked)
        for full_kv, actual_kv in zip(full, chunked):
            for expected, actual in zip(full_kv, actual_kv):
                torch.testing.assert_close(actual, expected)
        noise = torch.randn(4, 16)
        a = self.model(noise, torch.arange(7, 11), full)
        b = self.model(noise, torch.arange(7, 11), chunked)
        torch.testing.assert_close(a, b)
        self.model(noise + 100, torch.arange(7, 11), chunked)
        for old, actual in zip(saved, chunked):
            for x, y in zip(old, actual):
                torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_noise_attention_mask_is_causal_only_for_sliding_layers(self):
        cache = self.model.append_context(torch.randn(5, 32), torch.arange(5))
        x, positions = torch.randn(4, 16), torch.arange(5, 9)
        changed = x.clone()
        changed[-1] += 10
        # A sliding causal layer cannot see changed future noise positions.
        attn = self.model.layers[0].self_attn
        a, b = attn(x, positions, cache[0]), attn(changed, positions, cache[0])
        torch.testing.assert_close(a[:-1], b[:-1], atol=1e-6, rtol=1e-6)
        # Full draft attention deliberately sees all positions in its block.
        attn = self.model.layers[1].self_attn
        self.assertGreater(float((attn(x, positions, cache[1])[0]
                                  - attn(changed, positions, cache[1])[0]).detach().abs().max()), 1e-4)

    def test_config_rejects_wrong_target_and_unsupported_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "config.json"
            file.write_text(json.dumps(tiny_raw()))
            target = SimpleNamespace(hidden_size=16, vocab_size=33, num_hidden_layers=6)
            with self.assertRaisesRegex(ValueError, "does not match"):
                read_dflash_config(directory, target)
            for changes in ({"architectures": ["DFlash2DraftModel"]},
                            {"rope_parameters": {"rope_type": "yarn"}},
                            {"dflash_config": {"target_layer_ids": [8], "mask_token_id": 31}}):
                raw = tiny_raw()
                raw.update(changes)
                file.write_text(json.dumps(raw))
                with self.assertRaises(ValueError):
                    read_dflash_config(directory)

    def test_checkpoint_headers_detect_missing_or_wrong_shaped_weights(self):
        weights = {}
        for name, value in self.model.state_dict().items():
            if "gate_up_proj" in name:
                gate, up = value.chunk(2, 0)
                weights[name.replace("gate_up_proj", "gate_proj")] = gate.clone()
                weights[name.replace("gate_up_proj", "up_proj")] = up.clone()
            else:
                weights[name] = value
        with tempfile.TemporaryDirectory() as directory:
            file = str(Path(directory) / "model.safetensors")
            save_file(weights, file)
            validate_dflash_checkpoint(directory, self.model.config)
            original = weights["fc.weight"]
            weights["fc.weight"] = original.to(torch.int8)
            save_file(weights, file)
            with self.assertRaisesRegex(ValueError, "unsupported_dtype"):
                validate_dflash_checkpoint(directory, self.model.config)
            weights["fc.weight"] = torch.zeros(1, 1)
            save_file(weights, file)
            with self.assertRaisesRegex(ValueError, "wrong_shape"):
                validate_dflash_checkpoint(directory, self.model.config)
            del weights["fc.weight"]
            save_file(weights, file)
            with self.assertRaisesRegex(ValueError, "missing"):
                validate_dflash_checkpoint(directory, self.model.config)


if __name__ == "__main__":
    unittest.main()
