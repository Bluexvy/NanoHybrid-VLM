"""Offline v2 math, real config/header, conditional q, and state-routing checks."""
import json
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from helpers import DFlashDraftModel, tiny_config
from nanovllm.models.dflash import (CandidateSelector, GroupedDynamicCausalConv,
    dflash_checkpoint_shapes, read_dflash_config, validate_dflash_checkpoint)
from nanovllm.utils.loader import load_model
from nanovllm.engine.hybrid_state import HybridCacheSpec
from test_dflash_runtime import Runner, modules, batch
from download_qwen38 import describe_pair

FIXTURES = Path(__file__).parent / "fixtures"


def checkpoint_weights(model):
    weights = {}
    for name, value in model.state_dict().items():
        if "gate_up_proj" in name:
            for part, tensor in zip(("gate_proj", "up_proj"), value.chunk(2, 0)):
                weights[name.replace("gate_up_proj", part)] = tensor.clone()
        else:
            weights[name] = value.clone()
    return weights


class DFlash2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(202)

    def test_real_27b_config_shapes_tp_and_layer_routing_without_allocating_weights(self):
        from transformers import Qwen3_5Config
        root = Qwen3_5Config(**json.loads((FIXTURES / "qwen38_27b.json").read_text()))
        cfg = root.get_text_config(decoder=True)
        self.assertEqual((cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size), (64, 5120, 17408))
        spec = HybridCacheSpec.from_text_config(cfg, tensor_parallel_size=2)
        self.assertEqual(len(spec.gdn_layer_indices), 48)
        self.assertEqual(spec.full_attention_layer_indices, tuple(range(3, 64, 4)))
        self.assertEqual((spec.num_kv_heads, spec.num_gdn_value_heads, spec.conv_dim), (2, 24, 5120))
        self.assertEqual((spec.gdn_key_head_dim, spec.gdn_value_head_dim), (128, 128))
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_bytes((FIXTURES / "qwen38_27b_dflash2.json").read_bytes())
            draft = read_dflash_config(directory, cfg, tp_size=2)
        self.assertEqual(draft.target_layer_ids, (5, 19, 33, 47, 61))
        self.assertEqual((draft.block_size, draft.mask_token_id, draft.is_causal), (8, 248070, False))
        header = json.loads((FIXTURES / "qwen38_27b_dflash2_shapes.json").read_text())
        self.assertEqual(dflash_checkpoint_shapes(draft), {k: tuple(v["shape"]) for k, v in header.items()})
        # Construct every real-size draft parameter on meta, so shape checks
        # exercise module creation without allocating the 2B weight payload.
        with torch.device("meta"):
            model = DFlashDraftModel(draft, tp_rank=0, tp_size=2)
        self.assertEqual(tuple(model.layers[0].self_attn.q_proj.weight.shape), (2048, 5120))
        self.assertEqual(tuple(model.layers[0].self_attn.k_proj.weight.shape), (512, 5120))
        self.assertEqual(tuple(model.layers[0].mlp.down_proj.weight.shape), (5120, 8704))
        self.assertEqual(tuple(model.candidate_selector.predecessor_codebook.shape), (248320, 256))

    def test_real_27b_attention_projection_shapes_and_feature_layers(self):
        # Run the actual target Attention constructor with metadata-only tensors
        # and fake attention/RoPE backends; TP linear implementations are real.
        import ast
        from nanovllm.layers.linear import ColumnParallelLinear, RowParallelLinear
        source = batch.base.ROOT / "nanovllm/models/qwen3_5.py"
        definition = next(node for node in ast.parse(source.read_text(encoding="utf-8")).body
                          if isinstance(node, ast.ClassDef) and node.name == "Qwen3_5Attention")
        environment = {"torch": torch, "nn": torch.nn,
            "dist": SimpleNamespace(get_world_size=lambda: 2),
            "ColumnParallelLinear": ColumnParallelLinear, "RowParallelLinear": RowParallelLinear,
            "Qwen3_5RMSNorm": lambda width, **kwargs: torch.nn.LayerNorm(width),
            "get_rope": lambda **kwargs: torch.nn.Identity(), "Attention": lambda **kwargs: torch.nn.Identity()}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), environment)
        cfg = SimpleNamespace(**json.loads((FIXTURES / "qwen38_27b.json").read_text())["text_config"])
        with patch("torch.distributed.get_rank", return_value=0), \
             patch("torch.distributed.get_world_size", return_value=2), torch.device("meta"):
            attention = environment["Qwen3_5Attention"](cfg, 3)
        self.assertEqual((attention.num_heads, attention.num_kv_heads), (12, 2))
        self.assertEqual(tuple(attention.q_proj.weight.shape), (6144, 5120))  # Q + gate
        self.assertEqual(tuple(attention.o_proj.weight.shape), (5120, 3072))
        # Verify actual extraction indices across all 64 layers, not old 32.
        class Layer:
            block_type = "full_attention"
            def __call__(self, **kwargs):
                return kwargs["hidden_states"] + 1, None, None
        forward = batch.base.method("nanovllm/models/qwen3_5.py", "Qwen3_5TextModel", "forward", {"torch": torch})
        target = SimpleNamespace(hidden_size=1, num_hidden_layers=64, layers=[Layer() for _ in range(64)],
            embed_tokens=lambda ids: torch.zeros(len(ids), 1), norm=lambda x: x * 2,
            dflash_target_layer_ids=(5, 19, 33, 47, 61))
        forward(target, torch.tensor([7]), torch.tensor([0]))
        torch.testing.assert_close(target.last_dflash_hidden_states, torch.tensor([[6., 20., 34., 48., 62.]]))

    def test_dynamic_conv_matches_scalar_reference_and_uses_pre_input_for_post_kernel(self):
        conv = GroupedDynamicCausalConv(8, 2, 4)
        conv.base_kernel.data.normal_()
        x = torch.randn(4, 8)
        pre, dynamic = conv.prepare(x)
        all_dynamic = conv.kernel_projection(x).reshape(4, 2, 2, 2)
        def reference(value, which):
            out = torch.zeros_like(value)
            for t in range(4):
                for c in range(8):
                    for offset in range(2):
                        if t >= offset:
                            out[t, c] += (conv.base_kernel[which, offset, c]
                                + all_dynamic[t, which, offset, c // 4]) * value[t-offset, c]
            return out
        torch.testing.assert_close(pre, reference(x, 0))
        middle = torch.randn_like(x)
        torch.testing.assert_close(conv.finish(middle, dynamic), reference(middle, 1))
        # Two separate requests must never mix their last/first draft positions.
        altered = x.clone(); altered[-1] += 100
        torch.testing.assert_close(conv.prepare(altered)[0][0], pre[0])

    def test_selector_conditioning_greedy_and_sampling_q_match_manual_scores(self):
        cfg = tiny_config(2)
        selector = CandidateSelector(cfg)
        selector.predecessor_codebook.data.normal_()
        selector.successor_codebook.data.normal_()
        hidden, logits = torch.randn(3, 16), torch.randn(3, 32)
        unary, candidates = logits.topk(4, sorted=False)
        for temperature in (0, .8):
            tokens, distributions = selector.select(hidden, logits, 7, temperature)
            predecessor = 7
            for i, token in enumerate(tokens):
                score = unary[i] + ((selector.predecessor_codebook[predecessor]
                    * selector.hidden_projection(hidden)[i]) * selector.successor_codebook[candidates[i]]).sum(-1)
                if temperature:
                    expected = torch.zeros(32).scatter(0, candidates[i], torch.softmax(score.float() / temperature, -1))
                    torch.testing.assert_close(distributions[i], expected)
                    self.assertGreater(float(distributions[i][token].detach()), 0)
                    self.assertEqual(int(torch.count_nonzero(distributions[i])), 4)
                else:
                    self.assertEqual(token, int(candidates[i, score.argmax()]))
                predecessor = token

    def test_strict_checkpoint_loader_accepts_actual_raw_codebook_names(self):
        cfg = tiny_config(2)
        original = DFlashDraftModel(cfg, tp_rank=0, tp_size=1)
        weights = checkpoint_weights(original)
        self.assertIn("candidate_selector.predecessor_codebook", weights)
        with tempfile.TemporaryDirectory() as directory:
            file = str(Path(directory) / "model.safetensors")
            save_file(weights, file)
            validate_dflash_checkpoint(directory, cfg)
            loaded = DFlashDraftModel(cfg, tp_rank=0, tp_size=1)
            load_model(loaded, directory)
            for name, value in original.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)
            del weights["candidate_selector.predecessor_codebook"]
            save_file(weights, file)
            with self.assertRaisesRegex(ValueError, "missing"):
                validate_dflash_checkpoint(directory, cfg)

    def test_v2_runtime_proposes_conditional_q_and_commits_only_accepted_context(self):
        runner, seqs = Runner(), batch.requests()
        runner.dflash = DFlashDraftModel(tiny_config(2), tp_rank=0, tp_size=1).eval()
        # Fixture target emits only 16 logits, so use a matching candidate budget.
        for seq in seqs:
            runner._advance_mtp(seq, [0, 0], torch.randn(2, 32), 0)
            seq.temperature = .8
        with torch.inference_mode(), patch.dict(__import__('sys').modules, modules()):
            tokens, q = runner._draft_mtp(seqs[0], 2)
            self.assertEqual(len(tokens), 2)
            self.assertTrue(all(row.shape == (16,) for row in q))
            result = runner.run_mtp_batch(seqs, [2, 1, 0])
        for seq, emitted in zip(seqs, result):
            self.assertEqual(runner.dflash_caches[seq.seq_id][0][0].shape[1], 2 + len(emitted))

    def test_no_gpu_download_plan_and_pair_checks(self):
        target = json.loads((FIXTURES / "qwen38_27b.json").read_text())
        draft = json.loads((FIXTURES / "qwen38_27b_dflash2.json").read_text())
        self.assertEqual(describe_pair(target, draft)["local_gdn_value_heads"], 24)
        draft["hidden_size"] = 4096
        with self.assertRaisesRegex(ValueError, "dimensions"):
            describe_pair(target, draft)
        script = Path(__file__).parent / "download_qwen38.py"
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "not created"
            # No torch/huggingface import is permitted even if installed here.
            code = ("import builtins,runpy,sys; original=builtins.__import__; "
                "builtins.__import__=lambda name,*a,**k: (_ for _ in ()).throw(ImportError(name)) "
                "if name.split('.')[0] in ('torch','huggingface_hub') else original(name,*a,**k); "
                "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')")
            result = subprocess.run([sys.executable, "-c", code, str(script), "--root", str(destination), "--dry-run"],
                                    capture_output=True, text=True, check=True)
            self.assertIn("incoai/Qwen3.8-27B-DFlash2", result.stdout)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
