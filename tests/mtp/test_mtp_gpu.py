"""Small actual PyTorch CUDA Graph/runtime checks, without a 9B checkpoint."""
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
import torch
import test_mtp_cpu as base

# Avoid the public LLM entry point, which requires Linux flash-attn/Triton.
ROOT = Path(__file__).resolve().parents[2]
for name, relative in (("nanovllm", "nanovllm"), ("nanovllm.engine", "nanovllm/engine"),
                       ("nanovllm.utils", "nanovllm/utils")):
    if name not in sys.modules:
        package = ModuleType(name)
        package.__path__ = [str(ROOT / relative)]
        sys.modules[name] = package
from nanovllm.engine.hybrid_state import HybridCacheSpec, HybridStateManager
from nanovllm.engine.mtp_runner import MTPRunnerMixin
from nanovllm.engine.mtp_graph import MTPVerificationGraphMixin
from nanovllm.engine.gdn_replay import capture_packed_gdn_inputs
from nanovllm.utils.context import get_context, set_context, reset_context
from nanovllm.kernels.gdn_verification import verification_conv, verification_recurrent


namespace = {"torch": torch, "F": torch.nn.functional, "get_context": get_context,
    "verification_conv": verification_conv, "verification_recurrent": verification_recurrent,
    "capture_packed_gdn_inputs": capture_packed_gdn_inputs,
    "torch_causal_conv1d_reference": base._conv_reference,
    "torch_recurrent_gated_delta_rule": base._recurrent_reference}


class GPUGDN(base.ToyGDN):
    forward = base.method("nanovllm/layers/gated_delta_net.py", "Qwen3_5GatedDeltaNet", "forward", namespace)
    _run_causal_conv1d = base.method("nanovllm/layers/gated_delta_net.py", "Qwen3_5GatedDeltaNet",
                                    "_run_causal_conv1d", namespace)
    _run_gated_delta_rule = base.method("nanovllm/layers/gated_delta_net.py", "Qwen3_5GatedDeltaNet",
                                      "_run_gated_delta_rule", namespace)
    def __init__(self, layer):
        super().__init__(layer)
        self.activation = "silu"
        self.conv1d = torch.nn.Conv1d(10, 10, 4, groups=10, bias=False)
        self.conv1d.weight.data.copy_(self.conv_weight.unsqueeze(1))


class Target(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 8)
        self.layers = torch.nn.ModuleList([GPUGDN(1), GPUGDN(4)])
        self.lm_head = torch.nn.Linear(8, 16, bias=False)
        config = SimpleNamespace(hidden_size=8, intermediate_size=12, num_attention_heads=2,
                                 num_key_value_heads=1, head_dim=4, rms_norm_eps=1e-6)
        self.mtp = base.head.Qwen3_5MTP(config)

    def embed_input_ids(self, ids):
        return self.embedding(ids)

    def forward(self, input_ids, positions, gdn_states):
        hidden = self.embedding(input_ids).unsqueeze(0)
        updated = [None] * 5
        for layer in self.layers:
            initial = gdn_states[layer.layer_idx]
            hidden, conv, state = layer(hidden, initial.conv_state, initial.recurrent_state,
                prefill_seqlens=get_context().prefill_seqlens)
            updated[layer.layer_idx] = type(initial)(conv_state=conv, recurrent_state=state)
        return hidden.squeeze(0), updated


class Runner(MTPVerificationGraphMixin, MTPRunnerMixin):
    def __init__(self):
        self.initialize_mtp_runtime()
        self.world_size, self.rank, self.block_size = 1, 0, 4
        self.uses_multimodal_rope = True
        self.config = SimpleNamespace(enforce_eager=False, hybrid_cuda_graph_batch_sizes=(1, 2),
            mtp_gdn_rollback="replay_inputs", mtp_gdn_replay_backend="reference",
            mtp_num_speculative_tokens=2, max_model_len=16, eos=-1, mtp_cuda_graph_max_entries=4)
        self.mtp_graph_memory_budget = 64 * 1024 * 1024
        self.model = Target().cuda().eval()
        config = SimpleNamespace(num_key_value_heads=1, linear_num_key_heads=1,
            linear_num_value_heads=2,
            layer_types=("full_attention", "linear_attention", "full_attention", "full_attention", "linear_attention"),
            num_hidden_layers=5, num_attention_heads=2, hidden_size=8, head_dim=4,
            linear_key_head_dim=3, linear_value_head_dim=2, linear_conv_kernel_dim=4, dtype=torch.float32)
        self.hybrid_state_manager = HybridStateManager(HybridCacheSpec.from_text_config(config), 3, "cuda:0")
        self.hybrid_state_manager.conv_state_pool.normal_()
        self.hybrid_state_manager.recurrent_state_pool.normal_()
        self.hybrid_state_manager.initialized_slots[:] = [True] * 3

    def _mtp_rotary(self):
        return lambda positions, q, k: (q, k)

    def prepare_prefill(self, seqs):
        lengths = tuple(seq.num_scheduled_tokens for seq in seqs)
        ids = [token for seq in seqs for token in seq[
            seq.num_cached_tokens:seq.num_cached_tokens + seq.num_scheduled_tokens]]
        cumulative_q, cumulative_k = [0], [0]
        for seq in seqs:
            cumulative_q.append(cumulative_q[-1] + seq.num_scheduled_tokens)
            cumulative_k.append(cumulative_k[-1] + seq.num_cached_tokens + seq.num_scheduled_tokens)
        ids = torch.tensor(ids, device="cuda")
        positions = torch.arange(ids.numel(), device="cuda").unsqueeze(0).expand(3, -1).contiguous()
        tables = torch.zeros(len(seqs), 2, dtype=torch.int32, device="cuda")
        # Shape must match the largest real context, just as ModelRunner prepares.
        max_end = max(seq.num_cached_tokens + seq.num_scheduled_tokens for seq in seqs)
        tables = tables[:, : (max_end + 3) // 4].contiguous()
        set_context(True, cu_seqlens_q=torch.tensor(cumulative_q, dtype=torch.int32, device="cuda"),
            cu_seqlens_k=torch.tensor(cumulative_k, dtype=torch.int32, device="cuda"),
            max_seqlen_q=max(lengths), max_seqlen_k=max_end,
            slot_mapping=torch.arange(ids.numel(), dtype=torch.int32, device="cuda"),
            block_tables=tables, prefill_seqlens=lengths,
            gdn_cu_seqlens=torch.tensor(cumulative_q, dtype=torch.long, device="cuda"))
        return ids, positions


def requests():
    seqs = [base.FakeSequence(), base.FakeSequence()]
    seqs[0].seq_id, seqs[0].state_slot = 11, 0
    seqs[1].seq_id, seqs[1].state_slot = 22, 2
    seqs[1].token_ids[-1] = 7
    return seqs


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA PyTorch")
class GPUIntegrationTests(unittest.TestCase):
    def test_draft_graph_dynamic_lengths_dtypes_and_request_cache_isolation(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            torch.manual_seed(149)
            runner = Runner()
            runner.model.mtp.to(dtype=dtype)
            saved = []
            for length in (0, 1, 2, 3):
                attention = runner.model.mtp.layers[0].self_attn
                past = None if length == 0 else (
                    torch.randn(attention.kv_heads, length, attention.dim, device="cuda", dtype=dtype),
                    torch.randn(attention.kv_heads, length, attention.dim, device="cuda", dtype=dtype))
                embeddings = torch.randn(1, 8, device="cuda", dtype=dtype)
                features = torch.randn_like(embeddings)
                positions = torch.full((3, 1), length + 1, device="cuda", dtype=torch.long)
                with torch.inference_mode():
                    expected, expected_cache = runner.model.mtp(positions, embeddings, features, past, runner._mtp_rotary())
                    actual, actual_cache = runner.mtp_head_with_graph(positions, embeddings, features, past)
                tolerance = (3e-5, 3e-6) if dtype == torch.float32 else (.02, .005)
                torch.testing.assert_close(actual, expected, rtol=tolerance[0], atol=tolerance[1])
                for tensor, wanted in zip(actual_cache, expected_cache):
                    torch.testing.assert_close(tensor, wanted, rtol=tolerance[0], atol=tolerance[1])
                for tensors, original in saved:
                    for tensor, wanted in zip(tensors, original):
                        torch.testing.assert_close(tensor, wanted, rtol=0, atol=0)
                saved.append((actual_cache, tuple(tensor.clone() for tensor in actual_cache)))
            self.assertEqual(runner.mtp_stats["draft_graph_captures"], 3)
            self.assertEqual(runner.mtp_stats["draft_graph_replays"], 4)

    def test_actual_capture_replay_changed_inputs_and_state_slots(self):
        torch.manual_seed(113)
        runner, seqs = Runner(), requests()
        manager = runner.hybrid_state_manager
        before_conv, before_state = manager.conv_state_pool.clone(), manager.recurrent_state_pool.clone()
        for iteration in range(2):
            if iteration:
                seqs.reverse()
                seqs[0].token_ids[-1] = 5
            proposals = [[4, 5], [8, 9]]
            with torch.inference_mode():
                expected_logits, expected_hidden, expected_states, _ = runner._verify_mtp_batch(seqs, proposals)
                actual_logits, actual_hidden, actual_states, _ = runner.verify_mtp_with_graph(seqs, proposals)
            torch.testing.assert_close(actual_logits, expected_logits, rtol=3e-5, atol=3e-6)
            torch.testing.assert_close(actual_hidden, expected_hidden, rtol=3e-5, atol=3e-6)
            for layer in (1, 4):
                torch.testing.assert_close(actual_states[layer].conv_state, expected_states[layer].conv_state)
                torch.testing.assert_close(actual_states[layer].recurrent_state, expected_states[layer].recurrent_state,
                                           rtol=3e-5, atol=3e-6)
        self.assertEqual(runner.mtp_stats["verification_graph_captures"], 1)
        self.assertEqual(runner.mtp_stats["verification_graph_replays"], 2)
        torch.testing.assert_close(manager.conv_state_pool, before_conv, rtol=0, atol=0)
        torch.testing.assert_close(manager.recurrent_state_pool, before_state, rtol=0, atol=0)

    def test_actual_graph_accept_reject_commits_correct_gdn_boundaries(self):
        torch.manual_seed(127)
        runner, seqs = Runner(), requests()
        wanted = {}
        with torch.inference_mode():
            for row, seq in enumerate(seqs):
                runner._advance_mtp(seq, [2, seq.token_ids[-1]], torch.randn(2, 8, device="cuda"), 0)
                scores, _, _, _ = runner._verify_mtp_inputs(seq, [])
                prediction = int(scores[0].argmax())
                proposal = prediction if row == 0 else (prediction + 1) % 16
                runner.mtp_proposals[seq.seq_id] = proposal
                _, _, states, _ = runner._verify_mtp_inputs(seq, [proposal] if row == 0 else [])
                wanted[seq.seq_id] = states
            result = runner.run_mtp_batch(seqs, [1, 1])
        self.assertEqual([len(tokens) for tokens in result], [2, 1])
        self.assertEqual(runner.mtp_stats["verification_graph_replays"], 1)
        for seq in seqs:
            actual = runner.hybrid_state_manager.read_states(seq.state_slot)
            for layer in (1, 4):
                torch.testing.assert_close(actual[layer].conv_state, wanted[seq.seq_id][layer].conv_state)
                torch.testing.assert_close(actual[layer].recurrent_state, wanted[seq.seq_id][layer].recurrent_state,
                                           rtol=3e-5, atol=3e-6)


if __name__ == "__main__":
    unittest.main()
