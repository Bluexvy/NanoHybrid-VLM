"""Offline checks for diagnostics, including restoration after a failed probe."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mtp"))
import cpu_runtime
from nanovllm.engine.numerical_diagnostics import (
    anchor_rows, diagnose_hybrid_step, logits_difference, tensor_difference,
    estimate_decode_splits,
    padded_decode_table,
    anchor_token_rows, trace_first_gdn,
)


class Gate(torch.nn.Module):
    def forward(self, hidden, gate):
        return hidden + gate


class Qwen3_5GatedDeltaNet(torch.nn.Module):
    """CPU fixture for the actual GDN observation boundaries, not GDN math."""
    def __init__(self):
        super().__init__()
        self.value_dim = 4
        self.conv_delta = 0.
        self.fail_conv = False
        for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
            setattr(self, name, torch.nn.Identity())
        self.norm = Gate()

    def _run_causal_conv1d(self, mixed_qkv, conv_state=None, prefill_seqlens=None):
        if self.fail_conv:
            raise RuntimeError("GDN convolution failure")
        return mixed_qkv + self.conv_delta, conv_state

    def forward(self, hidden_states):
        raw = self.in_proj_qkv(hidden_states).transpose(1, 2)
        core, _ = self._run_causal_conv1d(mixed_qkv=raw)
        core = core.transpose(1, 2)
        z = self.in_proj_z(hidden_states)
        self.in_proj_b(hidden_states)
        self.in_proj_a(hidden_states)
        output = self.norm((core * .5).reshape(-1, 2), z.reshape(-1, 2))
        return self.out_proj(output.reshape_as(hidden_states))


class Pool:
    def __init__(self, batch):
        self.value = torch.zeros(batch, 1, 2, 2)

    def read_batched_states(self, slots):
        value = self.value.clone()
        return [SimpleNamespace(conv_state=value[..., 0].clone(), recurrent_state=value)]

    def write_batched_states(self, slots, states):
        self.value.copy_(states[0].recurrent_state)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        language = SimpleNamespace(layers=torch.nn.ModuleList([torch.nn.Linear(4, 4, bias=False)]),
                                   norm=torch.nn.Identity())
        self.model = SimpleNamespace(language_model=language)

    def forward(self, hidden):
        language = self.model.language_model
        for layer in language.layers:
            hidden = layer(hidden)
        return language.norm(hidden)

    def compute_logits(self, hidden):
        return hidden


class Runner:
    def __init__(self, batch=2):
        self.hybrid_state_manager = Pool(batch)
        self.model = Model()
        self.rank, self.world_size, self.block_size = 0, 1, 256
        self.enforce_eager = False
        self.hybrid_graphs = {}
        self.fail_graph = False
        self.fail_padded = False
        self.seen_table_widths = []
        self.hybrid_graph_workspace = SimpleNamespace(
            hidden_states=torch.zeros(batch, 4), slot_mapping=torch.zeros(batch, dtype=torch.int32),
            context_lens=torch.ones(batch, dtype=torch.int32), block_tables=torch.zeros(batch, 1, dtype=torch.int32),
            copy_decode_inputs=lambda **kwargs: batch)

    def prepare_decode(self, seqs):
        from nanovllm.utils.context import set_context
        workspace = self.hybrid_graph_workspace
        set_context(False, slot_mapping=workspace.slot_mapping, context_lens=workspace.context_lens,
                    block_tables=workspace.block_tables)
        return torch.arange(len(seqs)), torch.zeros(3, len(seqs), dtype=torch.long)

    def run_hybrid_graph_body(self, batch):
        hidden = self.model(torch.eye(4)[:batch])
        self.hybrid_state_manager.value.add_(1)
        self.hybrid_graph_workspace.hidden_states.copy_(hidden)

    def run_hybrid_decode(self, seqs):
        from nanovllm.utils.context import get_context
        self.prepare_decode(seqs)
        width = get_context().block_tables.shape[1]
        self.seen_table_widths.append(width)
        if self.fail_padded and self.enforce_eager and width == 4:
            raise RuntimeError("diagnostic padded table failure")
        if not self.enforce_eager and self.fail_graph:
            raise RuntimeError("diagnostic graph failure")
        hidden = self.model(torch.eye(4)[:len(seqs)])
        self.hybrid_state_manager.value.add_(1)
        return self.model.compute_logits(hidden).argmax(-1).tolist()

    def _verify_mtp_batch(self, seqs, drafts):
        width = 1 + len(drafts[0])
        hidden = self.model(torch.eye(4)[:len(seqs)].repeat_interleave(width, 0))
        states = self.hybrid_state_manager.read_batched_states([s.state_slot for s in seqs])
        states[0].conv_state.add_(width)
        states[0].recurrent_state.add_(width)
        return hidden, hidden, states, {0: tuple(None for _ in seqs)}

    def _replay_gdn_inputs(self, seq, records, count):
        value = self.hybrid_state_manager.value[seq.state_slot:seq.state_slot+1].clone() + count
        return [SimpleNamespace(conv_state=value[..., 0], recurrent_state=value)]


def requests():
    return [SimpleNamespace(state_slot=i, num_cached_tokens=3, num_tokens=4, block_table=[i]) for i in range(2)]


class DiagnosticTests(unittest.TestCase):
    def test_token_anchors_align_packed_sequence_and_flattened_heads(self):
        tokens = torch.arange(24).reshape(1, 6, 4)
        self.assertEqual(anchor_token_rows(tokens, 2).tolist(), [[0, 1, 2, 3], [12, 13, 14, 15]])
        self.assertTrue(torch.equal(anchor_token_rows(tokens, 2),
                                    anchor_token_rows(tokens.reshape(-1, 2), 2, features=4)))

    def test_first_gdn_trace_aligns_all_stages_without_changing_output(self):
        gdn = Qwen3_5GatedDeltaNet()
        hidden = torch.arange(8).reshape(2, 1, 4).float()
        expected = gdn(hidden_states=hidden)
        with trace_first_gdn(gdn, 2) as stages:
            actual = gdn(hidden_states=hidden)
            self.assertTrue(torch.equal(actual, expected))
            baseline = dict(stages)
            stages.clear()
            packed = hidden.reshape(2, 4).repeat_interleave(3, 0).reshape(1, 6, 4)
            gdn(hidden_states=packed)
            self.assertEqual(len(stages), 11)
            for name, value in stages.items():
                self.assertTrue(torch.equal(value, baseline[name]), name)
        self.assertNotIn("_run_causal_conv1d", gdn.__dict__)
        for part in gdn.modules():
            self.assertFalse(part._forward_hooks)
            self.assertFalse(part._forward_pre_hooks)

    def test_first_gdn_trace_distinguishes_conv_output_from_history(self):
        gdn = Qwen3_5GatedDeltaNet()
        hidden = torch.arange(8).reshape(2, 1, 4).float()
        with trace_first_gdn(gdn, 2) as stages:
            gdn(hidden_states=hidden)
            baseline = dict(stages)
            stages.clear()
            gdn.conv_delta = .25
            gdn(hidden_states=hidden)
            first = next(name for name, value in stages.items() if not torch.equal(value, baseline[name]))
            self.assertEqual(first, "conv.output")
            self.assertTrue(torch.equal(stages["conv.input"], baseline["conv.input"]))

    def test_first_gdn_trace_failure_restores_existing_method_override(self):
        gdn = Qwen3_5GatedDeltaNet()
        original = gdn._run_causal_conv1d
        gdn._run_causal_conv1d = original
        gdn.fail_conv = True
        with self.assertRaisesRegex(RuntimeError, "GDN convolution failure"):
            with trace_first_gdn(gdn, 2):
                gdn(hidden_states=torch.ones(2, 1, 4))
        self.assertIs(gdn._run_causal_conv1d, original)
        for part in gdn.modules():
            self.assertFalse(part._forward_hooks)
            self.assertFalse(part._forward_pre_hooks)

    def test_splitkv_estimate_changes_partition_for_batch16(self):
        for batch in (1, 4):
            a = estimate_decode_splits(batch, 12, 2, 256, 256, 108)
            b = estimate_decode_splits(batch, 12, 2, 256, 1024, 108)
            self.assertEqual(a["tokens_per_split_estimate"], 64)
            self.assertEqual(b["tokens_per_split_estimate"], 64)
        self.assertEqual(estimate_decode_splits(16, 12, 2, 256, 256, 108)["num_splits_estimate"], 4)
        self.assertEqual(estimate_decode_splits(16, 12, 2, 256, 1024, 108)["num_splits_estimate"], 6)

    def test_metrics_report_ties_and_nonfinite_values(self):
        a = torch.tensor([[2., 2., 1.], [0., 4., 2.]])
        b = torch.tensor([[2., 2.01, 1.], [0., 4., 2.]])
        value = logits_difference(a, b)
        self.assertEqual(value["argmax_mismatches"], 1)
        self.assertEqual(value["rows"][0]["expected_margin"], 0.)
        self.assertFalse(tensor_difference(a, a * float("nan"))["finite"])
        self.assertIn("shape_mismatch", tensor_difference(a, b[:1]))

    def test_anchor_rows_select_request_starts(self):
        self.assertEqual(anchor_rows(torch.arange(12).reshape(6, 2), 2).tolist(), [[0, 1], [6, 7]])
        with self.assertRaises(ValueError):
            anchor_rows(torch.ones(5, 2), 2)

    def test_successful_comparisons_leave_only_baseline_step_committed(self):
        runner = Runner()
        original = runner.model.compute_logits
        result = diagnose_hybrid_step(runner, requests(), [[5, 6, 7], [8, 9, 10]])
        for mode in ("verify_one", "verify_block"):
            value = result["ranks"][0]["comparisons"][mode]
            self.assertEqual(value["logits"]["argmax_mismatches"], 0)
            self.assertTrue(value["states"][0]["recurrent"]["exact"])
            self.assertIsNone(value["first_nonidentical_layer"])
        self.assertTrue(torch.equal(runner.hybrid_state_manager.value, torch.ones_like(runner.hybrid_state_manager.value)))
        self.assertFalse(runner.enforce_eager)
        self.assertEqual(runner.model.compute_logits, original)
        self.assertFalse(runner.model.model.language_model.layers[0]._forward_hooks)

    def test_failed_probe_restores_baseline_step_and_removes_hooks(self):
        runner = Runner()
        runner.hybrid_graphs[2] = True
        runner.fail_graph = True
        original = runner.model.compute_logits
        with self.assertRaisesRegex(RuntimeError, "diagnostic graph failure"):
            diagnose_hybrid_step(runner, requests(), [[5], [6]])
        self.assertTrue(torch.equal(runner.hybrid_state_manager.value, torch.ones_like(runner.hybrid_state_manager.value)))
        self.assertFalse(runner.enforce_eager)
        self.assertEqual(runner.model.compute_logits, original)
        self.assertFalse(runner.model.model.language_model.layers[0]._forward_hooks)

    def test_static_workspace_and_graph_comparison_routes(self):
        runner = Runner()
        runner.hybrid_graphs[2] = True
        value = diagnose_hybrid_step(runner, requests(), [[5], [6]])
        results = value["ranks"][0]["comparisons"]
        self.assertEqual(set(results), {"decode_graph", "decode_padded_table", "decode_static_eager", "verify_one", "verify_block"})
        for mode, result in results.items():
            self.assertTrue(result["hidden"]["exact"], mode)
            self.assertEqual(result["logits"]["argmax_mismatches"], 0, mode)
        self.assertTrue(torch.equal(runner.hybrid_state_manager.value, torch.ones_like(runner.hybrid_state_manager.value)))
        self.assertTrue(results["decode_padded_table"]["against_graph"]["hidden"]["exact"])

    def test_table_padding_preserves_live_ids_lengths_and_slots(self):
        from nanovllm.utils.context import get_context, reset_context
        runner = Runner()
        runner.hybrid_graph_workspace.block_tables[:, 0] = torch.tensor([7, 11])
        original = runner.prepare_decode
        try:
            with padded_decode_table(runner, 4):
                runner.prepare_decode(requests())
                context = get_context()
                self.assertEqual(context.block_tables.tolist(), [[7, 0, 0, 0], [11, 0, 0, 0]])
                self.assertIs(context.context_lens, runner.hybrid_graph_workspace.context_lens)
                self.assertIs(context.slot_mapping, runner.hybrid_graph_workspace.slot_mapping)
                self.assertEqual(runner.hybrid_graph_workspace.block_tables.tolist(), [[7], [11]])
                raise RuntimeError("restore preparation")
        except RuntimeError:
            pass
        finally:
            reset_context()
        self.assertEqual(runner.prepare_decode, original)

    def test_failed_padding_probe_restores_preparation_and_baseline_state(self):
        runner = Runner()
        runner.hybrid_graphs[2] = True
        runner.hybrid_graph_workspace.block_tables = torch.zeros(2, 4, dtype=torch.int32)
        # Real prepare_decode creates the compact dynamic table independently.
        original = runner.prepare_decode
        def compact(seqs):
            from nanovllm.utils.context import get_context
            result = original(seqs)
            get_context().block_tables = torch.zeros(2, 1, dtype=torch.int32)
            return result
        runner.prepare_decode = compact
        runner.fail_padded = True
        with self.assertRaisesRegex(RuntimeError, "diagnostic padded table failure"):
            diagnose_hybrid_step(runner, requests(), [[5], [6]])
        self.assertEqual(runner.seen_table_widths, [1, 1, 4])
        self.assertIs(runner.prepare_decode, compact)
        self.assertTrue(torch.equal(runner.hybrid_state_manager.value, torch.ones_like(runner.hybrid_state_manager.value)))
        self.assertFalse(runner.enforce_eager)
        self.assertFalse(runner.model.model.language_model.layers[0]._forward_hooks)

    def test_non_probe_step_only_advances_the_baseline(self):
        runner = Runner()
        value = diagnose_hybrid_step(runner, requests(), [[5], [6]], False)
        self.assertIn("baseline_ids", value)
        self.assertNotIn("ranks", value)
        self.assertFalse(runner.enforce_eager)


if __name__ == "__main__":
    unittest.main()
