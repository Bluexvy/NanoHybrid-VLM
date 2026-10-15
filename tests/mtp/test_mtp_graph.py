"""CPU checks of graph metadata routing and its actual batched GDN math."""
from types import SimpleNamespace
import unittest
import torch
import cpu_runtime
import test_mtp_cpu as base
from nanovllm.engine.mtp_graph import MTPVerificationWorkspace, MTPVerificationGraphMixin
from nanovllm.utils.context import get_context, set_context, reset_context
from nanovllm.kernels.gdn_verification import verification_conv, verification_recurrent


class GraphTests(unittest.TestCase):
    def test_workspace_addresses_stable_across_request_reordering(self):
        workspace = MTPVerificationWorkspace.allocate(2, 3, 16, 4, 3, "cpu")
        fields = ("input_ids", "positions", "slot_mapping", "block_tables", "state_slots", "cu_q", "cu_k", "gdn_cu")
        before = {name: getattr(workspace, name).data_ptr() for name in fields}
        context = SimpleNamespace(slot_mapping=torch.arange(6, dtype=torch.int32),
            block_tables=torch.tensor([[1, 2], [4, 5]], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 6, 13], dtype=torch.int32))
        workspace.copy_inputs(torch.arange(6), torch.arange(18).reshape(3, 6), context, [7, 3])
        context.block_tables = torch.tensor([[9], [6]], dtype=torch.int32)
        workspace.copy_inputs(torch.arange(6) + 10, torch.zeros(3, 6, dtype=torch.long), context, [3, 8])
        self.assertEqual(workspace.state_slots.tolist(), [3, 8])
        self.assertEqual(workspace.block_tables.tolist(), [[9, -1, -1, -1], [6, -1, -1, -1]])
        self.assertEqual(before, {name: getattr(workspace, name).data_ptr() for name in fields})
        bound = workspace.bind_context()
        self.assertTrue(bound.gdn_mtp_verification)
        self.assertEqual(bound.prefill_seqlens, (3, 3))
        self.assertIs(bound.block_tables, workspace.block_tables)
        reset_context()

    def test_duplicate_state_slots_are_not_padded_into_graph(self):
        workspace = MTPVerificationWorkspace.allocate(2, 2, 8, 4, 0, "cpu")
        context = SimpleNamespace(slot_mapping=torch.arange(4, dtype=torch.int32),
            block_tables=torch.zeros(2, 1, dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 3, 6], dtype=torch.int32))
        with self.assertRaisesRegex(ValueError, "exact"):
            workspace.copy_inputs(torch.arange(4), torch.arange(4), context, [1, 1])

    def test_ragged_verification_uses_eager(self):
        runner = SimpleNamespace(config=SimpleNamespace(enforce_eager=False,
            hybrid_cuda_graph_batch_sizes=(1, 2), mtp_gdn_rollback="replay_inputs"),
            hybrid_state_manager=SimpleNamespace(device=torch.device("cpu")),
            mtp_stats={"verification_graph_fallbacks": 0},
            _verify_mtp_batch=lambda seqs, drafts: ("eager", seqs, drafts))
        result = MTPVerificationGraphMixin.verify_mtp_with_graph(runner, [1, 2], [[3], []])
        self.assertEqual(result[0], "eager")
        self.assertEqual(runner.mtp_stats["verification_graph_fallbacks"], 1)

    def test_uniform_packed_backend_matches_independent_requests(self):
        torch.manual_seed(89)
        count, length, channels, heads, key_dim, value_dim = 3, 4, 5, 2, 3, 2
        initial_conv = torch.randn(count, channels, 4)
        initial_state = torch.randn(count, heads, key_dim, value_dim)
        raw = torch.randn(1, channels, count * length)
        weight = torch.randn(channels, 4)
        q = torch.randn(1, count * length, heads, key_dim)
        k = torch.randn_like(q)
        v = torch.randn(1, count * length, heads, value_dim)
        g = -torch.rand(1, count * length, heads)
        beta = torch.rand_like(g)
        namespace = {"torch": torch, "F": torch.nn.functional, "get_context": get_context,
                     "verification_conv": verification_conv, "verification_recurrent": verification_recurrent,
                     "torch_causal_conv1d_reference": base._conv_reference,
                     "torch_recurrent_gated_delta_rule": base._recurrent_reference}
        conv = base.method("nanovllm/layers/gated_delta_net.py", "Qwen3_5GatedDeltaNet",
                           "_run_causal_conv1d", namespace)
        recurrent = base.method("nanovllm/layers/gated_delta_net.py", "Qwen3_5GatedDeltaNet",
                                "_run_gated_delta_rule", namespace)
        layer = SimpleNamespace(conv1d=SimpleNamespace(weight=weight.unsqueeze(1), bias=None), activation="silu")
        set_context(True)
        get_context().gdn_mtp_verification = True
        try:
            output_conv, state_conv = conv(layer, raw, initial_conv, (length,) * count)
            output, state = recurrent(layer, q, k, v, g, beta, initial_state, (length,) * count)
            for row in range(count):
                part = slice(row * length, (row + 1) * length)
                expected_conv, expected_conv_state = base._conv_reference(raw[:, :, part], weight,
                    initial_state=initial_conv[row:row + 1])
                expected, expected_state = base._recurrent_reference(q[:, part], k[:, part], v[:, part],
                    g[:, part], beta[:, part], initial_state=initial_state[row:row + 1],
                    output_final_state=True, use_qk_l2norm_in_kernel=True)
                torch.testing.assert_close(output_conv[:, :, part], expected_conv)
                torch.testing.assert_close(state_conv[row:row + 1], expected_conv_state)
                torch.testing.assert_close(output[:, part], expected)
                torch.testing.assert_close(state[row:row + 1], expected_state)
            torch.testing.assert_close(initial_state, initial_state.clone(), rtol=0, atol=0)
        finally:
            reset_context()


if __name__ == "__main__":
    unittest.main()
