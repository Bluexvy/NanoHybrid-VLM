"""Small CPU checks: python tests/mtp/test_mtp_cpu.py (no GPU weights)."""
import ast
from collections import deque
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


acceptance = load("mtp_acceptance_cpu", "nanovllm/engine/speculative.py")
head = load("mtp_head_cpu", "nanovllm/models/qwen3_5_mtp.py")
runner = load("mtp_runner_cpu", "nanovllm/engine/mtp_runner.py")
replay = load("gdn_replay_cpu", "nanovllm/engine/gdn_replay.py")


def method(path, cls_name, name, namespace=None):
    # Package __init__ imports CUDA dependencies; extract the actual lightweight
    # method instead of mocking or reimplementing its commit/allocation logic.
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    code = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    scope = {} if namespace is None else dict(namespace)
    exec(compile(code, path, "exec"), scope)
    return scope[name]


class AcceptanceTests(unittest.TestCase):
    def check(self, draft, target, output, accepted, **kwargs):
        result = acceptance.verify_greedy(draft, target, eos_token_id=0,
                  remaining_tokens=kwargs.pop("remaining", 20), **kwargs)
        self.assertEqual(result.token_ids, tuple(output))
        self.assertEqual(result.accepted_draft_tokens, accepted)
        self.assertEqual(result.num_computed_tokens, len(output))

    def test_accept_and_bonus(self):
        self.check([4, 5], [4, 5, 6], [4, 5, 6], 2)

    def test_reject(self):
        self.check([8], [4, 9], [4], 0)
        self.check([4, 8], [4, 5, 9], [4, 5], 1)

    def test_eos_and_budget(self):
        self.check([0], [0, 5], [0], 1)
        self.check([0], [0, 5], [0, 5], 1, ignore_eos=True)
        self.check([4, 5], [4, 5, 6], [4], 1, remaining=1)

    def test_invalid_and_missing_weights(self):
        with self.assertRaises(ValueError):
            acceptance.verify_greedy([4], [4], remaining_tokens=5, eos_token_id=0)
        with self.assertRaises(ValueError):
            acceptance.verify_greedy([], [4], remaining_tokens=0, eos_token_id=0)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"mtp.fc.weight": "part.safetensors"}}))
            with self.assertRaisesRegex(ValueError, "missing trained MTP"):
                acceptance.validate_mtp_checkpoint(directory)


class HeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        config = SimpleNamespace(hidden_size=8, intermediate_size=12,
            num_attention_heads=2, num_key_value_heads=1, head_dim=4, rms_norm_eps=1e-6)
        self.model = head.Qwen3_5MTP(config).eval()
        self.embed, self.features = torch.randn(7, 8), torch.randn(7, 8)

    @staticmethod
    def rotary(positions, q, k):
        return q, k

    def test_full_vs_chunked_attention(self):
        with torch.no_grad():
            full, _ = self.model(torch.arange(1, 8), self.embed, self.features, None, self.rotary)
            cache, outputs = None, []
            for start, end in ((0, 3), (3, 5), (5, 7)):
                out, cache = self.model(torch.arange(start + 1, end + 1), self.embed[start:end],
                                        self.features[start:end], cache, self.rotary)
                outputs.append(out)
            torch.testing.assert_close(torch.cat(outputs), full, atol=2e-6, rtol=2e-6)
            self.assertEqual(cache[0].shape, (1, 7, 4))

    def test_future_does_not_affect_prefix(self):
        with torch.no_grad():
            first, _ = self.model(torch.arange(1, 8), self.embed, self.features, None, self.rotary)
            changed = self.embed.clone()
            changed[4:] += 100
            second, _ = self.model(torch.arange(1, 8), changed, self.features, None, self.rotary)
            torch.testing.assert_close(first[:4], second[:4])

    def test_gate_up_loading(self):
        param = self.model.layers[0].mlp.gate_up_proj.weight
        param.weight_loader(param, torch.ones(12, 8), 0)
        param.weight_loader(param, torch.full((12, 8), 2.0), 1)
        torch.testing.assert_close(param[:12], torch.ones(12, 8))
        torch.testing.assert_close(param[12:], torch.full((12, 8), 2.0))

    def test_fixed_cache_steps_match_full_attention_and_mask_future(self):
        with torch.inference_mode():
            full, expected_cache = self.model(torch.arange(1, 8), self.embed, self.features, None, self.rotary)
            cache = (torch.full((1, 8, 4), 999.), torch.full((1, 8, 4), 999.))
            outputs = []
            for index in range(7):
                output = self.model.forward_step(torch.tensor([index + 1]), self.embed[index:index + 1],
                    self.features[index:index + 1], cache, torch.tensor(index), self.rotary)
                outputs.append(output)
            torch.testing.assert_close(torch.cat(outputs), full, rtol=2e-5, atol=2e-6)
            for actual, expected in zip(cache, expected_cache):
                torch.testing.assert_close(actual[:, :7], expected, rtol=2e-5, atol=2e-6)


class FakeSequence:
    def __init__(self):
        self.seq_id, self.state_slot = 1, 3
        self.token_ids = [1, 2, 3]
        self.num_tokens, self.num_cached_tokens, self.num_scheduled_tokens = 3, 2, 1
        self.num_prompt_tokens, self.temperature, self.max_tokens = 2, 0, 5
        self.ignore_eos, self.is_multimodal = False, False
        self.block_table = [0]

    def __getitem__(self, item):
        return self.token_ids[item]

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    def append_token(self, token):
        self.token_ids.append(token)
        self.num_tokens += 1


class StatePool:
    def __init__(self):
        self.value = torch.tensor(10)

    def read_batched_states(self, slots):
        return [self.value.clone()]

    def write_batched_states(self, slots, states):
        self.value.copy_(states[0])


class FakeTarget:
    def __init__(self):
        self.lm_head = SimpleNamespace(weight=torch.eye(16))

    def __call__(self, input_ids, positions, gdn_states):
        gdn_states[0].add_(len(input_ids))
        return torch.nn.functional.one_hot(input_ids + 1, 16).float(), gdn_states


class FakeRunner(runner.MTPRunnerMixin):
    def __init__(self, proposal, eos=-1):
        self.initialize_mtp_runtime()
        self.world_size, self.config = 1, SimpleNamespace(eos=eos)
        self.mtp_proposals[1] = proposal
        self.hybrid_state_manager, self.model, self.calls = StatePool(), FakeTarget(), []

    def prepare_prefill(self, seqs):
        seq = seqs[0]
        ids = seq[seq.num_cached_tokens:seq.num_cached_tokens + seq.num_scheduled_tokens]
        self.calls.append(ids)
        return torch.tensor(ids), torch.arange(len(ids))

    def _advance_mtp(self, seq, next_token_ids, target_features, start):
        self.committed = (next_token_ids, target_features.shape[0], start)


class RunnerTests(unittest.TestCase):
    def exercise(self, proposal, expected, state, calls, eos=-1):
        instance, seq = FakeRunner(proposal, eos), FakeSequence()
        with patch.dict(sys.modules, {"nanovllm.engine.speculative": acceptance,
                "nanovllm.utils.context": SimpleNamespace(reset_context=lambda: None,
                    get_context=lambda: SimpleNamespace())}):
            result = instance.run_mtp_decode(seq)
        self.assertEqual(result, expected)
        self.assertEqual(instance.hybrid_state_manager.value.item(), state)
        self.assertEqual(instance.calls, calls)
        self.assertEqual(seq.token_ids, [1, 2, 3])
        self.assertEqual(instance.committed[1], len(expected))

    def test_accept(self):
        self.exercise(4, [4, 5], 12, [[3, 4]])

    def test_reject(self):
        self.exercise(8, [4], 11, [[3, 8], [3]])

    def test_accepted_eos(self):
        self.exercise(4, [4], 11, [[3, 4], [3]], eos=4)

    def test_no_capacity_falls_back_to_one_input(self):
        instance = FakeRunner(8)
        with patch.dict(sys.modules, {"nanovllm.engine.speculative": acceptance,
                "nanovllm.utils.context": SimpleNamespace(reset_context=lambda: None,
                    get_context=lambda: SimpleNamespace())}):
            result = instance.run_mtp_decode(FakeSequence(), allow_draft=False)
        self.assertEqual(result, [4])
        self.assertEqual(instance.calls, [[3]])
        self.assertEqual(instance.mtp_stats["proposed_tokens"], 0)
        self.assertEqual(instance.mtp_stats["replay_tokens"], 0)


class CommitTests(unittest.TestCase):
    def test_token_and_hash_boundary(self):
        commit = method("nanovllm/engine/scheduler.py", "Scheduler", "postprocess_speculative",
                        {"SequenceStatus": SimpleNamespace(FINISHED="finished")})
        seq, events = FakeSequence(), []
        blocks = SimpleNamespace(
            trim_verification_blocks=lambda request, count: events.append(("trim", count)),
            hash_blocks=lambda request: events.append(("hash", tuple(request.token_ids),
                request.num_cached_tokens, request.num_scheduled_tokens)))
        commit(SimpleNamespace(eos=-1, block_manager=blocks), seq, [4, 5])
        self.assertEqual(seq.num_cached_tokens, 4)
        self.assertEqual(events, [("trim", 4), ("hash", (1, 2, 3, 4, 5), 2, 2)])

    def test_block_boundary_reserve_release(self):
        reserve = method("nanovllm/engine/block_manager.py", "BlockManager", "reserve_verification_tokens")
        trim = method("nanovllm/engine/block_manager.py", "BlockManager", "trim_verification_blocks")
        free = deque([1])
        manager = SimpleNamespace(block_size=4, free_block_ids=free,
            blocks=[SimpleNamespace(ref_count=1, cache_ref_count=0) for _ in range(2)],
            _allocate_block=lambda: free.popleft(), _deallocate_block=lambda block: free.append(block))
        seq = FakeSequence()
        seq.num_cached_tokens = 3
        self.assertTrue(reserve(manager, seq, 2))
        self.assertEqual(seq.block_table, [0, 1])
        trim(manager, seq, 4)
        self.assertEqual(seq.block_table, [0])
        self.assertEqual(list(free), [1])


def reference_function(name, namespace):
    path = "nanovllm/layers/gated_delta_net.py"
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    scope = dict(namespace)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])),
                 path, "exec"), scope)
    return scope[name]


_l2 = reference_function("l2norm", {"torch": torch})
_conv_reference = reference_function("torch_causal_conv1d_reference",
                                    {"torch": torch, "F": torch.nn.functional})
_recurrent_reference = reference_function("torch_recurrent_gated_delta_rule",
                                         {"torch": torch, "l2norm": _l2})
_capture_context = SimpleNamespace(is_prefill=True, gdn_replay_records=None,
                                  gdn_replay_capture_tokens=0)


class ToyGDN(torch.nn.Module):
    """Actual model forward/reference recurrence with small CPU projections."""
    forward = method("nanovllm/layers/gated_delta_net.py", "Qwen3_5GatedDeltaNet", "forward",
        {"torch": torch, "F": torch.nn.functional, "get_context": lambda: _capture_context,
         "capture_packed_gdn_inputs": replay.capture_packed_gdn_inputs})

    def __init__(self, layer):
        super().__init__()
        self.layer_idx, self.hidden_size = layer, 8
        self.num_k_heads, self.num_v_heads = 1, 2
        self.head_k_dim, self.head_v_dim = 3, 2
        self.key_dim, self.value_dim = 3, 4
        self.in_proj_qkv = torch.nn.Linear(8, 10, bias=False)
        self.in_proj_z = torch.nn.Linear(8, 4, bias=False)
        self.in_proj_a = torch.nn.Linear(8, 2, bias=False)
        self.in_proj_b = torch.nn.Linear(8, 2, bias=False)
        self.out_proj = torch.nn.Linear(4, 8, bias=False)
        self.register_buffer("A_log", torch.tensor([0.2, -0.3]))
        self.register_buffer("dt_bias", torch.tensor([0.1, 0.2]))
        self.register_buffer("conv_weight", torch.randn(10, 4))
        self.norm = lambda output, z: output * torch.sigmoid(z)

    def _run_causal_conv1d(self, mixed_qkv, conv_state, prefill_seqlens):
        return _conv_reference(mixed_qkv, self.conv_weight, initial_state=conv_state)

    def _run_gated_delta_rule(self, query, key, value, g, beta, recurrent_state, **kwargs):
        return _recurrent_reference(query, key, value, g, beta,
            initial_state=recurrent_state, output_final_state=True, use_qk_l2norm_in_kernel=True)


class ReplayPool:
    def __init__(self):
        self.spec = SimpleNamespace(gdn_layer_indices=(1, 4))
        self.states = [None] * 5
        for layer in (1, 4):
            self.states[layer] = SimpleNamespace(conv_state=torch.randn(1, 10, 4),
                                                recurrent_state=torch.randn(1, 2, 3, 2))

    def read_states(self, slot):
        return self.states.copy()  # Tensor views, exactly like the real pool API.

    def read_batched_states(self, slots):
        return [None if s is None else SimpleNamespace(conv_state=s.conv_state.clone(),
                recurrent_state=s.recurrent_state.clone()) for s in self.states]

    def write_batched_states(self, slots, states):
        for layer in self.spec.gdn_layer_indices:
            self.states[layer].conv_state.copy_(states[layer].conv_state)
            self.states[layer].recurrent_state.copy_(states[layer].recurrent_state)


class ReplayTarget:
    def __init__(self):
        self.layers = [ToyGDN(1), ToyGDN(4)]
        self.lm_head = SimpleNamespace(weight=torch.eye(16))
        self.forward_count = 0

    def __call__(self, input_ids, positions, gdn_states):
        self.forward_count += 1
        x = torch.nn.functional.one_hot(input_ids % 8, 8).float().unsqueeze(0)
        result = [None] * 5
        for layer in self.layers:
            old = gdn_states[layer.layer_idx]
            x, conv, recurrent = layer(x, old.conv_state, old.recurrent_state,
                                      prefill_seqlens=(len(input_ids),))
            result[layer.layer_idx] = SimpleNamespace(conv_state=conv, recurrent_state=recurrent)
        hidden = torch.nn.functional.one_hot(input_ids + 1, 16).float()
        return hidden, result


def reset_capture():
    _capture_context.gdn_replay_records = None
    _capture_context.gdn_replay_capture_tokens = 0


class CompactReplayTests(unittest.TestCase):
    def exercise(self, draft, expected_tokens, eos=-1):
        torch.manual_seed(19)
        reset_capture()
        instance = FakeRunner(draft, eos)
        instance.config.mtp_gdn_rollback = "replay_inputs"
        instance.hybrid_state_manager, instance.model = ReplayPool(), ReplayTarget()
        pool = instance.hybrid_state_manager
        before = pool.read_batched_states([3])
        expected_inputs = [3] if len(expected_tokens) == 1 else [3, draft]
        with torch.inference_mode():
            _, expected = instance.model(torch.tensor(expected_inputs), None,
                                         pool.read_batched_states([3]))
        instance.model.forward_count = 0
        modules = {"nanovllm.engine.speculative": acceptance,
                   "nanovllm.engine.gdn_replay": replay,
                   "nanovllm.utils.context": SimpleNamespace(
                       get_context=lambda: _capture_context, reset_context=reset_capture)}
        with patch.dict(sys.modules, modules):
            result = instance.run_mtp_decode(FakeSequence())
        self.assertEqual(result, expected_tokens)
        self.assertEqual(instance.model.forward_count, 1)
        self.assertEqual(instance.calls, [[3, draft]])
        self.assertEqual(instance.mtp_stats["replay_tokens"], 0)
        self.assertIsNone(_capture_context.gdn_replay_records)
        for layer in (1, 4):
            torch.testing.assert_close(pool.states[layer].conv_state, expected[layer].conv_state)
            torch.testing.assert_close(pool.states[layer].recurrent_state,
                                       expected[layer].recurrent_state, atol=1e-6, rtol=1e-6)
        return instance, before

    def test_rejected_draft_replays_only_gdn(self):
        instance, _ = self.exercise(8, [4])
        self.assertEqual(instance.mtp_stats["gdn_replay_tokens"], 1)
        self.assertEqual(instance.mtp_stats["gdn_replay_layers"], 2)
        self.assertGreater(instance.mtp_stats["gdn_replay_buffer_bytes_peak"], 0)

    def test_accepted_draft_skips_replay(self):
        instance, _ = self.exercise(4, [4, 5])
        self.assertEqual(instance.mtp_stats["gdn_replay_tokens"], 0)

    def test_accepted_eos_uses_gdn_boundary_replay(self):
        self.exercise(4, [4], eos=4)

    def test_capture_compact_prefix_and_read_only_checkpoint(self):
        torch.manual_seed(23)
        pool, layer = ReplayPool(), ToyGDN(4)
        initial = pool.states[4]
        before_conv, before_recurrent = initial.conv_state.clone(), initial.recurrent_state.clone()
        _capture_context.gdn_replay_records = {}
        _capture_context.gdn_replay_capture_tokens = 1
        try:
            with torch.inference_mode():
                layer(torch.randn(1, 2, 8), initial.conv_state, initial.recurrent_state,
                      prefill_seqlens=(2,))
            record = _capture_context.gdn_replay_records[4]
            self.assertEqual(record.conv_input.shape, (1, 10, 1))
            self.assertEqual(record.key.shape, (1, 1, 2, 3))
            self.assertEqual(record.value.shape, (1, 1, 2, 2))
            self.assertFalse(hasattr(record, "query"))
            self.assertEqual(record.conv_input.untyped_storage().nbytes(),
                             record.conv_input.numel() * record.conv_input.element_size())
            replay.replay_gdn_state(record, initial.conv_state, initial.recurrent_state, 1)
            torch.testing.assert_close(initial.conv_state, before_conv)
            torch.testing.assert_close(initial.recurrent_state, before_recurrent)
            with self.assertRaises(ValueError):
                replay.replay_gdn_state(record, initial.conv_state, initial.recurrent_state, 2)
        finally:
            reset_capture()

    def test_missing_layer_record_is_rejected_before_commit(self):
        instance = FakeRunner(8)
        instance.hybrid_state_manager = ReplayPool()
        before = instance.hybrid_state_manager.read_batched_states([3])
        with patch.dict(sys.modules, {"nanovllm.engine.gdn_replay": replay}):
            with self.assertRaisesRegex(RuntimeError, "every GDN layer"):
                instance._replay_gdn_inputs(FakeSequence(), {}, 1)
        for layer in (1, 4):
            torch.testing.assert_close(instance.hybrid_state_manager.states[layer].recurrent_state,
                                       before[layer].recurrent_state)

    def test_replay_backend_selection_on_cpu(self):
        record = replay.GDNReplayInputs(
            torch.randn(1, 3, 1), torch.randn(1, 1, 2, 3),
            torch.randn(1, 1, 2, 2), -torch.ones(1, 1, 2), torch.ones(1, 1, 2) * .5)
        conv, state = torch.randn(1, 3, 4), torch.randn(1, 2, 3, 2)
        expected = replay.replay_gdn_state(record, conv, state, 1, backend="reference")
        actual = replay.replay_gdn_state(record, conv, state, 1, backend="auto")
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "CUDA checkpoints"):
            replay.replay_gdn_state(record, conv, state, 1, backend="cuda")
        with self.assertRaisesRegex(ValueError, "Replay backend"):
            replay.replay_gdn_state(record, conv, state, 1, backend="invalid")


if __name__ == "__main__":
    unittest.main()
