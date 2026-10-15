import pickle
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
import test_mtp_cpu as base
from cpu_runtime import (Sequence, SamplingParams, MTPRunnerMixin, Qwen3_5MTP,
                         HybridCacheSpec, HybridStateManager, PrefixKey, PrefixStateCache, BlockManager)


class Runner(MTPRunnerMixin):
    IMAGE_TOKEN_TYPE = 1
    def __init__(self):
        self.initialize_mtp_runtime()
        self.rank, self.world_size = 0, 1
        self.uses_multimodal_rope = True
        config = SimpleNamespace(hidden_size=8, intermediate_size=12, num_attention_heads=2,
                                 num_key_value_heads=1, head_dim=4, rms_norm_eps=1e-6)
        self.model = SimpleNamespace(mtp=Qwen3_5MTP(config).eval(),
            lm_head=SimpleNamespace(weight=torch.randn(16, 8)),
            model=SimpleNamespace(language_model=SimpleNamespace(last_mtp_hidden_states=None)))
        self.embedding = torch.randn(16, 8)
        self.model.embed_input_ids = lambda ids: self.embedding[ids]
        self.visual = torch.randn(2, 8)
        self.vision_calls = 0

    def _mtp_rotary(self):
        return lambda positions, q, k: (q, k)

    def get_or_create_visual_embeddings(self, seq):
        self.vision_calls += 1
        return self.visual


def request(tokens):
    seq = Sequence(tokens, SamplingParams(temperature=0, max_tokens=8))
    seq.state_slot = 0
    seq.mtp_num_speculative_tokens = 3
    return seq


class FeatureTests(unittest.TestCase):
    def test_multidraft_does_not_mutate_committed_cache(self):
        torch.manual_seed(63)
        runner, seq = Runner(), request([1, 2, 3, 4])
        runner._advance_mtp(seq, [2, 3, 4], torch.randn(3, 8), 0)
        before = tuple(value.clone() for value in runner.mtp_caches[seq.seq_id])
        draft, probs = runner._draft_mtp(seq, 3)
        self.assertEqual(len(draft), 3)
        self.assertEqual(len(probs), 3)
        for actual, expected in zip(runner.mtp_caches[seq.seq_id], before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_prefix_restore_matches_full_prefill_with_different_suffix(self):
        torch.manual_seed(71)
        original, seq = Runner(), request([1, 2, 3, 4, 5])
        features = torch.randn(5, 8)
        original._advance_mtp(seq, [2, 3, 4], features[:3], 0)
        snapshot = original.mtp_prefix_snapshot(seq.seq_id, 3).clone()
        key = SimpleNamespace(num_cached_tokens=3)
        # Use a hashable key, just like PrefixKey in the real engine.
        key = ("prefix", 3)
        original.mtp_prefix_snapshots[key] = snapshot
        restored = request([1, 2, 3, 9, 8])
        restored.num_cached_tokens, restored.num_scheduled_tokens = 3, 2
        original.restore_mtp_prefix(key, restored.seq_id)
        original.model.model.language_model.last_mtp_hidden_states = features[3:]
        original.update_mtp_after_prefill([restored], [0])
        # Same weights/features, compute the new request without prefix caching.
        full = request([1, 2, 3, 9, 8])
        original._advance_mtp(full, [2, 3, 9, 8, 0], features, 0)
        for actual, expected in zip(original.mtp_caches[restored.seq_id], original.mtp_caches[full.seq_id]):
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        self.assertEqual(original.mtp_proposals[restored.seq_id], original.mtp_proposals[full.seq_id])
        torch.testing.assert_close(snapshot.key, original.mtp_prefix_snapshots[key].key, rtol=0, atol=0)

    def test_shifted_multimodal_embeddings_and_three_axis_positions(self):
        runner = Runner()
        seq = request([1, 2, 2, 3])
        seq.mm_token_type_ids = [0, 1, 1, 0]
        seq.mrope_position_ids = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 2], [0, 1, 0, 2]])
        seq.mrope_position_delta = -1
        values = runner._mtp_embeddings(seq, torch.tensor([2, 2, 3]), 1)
        torch.testing.assert_close(values[:2], runner.visual)
        torch.testing.assert_close(values[2], runner.embedding[3])
        positions = runner._mtp_positions(seq, 2, 3, "cpu")
        torch.testing.assert_close(positions[:, :2], seq.mrope_position_ids[:, 2:])
        torch.testing.assert_close(positions[:, 2], torch.tensor([3, 3, 3]))

    def test_worker_pickle_preserves_mtp_decode_and_multimodal_metadata(self):
        seq = request([1, 2, 3])
        seq.is_prefill = False
        seq.num_cached_tokens = 2
        seq.mm_token_type_ids = [0, 1, 0]
        seq.mrope_position_ids = torch.tensor([[0, 1, 2]] * 3)
        seq.mrope_position_delta = 0
        restored = pickle.loads(pickle.dumps(seq))
        self.assertFalse(restored.is_prefill)
        self.assertEqual(restored.token_ids, [3])
        self.assertEqual(restored.token_offset, 2)
        self.assertEqual(restored[2:3], [3])
        with self.assertRaises(IndexError):
            restored[0]
        self.assertEqual(restored.max_tokens, seq.max_tokens)
        self.assertEqual(restored.temperature, seq.temperature)
        self.assertIsNone(restored.mm_token_type_ids)
        view = MTPRunnerMixin._mtp_verify_view(restored, [4, 5])
        self.assertEqual(view.token_ids, [3, 4, 5])
        self.assertEqual(view[2:5], [3, 4, 5])
        self.assertEqual(view.num_tokens, 5)
        self.assertEqual(seq.mm_token_type_ids, [0, 1, 0])

    def test_prefix_lru_accounts_for_mtp_snapshot_and_releases_it(self):
        config = SimpleNamespace(num_key_value_heads=1, linear_num_key_heads=1,
            linear_num_value_heads=2, layer_types=("linear_attention", "full_attention"),
            num_hidden_layers=2, num_attention_heads=2, hidden_size=8, head_dim=4,
            linear_key_head_dim=3, linear_value_head_dim=2, linear_conv_kernel_dim=4,
            dtype=torch.float32)
        spec = HybridCacheSpec.from_text_config(config)
        states = HybridStateManager(spec, 1, "cpu")
        states.conv_state_pool.zero_()
        states.recurrent_state_pool.zero_()
        states.initialized_slots[0] = True
        blocks = BlockManager(4, 4, enable_kv_prefix_lookup=False, record_prefix_metadata=True)
        seq = request([1, 2, 3, 4, 5])
        seq.block_size = 4
        blocks.allocate(seq, 0)
        seq.num_scheduled_tokens = 4
        blocks.hash_blocks(seq)
        ids, block_hash = blocks.prefix_metadata_at_boundary(seq, 4)
        torch.manual_seed(77)
        runner = Runner()
        runner._advance_mtp(seq, [2, 3, 4, 5], torch.randn(4, 8), 0)
        snapshot = runner.mtp_prefix_snapshot(seq.seq_id, 4)
        cache = PrefixStateCache(blocks, states, "test", torch.float32, 32, 100000)
        key = PrefixKey("test", block_hash, 4)
        entry, created = cache.commit(key, ids, 0, mtp_snapshot=snapshot)
        self.assertTrue(created)
        self.assertEqual(cache.current_prefix_cache_capacity_bytes,
                         entry.gdn_snapshot_bytes + snapshot.nbytes + 32)
        self.assertNotEqual(entry.mtp_snapshot.key.data_ptr(), snapshot.key.data_ptr())
        cache.discard(key)
        self.assertEqual(cache.total_mtp_snapshot_bytes, 0)
        self.assertEqual(cache.current_prefix_cache_capacity_bytes, 0)
        self.assertEqual(blocks.blocks[ids[0]].cache_ref_count, 0)
        # A snapshot that fits without MTP must be rejected if MTP exceeds budget.
        cache.capacity_bytes = entry.gdn_snapshot_bytes + snapshot.nbytes + 31
        rejected, created = cache.commit(key, ids, 0, mtp_snapshot=snapshot)
        self.assertIsNone(rejected)
        self.assertFalse(created)
        self.assertEqual(cache.current_prefix_cache_capacity_bytes, 0)

    def test_compact_multimodal_worker_prefill_metadata_keeps_delta(self):
        seq = request([1, 2, 3])
        seq.is_prefill, seq.num_cached_tokens = False, 2
        seq.block_table = [0, 1]
        seq.mrope_position_delta = -1
        worker = pickle.loads(pickle.dumps(seq))
        view = MTPRunnerMixin._mtp_verify_view(worker, [4, 5])
        captured = {}
        def tensor(data, **kwargs):
            kwargs.pop("pin_memory", None)
            return torch.tensor(data, **kwargs)
        prepare = base.method("nanovllm/engine/model_runner.py", "ModelRunner", "prepare_prefill",
            {"torch": SimpleNamespace(tensor=tensor, int64=torch.int64, int32=torch.int32, long=torch.long),
             "set_context": lambda **kwargs: captured.update(kwargs)})
        runner = SimpleNamespace(uses_multimodal_rope=True, block_size=4,
            prepare_block_tables=lambda seqs: torch.tensor([seq.block_table for seq in seqs], dtype=torch.int32))
        with patch.object(torch.Tensor, "cuda", lambda value, **kwargs: value):
            ids, positions = prepare(runner, [view])
        self.assertEqual(ids.tolist(), [3, 4, 5])
        self.assertEqual(positions.tolist(), [[1, 2, 3]] * 3)
        self.assertEqual(captured["prefill_seqlens"], (3,))


if __name__ == "__main__":
    unittest.main()
