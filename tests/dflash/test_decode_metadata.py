"""Exercise actual runner metadata methods without loading CUDA/model weights."""
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mtp"))
import cpu_runtime
import test_mtp_cpu as base
from nanovllm.utils.context import get_context, reset_context, set_context


def request(length, pages, delta=None):
    seq = cpu_runtime.Sequence(list(range(length)), cpu_runtime.SamplingParams(max_tokens=8))
    seq.block_table = list(pages)
    seq.mrope_position_delta = delta
    return seq


def runner(hybrid=True, max_len=1024, eager=False):
    # Extract the production methods; only pinned allocation/CUDA transfer are
    # replaced. Keep actual page IDs, layout, positions, and context assembly.
    def tensor(data, **kwargs):
        kwargs.pop("pin_memory", None)
        return torch.tensor(data, **kwargs)
    scope = {"torch": SimpleNamespace(tensor=tensor, int64=torch.int64, int32=torch.int32),
             "set_context": set_context}
    value = SimpleNamespace(is_hybrid_model=hybrid, config=SimpleNamespace(max_model_len=max_len),
                            block_size=256, uses_multimodal_rope=hybrid, enforce_eager=eager)
    for name in ("prepare_block_tables", "prepare_decode"):
        function = base.method("nanovllm/engine/model_runner.py", "ModelRunner", name, scope)
        setattr(value, name, MethodType(function, value))
    return value


class DecodeMetadataTests(unittest.TestCase):
    def setUp(self):
        self.transfer = patch.object(torch.Tensor, "cuda", lambda value, **kwargs: value)
        self.transfer.start()

    def tearDown(self):
        self.transfer.stop()
        reset_context()

    def test_eager_and_graph_routes_use_same_capacity(self):
        for eager in (False, True):
            value = runner(eager=eager)
            ids, positions = value.prepare_decode([request(33, [7], delta=-2), request(257, [9, 12])])
            context = get_context()
            self.assertEqual(context.block_tables.tolist(), [[7, -1, -1, -1], [9, 12, -1, -1]])
            self.assertEqual(context.context_lens.tolist(), [33, 257])
            self.assertEqual(context.slot_mapping.tolist(), [7 * 256 + 32, 12 * 256])
            self.assertEqual(ids.tolist(), [32, 256])
            self.assertEqual(positions.tolist(), [[30, 256]] * 3)

    def test_request_reorder_and_batch_shrink_keep_live_mapping(self):
        value = runner()
        a, b = request(17, [23]), request(280, [4, 18])
        value.prepare_decode([a, b])
        value.prepare_decode([b])  # B=1 after removal/reorder; no padded request.
        context = get_context()
        self.assertEqual(context.block_tables.tolist(), [[4, 18, -1, -1]])
        self.assertEqual(context.context_lens.tolist(), [280])
        self.assertEqual(context.slot_mapping.tolist(), [18 * 256 + 23])
        self.assertEqual(b.block_table, [4, 18])

    def test_non_bucket_batch_still_uses_fixed_capacity(self):
        value = runner()
        seqs = [request(20 + i, [30 + i]) for i in range(3)]
        value.prepare_decode(seqs)
        context = get_context()
        self.assertEqual(tuple(context.block_tables.shape), (3, 4))
        self.assertEqual(context.context_lens.tolist(), [20, 21, 22])

    def test_nonhybrid_decode_keeps_compact_table(self):
        value = runner(hybrid=False)
        _, positions = value.prepare_decode([request(33, [7]), request(257, [9, 12])])
        self.assertEqual(get_context().block_tables.tolist(), [[7, -1], [9, 12]])
        self.assertEqual(positions.tolist(), [32, 256])

    def test_prefill_table_call_without_capacity_remains_compact(self):
        value = runner()
        table = value.prepare_block_tables([request(33, [7]), request(257, [9, 12])])
        self.assertEqual(table.tolist(), [[7, -1], [9, 12]])

    def test_capacity_rounds_up_like_graph_workspace(self):
        value = runner(max_len=1025)
        value.prepare_decode([request(33, [7])])
        self.assertEqual(tuple(get_context().block_tables.shape), (1, 5))

    def test_fixed_capacity_rejects_truncating_live_pages(self):
        value = runner()
        with self.assertRaisesRegex(ValueError, "truncate live KV pages"):
            value.prepare_block_tables([request(257, [9, 12])], num_columns=1)


if __name__ == "__main__":
    unittest.main()
