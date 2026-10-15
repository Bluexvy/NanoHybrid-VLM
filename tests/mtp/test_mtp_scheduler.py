"""Actual scheduler/block manager checks of budgets and committed KV hashes."""
from types import SimpleNamespace
import unittest
import cpu_runtime
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def scheduler():
    config = SimpleNamespace(scheduler_policy="decode_first", max_num_seqs=3,
        max_num_batched_tokens=8, mtp_num_speculative_tokens=2, max_prefill_wait_ms=0,
        eos=-1, kvcache_block_size=4, text_config=SimpleNamespace(layer_types=("linear_attention",)),
        enable_prefix_cache=False, hybrid_prefix_cache_mode="disabled",
        num_kvcache_blocks=32, num_state_slots=3)
    return Scheduler(config)


def request(tokens):
    seq = Sequence(tokens, SamplingParams(temperature=0, max_tokens=6, ignore_eos=True))
    seq.block_size = 4
    seq.mtp_num_speculative_tokens = 2
    return seq


class SchedulerTests(unittest.TestCase):
    def test_multirequest_token_budget_and_completion_release(self):
        runtime = scheduler()
        seqs = [request([1, 2, 3]), request([4, 5, 6, 7, 8]), request([9, 10, 11, 12])]
        for seq in seqs:
            runtime.add(seq)
        rounds = 0
        while not runtime.is_finished():
            plan = runtime.schedule()
            self.assertLessEqual(plan.total_num_tokens, 8)
            self.assertLessEqual(len(plan.decode_seqs) + len(plan.prefill_seqs), 3)
            self.assertFalse(plan.is_empty)
            for seq in plan.decode_seqs:
                remaining = seq.max_tokens - seq.num_completion_tokens
                width = min(3, remaining)
                self.assertTrue(runtime.block_manager.reserve_verification_tokens(seq, width))
                runtime.postprocess_speculative(seq, [13] * width)
            if plan.prefill_seqs:
                runtime.record_computed_block_metadata(plan.prefill_seqs)
                runtime.postprocess(plan.prefill_seqs, [13] * len(plan.prefill_seqs), True)
            rounds += 1
            self.assertLess(rounds, 20)
        self.assertTrue(all(seq.is_finished and seq.state_slot is None for seq in seqs))
        self.assertEqual(runtime.state_slot_allocator.num_used_slots, 0)
        self.assertEqual(len(runtime.block_manager.used_block_ids), 0)

    def test_rejected_drafts_never_enter_block_hash(self):
        runtime = scheduler()
        seq = request([1, 2, 3, 4])
        runtime.add(seq)
        plan = runtime.schedule()
        runtime.record_computed_block_metadata(plan.prefill_seqs)
        runtime.postprocess(plan.prefill_seqs, [5], True)
        plan = runtime.schedule()
        self.assertEqual(plan.decode_seqs, [seq])
        self.assertTrue(runtime.block_manager.reserve_verification_tokens(seq, 3))
        # Draft [6,99] was partially rejected; validated outputs are [6,7].
        runtime.postprocess_speculative(seq, [6, 7])
        self.assertEqual(seq.num_cached_tokens, 6)
        self.assertEqual(seq.token_ids, [1, 2, 3, 4, 5, 6, 7])
        self.assertTrue(all(99 not in block.token_ids for block in runtime.block_manager.blocks))
        self.assertEqual(runtime.block_manager.blocks[seq.block_table[0]].token_ids, [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
