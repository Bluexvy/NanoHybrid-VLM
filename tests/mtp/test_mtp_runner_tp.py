"""Actual gather/broadcast/commit protocol with multi-process CPU Gloo."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import test_mtp_batch as batch
import test_mtp_cpu as base
from cpu_runtime import Sequence, SamplingParams


def worker(rank, size, init_method, stochastic):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=size)
    try:
        torch.manual_seed(101 + rank)
        runner, seqs = batch.Runner(), batch.requests()
        payload = [None]
        if rank == 0:
            real = []
            for old in seqs:
                seq = Sequence(old.token_ids, SamplingParams(temperature=0, max_tokens=5))
                seq.seq_id, seq.state_slot = old.seq_id, old.state_slot
                seq.num_cached_tokens, seq.num_scheduled_tokens, seq.is_prefill = 2, 1, False
                seq.mtp_num_speculative_tokens = 2
                real.append(seq)
            seqs = real
            payload[0] = seqs
        dist.broadcast_object_list(payload, src=0)
        if rank != 0:
            seqs = payload[0]
            assert all(seq.token_offset == 2 and len(seq.token_ids) == 1 for seq in seqs)
        runner.rank, runner.world_size = rank, size
        # Same vocab partitioning as ParallelLMHead; hidden states are replicated.
        runner.model.lm_head.weight = torch.eye(16).chunk(size, dim=0)[rank].contiguous()
        before = {seq.seq_id: runner.hybrid_state_manager.read_batched_states([seq.state_slot]) for seq in seqs}
        if stochastic:
            for seq in seqs:
                seq.temperature = .8
            def draft(seq, count):
                tokens = runner.proposals[seq.seq_id][:count]
                distributions = []
                for token in tokens:
                    prob = torch.full((16,), .3 / 15)
                    prob[token] = .7
                    distributions.append(prob)
                return tokens, distributions
            runner._draft_mtp = draft
        base.reset_capture()
        modules = {"nanovllm.engine.speculative": base.acceptance,
                   "nanovllm.engine.gdn_replay": base.replay,
                   "nanovllm.utils.context": SimpleNamespace(get_context=lambda: base._capture_context,
                                                              reset_context=base.reset_capture)}
        with patch.dict(sys.modules, modules):
            result = runner.run_mtp_batch(seqs, [2, 1, 0])
        received = [result]
        dist.broadcast_object_list(received, src=0)
        result = received[0]
        if not stochastic:
            assert result == [[4, 5], [8, 9], [5]]
        for row, seq in enumerate(seqs):
            committed = runner.advances[seq.seq_id][0]
            assert committed == result[row]
            count = len(committed)
            inputs = [seq.token_ids[-1]] + runner.proposals[seq.seq_id][:count - 1]
            base.reset_capture()
            base._capture_context.prefill_seqlens = (len(inputs),)
            _, wanted = runner.model(torch.tensor(inputs), None, before[seq.seq_id])
            actual = runner.hybrid_state_manager.read_states(seq.state_slot)
            for layer in (1, 4):
                torch.testing.assert_close(actual[layer].conv_state, wanted[layer].conv_state)
                torch.testing.assert_close(actual[layer].recurrent_state, wanted[layer].recurrent_state,
                                           atol=2e-6, rtol=2e-6)
        stats = torch.tensor([runner.mtp_stats["accepted_tokens"], runner.mtp_stats["proposed_tokens"]])
        all_stats = [torch.empty_like(stats) for _ in range(size)]
        dist.all_gather(all_stats, stats)
        assert all(torch.equal(item, stats) for item in all_stats)
    finally:
        dist.destroy_process_group()


class RunnerTensorParallelTests(unittest.TestCase):
    def run_protocol(self, stochastic):
        with tempfile.TemporaryDirectory() as directory:
            uri = (Path(directory) / "rendezvous").as_uri()
            mp.spawn(worker, args=(2, uri, stochastic), nprocs=2, join=True)

    def test_greedy_tp2_decision_and_state_commit(self):
        self.run_protocol(False)

    def test_sampling_tp2_decision_and_state_commit(self):
        self.run_protocol(True)


if __name__ == "__main__":
    unittest.main()
