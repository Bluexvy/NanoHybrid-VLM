"""Real TP=2/4 Gloo projection and context-cache agreement."""
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from helpers import DFlashDraftModel, tiny_config
from test_dflash_runtime import Runner, batch, modules
from unittest.mock import patch
import sys
from types import SimpleNamespace


def worker(rank, size, method, version=1):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=method, rank=rank, world_size=size)
    try:
        torch.manual_seed(173)
        config = tiny_config(version)
        config.num_attention_heads, config.num_key_value_heads = 8, 4
        full = DFlashDraftModel(config, tp_rank=0, tp_size=1).eval()
        local = DFlashDraftModel(config, tp_rank=rank, tp_size=size).eval()
        for name, parameter in local.named_parameters():
            value = full.state_dict()[name]
            if "gate_up_proj" in name:
                for index, shard in enumerate(value.chunk(2, 0)):
                    parameter.weight_loader(parameter, shard, index)
            elif hasattr(parameter, "weight_loader"):
                parameter.weight_loader(parameter, value)
            else:
                parameter.data.copy_(value)
        full_cache, local_cache, start = None, None, 0
        with torch.inference_mode():
            for added in (5, 2):
                feature = torch.randn(added, 32)
                positions = torch.arange(start, start + added)
                full_cache = full.append_context(feature, positions, full_cache)
                local_cache = local.append_context(feature, positions, local_cache)
                noise = torch.randn(4, 16)
                positions = torch.arange(start + added, start + added + 4)
                expected, actual = full(noise, positions, full_cache), local(noise, positions, local_cache)
                torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
                for original, shard in zip(full_cache, local_cache):
                    for x, y in zip(original, shard):
                        torch.testing.assert_close(y, x.chunk(size, 0)[rank], atol=3e-6, rtol=3e-5)
                start += added
            # Exercise actual coordinator-only proposals, probability handling,
            # packed verification, decisions, and accepted cache commits.
            for temperature in (0, .8):
                torch.manual_seed(181)
                runner, seqs = Runner(), batch.requests()
                runner.rank, runner.world_size = rank, size
                runner.dflash = local
                runner.model.lm_head.weight = torch.eye(16).chunk(size, 0)[rank].contiguous()
                for seq in seqs:
                    seq.temperature = temperature
                    runner._advance_mtp(seq, [0, 0], torch.randn(2, 32), 0)
                with patch.dict(sys.modules, modules()):
                    result = runner.run_mtp_batch(seqs, [2, 1, 0])
                boundaries = torch.tensor([runner.dflash_caches[seq.seq_id][0][0].shape[1] for seq in seqs])
                gathered = [torch.empty_like(boundaries) for _ in range(size)]
                dist.all_gather(gathered, boundaries)
                for other in gathered:
                    torch.testing.assert_close(other, boundaries, atol=0, rtol=0)
                if rank == 0:
                    assert boundaries.tolist() == [2 + len(tokens) for tokens in result]
                else:
                    assert result is None
        # Verify the actual all-rank memory-report protocol with CPU Gloo.
        # GPU allocation values are supplied by the CUDA-only get_memory_stats.
        reset_calls = []
        memory = batch.base.method("nanovllm/engine/model_runner.py", "ModelRunner", "get_distributed_memory_stats",
            {"torch": SimpleNamespace(cuda=SimpleNamespace(reset_peak_memory_stats=reset_calls.append)), "dist": dist})
        holder = SimpleNamespace(rank=rank, world_size=size, get_memory_stats=lambda: {"bytes": 100 + rank})
        values = memory(holder, True)
        assert reset_calls == [rank]
        if rank == 0:
            assert values == [{"bytes": 100 + i, "rank": i} for i in range(size)]
        else:
            assert values is None
    finally:
        dist.destroy_process_group()


class TensorParallelTests(unittest.TestCase):
    def check(self, size, version=1):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(worker, args=(size, (Path(directory) / "rendezvous").as_uri(), version), nprocs=size, join=True)

    def test_tp2(self):
        self.check(2)

    def test_tp4(self):
        self.check(4)

    def test_v2_tp2(self):
        self.check(2, 2)

    def test_v2_tp4(self):
        self.check(4, 2)


if __name__ == "__main__":
    unittest.main()
