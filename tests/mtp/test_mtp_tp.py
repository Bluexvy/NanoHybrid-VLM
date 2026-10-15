"""Real multi-process Gloo checks of MTP TP=2/4 against the unsharded head."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("mtp_tp_head", ROOT / "nanovllm/models/qwen3_5_mtp.py")
head = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = head
spec.loader.exec_module(head)


def rotary(positions, q, k):
    return q, k


def worker(rank, size, init_method):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=size)
    try:
        torch.manual_seed(41)
        config = SimpleNamespace(hidden_size=16, intermediate_size=24,
            num_attention_heads=16, num_key_value_heads=4, head_dim=4,
            rms_norm_eps=1e-6, attention_bias=True)
        reference = head.Qwen3_5MTP(config, tp_rank=0, tp_size=1).eval()
        local = head.Qwen3_5MTP(config, tp_rank=rank, tp_size=size).eval()
        for name, parameter in local.named_parameters():
            value = reference.state_dict()[name]
            if "gate_up_proj" in name:
                gate, up = value.chunk(2, 0)
                parameter.weight_loader(parameter, gate, 0)
                parameter.weight_loader(parameter, up, 1)
            elif hasattr(parameter, "weight_loader"):
                parameter.weight_loader(parameter, value)
            else:
                parameter.data.copy_(value)
        embeddings, features = torch.randn(7, 16), torch.randn(7, 16)
        with torch.inference_mode():
            expected, full_cache = reference(torch.arange(1, 8), embeddings, features, None, rotary)
            actual, local_cache = local(torch.arange(1, 8), embeddings, features, None, rotary)
            torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
            for original, shard in zip(full_cache, local_cache):
                torch.testing.assert_close(shard, original.chunk(size, 0)[rank], rtol=3e-5, atol=3e-6)
            cache, pieces = None, []
            for start, end in ((0, 3), (3, 5), (5, 7)):
                result, cache = local(torch.arange(start + 1, end + 1),
                    embeddings[start:end], features[start:end], cache, rotary)
                pieces.append(result)
            torch.testing.assert_close(torch.cat(pieces), expected, rtol=3e-5, atol=3e-6)
            attention = local.layers[0].self_attn
            fixed = (torch.full((attention.kv_heads, 8, attention.dim), 999.),
                     torch.full((attention.kv_heads, 8, attention.dim), 999.))
            steps = []
            for index in range(7):
                steps.append(local.forward_step(torch.tensor([index + 1]), embeddings[index:index + 1],
                    features[index:index + 1], fixed, torch.tensor(index), rotary))
            torch.testing.assert_close(torch.cat(steps), expected, rtol=3e-5, atol=3e-6)
        # Coordinator decisions must be identical across workers.
        decision = torch.tensor([2, 4, 7] if rank == 0 else [0, 0, 0])
        dist.broadcast(decision, src=0)
        assert decision.tolist() == [2, 4, 7]
    finally:
        dist.destroy_process_group()


class TensorParallelTests(unittest.TestCase):
    def check_size(self, size):
        with tempfile.TemporaryDirectory() as directory:
            init_method = (Path(directory) / "rendezvous").as_uri()
            mp.spawn(worker, args=(size, init_method), nprocs=size, join=True)

    def test_tp2_matches_unsharded(self):
        self.check_size(2)

    def test_tp4_matches_unsharded(self):
        self.check_size(4)


if __name__ == "__main__":
    unittest.main()
