"""Original Decode CUDA bindings at Qwen3.8-27B TP=2 local dimensions."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mtp"))
import cpu_runtime
from nanovllm.kernels.gdn_verification import verification_conv, verification_recurrent
from nanovllm.kernels.state_aware_gdn_cuda import (
    state_aware_causal_conv1d_cuda, state_aware_gdn_decode_cuda,
)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA PyTorch and matching C++ compiler")
class StateAwareBindingTests(unittest.TestCase):
    @torch.inference_mode()
    def test_local_shapes_slot_routing_and_graph(self):
        torch.manual_seed(714)
        heads, dim, channels, layers, layer = 24, 128, 5120, 3, 1
        for batch in (1, 4, 16):
            for graph_mode in (False, True):
                with self.subTest(batch=batch, graph=graph_mode):
                    slots = torch.arange(batch, device="cuda", dtype=torch.long).flip(0) + 1
                    conv = torch.randn(batch + 2, layers, channels, 4, device="cuda", dtype=torch.bfloat16) * .1
                    state = torch.randn(batch + 2, layers, heads, dim, dim, device="cuda") * .1
                    initial_conv, initial_state = conv.clone(), state.clone()
                    q = torch.randn(batch, 1, heads, dim, device="cuda", dtype=torch.bfloat16) * .1
                    k = torch.randn_like(q)
                    v = torch.randn_like(q) * .1
                    beta = torch.rand(batch, 1, heads, device="cuda", dtype=torch.bfloat16)
                    g = -torch.rand(batch, 1, heads, device="cuda") * .1
                    raw = torch.randn(batch, channels, 1, device="cuda", dtype=torch.bfloat16) * .1
                    weight = torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16) * .1
                    rec_out, conv_out = torch.empty_like(v), torch.empty_like(raw)

                    def body():
                        state_aware_causal_conv1d_cuda(raw, weight, conv, slots, layer, output=conv_out)
                        state_aware_gdn_decode_cuda(q, k, v, g, beta, state, slots, layer, output=rec_out)

                    # Compile before capture. Capture may also update state; reset it afterwards.
                    for _ in range(3):
                        body()
                    graph = None
                    if graph_mode:
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            body()
                    conv.copy_(initial_conv)
                    state.copy_(initial_state)
                    reference_conv, reference_state = initial_conv.clone(), initial_state.clone()
                    for step in range(3):
                        if step:
                            slots.copy_(slots.roll(1))
                            q.add_(.01)
                            raw.add_(.01)
                        expected_conv, next_conv = verification_conv(raw, weight, None,
                            reference_conv.index_select(0, slots)[:, layer], backend="reference")
                        expected_rec, next_state = verification_recurrent(q, k, v, g, beta,
                            reference_state.index_select(0, slots)[:, layer], backend="reference")
                        reference_conv[:, layer].index_copy_(0, slots, next_conv)
                        reference_state[:, layer].index_copy_(0, slots, next_state)
                        graph.replay() if graph is not None else body()
                        torch.cuda.synchronize()
                        torch.testing.assert_close(conv, reference_conv, rtol=0, atol=0)
                        torch.testing.assert_close(state, reference_state, rtol=5e-5, atol=5e-5)
                        torch.testing.assert_close(conv_out, expected_conv, rtol=.02, atol=.02)
                        torch.testing.assert_close(rec_out, expected_rec, rtol=.02, atol=.02)
                        # Inactive layers and never-addressed slots must remain bit-identical.
                        for untouched in (0, 2):
                            torch.testing.assert_close(state[:, untouched], initial_state[:, untouched], rtol=0, atol=0)
                            torch.testing.assert_close(conv[:, untouched], initial_conv[:, untouched], rtol=0, atol=0)
                        for unused in (0, batch + 1):
                            torch.testing.assert_close(state[unused], initial_state[unused], rtol=0, atol=0)
                            torch.testing.assert_close(conv[unused], initial_conv[unused], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
