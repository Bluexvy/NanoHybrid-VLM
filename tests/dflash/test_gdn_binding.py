"""Exercise actual PyTorch C++ bindings on Linux CUDA, including Graph replay."""
import unittest
from pathlib import Path
import sys

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mtp"))
import cpu_runtime
from nanovllm.kernels.gdn_verification import verification_conv, verification_recurrent


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA PyTorch and matching compiler")
class BindingTests(unittest.TestCase):
    def test_verification_bindings_match_reference_and_graph(self):
        torch.manual_seed(313)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for tokens in (3, 16):
                batch, heads, key_dim, value_dim, channels = 2, 2, 17, 19, 12
                q = torch.randn(batch, tokens, heads, key_dim, device="cuda", dtype=dtype)
                k = torch.randn_like(q)
                v = torch.randn(batch, tokens, heads, value_dim, device="cuda", dtype=dtype)
                g = -torch.rand(batch, tokens, heads, device="cuda")
                beta = torch.rand(batch, tokens, heads, device="cuda", dtype=dtype)
                recurrent = torch.randn(batch, heads, key_dim, value_dim, device="cuda") * .1
                raw = torch.randn(batch, channels, tokens, device="cuda", dtype=dtype)
                conv = torch.randn(batch, channels, 4, device="cuda", dtype=dtype)
                weight = torch.randn(channels, 4, device="cuda", dtype=dtype)
                bias = torch.randn(channels, device="cuda", dtype=dtype)
                original = recurrent.clone(), conv.clone()
                def body(backend):
                    rec = verification_recurrent(q, k, v, g, beta, recurrent, backend=backend)
                    short = verification_conv(raw, weight, bias, conv, backend=backend, activation="silu")
                    return (*rec, *short)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.inference_mode(), torch.cuda.stream(stream):
                    for _ in range(3): body("cuda")
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.inference_mode(), torch.cuda.graph(graph, stream=stream):
                    captured = body("cuda")
                torch.cuda.current_stream().wait_stream(stream)
                for change in (False, True):
                    if change:
                        q.add_(.03)
                        raw.add_(.02)
                    with torch.inference_mode():
                        expected = body("reference")
                        graph.replay()
                    torch.cuda.synchronize()
                    for actual, wanted in zip(captured, expected):
                        tolerance = (3e-5, 3e-6) if dtype == torch.float32 else (.03, .02)
                        torch.testing.assert_close(actual, wanted, rtol=tolerance[0], atol=tolerance[1])
                    torch.testing.assert_close(recurrent, original[0], rtol=0, atol=0)
                    torch.testing.assert_close(conv, original[1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
