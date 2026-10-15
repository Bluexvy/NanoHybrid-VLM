"""CPU checks for the exact target distribution and speculative boundaries."""
import importlib.util
from pathlib import Path
import sys
import unittest
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("speculative_sampling", ROOT / "nanovllm/engine/speculative.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class SamplingTests(unittest.TestCase):
    def test_rejection_uses_residual(self):
        q = torch.tensor([.8, .2])
        p = torch.tensor([[.2, .8], [.5, .5]])
        result = module.verify_sampling([0], [q], p, remaining_tokens=8,
            eos_token_id=-1, uniforms=[.9])
        self.assertEqual(result.token_ids, (1,))
        self.assertEqual(result.accepted_draft_tokens, 0)

    def test_accepted_prefix_bonus_and_budget(self):
        q = torch.tensor([0., 1.])
        p = torch.tensor([[0., 1.], [0., 1.], [1., 0.]])
        result = module.verify_sampling([1, 1], [q, q], p, remaining_tokens=8,
            eos_token_id=-1, uniforms=[0., 0.])
        self.assertEqual(result.token_ids, (1, 1, 0))
        limited = module.verify_sampling([1, 1], [q, q], p, remaining_tokens=1,
            eos_token_id=-1, uniforms=[0., 0.])
        self.assertEqual(limited.token_ids, (1,))

    def test_eos_stops_at_accepted_boundary(self):
        p = torch.tensor([[0., 1.], [1., 0.]])
        result = module.verify_sampling([1], [p[0]], p, remaining_tokens=8,
            eos_token_id=1, uniforms=[0.])
        self.assertEqual(result.token_ids, (1,))
        self.assertEqual(result.num_computed_tokens, 1)

    def test_empirical_first_token_matches_target(self):
        q = torch.tensor([.8, .2])
        p = torch.tensor([[.2, .8], [.5, .5]])
        generator = torch.Generator().manual_seed(81)
        ones = 0
        for _ in range(4000):
            token = int(torch.multinomial(q, 1, generator=generator))
            result = module.verify_sampling([token], [q], p, remaining_tokens=1,
                eos_token_id=-1, generator=generator)
            ones += result.token_ids[0]
        self.assertLess(abs(ones / 4000 - .8), .035)

    def test_temperature_and_invalid_probability(self):
        torch.testing.assert_close(module.token_probabilities(torch.tensor([[1., 2.]]), 0),
                                   torch.tensor([[0., 1.]]))
        with self.assertRaises(ValueError):
            module.verify_sampling([0], [torch.tensor([0., 1.])],
                torch.tensor([[.5, .5], [.5, .5]]), remaining_tokens=2, eos_token_id=-1)


if __name__ == "__main__":
    unittest.main()
