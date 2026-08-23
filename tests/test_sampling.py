import unittest

import torch

from nanovllm.layers.sampler import Sampler
from nanovllm.sampling_params import SamplingParams

class TestSamplingParams(unittest.TestCase):

    def test_zero_temperature_is_valid(self):
        params = SamplingParams(
            temperature=0.0,
        )

        self.assertEqual(
            params.temperature,
            0.0,
        )

    def test_positive_temperature_is_valid(self):
        params = SamplingParams(
            temperature=0.6,
        )

        self.assertEqual(
            params.temperature,
            0.6,
        )

    def test_invalid_temperatures_are_rejected(self):
        invalid_temperatures = [
            -0.1,
            float("nan"),
            float("inf"),
            -float("inf"),
        ]

        for temperature in invalid_temperatures:
            with self.subTest(
                temperature=temperature,
            ):
                with self.assertRaises(ValueError):
                    SamplingParams(
                        temperature=temperature,
                    )

class TestSampler(unittest.TestCase):

    def test_zero_temperature_uses_argmax(self):
        
        sampler = Sampler()

        logits = torch.tensor([
            [0.1, 3.0, -1.0],
            [4.0, 0.2, 5.0],
        ])

        temperatures = torch.tensor([
            0.0,
            0.0,
        ])

        expected = torch.tensor([
            1,
            2,
        ])

        for seed in [0, 1, 1234]:
            torch.manual_seed(seed)

            actual = sampler(
                logits,
                temperatures,
            )

            torch.testing.assert_close(
                actual,
                expected,
            )
            
    def test_mixed_greedy_and_sampling_batch(self):
        sampler = Sampler()

        logits = torch.tensor([
            [0.1, 3.0, -1.0],
            [4.0, 0.2, 5.0],
        ])

        temperatures = torch.tensor([
            0.0,
            0.6,
        ])

        token_ids = sampler(
            logits,
            temperatures,
        )

        self.assertEqual(
            token_ids[0].item(),
            1,
        )
    
    def test_all_greedy_does_not_advance_rng_state(self):
        sampler = Sampler()

        logits = torch.tensor([
            [0.1, 3.0, -1.0],
            [4.0, 0.2, 5.0],
        ])

        temperatures = torch.tensor([
            0.0,
            0.0,
        ])

        # 先调用一次，让 torch.compile 完成首次编译。
        sampler(logits, temperatures)

        torch.manual_seed(2026)
        state_before = torch.random.get_rng_state().clone()

        sampler(logits, temperatures)

        state_after = torch.random.get_rng_state()

        self.assertTrue(
            torch.equal(
                state_before,
                state_after,
            )
        )