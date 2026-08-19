import unittest

from nanovllm.sampling_params import SamplingParams


class TestSamplingParams(unittest.TestCase):

    def test_zero_temperature_enables_greedy(self):
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


if __name__ == "__main__":
    unittest.main()