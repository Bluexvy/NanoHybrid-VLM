import math
from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        if not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
