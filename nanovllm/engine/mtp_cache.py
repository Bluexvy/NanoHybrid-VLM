"""Read-only MTP prefix boundary, independent of the next request's suffix."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class MTPPrefixSnapshot:
    num_cached_tokens: int
    key: torch.Tensor
    value: torch.Tensor
    target_feature: torch.Tensor

    def __post_init__(self):
        if (self.num_cached_tokens <= 0 or self.key.shape != self.value.shape
                or self.key.ndim != 3 or self.key.shape[1] != self.num_cached_tokens - 1
                or self.target_feature.ndim != 2 or self.target_feature.shape[0] != 1):
            raise ValueError("Invalid MTP prefix snapshot boundary")

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.key, self.value, self.target_feature))

    def clone(self):
        return type(self)(self.num_cached_tokens, self.key.detach().clone(),
                          self.value.detach().clone(), self.target_feature.detach().clone())
