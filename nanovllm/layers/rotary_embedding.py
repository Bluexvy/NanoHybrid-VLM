from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        # self.head_size = head_size
        # assert rotary_dim == head_size
        self.head_size = head_size
        self.rotary_dim = rotary_dim

        if not 0 < rotary_dim <= head_size:
            raise ValueError(
                "rotary_dim must satisfy "
                "0 < rotary_dim <= head_size"
            )

        if rotary_dim % 2 != 0:
            raise ValueError(
                "rotary_dim must be even"
            )
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)

        # 只取出需要应用 RoPE 的前 rotary_dim 维。
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]

        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]

        query_rot = apply_rotary_emb(
            query_rot,
            cos,
            sin,
        )

        key_rot = apply_rotary_emb(
            key_rot,
            cos,
            sin,
        )

        # 把未参与旋转的部分原样拼回来。
        query = torch.cat(
            [query_rot, query_pass],
            dim=-1,
        )

        key = torch.cat(
            [key_rot, key_pass],
            dim=-1,
        )

        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
