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
        mrope_section: (
            tuple[int, int, int] | None
        ) = None,
    ) -> None:
        super().__init__()
        # self.head_size = head_size
        # assert rotary_dim == head_size
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.mrope_section = (
            mrope_section
        )

        if not 0 < rotary_dim <= head_size:
            raise ValueError(
                "rotary_dim must satisfy "
                "0 < rotary_dim <= head_size"
            )

        if rotary_dim % 2 != 0:
            raise ValueError(
                "rotary_dim must be even"
            )
        if self.mrope_section is not None:
            if len(self.mrope_section) != 3:
                raise ValueError(
                    "mrope_section must contain "
                    "three values"
                )

            if any(
                (
                    not isinstance(section, int)
                    or section <= 0
                )
                for section
                in self.mrope_section
            ):
                raise ValueError(
                    "Every mRoPE section must be "
                    "a positive integer"
                )

            num_rotary_frequencies = (
                rotary_dim // 2
            )

            if (
                sum(self.mrope_section)
                != num_rotary_frequencies
            ):
                raise ValueError(
                    "The sum of mrope_section must "
                    "equal rotary_dim // 2: "
                    f"{sum(self.mrope_section)} != "
                    f"{num_rotary_frequencies}"
                )
                
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def select_interleaved_mrope(
        self,
        values_by_axis: torch.Tensor,
    ) -> torch.Tensor:
        """
        将三个轴的频率交错组合。

        输入：
            values_by_axis
            [3, num_tokens, 1, num_frequencies]

        输出：
            [num_tokens, 1, num_frequencies]
        """

        if self.mrope_section is None:
            raise RuntimeError(
                "mrope_section is required for "
                "three-axis positions"
            )

        if (
            values_by_axis.ndim != 4
            or values_by_axis.shape[0] != 3
        ):
            raise ValueError(
                "mRoPE axis values must have shape "
                "[3, num_tokens, 1, "
                "num_frequencies]"
            )

        # 先默认全部使用 temporal 轴。
        selected = (
            values_by_axis[0].clone()
        )

        # 然后把 H/W 对应的交错频率替换进去。
        #
        # height:
        #   1, 4, 7, 10, ...
        #
        # width:
        #   2, 5, 8, 11, ...
        for axis, offset in (
            (1, 1),
            (2, 2),
        ):
            section_length = (
                self.mrope_section[axis]
            )

            frequency_indices = slice(
                offset,
                section_length * 3,
                3,
            )

            selected[
                ...,
                frequency_indices,
            ] = values_by_axis[
                axis,
                ...,
                frequency_indices,
            ]

        return selected

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        if query.ndim != 3:
            raise ValueError(
                "query must have shape "
                "[num_tokens, num_heads, head_dim]"
            )

        if key.ndim != 3:
            raise ValueError(
                "key must have shape "
                "[num_tokens, num_kv_heads, head_dim]"
            )

        num_tokens = query.shape[0]

        if key.shape[0] != num_tokens:
            raise ValueError(
                "query and key must contain "
                "the same number of tokens"
            )

        # =====================================
        # 普通一维文本位置
        # =====================================

        if positions.ndim == 1:
            if positions.shape != (
                num_tokens,
            ):
                raise ValueError(
                    "1D positions must have shape "
                    "[num_tokens]"
                )

            cos_sin = (
                self.cos_sin_cache[
                    positions
                ]
            )

            cos, sin = cos_sin.chunk(
                2,
                dim=-1,
            )

        # =====================================
        # 图文三轴位置
        # =====================================

        elif positions.ndim == 2:
            if positions.shape != (
                3,
                num_tokens,
            ):
                raise ValueError(
                    "3D mRoPE positions must have "
                    "shape [3, num_tokens]"
                )

            if self.mrope_section is None:
                raise ValueError(
                    "Three-axis positions require "
                    "mrope_section"
                )

            # [3,T] 查表后得到：
            #
            # [3,T,1,rotary_dim]
            cos_sin_by_axis = (
                self.cos_sin_cache[
                    positions
                ]
            )

            (
                cos_by_axis,
                sin_by_axis,
            ) = cos_sin_by_axis.chunk(
                2,
                dim=-1,
            )

            # 分别从 T/H/W 三个轴中选择
            # interleaved frequency。
            cos = (
                self.select_interleaved_mrope(
                    cos_by_axis
                )
            )

            sin = (
                self.select_interleaved_mrope(
                    sin_by_axis
                )
            )

        else:
            raise ValueError(
                "positions must have shape "
                "[num_tokens] or [3, num_tokens]"
            )

        # =====================================
        # Partial RoPE
        # =====================================

        query_rot = query[
            ...,
            :self.rotary_dim,
        ]

        query_pass = query[
            ...,
            self.rotary_dim:,
        ]

        key_rot = key[
            ...,
            :self.rotary_dim,
        ]

        key_pass = key[
            ...,
            self.rotary_dim:,
        ]

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

        # 前 rotary_dim 维已经旋转，
        # 后面的维度保持原样。
        query = torch.cat(
            [
                query_rot,
                query_pass,
            ],
            dim=-1,
        )

        key = torch.cat(
            [
                key_rot,
                key_pass,
            ],
            dim=-1,
        )

        return query, key


@lru_cache(2)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    mrope_section: (
        tuple[int, int, int] | None
    ) = None,
):
    rotary_emb = RotaryEmbedding(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=(
            max_position
        ),
        base=base,
        mrope_section=mrope_section,
    )

    return rotary_emb