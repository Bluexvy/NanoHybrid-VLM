import torch
from torch import nn
from math import isqrt

from flash_attn import flash_attn_varlen_func
from transformers.vision_utils import (
    get_vision_bilinear_indices_and_weights,
    get_vision_cu_seqlens,
    get_vision_position_ids,
)



class Qwen3_5VisionPatchEmbed(nn.Module):
    """
    把 Processor 产生的原始图像 patch，
    投影成 Vision Transformer hidden states。

    输入：
        pixel_values
        [num_patches, patch_dim]

    输出：
        hidden_states
        [num_patches, hidden_size]
    """

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()

        self.in_channels = config.in_channels

        self.temporal_patch_size = (
            config.temporal_patch_size
        )

        self.patch_size = config.patch_size

        self.hidden_size = config.hidden_size

        # 一个展平 patch 中包含的元素数量：
        #
        # RGB × temporal × height × width
        self.patch_dim = (
            self.in_channels
            * self.temporal_patch_size
            * self.patch_size
            * self.patch_size
        )

        kernel_size = (
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )

        # kernel_size 和 stride 相同。
        #
        # 每个输入 patch 被独立投影，
        # 不让两个 patch 在这里发生重叠。
        self.proj = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:

        if pixel_values.ndim != 2:
            raise ValueError(
                "pixel_values must have shape "
                "[num_patches, patch_dim]"
            )

        if (
            pixel_values.shape[-1]
            != self.patch_dim
        ):
            raise ValueError(
                "Invalid flattened patch dimension: "
                f"expected {self.patch_dim}, "
                f"got {pixel_values.shape[-1]}"
            )

        num_patches = pixel_values.shape[0]

        # Processor 当前给出 float32，
        # 但模型权重可能是 BF16。
        #
        # 输入必须转换成卷积权重的 dtype。
        target_dtype = self.proj.weight.dtype

        pixel_values = pixel_values.to(
            dtype=target_dtype
        )

        # 原来：
        #
        # [num_patches, 1536]
        #
        # 恢复成 Conv3d 需要的五维格式：
        #
        # [num_patches, 3, 2, 16, 16]
        pixel_values = pixel_values.reshape(
            num_patches,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )

        # 输出：
        #
        # [num_patches, 1152, 1, 1, 1]
        hidden_states = self.proj(
            pixel_values
        )

        # 去掉三个大小为 1 的空间/时间维度：
        #
        # [num_patches, 1152, 1, 1, 1]
        #                 ↓
        # [num_patches, 1152]
        hidden_states = hidden_states.reshape(
            num_patches,
            self.hidden_size,
        )

        return hidden_states
    
    
class Qwen3_5VisionRotaryEmbedding(nn.Module):
    """
    根据二维 position IDs 生成 Vision RoPE 角度。

    输入：
        position_ids
        [num_patches, 2]

        最后一维的两个值分别是：
            height position
            width position

    输出：
        rotary_frequencies
        [num_patches, head_dim // 2]
    """

    def __init__(
        self,
        dim: int,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()

        if dim <= 0:
            raise ValueError(
                "Vision rotary dim must be positive"
            )

        if dim % 2 != 0:
            raise ValueError(
                "Vision rotary dim must be even"
            )

        self.dim = dim
        self.theta = theta

        # 生成 RoPE 的逆频率：
        #
        # inv_freq[i]
        # =
        # 1 / theta^(2i / dim)
        #
        # arange(0, dim, 2) 表示每两个维度
        # 使用同一组旋转频率。
        inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(
                    0,
                    dim,
                    2,
                    dtype=torch.float32,
                )
                / dim
            )
        )

        # persistent=False：
        #
        # inv_freq 是由配置计算出来的常量，
        # 不需要出现在 checkpoint state_dict 中。
        self.register_buffer(
            "inv_freq",
            inv_freq,
            persistent=False,
        )

    def forward(
        self,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:

        if (
            position_ids.ndim != 2
            or position_ids.shape[-1] != 2
        ):
            raise ValueError(
                "Vision position_ids must have "
                "shape [num_patches, 2]"
            )

        # position_ids:
        # [num_patches, 2]
        #
        # unsqueeze 后：
        # [num_patches, 2, 1]
        #
        # inv_freq：
        # [dim / 2]
        #
        # 广播相乘后：
        # [num_patches, 2, dim / 2]
        frequencies = (
            position_ids
            .to(dtype=self.inv_freq.dtype)
            .unsqueeze(dim=-1)
            * self.inv_freq
        )

        # 把 height 和 width 两组频率拼到一起：
        #
        # [num_patches, 2, dim / 2]
        #                 ↓
        # [num_patches, dim]
        frequencies = frequencies.flatten(
            start_dim=1
        )

        return frequencies
    
    
def prepare_vision_position_embeddings(
    grid_thw: torch.Tensor,
    *,
    spatial_merge_size: int,
    num_grid_per_side: int,
    pos_embed: nn.Embedding,
    rotary_pos_emb: (
        Qwen3_5VisionRotaryEmbedding
    ),
) -> tuple[
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
]:
    """
    根据图片 patch 网格构造三类数据：

    1. absolute_pos_embeds
       加到 patch hidden states 上。

    2. position_embeddings = (cos, sin)
       交给每个 Vision Attention 层。

    3. cu_seqlens
       描述每张图片/每帧的 patch 边界。
    """

    if (
        grid_thw.ndim != 2
        or grid_thw.shape[-1] != 3
    ):
        raise ValueError(
            "grid_thw must have shape "
            "[num_images_or_videos, 3]"
        )

    if grid_thw.numel() == 0:
        raise ValueError(
            "grid_thw must not be empty"
        )

    if torch.any(grid_thw <= 0):
        raise ValueError(
            "Every T/H/W value must be positive"
        )

    if spatial_merge_size <= 0:
        raise ValueError(
            "spatial_merge_size must be positive"
        )

    # Patch Merger 要把 merge_size × merge_size
    # 的 patch 分为一组，因此 H/W 必须整除。
    if torch.any(
        grid_thw[:, 1:]
        % spatial_merge_size
        != 0
    ):
        raise ValueError(
            "Vision grid height and width must be "
            "divisible by spatial_merge_size"
        )

    # =====================================
    # 第一部分：可学习绝对位置 embedding
    # =====================================

    (
        bilinear_indices,
        bilinear_weights,
    ) = (
        get_vision_bilinear_indices_and_weights(
            grid_thw=grid_thw,
            num_grid_per_side=(
                num_grid_per_side
            ),
            spatial_merge_size=(
                spatial_merge_size
            ),
        )
    )

    # bilinear_indices：
    # [4, total_patches]
    #
    # 每个 patch 对应位置表中的四个角。
    #
    # lookup 后：
    # [4, total_patches, hidden_size]
    corner_embeddings = pos_embed(
        bilinear_indices
    )

    # bilinear_weights：
    # [4, total_patches]
    #
    # 增加 hidden_size 广播维：
    # [4, total_patches, 1]
    weighted_corner_embeddings = (
        corner_embeddings
        * bilinear_weights.unsqueeze(dim=-1)
    )

    # 将四个角加权求和：
    #
    # [4, total_patches, hidden_size]
    #                   ↓ sum(dim=0)
    # [total_patches, hidden_size]
    absolute_pos_embeds = (
        weighted_corner_embeddings.sum(dim=0)
    )

    # =====================================
    # 第二部分：二维 Vision RoPE
    # =====================================

    position_ids = get_vision_position_ids(
        grid_thw=grid_thw,
        spatial_merge_size=(
            spatial_merge_size
        ),
    )

    # position_ids：
    # [total_patches, 2]
    #
    # 两列分别是：
    # [height_position, width_position]
    rotary_frequencies = rotary_pos_emb(
        position_ids
    )

    # 当前：
    # [total_patches, head_dim // 2]
    #
    # 复制一次：
    # [total_patches, head_dim]
    rotary_frequencies = torch.cat(
        (
            rotary_frequencies,
            rotary_frequencies,
        ),
        dim=-1,
    )

    position_embeddings = (
        rotary_frequencies.cos(),
        rotary_frequencies.sin(),
    )

    # =====================================
    # 第三部分：Vision 序列边界
    # =====================================

    cu_seqlens = get_vision_cu_seqlens(
        grid_thw
    )

    return (
        absolute_pos_embeds,
        position_embeddings,
        cu_seqlens,
    )
    
def rotate_half(
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """
    将向量的前后两半执行二维旋转所需的重排。

    输入最后一维：
        [x1, x2]

    输出：
        [-x2, x1]
    """

    if hidden_states.shape[-1] % 2 != 0:
        raise ValueError(
            "RoPE head dimension must be even"
        )

    half_dim = (
        hidden_states.shape[-1] // 2
    )

    first_half = hidden_states[
        ...,
        :half_dim,
    ]

    second_half = hidden_states[
        ...,
        half_dim:,
    ]

    return torch.cat(
        (
            -second_half,
            first_half,
        ),
        dim=-1,
    )
    
def apply_vision_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """
    对 Vision Attention 的 Q/K 应用二维 RoPE。

    query/key:
        [total_patches, num_heads, head_dim]

    cos/sin:
        [total_patches, head_dim]
    """

    if query.shape != key.shape:
        raise ValueError(
            "Vision query and key must have "
            "the same shape"
        )

    if query.ndim != 3:
        raise ValueError(
            "Vision query/key must have shape "
            "[total_patches, num_heads, head_dim]"
        )

    (
        total_patches,
        _,
        head_dim,
    ) = query.shape

    expected_position_shape = (
        total_patches,
        head_dim,
    )

    if tuple(cos.shape) != expected_position_shape:
        raise ValueError(
            "Invalid Vision RoPE cos shape: "
            f"expected {expected_position_shape}, "
            f"got {tuple(cos.shape)}"
        )

    if tuple(sin.shape) != expected_position_shape:
        raise ValueError(
            "Invalid Vision RoPE sin shape: "
            f"expected {expected_position_shape}, "
            f"got {tuple(sin.shape)}"
        )

    query_dtype = query.dtype
    key_dtype = key.dtype

    # 增加 head 广播维：
    #
    # [T,D]
    #   ↓
    # [T,1,D]
    #
    # 同一个 patch 的所有 Attention head
    # 使用相同位置角度。
    cos = cos.unsqueeze(dim=1).float()
    sin = sin.unsqueeze(dim=1).float()

    # RoPE 计算使用 FP32，完成后转回模型精度。
    query_fp32 = query.float()
    key_fp32 = key.float()

    rotated_query = (
        query_fp32 * cos
        + rotate_half(query_fp32) * sin
    )

    rotated_key = (
        key_fp32 * cos
        + rotate_half(key_fp32) * sin
    )

    return (
        rotated_query.to(query_dtype),
        rotated_key.to(key_dtype),
    )
    

class Qwen3_5VisionAttention(nn.Module):
    """
    Qwen3.5 Vision Transformer 的非因果
    Variable-length Self-Attention。
    """

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads

        if (
            self.hidden_size
            % self.num_heads
            != 0
        ):
            raise ValueError(
                "Vision hidden_size must be "
                "divisible by num_heads"
            )

        self.head_dim = (
            self.hidden_size
            // self.num_heads
        )

        if self.head_dim % 2 != 0:
            raise ValueError(
                "Vision head_dim must be even "
                "for RoPE"
            )

        self.scaling = (
            self.head_dim ** -0.5
        )

        # 一次矩阵乘法同时产生 Q、K、V。
        #
        # 输入：
        # [T, hidden_size]
        #
        # 输出：
        # [T, 3 * hidden_size]
        self.qkv = nn.Linear(
            self.hidden_size,
            self.hidden_size * 3,
            bias=True,
        )

        # 将多头 Attention 输出投影回 hidden_size。
        self.proj = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:

        if hidden_states.ndim != 2:
            raise ValueError(
                "Vision hidden_states must have shape "
                "[total_patches, hidden_size]"
            )

        (
            total_patches,
            hidden_size,
        ) = hidden_states.shape

        if hidden_size != self.hidden_size:
            raise ValueError(
                "Invalid Vision hidden size: "
                f"expected {self.hidden_size}, "
                f"got {hidden_size}"
            )

        if (
            cu_seqlens.ndim != 1
            or cu_seqlens.numel() < 2
        ):
            raise ValueError(
                "Vision cu_seqlens must be a "
                "one-dimensional tensor with at "
                "least two elements"
            )

        if cu_seqlens.dtype != torch.int32:
            raise TypeError(
                "Vision cu_seqlens must use "
                "torch.int32"
            )

        if cu_seqlens[0].item() != 0:
            raise ValueError(
                "Vision cu_seqlens must start at 0"
            )

        if cu_seqlens[-1].item() != total_patches:
            raise ValueError(
                "The final Vision cu_seqlens value "
                "must equal total_patches"
            )

        if not hidden_states.is_cuda:
            raise RuntimeError(
                "Vision FlashAttention requires "
                "CUDA hidden states"
            )

        if cu_seqlens.device != hidden_states.device:
            raise ValueError(
                "Vision cu_seqlens and hidden_states "
                "must be on the same device"
            )

        if hidden_states.dtype not in {
            torch.float16,
            torch.bfloat16,
        }:
            raise TypeError(
                "Vision FlashAttention requires "
                "FP16 or BF16 hidden states"
            )

        # =====================================
        # 第一部分：QKV 投影
        # =====================================

        # [T,1152]
        #     ↓
        # [T,3456]
        qkv = self.qkv(
            hidden_states
        )

        # [T,3456]
        #     ↓
        # [T,3,16,72]
        qkv = qkv.reshape(
            total_patches,
            3,
            self.num_heads,
            self.head_dim,
        )

        # 沿 QKV 维度拆分：
        #
        # 每一个都是：
        # [T,16,72]
        query, key, value = qkv.unbind(
            dim=1
        )

        # =====================================
        # 第二部分：二维 Vision RoPE
        # =====================================

        if len(position_embeddings) != 2:
            raise ValueError(
                "position_embeddings must be "
                "a (cos, sin) tuple"
            )

        cos, sin = position_embeddings

        if (
            cos.device != hidden_states.device
            or sin.device != hidden_states.device
        ):
            raise ValueError(
                "Vision RoPE tensors and "
                "hidden_states must be on the "
                "same device"
            )

        query, key = (
            apply_vision_rotary_pos_emb(
                query=query,
                key=key,
                cos=cos,
                sin=sin,
            )
        )

        # =====================================
        # 第三部分：变长非因果 Attention
        # =====================================

        sequence_lengths = (
            cu_seqlens[1:]
            - cu_seqlens[:-1]
        )

        if torch.any(sequence_lengths <= 0):
            raise ValueError(
                "Every Vision sequence must "
                "contain at least one patch"
            )

        max_seqlen = int(
            sequence_lengths.max().item()
        )

        # 输入：
        #
        # query/key/value
        # [total_patches, num_heads, head_dim]
        #
        # 输出：
        # [total_patches, num_heads, head_dim]
        attention_output = (
            flash_attn_varlen_func(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=0.0,
                softmax_scale=self.scaling,
                causal=False,
            )
        )

        # =====================================
        # 第四部分：多头合并与输出投影
        # =====================================

        # [T,16,72]
        #     ↓
        # [T,1152]
        attention_output = (
            attention_output.reshape(
                total_patches,
                self.hidden_size,
            )
        )

        # [T,1152] → [T,1152]
        output = self.proj(
            attention_output
        )

        return output
    

class Qwen3_5VisionMLP(nn.Module):
    """
    Qwen3.5 Vision Transformer 中的前馈网络。

    结构：
        Linear
        GELU-tanh
        Linear
    """

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size

        self.intermediate_size = (
            config.intermediate_size
        )

        if (
            config.hidden_act
            != "gelu_pytorch_tanh"
        ):
            raise ValueError(
                "Qwen3.5 Vision MLP currently "
                "supports only "
                "'gelu_pytorch_tanh'"
            )

        # [T,1152] → [T,4304]
        self.linear_fc1 = nn.Linear(
            self.hidden_size,
            self.intermediate_size,
            bias=True,
        )

        # gelu_pytorch_tanh 精确对应：
        #
        # torch GELU(approximate="tanh")
        self.act_fn = nn.GELU(
            approximate="tanh"
        )

        # [T,4304] → [T,1152]
        self.linear_fc2 = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:

        if hidden_states.ndim != 2:
            raise ValueError(
                "Vision MLP hidden_states must "
                "have shape "
                "[total_patches, hidden_size]"
            )

        if (
            hidden_states.shape[-1]
            != self.hidden_size
        ):
            raise ValueError(
                "Invalid Vision MLP hidden size: "
                f"expected {self.hidden_size}, "
                f"got {hidden_states.shape[-1]}"
            )

        # [T,1152] → [T,4304]
        hidden_states = self.linear_fc1(
            hidden_states
        )

        # [T,4304] → [T,4304]
        hidden_states = self.act_fn(
            hidden_states
        )

        # [T,4304] → [T,1152]
        hidden_states = self.linear_fc2(
            hidden_states
        )

        return hidden_states
    
    
class Qwen3_5VisionBlock(nn.Module):
    """
    一个完整的 Qwen3.5 Vision Transformer Block。

    结构：
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
    """

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size

        # Attention 前的 LayerNorm。
        self.norm1 = nn.LayerNorm(
            self.hidden_size,
            eps=1e-6,
        )

        # Vision Self-Attention。
        self.attn = Qwen3_5VisionAttention(
            config
        )

        # MLP 前的 LayerNorm。
        self.norm2 = nn.LayerNorm(
            self.hidden_size,
            eps=1e-6,
        )

        # Vision 前馈网络。
        self.mlp = Qwen3_5VisionMLP(
            config
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:

        if hidden_states.ndim != 2:
            raise ValueError(
                "Vision Block hidden_states must "
                "have shape "
                "[total_patches, hidden_size]"
            )

        if (
            hidden_states.shape[-1]
            != self.hidden_size
        ):
            raise ValueError(
                "Invalid Vision Block hidden size: "
                f"expected {self.hidden_size}, "
                f"got {hidden_states.shape[-1]}"
            )

        # =====================================
        # 第一部分：Vision Self-Attention
        # =====================================

        # 保存未经归一化的输入。
        residual = hidden_states

        # [T,1152] → [T,1152]
        normalized_states = self.norm1(
            hidden_states
        )

        # 非因果 Variable-length Attention。
        attention_output = self.attn(
            hidden_states=normalized_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=(
                position_embeddings
            ),
        )

        # 第一条残差连接：
        #
        # x1 = x + Attention(LN(x))
        hidden_states = (
            residual + attention_output
        )

        # =====================================
        # 第二部分：Vision MLP
        # =====================================

        residual = hidden_states

        normalized_states = self.norm2(
            hidden_states
        )

        mlp_output = self.mlp(
            normalized_states
        )

        # 第二条残差连接：
        #
        # x2 = x1 + MLP(LN(x1))
        hidden_states = (
            residual + mlp_output
        )

        return hidden_states
    
    
class Qwen3_5VisionPatchMerger(nn.Module):
    """
    将 spatial_merge_size × spatial_merge_size
    个相邻 patch 合并成一个语言模型视觉 token。

    输入：
        [total_patches, vision_hidden_size]

    输出：
        [
            total_patches / merge_unit,
            text_hidden_size,
        ]
    """

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()

        self.vision_hidden_size = (
            config.hidden_size
        )

        self.spatial_merge_size = (
            config.spatial_merge_size
        )

        if self.spatial_merge_size <= 0:
            raise ValueError(
                "spatial_merge_size must be "
                "positive"
            )

        # spatial_merge_size=2：
        #
        # merge_unit = 2 × 2 = 4
        self.merge_unit = (
            self.spatial_merge_size
            * self.spatial_merge_size
        )

        # 4 个 1152 维 patch 拼接：
        #
        # 4 × 1152 = 4608
        self.merged_hidden_size = (
            self.vision_hidden_size
            * self.merge_unit
        )

        self.output_hidden_size = (
            config.out_hidden_size
        )

        # 注意：
        #
        # LayerNorm 在 patch 拼接之前执行，
        # 因此维度仍然是 1152。
        self.norm = nn.LayerNorm(
            self.vision_hidden_size,
            eps=1e-6,
        )

        # [num_visual_tokens,4608]
        #                 ↓
        # [num_visual_tokens,4608]
        self.linear_fc1 = nn.Linear(
            self.merged_hidden_size,
            self.merged_hidden_size,
            bias=True,
        )

        # Patch Merger 使用默认 GELU。
        #
        # 这里不是 Vision Block MLP 使用的
        # approximate="tanh"。
        self.act_fn = nn.GELU()

        # [num_visual_tokens,4608]
        #                 ↓
        # [num_visual_tokens,4096]
        self.linear_fc2 = nn.Linear(
            self.merged_hidden_size,
            self.output_hidden_size,
            bias=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:

        if hidden_states.ndim != 2:
            raise ValueError(
                "Patch Merger hidden_states must "
                "have shape "
                "[total_patches, vision_hidden_size]"
            )

        (
            total_patches,
            hidden_size,
        ) = hidden_states.shape

        if hidden_size != self.vision_hidden_size:
            raise ValueError(
                "Invalid Patch Merger hidden size: "
                f"expected "
                f"{self.vision_hidden_size}, "
                f"got {hidden_size}"
            )

        if total_patches % self.merge_unit != 0:
            raise ValueError(
                "The total number of patches must "
                "be divisible by merge_unit"
            )

        # =====================================
        # 第一部分：每个 patch 单独归一化
        # =====================================

        # [320,1152] → [320,1152]
        hidden_states = self.norm(
            hidden_states
        )

        # =====================================
        # 第二部分：每四个 patch 拼接
        # =====================================

        num_visual_tokens = (
            total_patches
            // self.merge_unit
        )

        # [320,1152]
        #      ↓
        # [80,4608]
        hidden_states = hidden_states.reshape(
            num_visual_tokens,
            self.merged_hidden_size,
        )

        # =====================================
        # 第三部分：投影到文本 hidden size
        # =====================================

        # [80,4608] → [80,4608]
        hidden_states = self.linear_fc1(
            hidden_states
        )

        hidden_states = self.act_fn(
            hidden_states
        )

        # [80,4608] → [80,4096]
        hidden_states = self.linear_fc2(
            hidden_states
        )

        return hidden_states
    
    
class Qwen3_5VisionModel(nn.Module):
    """
    Qwen3.5 的完整 Vision Transformer。

    输入：
        pixel_values
        [total_patches, patch_dim]

        grid_thw
        [num_images_or_videos, 3]

    输出：
        visual_embeddings
        [num_visual_tokens, text_hidden_size]
    """

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()

        self.config = config

        self.hidden_size = config.hidden_size

        self.depth = config.depth

        self.num_heads = config.num_heads

        self.spatial_merge_size = (
            config.spatial_merge_size
        )

        self.spatial_merge_unit = (
            self.spatial_merge_size
            * self.spatial_merge_size
        )

        if self.depth <= 0:
            raise ValueError(
                "Vision depth must be positive"
            )

        if (
            self.hidden_size
            % self.num_heads
            != 0
        ):
            raise ValueError(
                "Vision hidden_size must be "
                "divisible by num_heads"
            )

        # 当前项目首版没有实现 DeepStack
        # 中间视觉特征注入。
        deepstack_visual_indexes = tuple(
            getattr(
                config,
                "deepstack_visual_indexes",
                (),
            )
        )

        if deepstack_visual_indexes:
            raise NotImplementedError(
                "DeepStack visual feature "
                "injection is not supported"
            )

        # =====================================
        # 第一部分：Patch Embedding
        # =====================================

        self.patch_embed = (
            Qwen3_5VisionPatchEmbed(
                config
            )
        )

        # =====================================
        # 第二部分：位置编码
        # =====================================

        self.pos_embed = nn.Embedding(
            config.num_position_embeddings,
            config.hidden_size,
        )

        self.num_grid_per_side = isqrt(
            config.num_position_embeddings
        )

        if (
            self.num_grid_per_side
            * self.num_grid_per_side
            != config.num_position_embeddings
        ):
            raise ValueError(
                "num_position_embeddings must "
                "describe a square position grid"
            )

        head_dim = (
            self.hidden_size
            // self.num_heads
        )

        self.rotary_pos_emb = (
            Qwen3_5VisionRotaryEmbedding(
                head_dim // 2
            )
        )

        # =====================================
        # 第三部分：27 个 Vision Block
        # =====================================

        self.blocks = nn.ModuleList(
            [
                Qwen3_5VisionBlock(config)
                for _ in range(self.depth)
            ]
        )

        # =====================================
        # 第四部分：Patch Merger
        # =====================================

        self.merger = (
            Qwen3_5VisionPatchMerger(
                config
            )
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:

        if (
            grid_thw.ndim != 2
            or grid_thw.shape[-1] != 3
        ):
            raise ValueError(
                "grid_thw must have shape "
                "[num_images_or_videos, 3]"
            )

        if grid_thw.numel() == 0:
            raise ValueError(
                "grid_thw must not be empty"
            )

        if torch.any(grid_thw <= 0):
            raise ValueError(
                "Every grid T/H/W value must "
                "be positive"
            )

        if (
            pixel_values.device
            != grid_thw.device
        ):
            raise ValueError(
                "pixel_values and grid_thw must "
                "be on the same device"
            )

        # 每一行：
        #
        # T × H × W
        #
        # 多张图片时再求和。
        expected_num_patches = int(
            grid_thw.prod(dim=1).sum().item()
        )

        if (
            pixel_values.shape[0]
            != expected_num_patches
        ):
            raise ValueError(
                "pixel_values patch count does "
                "not match grid_thw: "
                f"expected "
                f"{expected_num_patches}, "
                f"got {pixel_values.shape[0]}"
            )

        # =====================================
        # 第一阶段：Patch Embedding
        # =====================================

        # [num_patches,1536]
        #          ↓
        # [num_patches,1152]
        hidden_states = self.patch_embed(
            pixel_values
        )

        # =====================================
        # 第二阶段：位置编码
        # =====================================

        (
            absolute_pos_embeds,
            position_embeddings,
            cu_seqlens,
        ) = prepare_vision_position_embeddings(
            grid_thw,
            spatial_merge_size=(
                self.spatial_merge_size
            ),
            num_grid_per_side=(
                self.num_grid_per_side
            ),
            pos_embed=self.pos_embed,
            rotary_pos_emb=(
                self.rotary_pos_emb
            ),
        )

        # 可学习绝对位置编码直接加到
        # patch hidden states。
        hidden_states = (
            hidden_states
            + absolute_pos_embeds.to(
                dtype=hidden_states.dtype
            )
        )

        # =====================================
        # 第三阶段：Vision Transformer
        # =====================================

        for block in self.blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=(
                    position_embeddings
                ),
            )

        # 此时：
        #
        # hidden_states
        # [total_patches,1152]

        # =====================================
        # 第四阶段：Patch Merger
        # =====================================

        visual_embeddings = self.merger(
            hidden_states
        )

        expected_visual_tokens = (
            expected_num_patches
            // self.spatial_merge_unit
        )

        if (
            visual_embeddings.shape[0]
            != expected_visual_tokens
        ):
            raise RuntimeError(
                "Patch Merger produced an "
                "unexpected number of visual tokens"
            )

        return visual_embeddings