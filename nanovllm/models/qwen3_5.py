import torch
from nanovllm.utils.context import get_context
from torch import nn

from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import Qwen3_5RMSNorm
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from nanovllm.layers.gated_delta_net import (
    Qwen3_5GatedDeltaNet,
)
from nanovllm.engine.hybrid_state import (
    GDNLayerState,
)
from nanovllm.layers.embed_head import (
    VocabParallelEmbedding,
    ParallelLMHead,
)
from nanovllm.models.qwen3_5_vision import (
    Qwen3_5VisionModel,
)

    # Qwen3_5ForConditionalGeneration.forward()
    #              ↓
    # Qwen3_5Model.forward()
    #              ↓
    # Qwen3_5TextModel.forward()
class Qwen3_5Attention(nn.Module):

    def __init__(
        self,
        config,
        layer_idx: int,
    ) -> None:
        super().__init__()

        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]

        if self.layer_type != "full_attention":
            raise ValueError(
                f"Layer {layer_idx} is not a "
                "full_attention layer"
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        # Qwen3.5 必须优先读取显式 head_dim。
        # 0.8B 的值是 256，不是 1024 / 8。
        self.head_dim = getattr(
            config,
            "head_dim",
            self.hidden_size // self.num_heads,
        )

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible "
                "by num_key_value_heads"
            )

        self.q_size = (
            self.num_heads * self.head_dim
        )

        self.kv_size = (
            self.num_kv_heads * self.head_dim
        )

        self.scaling = self.head_dim ** -0.5

        attention_bias = getattr(
            config,
            "attention_bias",
            False,
        )

        # q_proj 同时生成 Q 和 attention output gate。
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.q_size * 2,
            bias=attention_bias,
        )

        self.k_proj = nn.Linear(
            self.hidden_size,
            self.kv_size,
            bias=attention_bias,
        )

        self.v_proj = nn.Linear(
            self.hidden_size,
            self.kv_size,
            bias=attention_bias,
        )

        self.o_proj = nn.Linear(
            self.q_size,
            self.hidden_size,
            bias=attention_bias,
        )

        # Q/K 按每个 head 的 head_dim 做归一化。
        self.q_norm = Qwen3_5RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
        )

        self.k_norm = Qwen3_5RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
        )

        rope_parameters = getattr(
            config,
            "rope_parameters",
            None,
        ) or {}

        partial_rotary_factor = (
            rope_parameters.get(
                "partial_rotary_factor",
                1.0,
            )
        )

        self.rotary_dim = int(
            self.head_dim
            * partial_rotary_factor
        )

        rope_theta = rope_parameters.get(
            "rope_theta",
            getattr(config, "rope_theta", 10000.0),
        )
        
        mrope_interleaved = (
            rope_parameters.get(
                "mrope_interleaved",
                False,
            )
        )

        if not mrope_interleaved:
            raise NotImplementedError(
                "Qwen3.5 currently requires "
                "interleaved mRoPE"
            )

        raw_mrope_section = (
            rope_parameters.get(
                "mrope_section",
                None,
            )
        )

        if raw_mrope_section is None:
            raise ValueError(
                "Qwen3.5 config does not provide "
                "mrope_section"
            )

        mrope_section = tuple(
            int(section)
            for section in raw_mrope_section
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=(
                config.max_position_embeddings
            ),
            base=rope_theta,
            mrope_section=mrope_section,
        )
        
        # 复用 nano-vLLM 已有的：
        # FlashAttention
        # Paged KV Cache
        # Prefill/Decode 分支
        self.attn = Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:

        if hidden_states.ndim != 2:
            raise ValueError(
                "hidden_states must have shape "
                "[num_tokens, hidden_size]"
            )

        num_tokens = hidden_states.shape[0]

        # [T,1024] -> [T,4096]
        q_gate = self.q_proj(hidden_states)

        # [T,4096] -> [T,8,512]
        #
        # 每个 head 的 512 维实际排列为：
        # [Q head 的 256 维, gate head 的 256 维]
        q_gate = q_gate.view(
            num_tokens,
            self.num_heads,
            self.head_dim * 2,
        )

        # 分别得到：
        # q    [T,8,256]
        # gate [T,8,256]
        query, gate = q_gate.chunk(
            2,
            dim=-1,
        )

        # gate 最终需要和展平后的 attention output 相乘。
        gate = gate.reshape(
            num_tokens,
            self.q_size,
        )

        # [T,1024] -> [T,512] -> [T,2,256]
        key = self.k_proj(hidden_states).view(
            num_tokens,
            self.num_kv_heads,
            self.head_dim,
        )

        value = self.v_proj(hidden_states).view(
            num_tokens,
            self.num_kv_heads,
            self.head_dim,
        )

        query = self.q_norm(query)
        key = self.k_norm(key)

        # 只旋转 Q/K 前 rotary_dim=64 维。
        query, key = self.rotary_emb(
            positions,
            query,
            key,
        )

        # Prefill/Decode、KV Cache 写入和 causal attention
        # 都由底层 Attention 根据 Context 处理。
        output = self.attn(
            query,
            key,
            value,
        )

        # [T,8,256] -> [T,2048]
        output = output.flatten(
            start_dim=1,
        )

        # Full Attention 使用 sigmoid gate。
        output = output * torch.sigmoid(gate)

        # [T,2048] -> [T,1024]
        output = self.o_proj(output)

        return output
    

class Qwen3_5MLP(nn.Module):

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            config.intermediate_size
        )

        if config.hidden_act != "silu":
            raise ValueError(
                "Qwen3.5 MLP currently supports "
                "only SiLU activation"
            )

        # 把 gate_proj 和 up_proj 合并成一次矩阵乘法。
        #
        # 输入：
        # [T, hidden_size]
        #
        # 输出：
        # [T, 2 * intermediate_size]
        self.gate_up_proj = (
            MergedColumnParallelLinear(
                input_size=self.hidden_size,
                output_sizes=[
                    self.intermediate_size,
                    self.intermediate_size,
                ],
                bias=False,
            )
        )

        # 将合并输出拆成 gate 和 up：
        #
        # SiLU(gate) * up
        self.act_fn = SiluAndMul()

        # [T, intermediate_size]
        #     ↓
        # [T, hidden_size]
        self.down_proj = RowParallelLinear(
            input_size=self.intermediate_size,
            output_size=self.hidden_size,
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:

        # [T,H] -> [T,2I]
        gate_up = self.gate_up_proj(
            hidden_states
        )

        # [T,2I]
        #   ↓ split
        # gate [T,I], up [T,I]
        #   ↓
        # SiLU(gate) * up
        hidden_states = self.act_fn(
            gate_up
        )

        # [T,I] -> [T,H]
        hidden_states = self.down_proj(
            hidden_states
        )

        return hidden_states
    
    
class Qwen3_5DecoderLayer(nn.Module):

    def __init__(
        self,
        config,
        layer_idx: int,
        gdn_backend: str = "fla",
    ) -> None:
        super().__init__()

        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.block_type = (
            config.layer_types[layer_idx]
        )

        # 每层只创建与自身类型对应的 Token Mixer。
        if self.block_type == "linear_attention":
            self.linear_attn = (
                Qwen3_5GatedDeltaNet(
                    config=config,
                    layer_idx=layer_idx,
                    backend=gdn_backend,
                )
            )

        elif self.block_type == "full_attention":
            self.self_attn = Qwen3_5Attention(
                config=config,
                layer_idx=layer_idx,
            )

        else:
            raise ValueError(
                f"Unsupported layer type "
                f"{self.block_type!r} at layer "
                f"{layer_idx}"
            )

        # Token Mixer 前的 Pre-Norm。
        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # MLP 前的第二次 Pre-Norm。
        self.post_attention_layernorm = (
            Qwen3_5RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
        )

        self.mlp = Qwen3_5MLP(config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor | None = None,
        recurrent_state: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:

        if hidden_states.ndim != 2:
            raise ValueError(
                "hidden_states must have shape "
                "[num_tokens, hidden_size]"
            )

        num_tokens, hidden_size = (
            hidden_states.shape
        )

        if hidden_size != self.hidden_size:
            raise ValueError(
                f"Expected hidden_size "
                f"{self.hidden_size}, got "
                f"{hidden_size}"
            )

        has_text_positions = (
            positions.ndim == 1
            and positions.shape
            == (num_tokens,)
        )

        has_mrope_positions = (
            positions.ndim == 2
            and positions.shape
            == (3, num_tokens)
        )

        if not (
            has_text_positions
            or has_mrope_positions
        ):
            raise ValueError(
                "positions must have shape "
                "[num_tokens] or "
                "[3, num_tokens]"
            )

        # =====================================
        # 第一部分：Token Mixer
        # =====================================

        # 保存未经归一化的输入，用于第一条残差。
        residual = hidden_states

        # Pre-Norm：
        # [T,H] -> [T,H]
        normalized_states = (
            self.input_layernorm(
                hidden_states
            )
        )

        if self.block_type == "linear_attention":
            context = get_context()
            
            prefill_seqlens = (
                context.prefill_seqlens
                if context.is_prefill
                else None
            )
            gdn_cu_seqlens = (
                context.gdn_cu_seqlens
                if context.is_prefill
                else None
            )
            
            if context.is_prefill:
                # Prefill：
                #
                # normalized_states [T, H]
                #
                # T 表示一条 Sequence 中的多个 token。
                #
                # GDN 输入：
                # [1, T, H]
                gdn_input = normalized_states.unsqueeze(
                    dim=0
                )

            else:
                # Batched Decode：
                #
                # normalized_states [B, H]
                #
                # B 表示多条 Sequence，
                # 每条 Sequence 本轮只有一个 token。
                #
                # GDN 输入：
                # [B, 1, H]
                gdn_input = normalized_states.unsqueeze(
                    dim=1
                )

            (
                token_mixer_output,
                new_conv_state,
                new_recurrent_state,
            ) = self.linear_attn(
                hidden_states=gdn_input,
                conv_state=conv_state,
                recurrent_state=recurrent_state,
                prefill_seqlens=prefill_seqlens,
                gdn_cu_seqlens=gdn_cu_seqlens,
            )
            
            if context.is_prefill:
                # [1, T, H] -> [T, H]
                token_mixer_output = (
                    token_mixer_output.squeeze(dim=0)
                )
            else:
                # [B, 1, H] -> [B, H]
                token_mixer_output = (
                    token_mixer_output.squeeze(dim=1)
                )
        else:
            # Full Attention 层不应该收到 GDN state。
            if (
                conv_state is not None
                or recurrent_state is not None
            ):
                raise ValueError(
                    "Full Attention layers do not "
                    "accept GDN states"
                )

            token_mixer_output = self.self_attn(
                positions=positions,
                hidden_states=normalized_states,
            )

            new_conv_state = None
            new_recurrent_state = None

        # 第一条残差连接：
        #
        # x1 = x + TokenMixer(RMSNorm(x))
        hidden_states = (
            residual + token_mixer_output
        )

        # =====================================
        # 第二部分：SwiGLU MLP
        # =====================================

        # 保存 Token Mixer 之后的结果，
        # 用于第二条残差连接。
        residual = hidden_states

        normalized_states = (
            self.post_attention_layernorm(
                hidden_states
            )
        )

        mlp_output = self.mlp(
            normalized_states
        )

        # 第二条残差连接：
        #
        # x2 = x1 + MLP(RMSNorm(x1))
        hidden_states = residual + mlp_output

        return (
            hidden_states,
            new_conv_state,
            new_recurrent_state,
        )
        
        
class Qwen3_5TextModel(nn.Module):

    def __init__(
        self,
        config,
        gdn_backend: str = "fla",
    ) -> None:
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = (
            config.num_hidden_layers
        )

        # token ID -> hidden vector
        self.embed_tokens = (
            VocabParallelEmbedding(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
            )
        )

        self.layers = nn.ModuleList(
            [
                Qwen3_5DecoderLayer(
                    config=config,
                    layer_idx=layer_idx,
                    gdn_backend=gdn_backend,
                )
                for layer_idx in range(
                    config.num_hidden_layers
                )
            ]
        )

        # 所有 DecoderLayer 执行结束后的最终归一化。
        self.norm = Qwen3_5RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        gdn_states: list[
            GDNLayerState | None
        ] | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        list[GDNLayerState | None],
    ]:

        # input_ids 和 inputs_embeds 必须二选一。
        #
        # 纯文本阶段：
        #     使用 input_ids
        #
        # 未来图文阶段：
        #     先合并文本与视觉 embedding，
        #     再使用 inputs_embeds
        if (
            (input_ids is None)
            == (inputs_embeds is None)
        ):
            raise ValueError(
                "Exactly one of input_ids and "
                "inputs_embeds must be provided"
            )

        if inputs_embeds is None:
            if input_ids.ndim != 1:
                raise ValueError(
                    "input_ids must have shape "
                    "[num_tokens]"
                )

            hidden_states = self.embed_tokens(
                input_ids
            )

        else:
            if (
                inputs_embeds.ndim != 2
                or inputs_embeds.shape[-1]
                != self.hidden_size
            ):
                raise ValueError(
                    "inputs_embeds must have shape "
                    "[num_tokens, hidden_size]"
                )

            hidden_states = inputs_embeds

        num_tokens = hidden_states.shape[0]

        has_text_positions = (
            positions.ndim == 1
            and positions.shape
            == (num_tokens,)
        )

        has_mrope_positions = (
            positions.ndim == 2
            and positions.shape
            == (3, num_tokens)
        )

        if not (
            has_text_positions
            or has_mrope_positions
        ):
            raise ValueError(
                "positions must have shape "
                "[num_tokens] or "
                "[3, num_tokens]"
            )
            
        # 首次 Prefill 时调用方还没有任何 GDN state。
        if gdn_states is None:
            gdn_states = [
                None
                for _ in range(
                    self.num_hidden_layers
                )
            ]

        elif (
            len(gdn_states)
            != self.num_hidden_layers
        ):
            raise ValueError(
                "gdn_states must contain one "
                "entry per DecoderLayer"
            )

        updated_gdn_states: list[
            GDNLayerState | None
        ] = [
            None
            for _ in range(
                self.num_hidden_layers
            )
        ]

        for layer_idx, layer in enumerate(
            self.layers
        ):
            layer_state = gdn_states[layer_idx]

            if layer.block_type == "linear_attention":

                if layer_state is None:
                    conv_state = None
                    recurrent_state = None

                else:
                    conv_state = (
                        layer_state.conv_state
                    )
                    recurrent_state = (
                        layer_state.recurrent_state
                    )

                (
                    hidden_states,
                    new_conv_state,
                    new_recurrent_state,
                ) = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    conv_state=conv_state,
                    recurrent_state=recurrent_state,
                )

                updated_gdn_states[layer_idx] = (
                    GDNLayerState(
                        conv_state=new_conv_state,
                        recurrent_state=(
                            new_recurrent_state
                        ),
                    )
                )

            else:
                # Full Attention 层对应的列表项必须为空。
                if layer_state is not None:
                    raise ValueError(
                        "Full Attention layer "
                        f"{layer_idx} received a "
                        "GDN state"
                    )

                (
                    hidden_states,
                    unused_conv_state,
                    unused_recurrent_state,
                ) = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    conv_state=None,
                    recurrent_state=None,
                )

                if (
                    unused_conv_state is not None
                    or unused_recurrent_state is not None
                ):
                    raise RuntimeError(
                        "Full Attention layer "
                        "returned a GDN state"
                    )

        # 最后一个 DecoderLayer 后还有一次 RMSNorm。
        hidden_states = self.norm(
            hidden_states
        )

        return (
            hidden_states,
            updated_gdn_states,
        )
        
        
class Qwen3_5Model(nn.Module):

    def __init__(
        self,
        config,
        gdn_backend: str = "fla",
    ) -> None:
        super().__init__()

        self.config = config

        text_config = config.get_text_config(
            decoder=True
        )

        vision_config = getattr(
            config,
            "vision_config",
            None,
        )

        if vision_config is None:
            raise ValueError(
                "Qwen3.5 multimodal model must "
                "define vision_config"
            )

        # Patch Merger 最终输出必须能够直接
        # 替换文本 token embedding。
        if (
            vision_config.out_hidden_size
            != text_config.hidden_size
        ):
            raise ValueError(
                "Vision out_hidden_size must equal "
                "text hidden_size: "
                f"{vision_config.out_hidden_size} "
                f"!= {text_config.hidden_size}"
            )

        # 模块名必须叫 visual，才能对应：
        #
        # model.visual.*
        self.visual = Qwen3_5VisionModel(
            vision_config
        )

        # 模块名必须叫 language_model，
        # 才能对应：
        #
        # model.language_model.*
        self.language_model = (
            Qwen3_5TextModel(
                config=text_config,
                gdn_backend=gdn_backend,
            )
        )

    def get_visual_embeddings(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        执行 Vision Tower。

        输入：
            pixel_values
            [total_patches, patch_dim]

            image_grid_thw
            [num_images, 3]

        输出：
            visual_embeddings
            [num_visual_tokens, text_hidden_size]
        """

        return self.visual(
            pixel_values=pixel_values,
            grid_thw=image_grid_thw,
        )        

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        gdn_states: list[
            GDNLayerState | None
        ] | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        list[GDNLayerState | None],
    ]:

        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            gdn_states=gdn_states,
            inputs_embeds=inputs_embeds,
        )
        
class Qwen3_5ForConditionalGeneration(nn.Module):

    # 只合并 MLP 的 gate/up。
    #
    # 注意不要把 q_proj/k_proj/v_proj
    # 映射到旧 Qwen3 的 qkv_proj。
    packed_modules_mapping = {
        "gate_proj": (
            "gate_up_proj",
            0,
        ),
        "up_proj": (
            "gate_up_proj",
            1,
        ),
    }

    # MTP 暂不属于首版范围。
    #
    # Vision Tower 已实现，因此 model.visual.*
    # 必须接受严格的权重加载和完整性检查。
    ignored_weight_prefixes = (
        "mtp.",
    )

    def __init__(
        self,
        config,
        gdn_backend: str = "fla",
    ) -> None:
        super().__init__()

        self.config = config

        text_config = config.get_text_config(
            decoder=True
        )

        self.text_config = text_config

        self.model = Qwen3_5Model(
            config=config,
            gdn_backend=gdn_backend,
        )

        self.lm_head = ParallelLMHead(
            num_embeddings=text_config.vocab_size,
            embedding_dim=text_config.hidden_size,
            bias=False,
        )

        # 0.8B 和 4B checkpoint 使用共享权重。
        if text_config.tie_word_embeddings:
            self.lm_head.weight = (
                self.model
                .language_model
                .embed_tokens
                .weight
            )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        把文本 token ID 转换成文本 embedding。

        input_ids:
            [num_tokens]

        return:
            [num_tokens, hidden_size]
        """

        if input_ids.ndim != 1:
            raise ValueError(
                "input_ids must have shape [num_tokens]"
            )

        return (
            self.model
            .language_model
            .embed_tokens(input_ids)
        )

    def get_visual_embeddings(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:

        return self.model.get_visual_embeddings(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        gdn_states: list[
            GDNLayerState | None
        ] | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        list[GDNLayerState | None],
    ]:

        return self.model(
            input_ids=input_ids,
            positions=positions,
            gdn_states=gdn_states,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:

        return self.lm_head(
            hidden_states
        )