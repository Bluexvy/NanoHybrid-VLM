import torch
import torch.nn.functional as F
from torch import nn

try:
    from causal_conv1d import (
        causal_conv1d_fn as causal_conv1d_cuda,
        causal_conv1d_update as causal_conv1d_update_cuda,
    )
except ImportError:
    causal_conv1d_cuda = None
    causal_conv1d_update_cuda = None


try:
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule,
    )
except ImportError:
    chunk_gated_delta_rule = None
    fused_recurrent_gated_delta_rule = None


"""Q、K 的 L2 归一化"""
# 最后一维度才是每个head的向量维度
def l2norm(
    x: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-6,
) -> torch.Tensor:
    # reverse sqrt 所以最后是使用乘法
    inv_norm = torch.rsqrt(
        (x * x).sum(
            dim=dim,
            keepdim=True,
        ) + eps # 加 eps 防止 某个向量全是 0
    )

    return x * inv_norm

"""短期局部状态 conv_state"""
def torch_causal_conv1d_reference(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
) -> tuple[torch.Tensor, torch.Tensor]:
    if hidden_states.ndim != 3:
        raise ValueError(
            "hidden_states must have shape "
            "[batch, channels, sequence]"
        )
    # 取出 B C L
    # B 条序列
    # 每条有 C 个 channel
    # 本轮每条处理 L 个 token
    batch_size, num_channels, sequence_length = (
        hidden_states.shape
    )

    if sequence_length == 0:
        raise ValueError(
            "sequence_length must be positive"
        )

    if (
        weight.ndim != 2
        or weight.shape[0] != num_channels
    ):
        raise ValueError(
            "weight must have shape "
            "[channels, kernel_size]"
        )

    # 取出kernel size
    kernel_size = weight.shape[-1]

    if kernel_size == 0:
        raise ValueError(
            "kernel_size must be positive"
        )

    if (
        bias is not None
        and bias.shape != (num_channels,)
    ):
        raise ValueError(
            "bias must have shape [channels]"
        )
    # 假设有以下参数
    # B=2
    # C=3
    # K=4
    # 每条序列
    # 每个 channel
    # 保存最近 4 个值
    expected_state_shape = (
        batch_size,
        num_channels,
        kernel_size,
    )

    if (
        initial_state is not None
        and initial_state.shape
        != expected_state_shape
    ):
        raise ValueError(
            "initial_state must have shape "
            f"{expected_state_shape}"
        )
        
    # 记录原始输入dtype
    input_dtype = hidden_states.dtype

    if initial_state is None:
        state = torch.zeros(
            expected_state_shape,
            dtype=input_dtype,
            device=hidden_states.device,
        )
    else:
        state = initial_state.to(
            device=hidden_states.device,
            dtype=input_dtype,
        ).clone()

    # 沿最后一个维度拼接历史和当前输入
    # state.shape         = [B,C,K]
    # hidden_states.shape = [B,C,L]
    # combined_states.shape = [B,C,K+L]
    combined_states = torch.cat(
        [state, hidden_states],
        dim=-1,
    )

    conv_output = F.conv1d(
        combined_states.to(weight.dtype),
        weight.unsqueeze(dim=1),
        bias,
        padding=0,
        groups=num_channels,
    )

    # combined_states 的第一个窗口完全属于旧 state。
    # 只保留本次新输入对应的 sequence_length 个输出。
    conv_output = conv_output[
        ...,
        -sequence_length:
    ]

    # 卷积其实还是一个线性变换，所以需要一个silu加入非线性能力
    if activation == "silu":
        conv_output = F.silu(conv_output)
    elif activation is not None:
        raise ValueError(
            f"Unsupported activation: {activation!r}"
        )

    # 只保留最后一个kernel的数据
    new_state = combined_states[
        ...,
        -kernel_size:
    ].clone()

    return (
        conv_output.to(input_dtype),
        new_state,
    )

"""长期压缩状态 recurrent_state"""
def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    
    # 判断输入维度
    if query.ndim != 4:
        raise ValueError(
            "query must have shape "
            "[batch, sequence, heads, key_dim]"
        )
        
    # 获取query形状 然后逐条判断
    batch_size, sequence_length, num_heads, key_dim = (
        query.shape
    )

    if sequence_length == 0:
        raise ValueError(
            "sequence_length must be positive"
        )

    if key.shape != query.shape:
        raise ValueError(
            "key must have the same shape as query"
        )

    if (
        value.ndim != 4
        or value.shape[:3]
        != (batch_size, sequence_length, num_heads)
    ):
        raise ValueError(
            "value must have shape "
            "[batch, sequence, heads, value_dim]"
        )

    expected_gate_shape = (
        batch_size,
        sequence_length,
        num_heads,
    )

    if g.shape != expected_gate_shape:
        raise ValueError(
            f"g must have shape {expected_gate_shape}"
        )

    if beta.shape != expected_gate_shape:
        raise ValueError(
            "beta must have the same shape as g"
        )

    # 获取value head的维度
    value_dim = value.shape[-1]
    
    # 每条 Sequence
    #     ↓
    # 每个 GDN Head
    #     ↓
    # 拥有一个 [Dk, Dv] 记忆矩阵
    # state.shape = [2, 4, 5, 6]
    # state[0, 0]：seq0 的 head0 记忆矩阵
    # state[0, 1]：seq0 的 head1 记忆矩阵
    # state[1, 0]：seq1 的 head0 记忆矩阵
    # ...
    expected_state_shape = (
        batch_size,
        num_heads,
        key_dim,
        value_dim,
    )

    if (
        initial_state is not None
        and initial_state.shape
        != expected_state_shape
    ):
        raise ValueError(
            "initial_state must have shape "
            f"{expected_state_shape}"
        )

    # 先记录输入type 后面的计算状态需要全部换成FP32 但是最终输出还是要转回模型原始精度
    input_dtype = query.dtype
    
    # state状态需要重复被计算重复被使用 所以需要用fp32提高精度     
    # GDN state 会被每个 token 反复更新：S₀ → S₁ → S₂ → S₃ → ... → S₁₀₀₀₀
    # Reference 路径统一在 FP32 中进行状态计算。
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)
    g = g.to(torch.float32)
    beta = beta.to(torch.float32)

    # 是否在函数内部对qk做l2归一化
    if use_qk_l2norm_in_kernel:
        query = l2norm(
            query,
            dim=-1,
            eps=1e-6,
        )

        key = l2norm(
            key,
            dim=-1,
            eps=1e-6,
        )

    # q除以根号dk 和标准的attention差不多
    query = query * (key_dim ** -0.5)

    # 没有旧状态时创建全零状态： 第一次处理一个新 Sequence 时，没有历史记忆 没有旧 conv_state 也没有旧 recurrent_state
    # 所以从全零矩阵开始
    if initial_state is None:
        state = torch.zeros(
            expected_state_shape,
            dtype=torch.float32,
            device=query.device,
        )
    # 有旧状态 先移动到相同设备 然后转成FP32 再复制一份 
    # 如果要实现高性能 Decode，可以选择原地修改缓存状态，但需要更谨慎地处理投机解码回滚。
    else:
        state = initial_state.to(
            device=query.device,
            dtype=torch.float32,
        ).clone()

    outputs = []

    # 逐token循环
    # GPU 每个 token 都要形成连续依赖，长 Prefill 时利用率不好。
    # 高性能 Prefill 一般使用：
    # chunk gated delta rule
    # 将多个 token 分块并行计算。
    
    # 假设目前是token0-5 目前token0进入循环
    for token_idx in range(sequence_length):
        query_t = query[:, token_idx]
        key_t = key[:, token_idx]
        value_t = value[:, token_idx]
        g_t = g[:, token_idx]
        beta_t = beta[:, token_idx]

        # 假设目前是token0 那么 先计算token0的遗忘率
        # g_t.shape = [B, H] 所以需要两次升维度 变成 [B, H, 1, 1] 从而可以和state相乘
        # 每条 Sequence、每个 head 都使用自己的衰减标量。
        decay_t = (
            g_t.exp()
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        # 遗忘一部分旧状态
        state = state * decay_t

        # 用 token0 的 key 查询旧状态 key_t：[B, H, Dk] 所以需要升维度
        # [B, H, Dk, Dv]
        # 然后按照dim=-2维度 也就是 Dk维度求和
        # 结果为[B, H, Dv]
        remembered_value = (
            state
            * key_t.unsqueeze(-1)
        ).sum(dim=-2)

        # 计算token0需要写入的误差
        delta = (
            value_t - remembered_value
        ) * beta_t.unsqueeze(-1)

        # 把token0的误差写进状态
        # state 已经不再是初始状态
        # 而是包含 token0 信息的新状态 S₀
        state = (
            state
            + key_t.unsqueeze(-1)
            * delta.unsqueeze(-2)
        )

        # 用 token0 的 Query 读取新状态 依旧沿着DK维度求和
        output_t = (
            state
            * query_t.unsqueeze(-1)
        ).sum(dim=-2)

        outputs.append(output_t)

    # 处理完所有token后 形状是 [B, H, Dv] 需要沿着第一个维度重新堆叠回去 [B, 5, H, Dv]
    # output[:, 0] → token0 的输出
    # output[:, 1] → token1 的输出
    # output[:, 2] → token2 的输出
    # output[:, 3] → token3 的输出
    # output[:, 4] → token4 的输出
    output = torch.stack(
        outputs,
        dim=1,
    ).to(input_dtype)

    final_state = (
        state
        if output_final_state
        else None
    )
    # output形状 [B, L, H, Dv] 这个output要往下继续走后面的流程
    # final_state形状 [B, H, Dk, Dv] 保存到当前 GDN Layer 的状态缓存 下一轮 Decode 时继续使用
    return output, final_state


"""先对core_output做 RMSNorm 再乘以SiLU(z)"""
class Qwen3_5RMSNormGated(nn.Module):
    def __init__(
        self,
        # 每个 value head 的维度 head_v_dim
        hidden_size: int,
        eps: float = 1e-6,
    ):
        super().__init__()

        # 创建相应维度的1矩阵
        self.weight = nn.Parameter(
            torch.ones(
                hidden_size,
                dtype=torch.float32,
            )
        )

        self.variance_epsilon = eps

    # 真正计算RMSNorm
    def forward(
        self,
        hidden_states: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.shape != gate.shape:
            raise ValueError(
                "hidden_states and gate must have the same shape"
            )

        input_dtype = hidden_states.dtype

        # 先转换成FP32
        hidden_states_fp32 = hidden_states.float()

        # 求均方
        # 假设 hidden_states.shape = [2,4,3]
        # 2 个 batch
        # 4 个位置
        # 每个位置 3 个数 mean(dim=-1, keepdim=True)
        # variance.shape = [2,4,1]
        # 最后的 1 被保留下来。 免得后面再用unsqueeze(-1)
        variance = hidden_states_fp32.pow(2).mean(
            dim=-1,
            keepdim=True,
        )

        normalized = hidden_states_fp32 * torch.rsqrt(
            variance + self.variance_epsilon
        )

        normalized = (
            self.weight
            * normalized.to(input_dtype)
        )

        gated_output = (
            normalized
            * F.silu(gate.float())
        )

        return gated_output.to(input_dtype)
    
    
class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
        backend: str = "torch",
    ):
        super().__init__()
        
        # 只有两套路径
        if backend not in {
            "torch",
            "fla",
        }:
            raise ValueError(
                "backend must be either "
                "'torch' or 'fla'"
            )

        if backend == "fla":
            required_functions = (
                causal_conv1d_cuda,
                causal_conv1d_update_cuda,
                chunk_gated_delta_rule,
                fused_recurrent_gated_delta_rule,
            )

            if any(
                function is None
                for function in required_functions
            ):
                raise ImportError(
                    "The FLA backend requires "
                    "flash-linear-attention and "
                    "causal-conv1d"
                )

        self.backend = backend

        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        self.num_k_heads = (
            config.linear_num_key_heads
        )
        self.num_v_heads = (
            config.linear_num_value_heads
        )

        self.head_k_dim = (
            config.linear_key_head_dim
        )
        self.head_v_dim = (
            config.linear_value_head_dim
        )

        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                "linear_num_value_heads must be divisible by "
                "linear_num_key_heads"
            )

        self.key_dim = (
            self.num_k_heads
            * self.head_k_dim
        )

        self.value_dim = (
            self.num_v_heads
            * self.head_v_dim
        )

        self.conv_dim = (
            self.key_dim * 2
            + self.value_dim
        )

        self.conv_kernel_size = (
            config.linear_conv_kernel_dim
        )

        self.activation = config.hidden_act
        self.layer_norm_epsilon = (
            config.rms_norm_eps
        )

        if self.activation != "silu":
            raise ValueError(
                "The reference GDN currently supports only SiLU"
            )

        # qwen3.5不是每一层都是GDN
        self.layer_type = (
            config.layer_types[layer_idx]
        )

        if self.layer_type != "linear_attention":
            raise ValueError(
                f"Layer {layer_idx} is not a "
                "linear_attention layer"
            )

        self.in_proj_qkv = nn.Linear(
            self.hidden_size,
            self.conv_dim,
            bias=False,
        )

        """
        a：当前 token 想忘多少
        b：当前 token 想写多少
        z：当前 token 想输出多少
        """

        self.in_proj_z = nn.Linear(
            self.hidden_size,
            self.value_dim,
            bias=False,
        )

        # 每个 token、每个 Value head 一个标量。
        self.in_proj_b = nn.Linear(
            self.hidden_size,
            self.num_v_heads,
            bias=False,
        )

        self.in_proj_a = nn.Linear(
            self.hidden_size,
            self.num_v_heads,
            bias=False,
        )

        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
            padding=self.conv_kernel_size - 1,
        )

        self.dt_bias = nn.Parameter(
            torch.ones(self.num_v_heads)
        )

        initial_a = torch.empty(
            self.num_v_heads,
            dtype=torch.float32,
        ).uniform_(0.01, 16)

        self.A_log = nn.Parameter(
            torch.log(initial_a)
        )

        self.norm = Qwen3_5RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
        )

        self.out_proj = nn.Linear(
            self.value_dim,
            self.hidden_size,
            bias=False,
        )
    
    def _run_variable_length_causal_conv1d(
        self,
        mixed_qkv: torch.Tensor,
        conv_state: torch.Tensor | None,
        prefill_seqlens: tuple[int, ...],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        对Packed Variable-length Prefill执行
        Depthwise Causal Convolution。

        mixed_qkv:
            [1, C, total_tokens]

        conv_state:
            [num_sequences, C, kernel_size]

        prefill_seqlens:
            每条请求本轮处理的token数量。

        返回：
            conv_output:
                [1, C, total_tokens]

            new_conv_state:
                [num_sequences, C, kernel_size]
        """

        if mixed_qkv.ndim != 3:
            raise ValueError(
                "mixed_qkv must have shape "
                "[1, channels, total_tokens]"
            )

        packed_batch_size = mixed_qkv.shape[0]
        num_channels = mixed_qkv.shape[1]
        total_tokens = mixed_qkv.shape[2]

        if packed_batch_size != 1:
            raise ValueError(
                "Variable-length packed convolution "
                "requires batch size 1"
            )

        num_sequences = len(prefill_seqlens)

        if num_sequences <= 1:
            raise ValueError(
                "Variable-length convolution requires "
                "more than one sequence"
            )

        if any(
            seqlen <= 0
            for seqlen in prefill_seqlens
        ):
            raise ValueError(
                "Every Prefill sequence length must "
                "be positive"
            )

        if sum(prefill_seqlens) != total_tokens:
            raise ValueError(
                "The sum of prefill_seqlens must equal "
                "the number of packed tokens"
            )

        kernel_size = self.conv_kernel_size

        expected_state_shape = (
            num_sequences,
            num_channels,
            kernel_size,
        )

        # 允许直接单独调用该函数进行验证。
        # 正式Runtime中，HybridStateManager会为新请求
        # Gather出对应的全零state行。
        if conv_state is None:
            conv_state = torch.zeros(
                expected_state_shape,
                dtype=mixed_qkv.dtype,
                device=mixed_qkv.device,
            )

        elif (
            tuple(conv_state.shape)
            != expected_state_shape
        ):
            raise ValueError(
                "Packed conv_state must have shape "
                f"{expected_state_shape}, got "
                f"{tuple(conv_state.shape)}"
            )

        conv_state = conv_state.to(
            device=mixed_qkv.device,
            dtype=mixed_qkv.dtype,
        )

        # Reference路径不追求性能。
        # 分别执行每条请求，再拼接输出。
        if self.backend == "torch":
            outputs = []
            final_states = []

            token_offset = 0

            for sequence_idx, seqlen in enumerate(
                prefill_seqlens
            ):
                token_end = token_offset + seqlen

                sequence_input = mixed_qkv[
                    :,
                    :,
                    token_offset:token_end,
                ]

                sequence_state = conv_state[
                    sequence_idx:
                    sequence_idx + 1
                ]

                (
                    sequence_output,
                    sequence_final_state,
                ) = torch_causal_conv1d_reference(
                    hidden_states=sequence_input,
                    weight=(
                        self.conv1d.weight.squeeze(1)
                    ),
                    bias=self.conv1d.bias,
                    initial_state=sequence_state,
                    activation=self.activation,
                )

                outputs.append(sequence_output)

                final_states.append(
                    sequence_final_state[0]
                )

                token_offset = token_end

            return (
                torch.cat(outputs, dim=-1),
                torch.stack(final_states, dim=0),
            )

        if not mixed_qkv.is_cuda:
            raise RuntimeError(
                "The FLA backend requires CUDA tensors"
            )

        # 每一段扩展为：
        #
        #     [旧conv state, 当前token]
        #
        # 然后使用seq_idx防止不同请求之间串状态。
        extended_inputs = []
        sequence_indices = []
        new_conv_states = []

        token_offset = 0

        for sequence_idx, seqlen in enumerate(
            prefill_seqlens
        ):
            token_end = token_offset + seqlen

            current_input = mixed_qkv[
                0,
                :,
                token_offset:token_end,
            ]

            history = conv_state[
                sequence_idx
            ]

            extended_input = torch.cat(
                [
                    history,
                    current_input,
                ],
                dim=-1,
            )

            extended_inputs.append(
                extended_input
            )

            sequence_indices.append(
                torch.full(
                    (
                        kernel_size
                        + seqlen,
                    ),
                    sequence_idx,
                    dtype=torch.int32,
                    device=mixed_qkv.device,
                )
            )

            # 保存当前请求最后kernel_size个原始输入。
            #
            # 这些数据是下一轮Chunked Prefill或Decode
            # 的卷积历史。
            new_conv_states.append(
                extended_input[
                    :,
                    -kernel_size:
                ].clone()
            )

            token_offset = token_end

        packed_extended_input = torch.cat(
            extended_inputs,
            dim=-1,
        )

        # causal-conv1d在使用seq_idx时要求：
        #
        #     x.shape     = [1, C, T]
        #     x.stride(1) = 1
        #
        # 先转成[T,C]并contiguous，
        # 再转回[C,T]，使channel维在物理内存中连续。
        packed_extended_input = (
            packed_extended_input
            .transpose(0, 1)
            .contiguous()
            .transpose(0, 1)
            .unsqueeze(dim=0)
        )

        if (
            packed_extended_input.stride(1)
            != 1
        ):
            raise RuntimeError(
                "Packed causal convolution input "
                "must use channel-last memory layout"
            )
        seq_idx = torch.cat(
            sequence_indices,
            dim=0,
        ).unsqueeze(dim=0)

        packed_extended_output = (
            causal_conv1d_cuda(
                x=packed_extended_input,
                weight=(
                    self.conv1d.weight.squeeze(1)
                ),
                bias=self.conv1d.bias,
                seq_idx=seq_idx,
                activation=self.activation,
            )
        )

        # 删除每条请求前面历史state位置对应的输出，
        # 只恢复当前Prefill token的输出。
        current_outputs = []

        extended_offset = 0

        for seqlen in prefill_seqlens:
            current_start = (
                extended_offset
                + kernel_size
            )

            current_end = (
                current_start
                + seqlen
            )

            current_outputs.append(
                packed_extended_output[
                    :,
                    :,
                    current_start:current_end,
                ]
            )

            extended_offset += (
                kernel_size + seqlen
            )

        conv_output = torch.cat(
            current_outputs,
            dim=-1,
        )

        final_conv_state = torch.stack(
            new_conv_states,
            dim=0,
        )

        return (
            conv_output,
            final_conv_state,
        )
    
    
    
    """
    输入：mixed_qkv [B,C,L]
    conv_state [B,C,K] 或 None
    
    输出：conv_output [B,C,L]
    new_conv_state [B,C,K]
    """
    def _run_causal_conv1d(
        self,
        mixed_qkv: torch.Tensor,
        conv_state: torch.Tensor | None,
        prefill_seqlens: (
            tuple[int, ...] | None
        ) = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        is_variable_length_prefill = (
            prefill_seqlens is not None
            and len(prefill_seqlens) > 1
        )

        if is_variable_length_prefill:
            return (
                self
                ._run_variable_length_causal_conv1d(
                    mixed_qkv=mixed_qkv,
                    conv_state=conv_state,
                    prefill_seqlens=(
                        prefill_seqlens
                    ),
                )
            )
        # 如果是 torch 就回到普通版本的因果卷积 否则使用FLA加速
        if self.backend == "torch":
            return torch_causal_conv1d_reference(
                hidden_states=mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                initial_state=conv_state,
                activation=self.activation,
            )


        # FLA需要cuda kernel实现
        if not mixed_qkv.is_cuda:
            raise RuntimeError(
                "The FLA backend requires CUDA tensors"
            )

        (
            batch_size,
            num_channels,
            sequence_length,
        ) = mixed_qkv.shape

        kernel_size = self.conv_kernel_size

        expected_state_shape = (
            batch_size,
            num_channels,
            kernel_size,
        )

        if (
            conv_state is not None
            and conv_state.shape
            != expected_state_shape
        ):
            raise ValueError(
                "conv_state must have shape "
                f"{expected_state_shape}"
            )

        weight = self.conv1d.weight.squeeze(1)

        # 在这里判断是Prefill/Chunked Prefill/Decode 还是 新请求只有一个token
        # 通过两个条件判断 
        # 有没有旧 conv_state
        # 当前有几个 token
        
        # decode阶段
        if (
            conv_state is not None
            and sequence_length == 1
        ):
            new_conv_state = conv_state.to(
                device=mixed_qkv.device,
                dtype=mixed_qkv.dtype,
            ).clone()

            conv_output = (
                # 旧：[10,20,30,40]
                # 新：[20,30,40,50]
                # 1. 丢掉最老的 10
                # 2. 把新值 50 放进状态
                # 3. 使用 [20,30,40,50] 计算卷积
                # 4. 返回当前 token 输出
                # 读取旧 conv_state
                # 加入当前一个新 token
                # 计算当前 token 卷积输出
                # 原地更新 conv_state
                causal_conv1d_update_cuda(
                    x=mixed_qkv,
                    conv_state=new_conv_state,
                    weight=weight,
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )
            )

            return (
                conv_output,
                new_conv_state,
            )

        # 首次Prefill
        if conv_state is None:
            combined_input = mixed_qkv
        else:
            # 把过去的conv_state和现在新的mixed_qkv拼在一起
            combined_input = torch.cat(
                [
                    conv_state.to(
                        device=mixed_qkv.device,
                        dtype=mixed_qkv.dtype,
                    ),
                    mixed_qkv,
                ],
                dim=-1,
            )
        # causal_conv1d_cuda 一次处理多个 token 的因果卷积
        conv_output = causal_conv1d_cuda(
            x=combined_input,
            weight=weight,
            bias=self.conv1d.bias,
            activation=self.activation,
        )

        # 只输出传入chunk长度的结果，负数是因为要倒着切片，靠右侧是新的
        conv_output = conv_output[
            ...,
            -sequence_length:
        ]

        # Prompt 长度小于 kernel_size 怎么保存状态
        # 如果长度不够kernel_size 在左边补0（左边代表过去 右边代表未来 只能补到左边）
        if combined_input.shape[-1] < kernel_size:
            new_conv_state = F.pad(
                combined_input,
                (
                    kernel_size
                    - combined_input.shape[-1],
                    0,
                ),
            )
        else:
            new_conv_state = combined_input[
                ...,
                -kernel_size:
            ]

        return (
            conv_output,
            new_conv_state.clone(),
        )
    
    def _run_variable_length_gated_delta_rule(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor | None,
        prefill_seqlens: tuple[int, ...],
        gdn_cu_seqlens: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        对Packed Variable-length Prefill执行
        Gated Delta Rule。

        query/key:
            [1, total_tokens, heads, key_dim]

        value:
            [1, total_tokens, heads, value_dim]

        recurrent_state:
            [num_sequences, heads, key_dim, value_dim]

        输出：
            output:
                [1, total_tokens, heads, value_dim]

            final_state:
                [num_sequences, heads, key_dim, value_dim]
        """

        if query.ndim != 4:
            raise ValueError(
                "Packed query must have shape "
                "[1, total_tokens, heads, key_dim]"
            )

        packed_batch_size = query.shape[0]
        total_tokens = query.shape[1]
        num_sequences = len(prefill_seqlens)

        if packed_batch_size != 1:
            raise ValueError(
                "Variable-length GDN requires packed "
                "batch size 1"
            )

        if num_sequences <= 1:
            raise ValueError(
                "Variable-length GDN requires more "
                "than one sequence"
            )

        if any(
            seqlen <= 0
            for seqlen in prefill_seqlens
        ):
            raise ValueError(
                "Every Prefill sequence length must "
                "be positive"
            )

        if sum(prefill_seqlens) != total_tokens:
            raise ValueError(
                "The sum of prefill_seqlens must equal "
                "the number of packed GDN tokens"
            )

        expected_state_shape = (
            num_sequences,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
        )

        if (
            recurrent_state is not None
            and tuple(recurrent_state.shape)
            != expected_state_shape
        ):
            raise ValueError(
                "Packed recurrent_state must have "
                f"shape {expected_state_shape}, got "
                f"{tuple(recurrent_state.shape)}"
            )

        # Reference路径逐Sequence运行，
        # 主要用于数值验证。
        if self.backend == "torch":
            outputs = []
            final_states = []

            token_offset = 0

            for sequence_idx, seqlen in enumerate(
                prefill_seqlens
            ):
                token_end = token_offset + seqlen

                sequence_state = (
                    None
                    if recurrent_state is None
                    else recurrent_state[
                        sequence_idx:
                        sequence_idx + 1
                    ]
                )

                (
                    sequence_output,
                    sequence_final_state,
                ) = (
                    torch_recurrent_gated_delta_rule(
                        query=query[
                            :,
                            token_offset:token_end,
                        ],
                        key=key[
                            :,
                            token_offset:token_end,
                        ],
                        value=value[
                            :,
                            token_offset:token_end,
                        ],
                        g=g[
                            :,
                            token_offset:token_end,
                        ],
                        beta=beta[
                            :,
                            token_offset:token_end,
                        ],
                        initial_state=sequence_state,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=True,
                    )
                )

                outputs.append(
                    sequence_output
                )

                final_states.append(
                    sequence_final_state
                )

                token_offset = token_end

            return (
                torch.cat(
                    outputs,
                    dim=1,
                ),
                torch.cat(
                    final_states,
                    dim=0,
                ),
            )

        if not query.is_cuda:
            raise RuntimeError(
                "The FLA backend requires CUDA tensors"
            )

        if gdn_cu_seqlens is None:
            raise ValueError(
                "gdn_cu_seqlens is required for "
                "Variable-length GDN Prefill"
            )

        if not gdn_cu_seqlens.is_cuda:
            raise ValueError(
                "gdn_cu_seqlens must be on CUDA"
            )

        if gdn_cu_seqlens.dtype != torch.long:
            raise TypeError(
                "gdn_cu_seqlens must use torch.long"
            )

        if (
            gdn_cu_seqlens.numel()
            != num_sequences + 1
        ):
            raise ValueError(
                "gdn_cu_seqlens must contain one more "
                "entry than the number of sequences"
            )

        output, final_state = (
            chunk_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                scale=self.head_k_dim ** -0.5,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=gdn_cu_seqlens,
            )
        )

        return (
            output,
            final_state,
        )
    
    # 卷积以后已经得到了QKV g 和 beta 下面需要执行长期状态递推
    def _run_gated_delta_rule(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: torch.Tensor | None,
        prefill_seqlens: (
            tuple[int, ...] | None
        ) = None,
        gdn_cu_seqlens: (
            torch.Tensor | None
        ) = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        is_variable_length_prefill = (
            prefill_seqlens is not None
            and len(prefill_seqlens) > 1
        )

        if is_variable_length_prefill:
            return (
                self
                ._run_variable_length_gated_delta_rule(
                    query=query,
                    key=key,
                    value=value,
                    g=g,
                    beta=beta,
                    recurrent_state=(
                        recurrent_state
                    ),
                    prefill_seqlens=(
                        prefill_seqlens
                    ),
                    gdn_cu_seqlens=(
                        gdn_cu_seqlens
                    ),
                )
            )
        # 同上 如果检测到是torch普通实现 就传回之前的reference版本 否则继续往下走 通过高性能Kernel实现
        if self.backend == "torch":
            output, final_state = (
                torch_recurrent_gated_delta_rule(
                    query=query,
                    key=key,
                    value=value,
                    g=g,
                    beta=beta,
                    initial_state=recurrent_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                )
            )

            return output, final_state

        if not query.is_cuda:
            raise RuntimeError(
                "The FLA backend requires CUDA tensors"
            )

        sequence_length = query.shape[1]

        if sequence_length == 1:
            delta_rule = (
                # 只处理当前一个 token
                # 在一个融合 Kernel 中更新 recurrent_state 并读取输出
                fused_recurrent_gated_delta_rule
            )
        else:
            delta_rule = (
                # 一次处理多个 token 的 Gated Delta Rule
                chunk_gated_delta_rule
            )

        output, final_state = delta_rule(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            scale=self.head_k_dim ** -0.5,
            initial_state=recurrent_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

        return output, final_state
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor | None = None,
        recurrent_state: torch.Tensor | None = None,
        prefill_seqlens: (tuple[int, ...] | None) = None,
        gdn_cu_seqlens: torch.Tensor | None = None
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape "
                "[batch, sequence, hidden_size]"
            )

        (
            batch_size,
            sequence_length,
            hidden_size,
        ) = hidden_states.shape

        if hidden_size != self.hidden_size:
            raise ValueError(
                f"Expected hidden_size {self.hidden_size}, "
                f"got {hidden_size}"
            )

        mixed_qkv = self.in_proj_qkv(
            hidden_states
        )

        mixed_qkv = (
            mixed_qkv
            .transpose(1, 2)
            .contiguous()
        )

        (
            mixed_qkv,
            new_conv_state,
        ) = self._run_causal_conv1d(
            mixed_qkv=mixed_qkv,
            conv_state=conv_state,
            prefill_seqlens=prefill_seqlens,
        )

        mixed_qkv = (
            mixed_qkv
            .transpose(1, 2)
            .contiguous()
        )

        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )

        query = query.reshape(
            batch_size,
            sequence_length,
            self.num_k_heads,
            self.head_k_dim,
        )

        key = key.reshape(
            batch_size,
            sequence_length,
            self.num_k_heads,
            self.head_k_dim,
        )

        value = value.reshape(
            batch_size,
            sequence_length,
            self.num_v_heads,
            self.head_v_dim,
        )

        z = self.in_proj_z(
            hidden_states
        ).reshape(
            batch_size,
            sequence_length,
            self.num_v_heads,
            self.head_v_dim,
        )

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        beta = torch.sigmoid(b)

        # g = -exp(A_log) * softplus(a + dt_bias) 一定是非正的，用来计算最终遗忘率exp(g)
        g = (
            -self.A_log.float().exp()
            * F.softplus(
                a.float()
                + self.dt_bias.float()
            )
        )

        head_repeat = (
            self.num_v_heads
            // self.num_k_heads
        )

        if head_repeat > 1:
            query = query.repeat_interleave(
                head_repeat,
                dim=2,
            )

            key = key.repeat_interleave(
                head_repeat,
                dim=2,
            )

        (
            core_output,
            new_recurrent_state,
        ) = self._run_gated_delta_rule(
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            recurrent_state=recurrent_state,
            prefill_seqlens=prefill_seqlens,
            gdn_cu_seqlens=gdn_cu_seqlens,
        )
        
        core_output = core_output.reshape(
            -1,
            self.head_v_dim,
        )

        z = z.reshape(
            -1,
            self.head_v_dim,
        )

        core_output = self.norm(
            core_output,
            z,
        )

        core_output = core_output.reshape(
            batch_size,
            sequence_length,
            self.value_dim,
        )

        output = self.out_proj(
            core_output
        )

        return (
            output,
            new_conv_state,
            new_recurrent_state,
        )