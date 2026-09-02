import torch

from causal_conv1d import (
    causal_conv1d_update,
)


def run_eager_sequence(
    token_stream: torch.Tensor,
    initial_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[
    list[torch.Tensor],
    torch.Tensor,
]:
    """
    使用普通 Eager 模式逐 token 更新状态。

    token_stream:
        [num_steps, B, C, 1]

    initial_state:
        [B, C, K]

    返回：
        每轮输出
        最终 conv_state
    """

    eager_state = initial_state.clone()

    outputs = []

    for token in token_stream:
        output = causal_conv1d_update(
            x=token,
            conv_state=eager_state,
            weight=weight,
            bias=bias,
            activation="silu",
        )

        outputs.append(
            output.clone()
        )

    return outputs, eager_state


def build_manual_final_state(
    token_stream: torch.Tensor,
    initial_state: torch.Tensor,
) -> torch.Tensor:
    """
    手工模拟卷积窗口更新。

    假设 K=4：

        旧状态：[x0, x1, x2, x3]
        新 token：x4

        新状态：[x1, x2, x3, x4]

    这里只检查状态移动和写入，
    不手工复现卷积输出。
    """

    manual_state = initial_state.clone()

    for token in token_stream:
        # 删除最老的位置，并把其他历史向左移动。
        manual_state[..., :-1].copy_(
            manual_state[..., 1:].clone()
        )

        # 当前 token 写入窗口最右边。
        manual_state[..., -1].copy_(
            token.squeeze(-1)
        )

    return manual_state


def test_batch_size(
    batch_size: int,
) -> None:
    """
    为一个固定 Batch Size 捕获独立 CUDA Graph。
    """

    device = torch.device("cuda")
    dtype = torch.bfloat16

    num_channels = 256
    kernel_size = 4
    num_steps = 16

    print(
        "\n"
        + "=" * 72
    )

    print(
        f"Testing batch_size={batch_size}"
    )

    print(
        "=" * 72
    )

    # ==========================================
    # 1. 创建卷积参数
    # ==========================================

    weight = torch.randn(
        num_channels,
        kernel_size,
        device=device,
        dtype=dtype,
    )

    bias = torch.randn(
        num_channels,
        device=device,
        dtype=dtype,
    )

    # 使用非零初始状态。
    #
    # 这样可以证明 Kernel 确实读取了旧历史，
    # 而不是每次从全零状态开始。
    initial_state = torch.randn(
        batch_size,
        num_channels,
        kernel_size,
        device=device,
        dtype=dtype,
    )

    # 模拟连续 16 个 Decode token。
    token_stream = torch.randn(
        num_steps,
        batch_size,
        num_channels,
        1,
        device=device,
        dtype=dtype,
    )

    # ==========================================
    # 2. 计算 Eager 基准
    # ==========================================

    (
        eager_outputs,
        eager_final_state,
    ) = run_eager_sequence(
        token_stream=token_stream,
        initial_state=initial_state,
        weight=weight,
        bias=bias,
    )

    # ==========================================
    # 3. 手工检查最终窗口
    # ==========================================

    manual_final_state = (
        build_manual_final_state(
            token_stream=token_stream,
            initial_state=initial_state,
        )
    )

    torch.testing.assert_close(
        eager_final_state.float(),
        manual_final_state.float(),
        rtol=0,
        atol=0,
    )

    print(
        "Eager state update matches "
        "the manual sliding window."
    )

    # ==========================================
    # 4. 创建 Graph 使用的静态 Tensor
    # ==========================================

    # 每轮都把当前 token copy_ 到这个地址。
    static_x = torch.zeros(
        batch_size,
        num_channels,
        1,
        device=device,
        dtype=dtype,
    )

    # Graph 内被原地更新的状态。
    graph_state = initial_state.clone()

    input_pointer = static_x.data_ptr()
    state_pointer = graph_state.data_ptr()

    print(
        "Static input pointer:",
        input_pointer,
    )

    print(
        "Static state pointer:",
        state_pointer,
    )

    # ==========================================
    # 5. 使用独立状态执行 Warmup
    # ==========================================

    # causal_conv1d_update 会修改 conv_state，
    # 所以 Warmup 不能使用正式 graph_state。
    warmup_x = torch.zeros_like(
        static_x
    )

    warmup_state = torch.zeros_like(
        graph_state
    )

    warmup_stream = torch.cuda.Stream()

    warmup_stream.wait_stream(
        torch.cuda.current_stream()
    )

    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            causal_conv1d_update(
                x=warmup_x,
                conv_state=warmup_state,
                weight=weight,
                bias=bias,
                activation="silu",
            )

    torch.cuda.current_stream().wait_stream(
        warmup_stream
    )

    torch.cuda.synchronize()

    # ==========================================
    # 6. Capture
    # ==========================================

    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        static_output = (
            causal_conv1d_update(
                x=static_x,
                conv_state=graph_state,
                weight=weight,
                bias=bias,
                activation="silu",
            )
        )

    torch.cuda.synchronize()

    output_pointer = static_output.data_ptr()

    print(
        "Capture succeeded."
    )

    print(
        "Static output pointer:",
        output_pointer,
    )

    # 重要：
    #
    # CUDA Graph Capture 不是“只录制、不执行”。
    # Capture 时 causal_conv1d_update 已经真实地
    # 修改了一次 graph_state。
    #
    # 所以正式测试前必须恢复初始状态。
    graph_state.copy_(
        initial_state
    )

    static_x.zero_()

    torch.cuda.synchronize()

    # ==========================================
    # 7. 连续 Replay
    # ==========================================

    graph_outputs = []

    for step, token in enumerate(
        token_stream
    ):
        # 更新固定输入地址中的内容。
        static_x.copy_(token)

        # 确认地址没有变化。
        if static_x.data_ptr() != input_pointer:
            raise AssertionError(
                "static_x address changed"
            )

        if graph_state.data_ptr() != state_pointer:
            raise AssertionError(
                "graph_state address changed"
            )

        # Replay 应当：
        #
        # 1. 读取新的 static_x；
        # 2. 读取上一轮更新后的 graph_state；
        # 3. 生成当前输出；
        # 4. 原地更新 graph_state。
        graph.replay()

        current_output = (
            static_output.clone()
        )

        graph_outputs.append(
            current_output
        )

        torch.cuda.synchronize()

        if (
            static_output.data_ptr()
            != output_pointer
        ):
            raise AssertionError(
                "static_output address changed"
            )

        output_error = (
            current_output.float()
            - eager_outputs[step].float()
        ).abs()

        print(
            f"Step {step:02d}: "
            f"output max error="
            f"{output_error.max().item():.8f}, "
            f"mean error="
            f"{output_error.mean().item():.8f}"
        )

        torch.testing.assert_close(
            current_output.float(),
            eager_outputs[step].float(),
            rtol=1e-2,
            atol=1e-2,
        )

    # ==========================================
    # 8. 比较最终状态
    # ==========================================

    torch.cuda.synchronize()

    state_error = (
        graph_state.float()
        - eager_final_state.float()
    ).abs()

    print(
        "\nFinal state max error:",
        state_error.max().item(),
    )

    print(
        "Final state mean error:",
        state_error.mean().item(),
    )

    torch.testing.assert_close(
        graph_state.float(),
        eager_final_state.float(),
        rtol=0,
        atol=0,
    )

    torch.testing.assert_close(
        graph_state.float(),
        manual_final_state.float(),
        rtol=0,
        atol=0,
    )

    # 确认输出不是每轮都停留在 Capture 时的结果。
    unique_outputs = 1

    for index in range(
        1,
        len(graph_outputs),
    ):
        if not torch.equal(
            graph_outputs[index],
            graph_outputs[index - 1],
        ):
            unique_outputs += 1

    if unique_outputs <= 1:
        raise AssertionError(
            "Graph output did not change when "
            "static_x changed"
        )

    print(
        "Different input tokens produced "
        f"{unique_outputs} changing outputs."
    )

    print(
        f"batch_size={batch_size} passed."
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available"
        )

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    print(
        "PyTorch:",
        torch.__version__,
    )

    print(
        "CUDA:",
        torch.version.cuda,
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "Compute capability:",
        torch.cuda.get_device_capability(0),
    )

    # 先验证最重要的 bucket=1，
    # 再验证一个小型 Batched Decode。
    for batch_size in (1, 4):
        test_batch_size(
            batch_size
        )

    print(
        "\nPart 1B passed: "
        "causal_conv1d_update supports "
        "stateful CUDA Graph replay."
    )


if __name__ == "__main__":
    main()