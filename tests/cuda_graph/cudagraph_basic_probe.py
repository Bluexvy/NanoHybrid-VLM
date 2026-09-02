import torch
import torch.nn.functional as F


def workload(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    模拟一小段 Decode GPU 计算：

        Linear
          ↓
        Bias Add
          ↓
        SiLU

    这里只使用固定形状的 GPU Tensor，
    不包含 CPU/GPU 同步和数据相关 Python 分支。
    """
    return F.silu(
        torch.matmul(x, weight) + bias
    )


def main() -> None:
    # ==========================================
    # 1. 检查 CUDA 环境
    # ==========================================

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available"
        )

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("PyTorch version:", torch.__version__)
    print("PyTorch CUDA:", torch.version.cuda)
    print(
        "GPU:",
        torch.cuda.get_device_name(device),
    )
    print(
        "Compute capability:",
        torch.cuda.get_device_capability(device),
    )

    # 固定随机数，方便重复测试。
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # ==========================================
    # 2. 创建固定地址的输入和模型参数
    # ==========================================

    batch_size = 4
    hidden_size = 256

    # static_input 是 Graph 永久使用的输入地址。
    #
    # 后续不能用：
    #     static_input = another_tensor
    #
    # 而应该用：
    #     static_input.copy_(another_tensor)
    static_input = torch.randn(
        batch_size,
        hidden_size,
        device=device,
        dtype=dtype,
    )

    weight = torch.randn(
        hidden_size,
        hidden_size,
        device=device,
        dtype=dtype,
    )

    bias = torch.randn(
        hidden_size,
        device=device,
        dtype=dtype,
    )

    input_pointer = static_input.data_ptr()
    weight_pointer = weight.data_ptr()
    bias_pointer = bias.data_ptr()

    print("\nStatic Tensor pointers:")
    print("input pointer:", input_pointer)
    print("weight pointer:", weight_pointer)
    print("bias pointer:", bias_pointer)

    # ==========================================
    # 3. 在非默认 Stream 上 Warmup
    # ==========================================

    warmup_stream = torch.cuda.Stream()

    # 让 warmup_stream 等待当前 Stream 中此前提交的工作。
    warmup_stream.wait_stream(
        torch.cuda.current_stream()
    )

    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            unused_output = workload(
                static_input,
                weight,
                bias,
            )

    # 当前 Stream 等待 Warmup 完成。
    torch.cuda.current_stream().wait_stream(
        warmup_stream
    )

    torch.cuda.synchronize()

    # 防止变量被误认为完全没有使用。
    del unused_output

    # ==========================================
    # 4. 记录 Capture 前显存
    # ==========================================

    memory_before_allocated = (
        torch.cuda.memory_allocated()
    )

    memory_before_reserved = (
        torch.cuda.memory_reserved()
    )

    # ==========================================
    # 5. Capture
    # ==========================================

    graph = torch.cuda.CUDAGraph()

    # 这里的 Python 代码只执行一次。
    #
    # GPU 操作会被 CUDA Graph 记录。
    with torch.cuda.graph(graph):
        static_output = workload(
            static_input,
            weight,
            bias,
        )

    torch.cuda.synchronize()

    memory_after_allocated = (
        torch.cuda.memory_allocated()
    )

    memory_after_reserved = (
        torch.cuda.memory_reserved()
    )

    output_pointer = static_output.data_ptr()

    print("\nCapture succeeded.")
    print("output pointer:", output_pointer)

    print(
        "Capture allocated delta MiB:",
        (
            memory_after_allocated
            - memory_before_allocated
        )
        / 1024**2,
    )

    print(
        "Capture reserved delta MiB:",
        (
            memory_after_reserved
            - memory_before_reserved
        )
        / 1024**2,
    )

    # ==========================================
    # 6. 使用不同输入多次 Replay
    # ==========================================

    test_inputs = [
        torch.zeros_like(static_input),
        torch.ones_like(static_input),
        torch.randn_like(static_input),
        torch.randn_like(static_input) * 2,
    ]

    previous_result = None

    for replay_index, real_input in enumerate(
        test_inputs
    ):
        # --------------------------------------
        # Eager 基准
        # --------------------------------------

        eager_output = workload(
            real_input,
            weight,
            bias,
        )

        # --------------------------------------
        # 更新静态 Graph 输入
        # --------------------------------------

        # static_input 的地址保持不变，
        # 只修改地址中的数据。
        static_input.copy_(real_input)

        if static_input.data_ptr() != input_pointer:
            raise AssertionError(
                "Static input address changed"
            )

        # --------------------------------------
        # Replay
        # --------------------------------------

        graph.replay()

        # static_output 本身属于 Graph，
        # clone 一份用于本轮比较。
        graph_output = static_output.clone()

        torch.cuda.synchronize()

        # BF16 转成 FP32 后计算误差，
        # 方便观察数值。
        error = (
            graph_output.float()
            - eager_output.float()
        ).abs()

        max_error = error.max().item()
        mean_error = error.mean().item()

        torch.testing.assert_close(
            graph_output.float(),
            eager_output.float(),
            rtol=1e-2,
            atol=1e-2,
        )

        if (
            static_output.data_ptr()
            != output_pointer
        ):
            raise AssertionError(
                "Static output address changed"
            )

        # 不同输入应该产生不同结果。
        if previous_result is not None:
            same_as_previous = torch.equal(
                graph_output,
                previous_result,
            )
        else:
            same_as_previous = False

        print(
            f"\nReplay {replay_index}:"
        )
        print(
            "  input pointer:",
            static_input.data_ptr(),
        )
        print(
            "  output pointer:",
            static_output.data_ptr(),
        )
        print(
            "  max error:",
            max_error,
        )
        print(
            "  mean error:",
            mean_error,
        )
        print(
            "  same as previous:",
            same_as_previous,
        )

        previous_result = graph_output.clone()

    # ==========================================
    # 7. 连续 Replay 稳定性
    # ==========================================

    stable_input = torch.randn_like(
        static_input
    )

    eager_reference = workload(
        stable_input,
        weight,
        bias,
    )

    static_input.copy_(stable_input)

    for _ in range(20):
        graph.replay()

    stable_graph_output = static_output.clone()

    torch.cuda.synchronize()

    torch.testing.assert_close(
        stable_graph_output.float(),
        eager_reference.float(),
        rtol=1e-2,
        atol=1e-2,
    )

    print(
        "\n20 repeated replays passed."
    )

    print(
        "\nPart 1A passed: "
        "basic CUDA Graph capture/replay is available."
    )


if __name__ == "__main__":
    main()