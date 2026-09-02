from __future__ import annotations

import argparse
import subprocess
import sys
import traceback

import torch


DTYPE = torch.bfloat16
DEVICE = torch.device("cuda")


def check_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> None:
    error = (
        actual.float()
        - expected.float()
    ).abs()

    max_error = error.max().item()
    mean_error = error.mean().item()

    print(
        f"{name}: "
        f"max_error={max_error:.8f}, "
        f"mean_error={mean_error:.8f}"
    )

    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        rtol=rtol,
        atol=atol,
    )


# ============================================================
# Case 1：FLA fused recurrent GDN
# ============================================================

def test_fla_recurrent() -> None:
    from fla.ops.gated_delta_rule import (
        fused_recurrent_gated_delta_rule,
    )

    print("Testing FLA fused recurrent GDN...")

    # 使用 Qwen3.5-9B 的实际 GDN 维度。
    batch_size = 1
    num_heads = 32
    key_dim = 128
    value_dim = 128
    num_steps = 8

    scale = key_dim ** -0.5

    initial_state = torch.randn(
        batch_size,
        num_heads,
        key_dim,
        value_dim,
        device=DEVICE,
        dtype=torch.float32,
    )

    q_stream = torch.randn(
        num_steps,
        batch_size,
        1,
        num_heads,
        key_dim,
        device=DEVICE,
        dtype=DTYPE,
    )

    k_stream = torch.randn_like(q_stream)

    v_stream = torch.randn(
        num_steps,
        batch_size,
        1,
        num_heads,
        value_dim,
        device=DEVICE,
        dtype=DTYPE,
    )

    # g 在当前模型实现中使用 FP32。
    g_stream = -torch.rand(
        num_steps,
        batch_size,
        1,
        num_heads,
        device=DEVICE,
        dtype=torch.float32,
    )

    beta_stream = torch.rand(
        num_steps,
        batch_size,
        1,
        num_heads,
        device=DEVICE,
        dtype=DTYPE,
    )

    # --------------------------------------------------------
    # Eager 基准
    # --------------------------------------------------------

    eager_state = initial_state.clone()
    eager_outputs = []
    eager_states = []

    for step in range(num_steps):
        (
            eager_output,
            eager_state,
        ) = fused_recurrent_gated_delta_rule(
            q=q_stream[step],
            k=k_stream[step],
            v=v_stream[step],
            g=g_stream[step],
            beta=beta_stream[step],
            scale=scale,
            initial_state=eager_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

        eager_outputs.append(
            eager_output.clone()
        )

        eager_states.append(
            eager_state.clone()
        )

    # --------------------------------------------------------
    # Warmup：使用独立状态，避免污染 Graph 状态
    # --------------------------------------------------------

    warmup_state = initial_state.clone()
    warmup_stream = torch.cuda.Stream()

    warmup_stream.wait_stream(
        torch.cuda.current_stream()
    )

    with torch.cuda.stream(warmup_stream):
        for step in range(3):
            (
                unused_output,
                warmup_state,
            ) = fused_recurrent_gated_delta_rule(
                q=q_stream[step],
                k=k_stream[step],
                v=v_stream[step],
                g=g_stream[step],
                beta=beta_stream[step],
                scale=scale,
                initial_state=warmup_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )

    torch.cuda.current_stream().wait_stream(
        warmup_stream
    )
    torch.cuda.synchronize()

    del unused_output
    del warmup_state

    # --------------------------------------------------------
    # 静态 Graph 输入
    # --------------------------------------------------------

    static_q = torch.zeros_like(q_stream[0])
    static_k = torch.zeros_like(k_stream[0])
    static_v = torch.zeros_like(v_stream[0])
    static_g = torch.zeros_like(g_stream[0])
    static_beta = torch.zeros_like(beta_stream[0])

    # 首版 Runtime 采用：
    #
    # graph_input_state
    #     ↓ Graph
    # graph_final_state
    #
    # 然后 Graph 外把 final state 写回状态池。
    static_initial_state = (
        initial_state.clone()
    )

    input_state_pointer = (
        static_initial_state.data_ptr()
    )

    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        (
            static_output,
            static_final_state,
        ) = fused_recurrent_gated_delta_rule(
            q=static_q,
            k=static_k,
            v=static_v,
            g=static_g,
            beta=static_beta,
            scale=scale,
            initial_state=static_initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

    torch.cuda.synchronize()

    print("FLA capture succeeded.")
    print(
        "input state pointer:",
        input_state_pointer,
    )
    print(
        "final state pointer:",
        static_final_state.data_ptr(),
    )

    # Capture 会真实执行，所以恢复初始状态。
    static_initial_state.copy_(
        initial_state
    )

    torch.cuda.synchronize()

    # --------------------------------------------------------
    # 多轮 Replay
    # --------------------------------------------------------

    for step in range(num_steps):
        static_q.copy_(q_stream[step])
        static_k.copy_(k_stream[step])
        static_v.copy_(v_stream[step])
        static_g.copy_(g_stream[step])
        static_beta.copy_(beta_stream[step])

        graph.replay()

        graph_output = static_output.clone()
        graph_final_state = (
            static_final_state.clone()
        )

        # 下一轮必须使用本轮产生的最终状态。
        static_initial_state.copy_(
            static_final_state
        )

        torch.cuda.synchronize()

        if (
            static_initial_state.data_ptr()
            != input_state_pointer
        ):
            raise AssertionError(
                "FLA static state address changed"
            )

        check_close(
            f"FLA step {step} output",
            graph_output,
            eager_outputs[step],
            rtol=2e-2,
            atol=2e-2,
        )

        check_close(
            f"FLA step {step} state",
            graph_final_state,
            eager_states[step],
            rtol=1e-4,
            atol=1e-4,
        )

    print("FLA recurrent Graph test passed.")


# ============================================================
# Case 2：Paged Attention + Triton KV Store
# ============================================================

def test_paged_attention() -> None:
    from flash_attn import (
        flash_attn_with_kvcache,
    )

    from nanovllm.layers.attention import (
        store_kvcache,
    )

    print(
        "Testing Paged Attention and "
        "Triton KV store..."
    )

    # Qwen3.5-9B Full Attention 实际维度。
    batch_size = 2
    num_query_heads = 16
    num_kv_heads = 4
    head_dim = 256

    block_size = 256
    num_blocks = batch_size
    num_steps = 4

    block_tables = torch.arange(
        batch_size,
        device=DEVICE,
        dtype=torch.int32,
    ).view(batch_size, 1)

    q_stream = torch.randn(
        num_steps,
        batch_size,
        num_query_heads,
        head_dim,
        device=DEVICE,
        dtype=DTYPE,
    )

    k_stream = torch.randn(
        num_steps,
        batch_size,
        num_kv_heads,
        head_dim,
        device=DEVICE,
        dtype=DTYPE,
    )

    v_stream = torch.randn_like(k_stream)

    initial_k_cache = torch.zeros(
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        device=DEVICE,
        dtype=DTYPE,
    )

    initial_v_cache = torch.zeros_like(
        initial_k_cache
    )

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    warmup_k_cache = initial_k_cache.clone()
    warmup_v_cache = initial_v_cache.clone()

    warmup_slots = (
        torch.arange(
            batch_size,
            device=DEVICE,
            dtype=torch.int32,
        )
        * block_size
    )

    warmup_context_lens = torch.ones(
        batch_size,
        device=DEVICE,
        dtype=torch.int32,
    )

    warmup_stream = torch.cuda.Stream()

    warmup_stream.wait_stream(
        torch.cuda.current_stream()
    )

    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            store_kvcache(
                k_stream[0],
                v_stream[0],
                warmup_k_cache,
                warmup_v_cache,
                warmup_slots,
            )

            unused_output = (
                flash_attn_with_kvcache(
                    q_stream[0].unsqueeze(1),
                    warmup_k_cache,
                    warmup_v_cache,
                    cache_seqlens=(
                        warmup_context_lens
                    ),
                    block_table=block_tables,
                    causal=True,
                )
            )

    torch.cuda.current_stream().wait_stream(
        warmup_stream
    )
    torch.cuda.synchronize()

    del unused_output
    del warmup_k_cache
    del warmup_v_cache

    # --------------------------------------------------------
    # 静态 Graph Tensor
    # --------------------------------------------------------

    static_q = torch.zeros_like(q_stream[0])
    static_k = torch.zeros_like(k_stream[0])
    static_v = torch.zeros_like(v_stream[0])

    static_slot_mapping = (
        warmup_slots.clone()
    )

    static_context_lens = (
        warmup_context_lens.clone()
    )

    static_block_tables = (
        block_tables.clone()
    )

    graph_k_cache = (
        initial_k_cache.clone()
    )

    graph_v_cache = (
        initial_v_cache.clone()
    )

    k_cache_pointer = graph_k_cache.data_ptr()
    v_cache_pointer = graph_v_cache.data_ptr()

    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        store_kvcache(
            static_k,
            static_v,
            graph_k_cache,
            graph_v_cache,
            static_slot_mapping,
        )

        static_output = (
            flash_attn_with_kvcache(
                static_q.unsqueeze(1),
                graph_k_cache,
                graph_v_cache,
                cache_seqlens=(
                    static_context_lens
                ),
                block_table=(
                    static_block_tables
                ),
                causal=True,
            )
        )

    torch.cuda.synchronize()

    print(
        "Paged Attention capture succeeded."
    )

    # Capture 已向 Cache 写过一次，恢复初始值。
    graph_k_cache.copy_(initial_k_cache)
    graph_v_cache.copy_(initial_v_cache)

    eager_k_cache = initial_k_cache.clone()
    eager_v_cache = initial_v_cache.clone()

    torch.cuda.synchronize()

    # --------------------------------------------------------
    # 多轮 Decode
    # --------------------------------------------------------

    block_ids = torch.arange(
        batch_size,
        device=DEVICE,
        dtype=torch.int32,
    )

    for step in range(num_steps):
        context_lens = torch.full(
            (batch_size,),
            step + 1,
            device=DEVICE,
            dtype=torch.int32,
        )

        slot_mapping = (
            block_ids * block_size + step
        )

        # Eager：先写入当前 K/V，再读完整历史。
        store_kvcache(
            k_stream[step],
            v_stream[step],
            eager_k_cache,
            eager_v_cache,
            slot_mapping,
        )

        eager_output = (
            flash_attn_with_kvcache(
                q_stream[step].unsqueeze(1),
                eager_k_cache,
                eager_v_cache,
                cache_seqlens=context_lens,
                block_table=block_tables,
                causal=True,
            )
        )

        static_q.copy_(q_stream[step])
        static_k.copy_(k_stream[step])
        static_v.copy_(v_stream[step])

        static_slot_mapping.copy_(
            slot_mapping
        )

        static_context_lens.copy_(
            context_lens
        )

        graph.replay()

        graph_output = static_output.clone()

        torch.cuda.synchronize()

        if graph_k_cache.data_ptr() != k_cache_pointer:
            raise AssertionError(
                "Graph K Cache address changed"
            )

        if graph_v_cache.data_ptr() != v_cache_pointer:
            raise AssertionError(
                "Graph V Cache address changed"
            )

        check_close(
            f"Attention step {step} output",
            graph_output,
            eager_output,
            rtol=2e-2,
            atol=2e-2,
        )

    check_close(
        "Final K Cache",
        graph_k_cache,
        eager_k_cache,
        rtol=0,
        atol=0,
    )

    check_close(
        "Final V Cache",
        graph_v_cache,
        eager_v_cache,
        rtol=0,
        atol=0,
    )

    print(
        "Paged Attention Graph test passed."
    )


# ============================================================
# Case 3：Linear/RMSNorm/SwiGLU/mRoPE
# ============================================================

def test_core_layers() -> None:
    from nanovllm.layers.activation import (
        SiluAndMul,
    )

    from nanovllm.layers.layernorm import (
        Qwen3_5RMSNorm,
    )

    from nanovllm.layers.rotary_embedding import (
        RotaryEmbedding,
    )

    print(
        "Testing Linear, RMSNorm, SwiGLU "
        "and mRoPE..."
    )

    batch_size = 4
    hidden_size = 128

    head_dim = 256
    rotary_dim = 64

    num_query_heads = 16
    num_kv_heads = 4

    linear_weight = torch.randn(
        hidden_size,
        hidden_size,
        device=DEVICE,
        dtype=DTYPE,
    )

    norm = Qwen3_5RMSNorm(
        hidden_size
    ).to(
        device=DEVICE,
        dtype=DTYPE,
    )

    with torch.no_grad():
        norm.weight.uniform_(
            -0.1,
            0.1,
        )

    activation = SiluAndMul().to(
        device=DEVICE
    )

    rotary = RotaryEmbedding(
        head_size=head_dim,
        rotary_dim=rotary_dim,
        max_position_embeddings=4096,
        base=10_000_000,
        mrope_section=(11, 11, 10),
    ).to(
        device=DEVICE
    )

    static_x = torch.zeros(
        batch_size,
        hidden_size,
        device=DEVICE,
        dtype=DTYPE,
    )

    static_gate_up = torch.zeros(
        batch_size,
        hidden_size * 2,
        device=DEVICE,
        dtype=DTYPE,
    )

    static_query = torch.zeros(
        batch_size,
        num_query_heads,
        head_dim,
        device=DEVICE,
        dtype=DTYPE,
    )

    static_key = torch.zeros(
        batch_size,
        num_kv_heads,
        head_dim,
        device=DEVICE,
        dtype=DTYPE,
    )

    static_positions = torch.zeros(
        3,
        batch_size,
        device=DEVICE,
        dtype=torch.long,
    )

    def workload():
        linear_output = torch.matmul(
            static_x,
            linear_weight,
        )

        norm_output = norm(
            linear_output
        )

        activation_output = activation(
            static_gate_up
        )

        query_output, key_output = rotary(
            static_positions,
            static_query,
            static_key,
        )

        return (
            norm_output,
            activation_output,
            query_output,
            key_output,
        )

    # torch.compile 修饰的激活层必须先 Warmup。
    for _ in range(3):
        unused_outputs = workload()

    torch.cuda.synchronize()
    del unused_outputs

    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        (
            static_norm_output,
            static_activation_output,
            static_query_output,
            static_key_output,
        ) = workload()

    torch.cuda.synchronize()

    print("Core layer capture succeeded.")

    for replay_index in range(4):
        real_x = torch.randn_like(static_x)
        real_gate_up = torch.randn_like(
            static_gate_up
        )
        real_query = torch.randn_like(
            static_query
        )
        real_key = torch.randn_like(
            static_key
        )

        base_position = replay_index * 7

        real_positions = torch.tensor(
            [
                [
                    base_position + i
                    for i in range(batch_size)
                ],
                [
                    base_position + 2 * i
                    for i in range(batch_size)
                ],
                [
                    base_position + 3 * i
                    for i in range(batch_size)
                ],
            ],
            device=DEVICE,
            dtype=torch.long,
        )

        eager_linear = torch.matmul(
            real_x,
            linear_weight,
        )

        eager_norm = norm(eager_linear)

        eager_activation = activation(
            real_gate_up
        )

        (
            eager_query,
            eager_key,
        ) = rotary(
            real_positions,
            real_query,
            real_key,
        )

        static_x.copy_(real_x)
        static_gate_up.copy_(real_gate_up)
        static_query.copy_(real_query)
        static_key.copy_(real_key)
        static_positions.copy_(real_positions)

        graph.replay()

        graph_norm = (
            static_norm_output.clone()
        )
        graph_activation = (
            static_activation_output.clone()
        )
        graph_query = (
            static_query_output.clone()
        )
        graph_key = (
            static_key_output.clone()
        )

        torch.cuda.synchronize()

        check_close(
            f"Core {replay_index} RMSNorm",
            graph_norm,
            eager_norm,
        )

        check_close(
            f"Core {replay_index} SwiGLU",
            graph_activation,
            eager_activation,
        )

        check_close(
            f"Core {replay_index} Query mRoPE",
            graph_query,
            eager_query,
        )

        check_close(
            f"Core {replay_index} Key mRoPE",
            graph_key,
            eager_key,
        )

    print("Core layer Graph test passed.")


# ============================================================
# Case 4：动态 state_slot_ids 的 Gather/Scatter
# ============================================================

def test_state_gather_scatter() -> None:
    print(
        "Testing state Gather/Scatter..."
    )

    num_slots = 8
    bucket_size = 2

    state_pool = torch.randn(
        num_slots,
        4,
        3,
        device=DEVICE,
        dtype=torch.float32,
    )

    initial_pool = state_pool.clone()

    static_slot_ids = torch.tensor(
        [0, 1],
        device=DEVICE,
        dtype=torch.long,
    )

    slot_pointer = static_slot_ids.data_ptr()
    pool_pointer = state_pool.data_ptr()

    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        # Gather：
        # 根据静态地址中的动态 slot ID
        # 读取对应状态。
        static_gathered = torch.index_select(
            state_pool,
            dim=0,
            index=static_slot_ids,
        )

        # 模拟模型对状态的更新。
        static_updated = (
            static_gathered + 1.0
        )

        # Scatter：
        # 把更新后的 Batch 状态写回原状态池。
        state_pool.index_copy_(
            0,
            static_slot_ids,
            static_updated,
        )

    torch.cuda.synchronize()

    print(
        "Gather/Scatter capture succeeded."
    )

    # Capture 已执行一次，恢复状态池。
    state_pool.copy_(initial_pool)

    torch.cuda.synchronize()

    expected_pool = initial_pool.clone()

    slot_sequences = [
        [1, 3],
        [2, 6],
        [1, 6],
        [0, 7],
    ]

    for replay_index, slot_values in enumerate(
        slot_sequences
    ):
        new_slot_ids = torch.tensor(
            slot_values,
            device=DEVICE,
            dtype=torch.long,
        )

        # 构造 Eager 期望结果。
        expected_gathered = (
            torch.index_select(
                expected_pool,
                dim=0,
                index=new_slot_ids,
            )
        )

        expected_updated = (
            expected_gathered + 1.0
        )

        expected_pool.index_copy_(
            0,
            new_slot_ids,
            expected_updated,
        )

        # 修改固定地址里的 slot ID。
        static_slot_ids.copy_(
            new_slot_ids
        )

        graph.replay()

        torch.cuda.synchronize()

        if (
            static_slot_ids.data_ptr()
            != slot_pointer
        ):
            raise AssertionError(
                "state_slot_ids address changed"
            )

        if state_pool.data_ptr() != pool_pointer:
            raise AssertionError(
                "state pool address changed"
            )

        check_close(
            f"Gather/Scatter replay {replay_index}",
            state_pool,
            expected_pool,
            rtol=0,
            atol=0,
        )

    print(
        "State Gather/Scatter Graph test passed."
    )


# ============================================================
# 独立子进程 Runner
# ============================================================

CASES = {
    "fla_recurrent": test_fla_recurrent,
    "paged_attention": test_paged_attention,
    "core_layers": test_core_layers,
    "state_gather_scatter": (
        test_state_gather_scatter
    ),
}


def run_one_case(
    case_name: str,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available"
        )

    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )
    print(
        "Compute capability:",
        torch.cuda.get_device_capability(0),
    )

    CASES[case_name]()

    print(
        f"\nCASE PASSED: {case_name}"
    )


def run_all_cases() -> None:
    failed_cases = []

    for case_name in CASES:
        print(
            "\n"
            + "#" * 80
        )
        print(
            "Running CUDA Graph case:",
            case_name,
        )
        print(
            "#" * 80
        )

        result = subprocess.run(
            [
                sys.executable,
                __file__,
                "--case",
                case_name,
            ],
            check=False,
        )

        if result.returncode != 0:
            failed_cases.append(
                case_name
            )

    print(
        "\n"
        + "=" * 80
    )

    if failed_cases:
        print(
            "FAILED CASES:",
            failed_cases,
        )

        raise SystemExit(1)

    print(
        "ALL REMAINING CUDA GRAPH "
        "COMPATIBILITY TESTS PASSED"
    )

    print(
        "=" * 80
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--case",
        choices=tuple(CASES),
        default=None,
    )

    args = parser.parse_args()

    if args.case is None:
        run_all_cases()
        return

    try:
        run_one_case(args.case)

    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()