import torch

from nanovllm.engine.hybrid_cuda_graph import (
    HybridDecodeStaticWorkspace,
)

from nanovllm.engine.hybrid_state import (
    GDNLayerState,
    HybridCacheSpec,
)


def make_states(
    spec: HybridCacheSpec,
    batch_size: int,
) -> list[GDNLayerState | None]:
    states = [
        None
        for _ in range(
            spec.num_hidden_layers
        )
    ]

    for global_layer_idx in (
        spec.gdn_layer_indices
    ):
        states[global_layer_idx] = (
            GDNLayerState(
                conv_state=torch.randn(
                    batch_size,
                    spec.conv_dim,
                    spec.conv_kernel_size,
                    device="cuda",
                    dtype=spec.conv_dtype,
                ),
                recurrent_state=torch.randn(
                    batch_size,
                    spec.num_gdn_value_heads,
                    spec.gdn_key_head_dim,
                    spec.gdn_value_head_dim,
                    device="cuda",
                    dtype=(
                        spec.recurrent_dtype
                    ),
                ),
            )
        )

    return states


def assert_state_views_equal(
    spec: HybridCacheSpec,
    actual: list[
        GDNLayerState | None
    ],
    expected: list[
        GDNLayerState | None
    ],
) -> None:
    for layer_idx in range(
        spec.num_hidden_layers
    ):
        actual_state = actual[layer_idx]
        expected_state = expected[layer_idx]

        if layer_idx in (
            spec.full_attention_layer_indices
        ):
            assert actual_state is None
            assert expected_state is None
            continue

        assert actual_state is not None
        assert expected_state is not None

        torch.testing.assert_close(
            actual_state.conv_state,
            expected_state.conv_state,
            rtol=0,
            atol=0,
        )

        torch.testing.assert_close(
            actual_state.recurrent_state,
            expected_state.recurrent_state,
            rtol=0,
            atol=0,
        )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available"
        )

    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    # 使用一个小型 Hybrid 模型配置测试数据结构。
    #
    # 全局层：
    # 0 → GDN compact 0
    # 1 → Full Attention compact 0
    # 2 → GDN compact 1
    # 3 → GDN compact 2
    spec = HybridCacheSpec(
        num_hidden_layers=4,
        full_attention_layer_indices=(1,),
        gdn_layer_indices=(0, 2, 3),
        full_attention_index_by_layer=(
            -1,
            0,
            -1,
            -1,
        ),
        gdn_index_by_layer=(
            0,
            -1,
            1,
            2,
        ),
        num_kv_heads=2,
        attention_head_dim=8,
        conv_dim=12,
        conv_kernel_size=4,
        num_gdn_value_heads=2,
        gdn_key_head_dim=4,
        gdn_value_head_dim=4,
        conv_dtype=torch.bfloat16,
        recurrent_dtype=torch.float32,
    )

    workspace = (
        HybridDecodeStaticWorkspace
        .allocate(
            spec=spec,
            max_batch_size=4,
            max_num_blocks=8,
            hidden_size=16,
            device="cuda",
        )
    )

    print(
        "Workspace MiB:",
        workspace.allocated_bytes / 1024**2,
    )

    original_pointers = (
        workspace.storage_pointers()
    )

    batch_size = 2

    input_ids = torch.tensor(
        [100, 200],
        device="cuda",
        dtype=torch.long,
    )

    positions = torch.tensor(
        [
            [10, 20],
            [10, 20],
            [10, 20],
        ],
        device="cuda",
        dtype=torch.long,
    )

    slot_mapping = torch.tensor(
        [300, 700],
        device="cuda",
        dtype=torch.int32,
    )

    context_lens = torch.tensor(
        [11, 21],
        device="cuda",
        dtype=torch.int32,
    )

    block_tables = torch.tensor(
        [
            [3, 4, 5],
            [7, 8, -1],
        ],
        device="cuda",
        dtype=torch.int32,
    )

    state_slot_ids = [5, 1]

    eager_input_states = make_states(
        spec,
        batch_size,
    )

    returned_batch_size = (
        workspace.copy_decode_inputs(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            state_slot_ids=state_slot_ids,
            gdn_states=eager_input_states,
        )
    )

    assert returned_batch_size == batch_size

    torch.testing.assert_close(
        workspace.input_ids[:batch_size],
        input_ids,
    )

    torch.testing.assert_close(
        workspace.positions[:, :batch_size],
        positions,
    )

    torch.testing.assert_close(
        workspace.slot_mapping[:batch_size],
        slot_mapping,
    )

    torch.testing.assert_close(
        workspace.context_lens[:batch_size],
        context_lens,
    )

    torch.testing.assert_close(
        workspace.block_tables[
            :batch_size,
            :block_tables.shape[1],
        ],
        block_tables,
    )

    torch.testing.assert_close(
        workspace.state_slot_ids[:batch_size],
        torch.tensor(
            state_slot_ids,
            device="cuda",
            dtype=torch.long,
        ),
    )

    static_input_states = (
        workspace.input_gdn_state_views(
            batch_size
        )
    )

    assert_state_views_equal(
        spec,
        static_input_states,
        eager_input_states,
    )

    # 模拟模型输出。
    fake_hidden_states = torch.randn(
        batch_size,
        workspace.hidden_size,
        device="cuda",
        dtype=spec.conv_dtype,
    )

    fake_updated_states = make_states(
        spec,
        batch_size,
    )

    workspace.copy_model_outputs(
        batch_size=batch_size,
        hidden_states=fake_hidden_states,
        updated_gdn_states=(
            fake_updated_states
        ),
    )

    torch.testing.assert_close(
        workspace.hidden_states[:batch_size],
        fake_hidden_states,
    )

    static_output_states = (
        workspace.output_gdn_state_views(
            batch_size
        )
    )

    assert_state_views_equal(
        spec,
        static_output_states,
        fake_updated_states,
    )

    # 再复制完全不同的数据，验证地址不变、
    # 内容可以变化。
    second_input_ids = torch.tensor(
        [999, 888],
        device="cuda",
        dtype=torch.long,
    )

    second_states = make_states(
        spec,
        batch_size,
    )

    workspace.copy_decode_inputs(
        input_ids=second_input_ids,
        positions=positions + 100,
        slot_mapping=slot_mapping + 10,
        context_lens=context_lens + 1,
        block_tables=block_tables,
        state_slot_ids=[2, 6],
        gdn_states=second_states,
    )

    current_pointers = (
        workspace.storage_pointers()
    )

    assert (
        current_pointers
        == original_pointers
    )

    torch.testing.assert_close(
        workspace.input_ids[:batch_size],
        second_input_ids,
    )

    assert_state_views_equal(
        spec,
        workspace.input_gdn_state_views(
            batch_size
        ),
        second_states,
    )

    print(
        "All static Tensor addresses stayed fixed."
    )

    print(
        "Part 2 passed: Hybrid Decode static "
        "Workspace copies dynamic inputs and "
        "states without changing storage addresses."
    )


if __name__ == "__main__":
    main()