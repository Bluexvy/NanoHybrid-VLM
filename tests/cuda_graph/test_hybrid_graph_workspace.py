import torch

from nanovllm.engine.hybrid_cuda_graph import (
    HybridDecodeStaticWorkspace,
)

from nanovllm.engine.hybrid_state import (
    HybridCacheSpec,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available"
        )

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

    # 清理后只应保留这些固定地址 Tensor。
    expected_pointer_names = {
        "input_ids",
        "positions",
        "slot_mapping",
        "context_lens",
        "block_tables",
        "state_slot_ids",
        "hidden_states",
    }

    original_pointers = (
        workspace.storage_pointers()
    )

    assert (
        set(original_pointers)
        == expected_pointer_names
    )

    # 确认大型 GDN 输入/输出 Workspace
    # 已经从数据结构中删除。
    removed_state_buffers = (
        "gdn_conv_states",
        "gdn_recurrent_states",
        "updated_gdn_conv_states",
        "updated_gdn_recurrent_states",
    )

    for name in removed_state_buffers:
        assert not hasattr(workspace, name)

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

    returned_batch_size = (
        workspace.copy_decode_inputs(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            state_slot_ids=state_slot_ids,
        )
    )

    assert returned_batch_size == batch_size

    torch.testing.assert_close(
        workspace.input_ids[:batch_size],
        input_ids,
    )

    torch.testing.assert_close(
        workspace.positions[
            :,
            :batch_size,
        ],
        positions,
    )

    torch.testing.assert_close(
        workspace.slot_mapping[
            :batch_size
        ],
        slot_mapping,
    )

    torch.testing.assert_close(
        workspace.context_lens[
            :batch_size
        ],
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
        workspace.state_slot_ids[
            :batch_size
        ],
        torch.tensor(
            state_slot_ids,
            device="cuda",
            dtype=torch.long,
        ),
    )

    # 第二轮写入完全不同的数据。
    # CUDA Graph 要求地址不变，但内容可以变化。
    second_input_ids = torch.tensor(
        [999, 888],
        device="cuda",
        dtype=torch.long,
    )

    second_block_tables = torch.tensor(
        [
            [9, 10],
            [11, 12],
        ],
        device="cuda",
        dtype=torch.int32,
    )

    workspace.copy_decode_inputs(
        input_ids=second_input_ids,
        positions=positions + 100,
        slot_mapping=slot_mapping + 10,
        context_lens=context_lens + 1,
        block_tables=second_block_tables,
        state_slot_ids=[2, 6],
    )

    current_pointers = (
        workspace.storage_pointers()
    )

    # 地址必须保持不变。
    assert current_pointers == original_pointers

    # 内容必须更新。
    torch.testing.assert_close(
        workspace.input_ids[:batch_size],
        second_input_ids,
    )

    torch.testing.assert_close(
        workspace.state_slot_ids[
            :batch_size
        ],
        torch.tensor(
            [2, 6],
            device="cuda",
            dtype=torch.long,
        ),
    )

    torch.testing.assert_close(
        workspace.block_tables[
            :batch_size,
            :2,
        ],
        second_block_tables,
    )

    # 第二轮 block table 只有两列。
    # 上一轮残留的第三列必须被清成 -1。
    assert torch.all(
        workspace.block_tables[
            :batch_size,
            2:
        ] == -1
    )

    # allocated_bytes 必须只统计仍然存在的 Tensor。
    expected_allocated_bytes = sum(
        tensor.numel()
        * tensor.element_size()
        for tensor in (
            workspace.input_ids,
            workspace.positions,
            workspace.slot_mapping,
            workspace.context_lens,
            workspace.block_tables,
            workspace.state_slot_ids,
            workspace.hidden_states,
        )
    )

    assert (
        workspace.allocated_bytes
        == expected_allocated_bytes
    )

    print(
        "Workspace bytes:",
        workspace.allocated_bytes,
    )

    print(
        "Workspace MiB:",
        workspace.allocated_bytes / 1024**2,
    )

    print(
        "All remaining static Tensor "
        "addresses stayed fixed."
    )

    print(
        "Large GDN input/output staging "
        "buffers were removed."
    )

    print(
        "Part passed: metadata-only CUDA "
        "Graph Workspace is correct."
    )


if __name__ == "__main__":
    main()