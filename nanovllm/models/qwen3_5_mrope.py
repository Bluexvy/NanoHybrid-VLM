from itertools import groupby

import torch


TEXT_TOKEN_TYPE = 0
IMAGE_TOKEN_TYPE = 1
VIDEO_TOKEN_TYPE = 2


def build_qwen35_mrope_positions(
    mm_token_type_ids: list[int],
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> tuple[torch.Tensor, int]:
    """
    为单条 Qwen3.5 图文序列构造 mRoPE 位置。

    返回：
        position_ids:
            [3, sequence_length]

        position_delta:
            Decode 阶段用于把普通 token index
            映射到压缩后的 mRoPE 位置。
    """

    if not mm_token_type_ids:
        raise ValueError(
            "mm_token_type_ids must not be empty"
        )

    if (
        image_grid_thw.ndim != 2
        or image_grid_thw.shape[1] != 3
    ):
        raise ValueError(
            "image_grid_thw must have shape "
            "[num_images, 3]"
        )

    if spatial_merge_size <= 0:
        raise ValueError(
            "spatial_merge_size must be positive"
        )

    position_parts: list[
        torch.Tensor
    ] = []

    current_position = 0
    image_index = 0

    # groupby 会把连续的类型分成若干段。
    #
    # 例如：
    # [0, 0, 1, 1, 1, 0, 0]
    #
    # 被分成：
    # 文本段、图像段、文本段。
    for modality_type, group in groupby(
        mm_token_type_ids
    ):
        group_length = sum(
            1 for _ in group
        )

        # =====================================
        # 文本位置
        # =====================================

        if modality_type == TEXT_TOKEN_TYPE:
            text_positions = torch.arange(
                current_position,
                current_position + group_length,
                dtype=torch.long,
            )

            # 文本的 T/H/W 三个位置完全相同。
            text_positions = (
                text_positions
                .unsqueeze(0)
                .expand(3, -1)
            )

            position_parts.append(
                text_positions
            )

            current_position += group_length
            continue

        # =====================================
        # 当前首版不支持视频
        # =====================================

        if modality_type == VIDEO_TOKEN_TYPE:
            raise NotImplementedError(
                "Video mRoPE is not supported"
            )

        if modality_type != IMAGE_TOKEN_TYPE:
            raise ValueError(
                "Unknown multimodal token type: "
                f"{modality_type}"
            )

        # =====================================
        # 图像位置
        # =====================================

        if image_index >= image_grid_thw.shape[0]:
            raise ValueError(
                "There are more image token groups "
                "than image grids"
            )

        grid_t, grid_h, grid_w = (
            int(value)
            for value in (
                image_grid_thw[image_index]
                .tolist()
            )
        )

        image_index += 1

        if (
            grid_h % spatial_merge_size != 0
            or grid_w % spatial_merge_size != 0
        ):
            raise ValueError(
                "Image grid dimensions must be "
                "divisible by spatial_merge_size"
            )

        # Qwen3.5 单图的 temporal merge 为 1。
        llm_grid_t = grid_t

        llm_grid_h = (
            grid_h // spatial_merge_size
        )

        llm_grid_w = (
            grid_w // spatial_merge_size
        )

        expected_image_tokens = (
            llm_grid_t
            * llm_grid_h
            * llm_grid_w
        )

        if (
            group_length
            != expected_image_tokens
        ):
            raise ValueError(
                "Image token group length does not "
                "match the merged image grid: "
                f"{group_length} != "
                f"{expected_image_tokens}"
            )

        temporal_positions = torch.arange(
            llm_grid_t,
            dtype=torch.long,
        )

        height_positions = torch.arange(
            llm_grid_h,
            dtype=torch.long,
        )

        width_positions = torch.arange(
            llm_grid_w,
            dtype=torch.long,
        )

        temporal_grid, height_grid, width_grid = (
            torch.meshgrid(
                temporal_positions,
                height_positions,
                width_positions,
                indexing="ij",
            )
        )

        vision_positions = torch.stack(
            [
                temporal_grid,
                height_grid,
                width_grid,
            ],
            dim=0,
        ).reshape(
            3,
            -1,
        )

        # 图像三个轴都从前面文本结束的位置开始。
        vision_positions = (
            vision_positions
            + current_position
        )

        position_parts.append(
            vision_positions
        )

        # 图像占用了 H×W 个 token，
        # 但位置空间只前进 max(H, W)。
        current_position += max(
            llm_grid_h,
            llm_grid_w,
        )

    if image_index != image_grid_thw.shape[0]:
        raise ValueError(
            "There are unused image grids: "
            f"used {image_index}, "
            f"provided {image_grid_thw.shape[0]}"
        )

    position_ids = torch.cat(
        position_parts,
        dim=1,
    ).contiguous()

    sequence_length = len(
        mm_token_type_ids
    )

    if position_ids.shape != (
        3,
        sequence_length,
    ):
        raise RuntimeError(
            "mRoPE position shape mismatch: "
            f"{tuple(position_ids.shape)}"
        )

    # 图像会压缩位置空间。
    #
    # 后续 Decode 的位置：
    #
    # ordinary_token_index + position_delta
    position_delta = (
        int(position_ids.max().item())
        + 1
        - sequence_length
    )

    return (
        position_ids,
        position_delta,
    )