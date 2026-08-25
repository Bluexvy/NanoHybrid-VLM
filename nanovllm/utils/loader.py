import os
from glob import glob

import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
) -> None:
    param.data.copy_(loaded_weight)


def load_model(
    model: nn.Module,
    path: str,
) -> None:

    packed_modules_mapping = getattr(
        model,
        "packed_modules_mapping",
        {},
    )

    ignored_weight_prefixes = getattr(
        model,
        "ignored_weight_prefixes",
        (),
    )

    checkpoint_files = sorted(
        glob(
            os.path.join(
                path,
                "*.safetensors",
            )
        )
    )

    if not checkpoint_files:
        raise FileNotFoundError(
            f"No safetensors files found in "
            f"{path!r}"
        )

    # named_parameters 默认会去除共享 Parameter 的重复名字。
    #
    # 对共享 Embedding/LM Head 来说，
    # canonical name 会是：
    # model.language_model.embed_tokens.weight
    model_parameters = dict(
        model.named_parameters()
    )

    # 普通参数是否已经被加载。
    loaded_parameters: set[str] = set()

    # 合并参数分别加载了哪些 shard。
    #
    # 例如：
    # {
    #   "model.language_model.layers.0."
    #   "mlp.gate_up_proj.weight": {0, 1}
    # }
    loaded_packed_shards: dict[
        str,
        set[str | int],
    ] = {}

    ignored_weights: list[str] = []
    unexpected_weights: list[str] = []

    for checkpoint_file in checkpoint_files:
        with safe_open(
            checkpoint_file,
            framework="pt",
            device="cpu",
        ) as checkpoint:

            for weight_name in checkpoint.keys():

                # 当前阶段明确允许忽略的 checkpoint 部分。
                if any(
                    weight_name.startswith(prefix)
                    for prefix
                    in ignored_weight_prefixes
                ):
                    ignored_weights.append(
                        weight_name
                    )
                    continue

                param_name = weight_name
                packed_shard_id = None

                # 尝试将 checkpoint 的独立参数映射到
                # nano-vLLM 的合并参数。
                for (
                    checkpoint_substring,
                    (
                        packed_substring,
                        shard_id,
                    ),
                ) in packed_modules_mapping.items():

                    if (
                        checkpoint_substring
                        in weight_name
                    ):
                        param_name = (
                            weight_name.replace(
                                checkpoint_substring,
                                packed_substring,
                                1,
                            )
                        )

                        packed_shard_id = shard_id
                        break

                if param_name not in model_parameters:
                    unexpected_weights.append(
                        weight_name
                    )
                    continue

                param = model_parameters[param_name]
                loaded_weight = checkpoint.get_tensor(
                    weight_name
                )

                if packed_shard_id is None:
                    weight_loader = getattr(
                        param,
                        "weight_loader",
                        default_weight_loader,
                    )

                    weight_loader(
                        param,
                        loaded_weight,
                    )

                    loaded_parameters.add(
                        param_name
                    )

                else:
                    weight_loader = getattr(
                        param,
                        "weight_loader",
                        None,
                    )

                    if weight_loader is None:
                        raise RuntimeError(
                            f"Packed parameter "
                            f"{param_name!r} does not "
                            "define weight_loader"
                        )

                    weight_loader(
                        param,
                        loaded_weight,
                        packed_shard_id,
                    )

                    loaded_packed_shards.setdefault(
                        param_name,
                        set(),
                    ).add(
                        packed_shard_id
                    )

    # =======================================
    # 检查合并参数的所有 shard 是否完整
    # =======================================

    required_packed_shards: dict[
        str,
        set[str | int],
    ] = {}

    for param_name in model_parameters:
        for (
            _,
            (
                packed_substring,
                shard_id,
            ),
        ) in packed_modules_mapping.items():

            if packed_substring in param_name:
                required_packed_shards.setdefault(
                    param_name,
                    set(),
                ).add(shard_id)

    incomplete_packed_parameters: dict[
        str,
        set[str | int],
    ] = {}

    for (
        param_name,
        required_shards,
    ) in required_packed_shards.items():

        actual_shards = (
            loaded_packed_shards.get(
                param_name,
                set(),
            )
        )

        missing_shards = (
            required_shards - actual_shards
        )

        if missing_shards:
            incomplete_packed_parameters[
                param_name
            ] = missing_shards
        else:
            loaded_parameters.add(
                param_name
            )

    missing_parameters = (
        set(model_parameters)
        - loaded_parameters
    )

    error_messages: list[str] = []

    if unexpected_weights:
        formatted = "\n  ".join(
            sorted(unexpected_weights)
        )

        error_messages.append(
            "Unexpected checkpoint weights:\n"
            f"  {formatted}"
        )

    if missing_parameters:
        formatted = "\n  ".join(
            sorted(missing_parameters)
        )

        error_messages.append(
            "Missing model parameters:\n"
            f"  {formatted}"
        )

    if incomplete_packed_parameters:
        formatted = "\n  ".join(
            (
                f"{param_name}: missing shards "
                f"{sorted(shards, key=str)}"
            )
            for param_name, shards
            in sorted(
                incomplete_packed_parameters.items()
            )
        )

        error_messages.append(
            "Incomplete packed parameters:\n"
            f"  {formatted}"
        )

    if error_messages:
        raise RuntimeError(
            "\n\n".join(error_messages)
        )