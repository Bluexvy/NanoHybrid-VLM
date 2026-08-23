from dataclasses import dataclass

# 根据字符串动态导入模块
from importlib import import_module

from torch import nn
from transformers.configuration_utils import PretrainedConfig

# Registry 只负责选择哪一个模型实现
# frozen 注册信息属于静态元数据，不应该在程序运行中被意外改变。
# slots=True 会限制对象只能拥有声明过的字段
@dataclass(frozen=True, slots=True)
class ModelEntry:
    """注册表中一条模型记录。"""
    # 这个模型允许的 checkpoint architecture
    architectures: tuple[str, ...]
    # 本地实现所在模块
    module: str
    # 模块里的模型类
    class_name: str

    # 内部维护一个字典 注册后类似：
    # {
    #     "qwen3": ModelEntry(
    #         architectures=("Qwen3ForCausalLM",),
    #         module="nanovllm.models.qwen3",
    #         class_name="Qwen3ForCausalLM",
    #     )
    # }
class ModelRegistry:
    """根据 Hugging Face 根配置查找 nano-vLLM 模型类。"""

    def __init__(self):
        self._entries: dict[str, ModelEntry] = {}

    def register(
        self,
        *,
        model_type: str,
        # 由任意数量字符串组成的 tuple
        architectures: tuple[str, ...],
        module: str,
        class_name: str,
    ) -> None:
        # 前导下划线 _entries 表示它是 Registry 内部状态，外部不应直接操作
        if model_type in self._entries:
            raise ValueError(
                f"Model type {model_type!r} is already registered"
            )
        # 拒绝空的 arch
        if not architectures:
            raise ValueError(
                "At least one architecture must be registered"
            )
        # 保存注册项
        self._entries[model_type] = ModelEntry(
            architectures=tuple(architectures),
            module=module,
            class_name=class_name,
        )

    # 输入 Huggingface的根配置 
    # 输出 nn.Module模型类 注意 返回的是类，而不是“实例”
    def resolve(
        self,
        root_config: PretrainedConfig,
    ) -> type[nn.Module]:
        # model_type是模型家族： qwen3 / qwen3_5 / llama / gemma3
        model_type = getattr(
            root_config,
            "model_type",
            None,
        )

        if model_type not in self._entries:
            supported = ", ".join(
                sorted(self._entries)
            )

            raise ValueError(
                f"Unsupported model_type {model_type!r}. "
                f"Supported model types: {supported}"
            )

        entry = self._entries[model_type]

        # architectures 表示 checkpoint 顶层结构
        # 例如Qwen3ForCausalLM
        # Qwen3_5ForConditionalGeneration
        # Qwen3_5TextModel
        checkpoint_architectures = getattr(
            root_config,
            "architectures",
            None,
        ) or ()

        # 统一转成 tuple
        if isinstance(checkpoint_architectures, str):
            checkpoint_architectures = (
                checkpoint_architectures,
            )

        # 判断 architecture 是否匹配
        architecture_matches = any(
            architecture in entry.architectures
            for architecture in checkpoint_architectures
        )

        if (
            checkpoint_architectures
            and not architecture_matches
        ):
            raise ValueError(
                f"Model type {model_type!r} does not support "
                "checkpoint architectures "
                f"{tuple(checkpoint_architectures)!r}. "
                "Supported architectures: "
                f"{entry.architectures!r}"
            )

        try:
            model_module = import_module(entry.module)
        except ImportError as error:
            raise ImportError(
                f"Failed to import model module "
                f"{entry.module!r}"
            ) from error

        try:
            model_class = getattr(
                model_module,
                entry.class_name,
            )
        except AttributeError as error:
            raise ImportError(
                f"Module {entry.module!r} does not contain "
                f"class {entry.class_name!r}"
            ) from error

        if (
            not isinstance(model_class, type)
            or not issubclass(model_class, nn.Module)
        ):
            raise TypeError(
                f"{entry.module}.{entry.class_name} "
                "must be an nn.Module subclass"
            )

        return model_class


MODEL_REGISTRY = ModelRegistry()

MODEL_REGISTRY.register(
    model_type="qwen3",
    architectures=("Qwen3ForCausalLM",),
    module="nanovllm.models.qwen3",
    class_name="Qwen3ForCausalLM",
)


def get_model_class(
    root_config: PretrainedConfig,
) -> type[nn.Module]:
    return MODEL_REGISTRY.resolve(root_config)