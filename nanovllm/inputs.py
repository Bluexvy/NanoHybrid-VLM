from dataclasses import dataclass
from typing import TypeAlias, TypedDict

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from nanovllm.config import Config
from nanovllm.models.qwen3_5_mrope import (
    build_qwen35_mrope_positions,
)

class MultiModalData(TypedDict):
    image: Image.Image


class MultiModalPrompt(TypedDict):
    prompt: str
    multi_modal_data: MultiModalData


PromptInput: TypeAlias = (
    str
    | list[int]
    | MultiModalPrompt
)


def validate_token_ids(
    token_ids: list[int],
) -> None:
    if not isinstance(token_ids, list):
        raise TypeError(
            "token_ids must be a list"
        )

    if not token_ids:
        raise ValueError(
            "token_ids must not be empty"
        )

    for token_id in token_ids:
        # bool 是 int 的子类，因此要单独排除。
        if (
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
        ):
            raise TypeError(
                "Every token ID must be an integer"
            )

        if token_id < 0:
            raise ValueError(
                "Token IDs must be non-negative"
            )


@dataclass(slots=True)
class ProcessedPrompt:
    """
    输入预处理后的统一结果。

    纯文本请求：
        只有 token_ids。

    图文请求：
        同时保存 token_ids、
        mm_token_type_ids、
        pixel_values 和 image_grid_thw。
    """

    token_ids: list[int]

    mm_token_type_ids: list[int] | None = None

    pixel_values: torch.Tensor | None = None

    image_grid_thw: torch.Tensor | None = None
    
    mrope_position_ids: (
        torch.Tensor | None
    ) = None

    mrope_position_delta: (
        int | None
    ) = None

    def __post_init__(self):
        validate_token_ids(self.token_ids)

        multimodal_fields = (
            self.mm_token_type_ids
                is not None,
            self.pixel_values
                is not None,
            self.image_grid_thw
                is not None,
            self.mrope_position_ids
                is not None,
            self.mrope_position_delta
                is not None,
        )

        # 三个多模态字段必须同时存在，
        # 或者同时不存在。
        if (
            any(multimodal_fields)
            and not all(multimodal_fields)
        ):
            raise ValueError(
                "Multimodal fields must be "
                "provided together"
            )

        if not self.is_multimodal:
            return

        if (
            len(self.mm_token_type_ids)
            != len(self.token_ids)
        ):
            raise ValueError(
                "mm_token_type_ids and token_ids "
                "must have the same length"
            )

        if self.pixel_values.ndim != 2:
            raise ValueError(
                "pixel_values must have shape "
                "[num_patches, patch_dim]"
            )

        if (
            self.image_grid_thw.ndim != 2
            or self.image_grid_thw.shape != (1, 3)
        ):
            raise ValueError(
                "image_grid_thw must have shape "
                "[1, 3] for one image"
            )

        # 等请求真正被 Scheduler 选中后，
        # ModelRunner 再把图像数据传到 GPU。
        if self.pixel_values.is_cuda:
            raise ValueError(
                "pixel_values must remain on CPU"
            )

        if self.image_grid_thw.is_cuda:
            raise ValueError(
                "image_grid_thw must remain on CPU"
            )
        
        if self.mrope_position_ids.shape != (
            3,
            len(self.token_ids),
        ):
            raise ValueError(
                "mrope_position_ids must have "
                "shape [3, num_tokens]"
            )

        if not isinstance(
            self.mrope_position_delta,
            int,
        ):
            raise TypeError(
                "mrope_position_delta must be int"
            )

    @property
    def is_multimodal(self) -> bool:
        return self.pixel_values is not None


class InputProcessor:

    # Qwen3.5 官方 get_rope_index() 使用：
    #
    # 0：文本
    # 1：图像
    # 2：视频
    IMAGE_TOKEN_TYPE = 1

    def __init__(self, config: Config):
        self.config = config
        self.processor = None
        self.image_token_id = None
        self.spatial_merge_size = None

        if config.vision_config is None:
            # 普通 Qwen3 等纯文本模型。
            self.tokenizer = (
                AutoTokenizer.from_pretrained(
                    config.model,
                    use_fast=True,
                )
            )
            return

        # Qwen3.5 图文模型使用官方 Processor。
        #
        # Processor 内部同时包含：
        # 1. tokenizer
        # 2. image processor
        # 3. chat template
        self.processor = (
            AutoProcessor.from_pretrained(
                config.model,
            )
        )

        self.tokenizer = (
            self.processor.tokenizer
        )

        self.image_token_id = getattr(
            config.hf_config,
            "image_token_id",
            None,
        )

        if not isinstance(
            self.image_token_id,
            int,
        ):
            raise ValueError(
                "The model config does not provide "
                "a valid image_token_id"
            )

        self.spatial_merge_size = getattr(
            config.vision_config,
            "spatial_merge_size",
            None,
        )

        if not isinstance(
            self.spatial_merge_size,
            int,
        ):
            raise ValueError(
                "vision_config does not provide "
                "spatial_merge_size"
            )

    def process(
        self,
        prompt: PromptInput,
    ) -> ProcessedPrompt:
        # 情况一：普通字符串。
        if isinstance(prompt, str):
            token_ids = self.tokenizer.encode(
                prompt,
            )

            return ProcessedPrompt(
                token_ids=token_ids,
            )

        # 情况二：用户已经完成 tokenize。
        if isinstance(prompt, list):
            validate_token_ids(prompt)

            return ProcessedPrompt(
                token_ids=prompt.copy(),
            )

        # 情况三：图文字典。
        if isinstance(prompt, dict):
            return self._process_multimodal(
                prompt,
            )

        raise TypeError(
            "prompt must be a string, "
            "a list of token IDs, "
            "or a multimodal prompt dictionary"
        )

    def _process_multimodal(
        self,
        prompt: MultiModalPrompt,
    ) -> ProcessedPrompt:
        if self.processor is None:
            raise ValueError(
                "The selected model does not "
                "support multimodal input"
            )

        if set(prompt.keys()) != {
            "prompt",
            "multi_modal_data",
        }:
            raise ValueError(
                "A multimodal prompt must contain "
                "only 'prompt' and "
                "'multi_modal_data'"
            )

        text = prompt["prompt"]
        multimodal_data = (
            prompt["multi_modal_data"]
        )

        if (
            not isinstance(text, str)
            or not text
        ):
            raise ValueError(
                "The multimodal prompt text must "
                "be a non-empty string"
            )

        if not isinstance(
            multimodal_data,
            dict,
        ):
            raise TypeError(
                "multi_modal_data must be a dict"
            )

        if set(multimodal_data.keys()) != {
            "image",
        }:
            raise ValueError(
                "Only one local PIL image "
                "is currently supported"
            )

        image = multimodal_data["image"]

        if not isinstance(image, Image.Image):
            raise TypeError(
                "image must be a PIL.Image.Image"
            )

        # 统一为 RGB，防止灰度图、RGBA 图片导致
        # patch embedding 输入通道数不一致。
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 构造 Qwen3.5 官方对话格式。
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": text,
                    },
                ],
            },
        ]

        formatted_prompt = (
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        batch = self.processor(
            text=[formatted_prompt],
            images=[image],
            padding=False,
            return_tensors="pt",
        )

        required_keys = {
            "input_ids",
            "attention_mask",
            "mm_token_type_ids",
            "pixel_values",
            "image_grid_thw",
        }

        missing_keys = (
            required_keys - set(batch.keys())
        )

        if missing_keys:
            raise RuntimeError(
                "AutoProcessor did not return "
                f"required fields: {missing_keys}"
            )

        input_ids = batch["input_ids"]

        attention_mask = (
            batch["attention_mask"]
        )

        mm_token_type_ids = (
            batch["mm_token_type_ids"]
        )

        pixel_values = (
            batch["pixel_values"]
        )

        image_grid_thw = (
            batch["image_grid_thw"]
        )

        # 当前一个请求只处理一条序列。
        if (
            input_ids.ndim != 2
            or input_ids.shape[0] != 1
        ):
            raise RuntimeError(
                "input_ids must have shape [1, L]"
            )

        if (
            mm_token_type_ids.shape
            != input_ids.shape
        ):
            raise RuntimeError(
                "mm_token_type_ids must have "
                "the same shape as input_ids"
            )

        # padding=False 且一次只有一个请求，
        # 因此 attention_mask 应当全部为 1。
        if not bool(
            attention_mask.bool().all().item()
        ):
            raise RuntimeError(
                "Unexpected padding in a "
                "single multimodal request"
            )

        if (
            image_grid_thw.ndim != 2
            or image_grid_thw.shape != (1, 3)
        ):
            raise RuntimeError(
                "Only one image is supported"
            )

        token_ids = input_ids[0].tolist()

        token_type_ids = (
            mm_token_type_ids[0].tolist()
        )

        # image_grid_thw = [T, H, W]。
        #
        # T * H * W 是进入 Vision Tower 的
        # 原始视觉 patch 数量。
        raw_patch_counts = (
            image_grid_thw.prod(dim=-1)
        )

        merge_area = (
            self.spatial_merge_size ** 2
        )

        if bool(
            (
                raw_patch_counts
                % merge_area
                != 0
            ).any().item()
        ):
            raise RuntimeError(
                "The number of vision patches "
                "is not divisible by the "
                "spatial merge area"
            )

        expected_image_tokens = int(
            (
                raw_patch_counts
                // merge_area
            ).sum().item()
        )

        placeholder_count = sum(
            token_id == self.image_token_id
            for token_id in token_ids
        )

        image_type_count = sum(
            token_type
            == self.IMAGE_TOKEN_TYPE
            for token_type in token_type_ids
        )

        if (
            placeholder_count
            != expected_image_tokens
        ):
            raise RuntimeError(
                "The number of image placeholders "
                "does not match the number of "
                "merged visual embeddings"
            )

        if (
            image_type_count
            != expected_image_tokens
        ):
            raise RuntimeError(
                "mm_token_type_ids does not match "
                "the number of image tokens"
            )

        # 确保 image_token_id 和图像类型位置
        # 一一对应。
        for token_id, token_type in zip(
            token_ids,
            token_type_ids,
        ):
            is_image_placeholder = (
                token_id
                == self.image_token_id
            )

            is_image_type = (
                token_type
                == self.IMAGE_TOKEN_TYPE
            )

            if (
                is_image_placeholder
                != is_image_type
            ):
                raise RuntimeError(
                    "Image placeholders and "
                    "mm_token_type_ids are not aligned"
                )

        (
            mrope_position_ids,
            mrope_position_delta,
        ) = build_qwen35_mrope_positions(
            mm_token_type_ids=token_type_ids,
            image_grid_thw=image_grid_thw,
            spatial_merge_size=(
                self.spatial_merge_size
            ),
        )

        return ProcessedPrompt(
            token_ids=token_ids,
            mm_token_type_ids=token_type_ids,
            pixel_values=(
                pixel_values
                .detach()
                .cpu()
                .contiguous()
            ),
            image_grid_thw=(
                image_grid_thw
                .detach()
                .cpu()
                .contiguous()
            ),
            mrope_position_ids=(
                mrope_position_ids
            ),
            mrope_position_delta=(
                mrope_position_delta
            ),
        )