from copy import copy
from enum import Enum, auto
from itertools import count
import torch
from torch import Tensor
from typing import TYPE_CHECKING

from nanovllm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from nanovllm.engine.prefix_cache import (
        PrefixKey,
    )

class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(
        self,
        token_ids: list[int],
        sampling_params=SamplingParams(),
        *,
        mm_token_type_ids: (
            list[int] | None
        ) = None,
        pixel_values: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        mrope_position_ids: (
            Tensor | None
        ) = None,
        mrope_position_delta: (
            int | None
        ) = None,
    ):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        # TP workers can retain only a decode suffix at this logical offset.
        self.token_offset = 0
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        # 当前请求成功创建了多少个新的
        # GDN-aware Prefix Entry。
        # duplicate commit 不会增加这个数字，
        # 因为它没有创建新的 GPU snapshot。
        self.num_prefix_snapshots_created = 0
        self.is_prefill = True
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.mtp_num_speculative_tokens = 0
        
        # 只保存整数 slot 编号，
        # 不保存任何 GPU Tensor。
        self.state_slot: int | None = None
        
        # Scheduler 是否已经为该请求执行过一次
        # Prefix State Cache lookup。
        self.prefix_lookup_completed = False

        # 命中的 CPU PrefixKey。
        #
        # 不在 Sequence 中保存 PrefixStateEntry，
        # 因为 Entry 含有大型 GPU Tensor Snapshot。
        self.prefix_cache_key: (
            "PrefixKey | None"
        ) = None

        # Scheduler 已经 attach KV 并分配 state slot，
        # 但 Engine 尚未把 GDN Snapshot 恢复进该 slot。
        self.prefix_restore_pending = False

        # 实际通过 Prefix Hit 跳过了多少 Prompt tokens。
        self.num_prefix_hit_tokens = 0
        
        multimodal_fields = (
            mm_token_type_ids is not None,
            pixel_values is not None,
            image_grid_thw is not None,
            mrope_position_ids is not None,
            mrope_position_delta is not None,
        )

        if (
            any(multimodal_fields)
            and not all(multimodal_fields)
        ):
            raise ValueError(
                "Multimodal Sequence fields "
                "must be provided together"
            )

        if (
            mm_token_type_ids is not None
            and len(mm_token_type_ids)
            != len(token_ids)
        ):
            raise ValueError(
                "mm_token_type_ids must align "
                "with token_ids"
            )

        if (
            mrope_position_ids is not None
            and mrope_position_ids.shape
            != (3, len(token_ids))
        ):
            raise ValueError(
                "mrope_position_ids must have "
                "shape [3, num_tokens]"
            )

        if (
            mrope_position_ids is not None
            and mrope_position_ids.dtype
            != torch.long
        ):
            raise TypeError(
                "mrope_position_ids must use "
                "torch.long"
            )

        if (
            mrope_position_ids is not None
            and mrope_position_ids.is_cuda
        ):
            raise ValueError(
                "mrope_position_ids must remain "
                "on CPU while waiting"
            )

        if (
            mrope_position_delta is not None
            and not isinstance(
                mrope_position_delta,
                int,
            )
        ):
            raise TypeError(
                "mrope_position_delta must be int"
            )

        self.mm_token_type_ids = (
            copy(mm_token_type_ids)
            if mm_token_type_ids is not None
            else None
        )

        self.pixel_values = pixel_values

        self.image_grid_thw = (
            image_grid_thw
        )
        
        self.mrope_position_ids = (
            mrope_position_ids.clone()
            if mrope_position_ids is not None
            else None
        )

        self.mrope_position_delta = (
            mrope_position_delta
        )

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        offset = getattr(self, "token_offset", 0)
        if offset:
            if isinstance(key, slice):
                indices = range(*key.indices(self.num_tokens))
                if indices and min(indices) < offset:
                    raise IndexError("TP decode payload does not contain historical token IDs")
                return [self.token_ids[index - offset] for index in indices]
            index = key + self.num_tokens if key < 0 else key
            if index < offset or index >= self.num_tokens:
                raise IndexError("Token index is outside the TP decode suffix")
            return self.token_ids[index - offset]
        return self.token_ids[key]


    @property
    def is_multimodal(self) -> bool:
        return self.pixel_values is not None
    
    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(
        self,
        token_id: int,
    ):
        self.token_ids.append(token_id)

        if self.mm_token_type_ids is not None:
            # 模型生成的 token 一定属于文本。
            self.mm_token_type_ids.append(0)

        if self.mrope_position_ids is not None:
            if self.mrope_position_delta is None:
                raise RuntimeError(
                    "A multimodal Sequence is "
                    "missing mRoPE position delta"
                )

            # 此时 self.num_tokens 还是追加前的长度。
            #
            # 例如：
            # prompt length = 96
            # delta = -70
            #
            # 第一个生成 token 的位置：
            # 96 - 70 = 26
            new_position = (
                self.num_tokens
                + self.mrope_position_delta
            )

            # 新生成的是文本 token，
            # 所以 T/H/W 三个位置完全相同。
            new_position_column = (
                self.mrope_position_ids
                .new_full(
                    (3, 1),
                    new_position,
                )
            )

            self.mrope_position_ids = (
                torch.cat(
                    [
                        self.mrope_position_ids,
                        new_position_column,
                    ],
                    dim=1,
                )
            )

        self.last_token = token_id
        self.num_tokens += 1
        
    def __getstate__(self):
        last_state = (
            self.last_token
            if not self.is_prefill
            else self.token_ids
        )

        if (
            self.is_prefill
            and self.is_multimodal
        ):
            # Prefill 需要完整图像和完整位置。
            multimodal_state = (
                self.mm_token_type_ids,
                self.pixel_values,
                self.image_grid_thw,
                self.mrope_position_ids,
                self.mrope_position_delta,
            )

        else:
            # Decode 不再运行 Vision Tower，
            # 也不需要完整三轴位置数组。
            #
            # 但它仍然需要 delta 来计算
            # 当前 Decode token 的位置。
            multimodal_state = (
                None,
                None,
                None,
                None,
                self.mrope_position_delta,
            )

        return (
            self.seq_id,
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.num_scheduled_tokens,
            self.block_table,
            self.state_slot,
            last_state,
            *multimodal_state,
            {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "ignore_eos": self.ignore_eos,
                "mtp_num_speculative_tokens": self.mtp_num_speculative_tokens,
                "prefix_cache_key": self.prefix_cache_key,
                "token_ids": (self.token_ids if self.is_prefill else [self.last_token])
                             if self.mtp_num_speculative_tokens else None,
                "token_offset": (getattr(self, "token_offset", 0) if self.is_prefill else self.num_tokens - 1)
                                if self.mtp_num_speculative_tokens else 0,
                "mm_token_type_ids": self.mm_token_type_ids if self.mtp_num_speculative_tokens and self.is_prefill else None,
                "mrope_position_ids": self.mrope_position_ids if self.mtp_num_speculative_tokens and self.is_prefill else None,
            },
        )
                
    def __setstate__(
        self,
        state,
    ):
        metadata = state[-1] if isinstance(state[-1], dict) else {}
        if metadata:
            state = state[:-1]
        (
            self.seq_id,
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.num_scheduled_tokens,
            self.block_table,
            self.state_slot,
            last_state,
            self.mm_token_type_ids,
            self.pixel_values,
            self.image_grid_thw,
            self.mrope_position_ids,
            self.mrope_position_delta,
        ) = state

        self.is_prefill = isinstance(
            last_state,
            list,
        )

        if self.is_prefill:
            self.token_ids = last_state
            self.last_token = (
                self.token_ids[-1]
            )
        else:
            self.token_ids = []
            self.last_token = last_state
        self.temperature = metadata.get("temperature", 0)
        self.max_tokens = metadata.get("max_tokens", 0)
        self.ignore_eos = metadata.get("ignore_eos", False)
        self.mtp_num_speculative_tokens = metadata.get("mtp_num_speculative_tokens", 0)
        self.prefix_cache_key = metadata.get("prefix_cache_key")
        self.token_offset = metadata.get("token_offset", 0)
        if metadata.get("token_ids") is not None:
            self.token_ids = metadata["token_ids"]
            self.mm_token_type_ids = metadata["mm_token_type_ids"]
            self.mrope_position_ids = metadata["mrope_position_ids"]
