from copy import copy
from enum import Enum, auto
from itertools import count
from torch import Tensor

from nanovllm.sampling_params import SamplingParams


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
    ):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        
        # 只保存整数 slot 编号，
        # 不保存任何 GPU Tensor。
        self.state_slot: int | None = None
        
        multimodal_fields = (
            mm_token_type_ids is not None,
            pixel_values is not None,
            image_grid_thw is not None,
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

        self.mm_token_type_ids = (
            copy(mm_token_type_ids)
            if mm_token_type_ids is not None
            else None
        )

        self.pixel_values = pixel_values

        self.image_grid_thw = (
            image_grid_thw
        )

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
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

        # 新生成的 token 一定是文本，
        # 所以对应的 multimodal type 是 0。
        if self.mm_token_type_ids is not None:
            self.mm_token_type_ids.append(0)

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
            multimodal_state = (
                self.mm_token_type_ids,
                self.pixel_values,
                self.image_grid_thw,
            )
        else:
            multimodal_state = (
                None,
                None,
                None,
            )

        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.num_scheduled_tokens,
            self.block_table,
            self.state_slot,
            last_state,
            *multimodal_state,
        )
        
    def __setstate__(
        self,
        state,
    ):
        (
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