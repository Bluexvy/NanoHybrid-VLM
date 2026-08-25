from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.hybrid_state import (
    StateSlotAllocator,
)

class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(
            num_blocks=config.num_kvcache_blocks,
            block_size=config.kvcache_block_size,
            enable_prefix_cache=(
                config.enable_prefix_cache
            ),
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        layer_types = getattr(
        config.text_config,
        "layer_types",
            (),
        )

        self.requires_gdn_state = any(
            layer_type == "linear_attention"
            for layer_type in layer_types
        )

        if self.requires_gdn_state:
            if config.num_state_slots <= 0:
                raise ValueError(
                    "ModelRunner must calculate a positive "
                    "num_state_slots before Scheduler "
                    "construction"
                )

            self.state_slot_allocator = (
                StateSlotAllocator(
                    config.num_state_slots
                )
            )

        else:
            self.state_slot_allocator = None

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                if not self.can_allocate_state_slot(seq):
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(
                    seq,
                    num_cached_blocks,
                )

                self.allocate_state_slot(seq)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(
        self,
        seq: Sequence,
    ):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True

        # 释放 Full Attention KV。
        self.block_manager.deallocate(seq)

        # 释放 GDN state slot。
        self.release_state_slot(seq)

        # num_cached_tokens 已被 BlockManager
        # 重置为 0。
        #
        # 恢复时必须从 token 0 重新 Prefill，
        # 重建 KV 和全部 GDN state。
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if ((not seq.ignore_eos and token_id == self.eos) or (seq.num_completion_tokens == seq.max_tokens)):
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.release_state_slot(seq)
                self.running.remove(seq)

    def can_allocate_state_slot(
        self,
        seq: Sequence,
    ) -> bool:

        if not self.requires_gdn_state:
            return True

        # 已经做过一个 Prefill chunk 的请求
        # 会继续持有原来的 slot。
        if seq.state_slot is not None:
            return True

        return (
            self.state_slot_allocator
            .can_allocate()
        )


    def allocate_state_slot(
        self,
        seq: Sequence,
    ) -> None:

        if not self.requires_gdn_state:
            return

        if seq.state_slot is not None:
            raise RuntimeError(
                f"Sequence {seq.seq_id} already "
                "owns state slot {seq.state_slot}"
            )

        seq.state_slot = (
            self.state_slot_allocator.allocate()
        )


    def release_state_slot(
        self,
        seq: Sequence,
    ) -> None:

        if not self.requires_gdn_state:
            return

        if seq.state_slot is None:
            raise RuntimeError(
                f"Sequence {seq.seq_id} has no "
                "state slot to release"
            )

        self.state_slot_allocator.release(
            seq.state_slot
        )

        seq.state_slot = None