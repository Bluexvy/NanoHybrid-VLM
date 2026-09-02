from dataclasses import dataclass

import torch


@dataclass(slots=True)
class Context:
    # Attention/GDN 根据它区分 Prefill 与 Decode。
    is_prefill: bool = False

    # FlashAttention Variable-length Prefill 元数据。
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0

    # Paged KV Cache 元数据。
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

    # Variable-length GDN Prefill 元数据。
    #
    # prefill_seqlens：
    #     每条请求当前 Prefill chunk 的长度。
    #
    # gdn_cu_seqlens：
    #     FLA 使用的累积序列边界。
    prefill_seqlens: tuple[int, ...] | None = None
    gdn_cu_seqlens: torch.Tensor | None = None


_CONTEXT = Context()


def get_context() -> Context:
    return _CONTEXT


def set_context(
    is_prefill: bool,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    slot_mapping: torch.Tensor | None = None,
    context_lens: torch.Tensor | None = None,
    block_tables: torch.Tensor | None = None,
    prefill_seqlens: tuple[int, ...] | None = None,
    gdn_cu_seqlens: torch.Tensor | None = None,
) -> None:
    global _CONTEXT

    _CONTEXT = Context(
        is_prefill=is_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        prefill_seqlens=prefill_seqlens,
        gdn_cu_seqlens=gdn_cu_seqlens,
    )


def reset_context() -> None:
    global _CONTEXT
    _CONTEXT = Context()