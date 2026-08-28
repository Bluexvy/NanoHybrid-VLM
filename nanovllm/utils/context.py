from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    # Prefill microbatch中每条Sequence
    # 本轮实际执行的token数量。
    #
    # 例如：
    #     prefill_seqlens = (5, 3, 7)
    #
    # 对应：
    #     cu_seqlens_q = [0, 5, 8, 15]
    #
    # 它保留在CPU上，避免GDN层为了获得序列
    # 长度而将cu_seqlens_q从GPU同步回CPU。
    prefill_seqlens: tuple[int, ...] | None = None
    # FLA Gated Delta Rule使用的变长边界。
    #
    # 数值与cu_seqlens_q相同，但dtype为torch.long。
    gdn_cu_seqlens: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
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
    prefill_seqlens: (
        tuple[int, ...] | None
    ) = None,
    gdn_cu_seqlens: (
        torch.Tensor | None
    ) = None,
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
    
def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
