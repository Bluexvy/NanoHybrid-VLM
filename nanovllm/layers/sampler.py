import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # 把 temperature 先通过 unsqueeze 升维度 然后再原地通过float做除法（每一行除以刚刚经过变换的temp，也就是每一行都对应做除法）
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
