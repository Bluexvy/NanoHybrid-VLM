import torch
from torch import nn

class Sampler(nn.Module):

    @torch.compile
    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
    ):
        logits = logits.float()

        # 先为所有请求计算 greedy 结果。
        # 正温度请求对应的位置稍后会被随机采样结果覆盖。
        # argmax返回的是最大值位置下标
        token_ids = logits.argmax(dim=-1)

        # 只选择真正需要随机采样的请求。得到展平后对应的位置下标
        sampling_indices = torch.nonzero(
            temperatures > 0,
            as_tuple=False,
        ).flatten()

        # 如flatten后的tensor里面没有元素，意味着整个 batch 都是 greedy：
        # 在任何 softmax 和随机数操作之前直接返回。
        if sampling_indices.numel() == 0:
            return token_ids

        sampling_logits = logits.index_select(
            dim=0,
            index=sampling_indices,
        )

        sampling_temperatures = temperatures.index_select(
            dim=0,
            index=sampling_indices,
        )

        scaled_logits = (
            sampling_logits
            / sampling_temperatures.unsqueeze(dim=1)
        )

        probs = torch.softmax(
            scaled_logits,
            dim=-1,
        )

        noise = (
            torch.empty_like(probs)
            .exponential_(1)
            .clamp_min_(1e-10)
        )

        sampled_tokens = (
            probs / noise
        ).argmax(dim=-1)

        # 用 scatter 写回原 batch
        return token_ids.scatter(
            dim=0,
            index=sampling_indices,
            src=sampled_tokens,
        )