# Qwen3.5 DFlash / Qwen3.8 DFlash2 接入

**Qwen3.8-27B + DFlash2 的无卡下载、TP=2 适配说明和一键测试，见 [QWEN38_GUIDE.md](QWEN38_GUIDE.md)。**

2×A800 的一键启动、接受率/性能统计与结果说明见 [A800_GUIDE.md](A800_GUIDE.md)。

本后端支持 DFlash v1/v2 的并行起草，保留原生 MTP。以下原有示例是 9B + v1。参考：

- 官方实现：https://github.com/z-lab/dflash （MIT）
- 对应草稿：https://huggingface.co/z-lab/Qwen3.5-9B-DFlash
- 论文：https://arxiv.org/abs/2602.06036

根据草稿 config.architectures 自动选择 v1 或 v2；没有重新训练草稿。
引擎只读取本地 config.json 和 safetensors，不执行仓库中的远程 Python。

## 使用

目标和草稿必须分别是本地完整目录。现有原生 MTP 参数继续有效。

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/models/Qwen3.5-9B",
    speculative_method="dflash",
    dflash_model="/models/Qwen3.5-9B-DFlash",
    dflash_num_speculative_tokens=7,
    tensor_parallel_size=2,
    max_num_seqs=4,
    max_num_batched_tokens=256,
    max_model_len=1024,
    enforce_eager=False,
    hybrid_cuda_graph_batch_sizes=(1, 2, 4),
)
try:
    result = llm.generate(["Explain speculative decoding."],
                          SamplingParams(temperature=0, max_tokens=64))
finally:
    llm.exit()
```

也可以从仓库根目录运行：

```text
python tests/dflash/example_dflash.py --model /models/Qwen3.5-9B --draft /models/Qwen3.5-9B-DFlash --draft-tokens 7 --tp 2 --batch 4 --graph
```

`--baseline` 禁用投机；`--temperature 0.8` 测试随机采样；`--prefix-cache` 开启联合前缀缓存。
示例入口输出的 elapsed 包含 Prefill，不能当成纯 Decode 延迟。

## 实际数据流

1. 读取草稿的 target_layer_ids；主模型执行对应层后保留输出，按指定顺序拼接。
   HF 的 hidden_states[i+1] 对应第 i 层输出，最后一层如果被选择则使用最终 Norm 后结果。
2. Prefill 每个 chunk 的特征经草稿 fc + 标准 Qwen3 RMSNorm 融合。
   每个草稿层把该上下文投影为独立的 K/V，并追加到对应请求的草稿缓存。
3. Decode 已有一个 pending token。构造 `[pending, MASK, ..., MASK]`，一次草稿前向得到多个候选。
   丢弃 anchor 的输出，使用各 MASK 位置的 Hidden State 经共享 LM Head 得到该位置的候选。
4. 主模型按照正常因果顺序验证 `[pending, draft_1, ..., draft_n]`。
   Greedy 使用最长匹配前缀；随机采样使用逐位置 p/q 接受及正残差分布。
5. 只提交实际处理的输入对应的 GDN 状态和多层特征。未接受的候选不进入草稿上下文。
   目标模型 Paged KV 的有效长度/块回收继续由已有 Scheduler 和 BlockManager 处理。

举例：pending 是“我”，候选是“想、去、北京”，主模型只接受“想”，随后给出“要”。
本轮输出“想、要”；已计算输入为“我、想”；GDN 和 DFlash 上下文只推进这两个输入。
“要”是下一轮 pending，尚未进入目标状态；“去、北京”的特征全部丢弃。

## 与 MTP 不同的地方

- 官方 9B 草稿配置是 6 层，读取主模型 `[1,5,9,13,17,21,25,29]` 共 8 层输出。
- DFlash 使用标准 RMSNorm 的 `weight`，目标 Qwen3.5 使用 `1 + weight`，不能混用。
- 草稿使用自己完整 Head 的 RoPE 配置；不能复用目标模型的部分旋转/mRoPE。
- 草稿因果性优先读取 is_causal；未指定时 sliding 默认因果、full 默认非因果。
  Qwen3.8 DFlash2 显式 is_causal=false，所以其 sliding 窗口内允许双向读取噪声块。
  目标模型验证始终保持正常因果性。
- 草稿缓存只保存已确认上下文的 K/V。当前 block 的 noise K/V 临时参与计算，不持久化。
- 原生 MTP Prefix Snapshot 保存 L-1 组移位输入和边界特征。
  DFlash Snapshot 保存 L 个已处理 token 的上下文 KV，不包含下一请求的后缀。
- 对外配置 DFlash 草稿数量；内部 mtp_num_speculative_tokens 作为通用调度预算沿用，
  mtp_stats 也沿用为通用投机计数，并增加 dflash_draft_forwards。

## TP、批量、Graph、Prefix

- 草稿 Q/K/V、gate/up 按输出切分；o/down 按输入切分并 AllReduce。
  fc、Norm 复制；共享目标 Embedding 和词表切分的 LM Head。
- 所有 rank 按同样请求顺序执行；Rank 0 收集 logits、采样、决定接受边界并广播。
- 不同请求可以有不同草稿长度；目标验证支持 packed/ragged，草稿目前逐请求运行。
- 目标验证复用固定地址 CUDA Graph，精确匹配 B、验证长度、上下文容量。
  多层特征作为该 Graph 的独立输出引用保留，避免 Eager Forward 覆盖 Python 属性后误读。
- 当前草稿本身使用 Eager PyTorch SDPA，不捕获 DFlash draft CUDA Graph。
- Prefix Cache 同时保存目标 KV、GDN Snapshot 和草稿 KV；草稿快照字节计入容量/LRU。
  恢复为请求私有副本，避免新请求更新污染共享前缀。
- 草稿权重、KV 和 Graph 均消耗额外显存；分配目标缓存前预留草稿上下文 KV 预算。

## 本地验证

```text
python -m unittest discover -s tests/dflash -p "test_*.py"
python -m unittest discover -s tests/mtp -p "test_*.py"
```

覆盖配置、checkpoint header/形状、上下文分块、因果/非因果掩码、MASK 位置预测、
随机草稿概率、packed 接受/拒绝的 GDN 和草稿提交边界、快照隔离、多层特征提取；
TP=2/4 为真实 CPU Gloo 多进程测试，并实际执行候选采样、验证与决策同步。

有 CUDA PyTorch 时：

```text
python tests/dflash/test_dflash_gpu.py
```

小 Hybrid Target 的真实 CUDA Graph capture/replay、请求顺序和输入更新、
部分接受后的 GDN/草稿状态一致性。使用 Reference GDN 后端，不替代 CUDA Extension Binding 验证。

官方数值对比需要单独取得可信的官方 `dflash/model.py`：

```text
python tests/dflash/compare_official.py --reference /path/to/official/model.py
python tests/dflash/compare_official.py --reference /path/to/official/model.py --device cuda
```

以相同的小权重比较 FP32/FP16/BF16，包含三次上下文续接及每层缓存。
官方 checkpoint 声明 Transformers 5.7；对照脚本在旧版 Transformers 下会明确打印
SDPA is_causal 兼容处理。该处理只作用于对照进程，不修改引擎和已安装的包。

真实权重环境的验收入口：

```text
python tests/dflash/validate_dflash_model.py --model /models/Qwen3.5-9B --draft-model /models/Qwen3.5-9B-DFlash --tp 2 --batch 4 --drafts 7 --require-graph
```

每项使用独立进程，对比 Baseline、DFlash Eager、DFlash Graph、DFlash Prefix 的 Greedy token，
检查实际 Graph Replay 与实际 Prefix Hit，输出 artifacts/dflash/model_validation.json。
只有实际运行通过后才能认为真实模型验收完成。

## 当前适用范围及未验证项

- Dense Qwen3.5、纯文本、BF16/FP16/FP32；1..15 个候选且不超过训练 block_size-1。
  图片请求明确拒绝；现有图文输入可继续使用原生 MTP。DFlash 2/量化权重未接入。
- 本地已执行小规模 CPU、3060 CUDA 和官方小权重数值对比。
- 没有运行完整 9B + DFlash 权重，没有验证 A800 多卡 NCCL Graph，没有测实际加速比。
- 草稿上下文 KV 暂用连续张量追加，全注意力层保留完整上下文；尚未改成 Paged KV。
  SDPA 显式构造窗口 mask，未接专门的 FlashAttention sliding-window 接口。
  这些会影响长上下文的时间/显存，不能据小模型验证宣称生产性能。
