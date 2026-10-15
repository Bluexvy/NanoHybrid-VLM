# Qwen3.8-27B + DFlash2：无卡下载与 2×A800 测试

分支：`codex/qwen38-dflash2`。沿用 `speculative_method="dflash"`，通过草稿的
`architectures=["DFlash2DraftModel"]` 自动启用第二代，不需要新的 speculative_method 名称。

## 1. 无卡模式先下载

确认下面的 `/workspace/models` 是开 GPU 后仍保留的数据盘。推荐准备至少 100GB 可用空间；
保留旧模型、下载缓存和测试环境时再增加余量。此阶段不需要 torch、CUDA 或 GPU。

```bash
cd /你的路径/NanoHybrid-VLM
git fetch origin
git switch codex/qwen38-dflash2
git pull --ff-only
python -m pip install -U huggingface_hub
python tests/dflash/download_qwen38.py --root /workspace/models
```

下载的配套模型固定为：

- `Qwen/Qwen3.8-27B` → `/workspace/models/Qwen3.8-27B`
- `incoai/Qwen3.8-27B-DFlash2` → `/workspace/models/Qwen3.8-27B-DFlash2`

脚本只下载 safetensors 与配置、tokenizer/processor 等所需文件，不下载/执行远程 Python。
每次先解析仓库 revision，再固定该 revision 下载；保存实际权重字节数和 revision 到
`*.download.json`。中断后重跑会利用 Hugging Face 的缓存/续传。
可用 `--only target` 或 `--only draft` 分开下载；`--dry-run` 只打印计划且不安装依赖、不联网。

只有模型文件下载完，才需要启动 GPU。下载模型不等于完整推理已经验收。

## 2. 切回 GPU 后先跑 smoke

使用之前可运行 Qwen3.5 的 Linux CUDA 环境。不要为了下载模型升级 torch/CUDA。
需要能解析 `qwen3_5` 配置的 Transformers；本地 CPU 配置检查使用 Transformers 5.7。
原有 flash-attn、FLA、causal-conv1d 及 CUDA Extension 编译依赖仍然需要。

```bash
python -m pip install -e . --no-deps
bash tests/dflash/run_qwen38_a800.sh \
  /workspace/models/Qwen3.8-27B \
  /workspace/models/Qwen3.8-27B-DFlash2
```

默认 GPU 0、1，TP=2，B=1、2，起草 3 个 token，最大上下文 1024，输出 64，三次测量。
先跑原有小张量回归与实际 C++ Binding 检查，再跑普通/投机 × Eager/Graph，
最后检查前缀命中、动态接入和状态释放。**不要加 torchrun**，引擎自己启动 TP worker。

通过后测完整 7 个候选（该草稿 block_size=8，含一个已确定的 anchor）：

```bash
SUITE_MODE=full BATCHES="1 4 16" DRAFTS="3 7" \
  bash tests/dflash/run_qwen38_a800.sh \
  /workspace/models/Qwen3.8-27B \
  /workspace/models/Qwen3.8-27B-DFlash2
```

报告在 `artifacts/dflash-a800/时间戳/cases/summary.md` 和 `summary.json`。
接受率、E2E/Decode-only 吞吐、TTFT/TPOT、每卡显存和计时口径沿用 [A800_GUIDE.md](A800_GUIDE.md)。
报告新增 draft_architecture，明确记录本次是否真正加载 DFlash2。
任何数值、状态、Graph 或编译失败都停止并保留日志，不当作性能结果。

可先不挂 GPU 检查启动参数：

```bash
DRY_RUN=1 bash tests/dflash/run_qwen38_a800.sh /workspace/models/Qwen3.8-27B /workspace/models/Qwen3.8-27B-DFlash2
```

## 3. 主模型的尺寸适配

Qwen3.8-27B 继续使用 `Qwen3_5ForConditionalGeneration` / `model_type=qwen3_5`。
已有主模型与状态池主要就是配置驱动，没有另造一个写死 27B 尺寸的实现。
此次补充真实配置、真实权重头和 metadata-only 参数构造，检查：

| 项目 | 全局 | TP=2 每卡 |
| --- | ---: | ---: |
| Decoder 层数 | 64（48 GDN + 16 Full Attention） | 同样执行全部 64 层 |
| Hidden Size | 5120 | 5120 |
| FFN Intermediate | 17408 | 8704 |
| Full Attention Q/KV heads | 24 / 4 | 12 / 2 |
| Attention Q+gate 投影权重 | `[12288,5120]` | `[6144,5120]` |
| Attention O 投影权重 | `[5120,6144]` | `[5120,3072]` |
| GDN Key/Value heads | 16 / 48 | 8 / 24 |
| GDN K/V head dim | 128 / 128 | 128 / 128 |
| Conv channels | 10240 | 5120 |
| 每请求每层 Recurrent State | `[48,128,128]` | `[24,128,128]`，FP32 |

层类型读 `layer_types`，不写死 24/8；全局层号映射到紧凑的 GDN/KV 层号。
主模型选择的特征层为 `[5,19,33,47,61]`，每次拼接得到 `[tokens,25600]`，
草稿的 fc 投影回 5120。读取的是这些 decoder 层的输出，遵循 HF hidden_states 的 +1 偏移。
Full Attention 计算沿用官方当前 forward 的 sigmoid 输出门控；不能只看 config 的
`output_gate_type="swish"` 字样就擅自把该运算换成 SiLU。

## 4. DFlash2 新增计算

1. 草稿是 5 层、Q/KV=32/8、head_dim=128，使用自己的 RoPE；与主模型 24/4、256 区分。
   Mask ID=248070，窗口=2048，显式非因果 draft attention；目标验证仍然是因果。
2. 每层 Attention 和 MLP 前后加入两点动态卷积。每 16 通道共享动态修正，
   post 卷积使用 pre 输入生成的另一组动态核，不从 post 输出重算。
   卷积仅作用于本轮 `[anchor,MASK...]`，首位置左侧补零；anchor 的特征会传到下一位置。
   它不修改目标 GDN Conv Pool，也不引入跨请求的卷积历史。
3. 每个 MASK 经过共享 LM Head 得到 logits，Rank 0 保留 Top-16，
   用 rank=256 的前驱/后继码本与隐藏特征计算相邻候选分数，然后选择一条路径。
   没有再次运行草稿 backbone，也没有做树形目标验证。
4. 随机采样保存每步**以已选前驱为条件**的 q 分布，Top-16 外概率为零。
   通用验证器用 min(1,p/q) 接受，拒绝后从 `(p-q)+` 重采样；不能继续用第一代的独立 q。

Checkpoint 里码本参数叫 `candidate_selector.predecessor_codebook` 和
`successor_codebook`，没有 `.weight` 后缀。加载和严格形状检查均按真实文件头处理，
不会漏加载 selector 后假装启用 DFlash2。

## 5. TP、状态和 Graph

- 草稿 QKV、gate/up 分片，o/down AllReduce；Conv、fc、Norm 和 selector 参数复制。
  selector 仅由 Rank 0 执行，候选 token 广播给各卡；复制参数显存已计入模型加载。
- KV/GDN 回滚沿用共享验证器。验证期间的候选不更新原始 GDN Pool；
  部分接受时只重放有效输入，目标 KV 有效长度与草稿 context KV 一起推进。
- 草稿只持久化已确认目标特征生成的 context KV；noise KV、卷积中间值和候选路径不持久化。
  前缀恢复仍需复制私有草稿 KV，避免后缀污染。
- 目标验证可捕获 CUDA Graph；DFlash2 草稿目前是逐请求 Eager PyTorch/SDPA，
  selector 在 Rank 0 顺序选择路径。没有声称完成融合 draft kernel 或树形投机。
- 此阶段的 DFlash2 仍是**纯文本投机**；原有多模态主模型路径保留，图片请求不走此草稿。

## 6. 本地验证与尚待 A800 验收

见 [LOCAL_VALIDATION.md](LOCAL_VALIDATION.md) 中 Qwen3.8/DFlash2 补充。
真实配置与 81 个权重 tensor 的 header 匹配；没有在本地下载或加载完整 27B 权重。
小模型数值、状态管理和 Graph 测试通过，不等于实际 27B、A800 NCCL 和接受率已验收。
真实速度、接受率和可用最大并发，由上面的服务器测试产生。

参考：https://github.com/z-lab/dflash/blob/main/dflash/model.py （MIT；许可证在 third_party）
以及官方目标/草稿 config。没有修改已安装的 Transformers 或执行模型仓库的远程代码。
