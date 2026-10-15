# 本地验证记录（2026-10-15）

这些是小模型/小张量验证结果，不代表完整 9B 模型验收或性能结果。

| 检查 | 执行结果 |
| --- | --- |
| DFlash CPU suite | 15 项中 13 项通过，2 项 CUDA 测试在 CPU 环境跳过 |
| DFlash CUDA suite | 2 项通过，真实 CUDA Graph capture/replay |
| 原 MTP CPU suite | 51 项中 48 项通过，3 项 CUDA 测试在 CPU 环境跳过 |
| 原 MTP CUDA suite | 3 项通过 |
| 官方 DFlash 小权重数值对比，CPU | FP32/FP16/BF16 均通过，每种包含三次上下文续接及缓存比较 |
| 官方 DFlash 小权重数值对比，CUDA | FP32/FP16/BF16 均通过，每种包含三次上下文续接及缓存比较 |
| TP=2/4 | 已包含在 DFlash CPU suite；真实 Gloo 多进程，含随机/Greedy 草稿及验证提交 |
| Python AST | 本次相关 18 个文件通过 |
| diff whitespace | git diff --check 通过 |
| 运行/完整模型验收入口 | --help 通过；未执行完整权重验收 |

CUDA 对照环境安装的是 Transformers 4.52.4。官方草稿参考代码按新版 Transformers
的 SDPA 语义工作，旧版包装器忽略 module.is_causal=False，因此对照脚本显式传入该标记。
此调整仅限对照脚本，不修改安装环境或引擎。CPU 对照使用支持新版接口的环境。

未完成的硬件验收：完整 Qwen3.5-9B + 已训练 DFlash 权重的 Greedy 对齐、A800 NCCL 多卡
Graph、真实接受率、显存峰值和吞吐。没有启动租卡或下载完整模型，没有生成加速比。

## A800 一键套件补充验证

- 新增 6 项 benchmark harness 测试通过：接受率/推进长度口径、预热计数扣除、动态请求和输出顺序、
  输入/Greedy 差异检测、测试矩阵、失败退出/超时和日志保留。
- 完整 CPU 回归：MTP 51 项中通过 48 项；DFlash/测试套件 22 项中通过 19 项。
  共 67 项通过；其余为 CUDA/实际 C++ Binding 检查，在 CPU 环境跳过。
- CUDA 环境：DFlash 15 项全部通过（包括 2 项 Graph）；原 MTP 的 3 项 GPU 测试通过。
- TP=2/4 的 CPU Gloo 测试也检查了新增的跨 rank 显存报告协议；显存值本身由 GPU 环境采集。
- 一键 Bash 入口已通过语法检查、带空格路径的 DRY_RUN；各 Python 入口 --help 与 AST 通过。
- 本地没有通过 PyTorch 编译/执行 C++ Binding；服务器脚本默认先跑这个门槛，再加载完整模型。

## Qwen3.8-27B / DFlash2 补充验证（2026-10-15）

- CPU：`tests/mtp` 51 项，通过 48 项、跳过 3 个 CUDA；`tests/dflash` 33 项，
  通过 28 项、跳过 4 个 CUDA 与 1 个实际 Binding。合计 **76 项 CPU 测试通过**。
- 新增 v2 检查覆盖：真实 64 层配置、48/16 层路由、真实尺寸 TP=2 Attention 投影、
  meta 上构造完整尺寸草稿、81 个真实 safetensors header 名称/形状、两点动态卷积标量对照、
  前驱条件 q、无 `.weight` 后缀的码本加载、接受边界、无 GPU 下载计划。
- TP=2/4：真实 CPU Gloo 多进程分别检查 v1/v2，含逐层输出、KV 分片、随机/Greedy
  起草、跨 rank 接受边界一致性和显存采集协议。
- RTX 3060 Laptop：4 项小 Hybrid Target CUDA 测试通过（v1/v2 各 2），
  包含真实验证 Graph replay、动态输入/请求顺序、部分接受后状态恢复。
- 与官方 `z-lab/dflash/dflash/model.py` 的小权重对照：CPU、CUDA 均通过
  FP32/FP16/BF16 × 三次上下文续接，共 18 个前向组合；同时核对 Greedy 路径和条件采样 q。
  FP32 最大绝对误差约 1.5e-8，低精度该组样例为 0；不是完整模型的误差界。
  参考源码 Git blob 为 `7821d66a7b932b4e0f15381a39f76ba1d7eceb8a`。
- CUDA 对照使用旧 Transformers 4.52.4，所以对照进程显式传入草稿 is_causal 标志，
  避免旧 SDPA wrapper 自动推断为 causal；没有修改引擎或已安装包。
- 主模型公开索引有 1199 个 tensor，主要命名沿用 `model.language_model.*`、
  `model.visual.*`、`lm_head.*`、`mtp.*`；DFlash 模式仍忽略原生 `mtp.*` 权重。
- 无卡下载入口 dry-run、Bash 语法、带空格路径启动计划均检查；不在本地下载完整权重。

还没有完成：完整 27B 的 HF/引擎 Greedy 对齐、A800 NCCL、实际 C++ Binding 编译运行、
接受率和吞吐。服务器脚本准备好了这些验收所需的引擎基线/投机对比和 Binding 门槛；
不能把本地小模型通过写成完整 27B 测试通过。
