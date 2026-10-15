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
