# 2×A800 一键测试 DFlash

本套件负责：真实权重 Greedy 对齐、接受率、推进长度、端到端/Decode 吞吐、TTFT/TPOT、
每个 rank 的显存、前缀命中和动态请求状态检查。它不会自动安装依赖或租用机器。

## 1. 拉取测试分支

```bash
cd /你的路径/NanoHybrid-VLM
git fetch origin
git switch codex/dflash-a800-tests
git pull --ff-only
```

首次 switch 会自动追踪 origin 上同名分支。无需合并到 main 后才能测试。

## 2. 使用原来能运行 Qwen3.5 的 CUDA 环境

```bash
conda activate 你的CUDA环境
python -m pip install -e . --no-deps
```

保留原有 torch/CUDA/flash-attn/FLA 组合，不要为测试盲目升级 torch。
需要支持 Qwen3.5 配置的 Transformers，以及 flash-attn、flash-linear-attention、
causal-conv1d、xxhash、ninja。默认 CUDA 后端还需要匹配 PyTorch 的 CUDA Toolkit 和 C++ 编译器。
安装缺失依赖时应沿用该服务器原本可工作的环境。

## 3. 先运行 smoke

如果目标权重已经在 `/workspace/models/Qwen3.5-9B`，草稿尚未下载：

```bash
DOWNLOAD_DRAFT=1 bash tests/dflash/run_a800.sh \
  /workspace/models/Qwen3.5-9B \
  /workspace/models/Qwen3.5-9B-DFlash
```

`DOWNLOAD_DRAFT=1` 明确下载 z-lab/Qwen3.5-9B-DFlash；只下载草稿，不重新下载目标模型。
下载结束后，所有测试强制离线读取本地权重。草稿已存在时去掉 DOWNLOAD_DRAFT=1。
目标目录应包含完整权重、config、tokenizer/processor 文件。

默认只使用可见 GPU 0、1，TP=2。**不要加 torchrun**，引擎自己启动另一个 TP 进程。

Smoke 默认测试：

- B=1、4，每轮最多 3 个草稿；每个性能 case 3 次测量，输出固定为 64 tokens。
- Baseline Eager、Baseline Graph、DFlash Eager、DFlash Graph。
- 额外 B=2 的前缀复用和动态接入检查，包含请求长度不同、完成后槽位复用。
- 每个测量批次前，使用相同输入及输出预算进行预热。
- 每个 case 使用独立进程；异常或超时会清理它自己启动的 TP 进程组，保留日志并停止。
- CUDA C++ Binding 数值/Graph 检查在完整模型测试之前执行；失败不会悄悄改用 Reference。

## 4. 看结果

终端会打印输出目录，默认为：

```text
artifacts/dflash-a800/时间戳/
├── run.log
├── environment.json
└── cases/
    ├── manifest.json
    ├── summary.md
    ├── summary.json
    ├── b1_dflash3_graph.log
    ├── b1_dflash3_graph.json
    └── ...
```

优先看 `cases/summary.md`，详细数据在 `summary.json`。summary.json 的 passed=true 才表示
所有计划项完成并通过 Greedy 对齐；中途失败时仍有已完成项的报告。
失败日志也保留在同一目录。**本地小测试通过不代表这次 A800 测试通过。**

报告口径：

| 指标 | 定义 |
| --- | --- |
| acceptance_rate | 接受的草稿 token 总数 / 实际起草 token 总数，不包含额外生成的 bonus |
| mean_emitted_tokens_per_request_round | 本轮产生的输出 token 数 / 请求级验证轮数，包含 correction/bonus |
| E2E tok/s | 输出 token 总数 / 从入队到请求完成的时间，包含 Prefill，排除加载/预热 |
| Decode-only tok/s | 只有 Decode、没有 Prefill 的同步 step 的输出数 / 这些 step 的总耗时 |
| TTFT | 每条请求到第一批输出可用的时间 |
| TPOT | 每条请求 (末 token 时间 − 首 token 时间)/(输出数 − 1) |
| memory_per_rank | 各卡 peak allocated/reserved、KV/GDN/草稿参数等，采集在计时区间之外 |

投机会成批产生 token，因此不使用重复时间戳去伪造逐 token ITL 的 P95。
性能 case 采用 Prefill-first 方便区分两个阶段；动态接入检查采用 Decode-first。
计时 step 在 Rank 0 同步 CUDA；各卡显存采集通过 TP 集体通信，在计时区间外同步。
这是固定并发批次的测试，不等同于带到达率的线上开放式服务压测。

两个对比都报告：DFlash Eager 对 Baseline Eager、DFlash Graph 对 Baseline Graph。
另给出相对 Baseline Eager 的总收益，避免把 Graph 收益全部算到 DFlash 上。
如果测量期间仍捕获新 Graph，trial 中 measurement_contains_graph_capture 会明确标记。

内置 workload 是少量可复现的问答、代码、数学题，用于 smoke 和初步接受率估计，
不应直接作为标准数据集结果或简历加速比。

## 5. Smoke 通过后再做完整矩阵

```bash
SUITE_MODE=full bash tests/dflash/run_a800.sh \
  /workspace/models/Qwen3.5-9B \
  /workspace/models/Qwen3.5-9B-DFlash
```

Full 默认 B=1、4、16，草稿数 3、7；每个性能 case 5 次测量，输出 128 tokens。
需要更长统计时可自行增加 REPEATS、OUTPUT_TOKENS。每次使用新输出目录。

自定义一组测试：

```bash
BATCHES="1 4 16" DRAFTS="3 7 15" REPEATS=5 OUTPUT_TOKENS=128 \
  bash tests/dflash/run_a800.sh /workspace/models/Qwen3.5-9B /workspace/models/Qwen3.5-9B-DFlash
```

自定义业务问题使用 JSONL，每行：

```json
{"id":"question-1","task":"qa","prompt":"请解释什么是张量并行，并举例。"}
```

```bash
PROMPTS_JSONL=/workspace/my_prompts.jsonl THINKING=1 \
  bash tests/dflash/run_a800.sh /workspace/models/Qwen3.5-9B /workspace/models/Qwen3.5-9B-DFlash
```

每个测量批次会保存输入 token 的 hash、长度、任务类型、完整输出 token，便于追查差异。

## 6. 常见失败

- 缺少 Qwen3.5 配置支持：检查 environment.json 的 Transformers 版本及完整错误。
- nvcc/编译器或 CUDA 版本不匹配：先修复环境；也可以先用下面的 Reference 模式验收功能。
- Greedy 不一致：查看 summary.json 的第一个不同 token 和对应 case 日志，不会当成功继续测。
- OOM：先用 BATCHES="1"，然后检查每卡可用显存和残留外部任务。脚本不会杀其他人的任务。
- 卡死：CASE_TIMEOUT 默认每 case 1200 秒；可通过环境变量调整。超时保留日志，返回失败。
- 端口 2333 或共享内存 nanovllm 已被占用：不要在同一台机器同时跑两套当前引擎测试。
- 下载失败：先完成草稿下载，再在离线测试环境运行；下载不属于性能计时。

只检查正确性的 Reference 回退：

```bash
BACKEND=reference RUN_BINDING_TESTS=0 BATCHES="1" \
  bash tests/dflash/run_a800.sh /workspace/models/Qwen3.5-9B /workspace/models/Qwen3.5-9B-DFlash
```

Reference 结果可以检查接受和状态管理，但不能冒充 CUDA 优化算子的性能。

只预览命令，不使用显卡、不加载权重：

```bash
DRY_RUN=1 bash tests/dflash/run_a800.sh /models/target /models/draft
```

结束后把整个结果目录打包发回来即可；无需发送模型权重。
