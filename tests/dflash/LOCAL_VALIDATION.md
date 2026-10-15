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

## A800 扩展编译错误修复（2026-10-16）

- 服务器实际编译暴露：PyTorch 扩展定义了 `__CUDA_NO_HALF_CONVERSIONS__` 和
  `__CUDA_NO_BFLOAT16_CONVERSIONS__`，`verify_conv` 的两处浮点写回不能调用隐式构造函数。
- 写回改用显式 `__float2half_rn` / `__float2bfloat16_rn`，FP32 写回保持原样。
  本地 NVRTC 检查现在使用同样的四个 PyTorch half/BF16 限制宏。
- 本地 CUDA 12.8 / RTX 3060 Laptop：8 组状态重放 Graph 检查、12 组融合验证检查通过，
  包含 FP32/FP16/BF16、非连续输入和 Graph；这些运行也编译了全部输入类型实例。
- 此轮没有在本地通过完整 PyTorch C++ Binding 或执行完整 27B；仍需服务器重新编译并验收。

## 聊天模板返回类型兼容修复（2026-10-16）

- A800 基线预热请求遇到 `prompt must be ...`：新版 Transformers 的聊天模板默认返回
  `BatchEncoding`，测试 worker 原先把它直接当作 token 列表传入引擎。
- worker 与 preflight 共用 `chat_ids`，显式设置 `return_dict=False`、`return_tensors=None`，
  并检查非空的一维非负整数列表，避免在完整模型加载后才发现不支持的输入类型。
- 在 Transformers 5.7.0 / 4.52.4 中分别运行 benchmark harness 的 8 项检查，全部通过。
  新增检查使用内存中构造的真实 fast tokenizer，核对 thinking 开关、实际 token ID 和请求提交；
  无需模型权重或网络。另检查非法返回格式会立即失败。
- 这里只验证了输入兼容性；完整 27B / A800 用例仍需服务器重跑。

## 可选性能继续模式（2026-10-16）

- A800 的 B=1 / 草稿 3 / Eager 在第 4 个 trial 出现 Greedy token 差异。
  差异原因尚未定位，不能认定只是低精度舍入或认定已经通过正确性。
- 增加默认关闭的 `ALLOW_TOKEN_MISMATCH=1` / `--allow-token-mismatch`：仅性能用例的
  token 差异可降为告警继续；状态管理输出差异、运行错误、输入和请求数错误仍然失败。
- 保留严格 `passed=false` 与全部错误；另外输出 `performance_completed`、
  `token_consistency_passed`、`fatal_errors`、`warnings`。有差异时报告 WARN，不输出整体 PASS。
- 本地 10 项 harness 检查通过，含模拟 worker 生成真实报告的完整调度检查、
  告警/致命错误分类与报告标记；Bash 语法和 full 模式开关透传 dry-run 通过。
- 没有更改模型、采样、接受规则或状态提交，也没有重新运行完整 27B。

## 显式选择原 Decode CUDA 后端（2026-10-16）

- 套件、preflight、worker 增加 `--gdn-decode-backend`，默认 `state_aware_cuda`；
  与短序列投机验证/重放的 `--backend` 独立。普通 Eager/Graph 均显式传入引擎。
- 每个用例和环境报告记录选中的 Decode 后端，worker 启动后核对 Config；
  报告拒绝把不同 Decode 后端直接作一致性通过。
- 启动完整权重之前新增原 Conv/Recurrent 绑定测试（B=1/4/16，H=24，Dk/Dv=128，
  C=5120），覆盖多步更新、变动槽位、固定输出、未寻址状态与 Graph。
- 本地 12 项 harness 检查通过，含配置/参数透传、禁止混用后端与报告策略回归；
  完整启动的 Bash 语法和双 CUDA 开关 dry-run 通过。绑定测试在 CPU PyTorch 环境明确跳过。
  原 CUDA 核心没有修改，真实 C++ 编译与新局部形状数值检查仍需在 A800 执行。

## 输出分歧定位入口（2026-10-16）

- 实际回传批次 `20261016_003713` 已完成全部20组，原state-aware CUDA Binding与重放
  检查通过，状态管理用例通过；性能用例仍有Greedy差异，未通过完整一致性。
- 新增显式调用的固定历史诊断方法，不在普通生成中执行；比较普通Eager、现有Graph、
  静态Workspace Eager、单token验证和多token验证的anchor输出及同边界GDN状态。
- 确认普通Decode的动态block table与静态Graph表宽不同。FA2.8.3源码以表宽估计容量并
  自动选SplitKV；108SM假设下，B=16的K分段宽度估计64对192，与B=1/4的模式不同。
  这是具体的数值路径假设，还未在完整27B运行时确认，不宣称已经修复根因。
- 本地7项诊断器小测试通过，检查top2/非有限值、anchor位置、异常/成功后的状态恢复、
  hook清理、静态路由、SplitKV heuristic估计。无完整权重下载、无27B实际诊断结果。

## A800数值报告复核与单因素页表对照（2026-10-16）

- 已复核用户回传的`numerics-b16.json`：7处固定历史、2个rank。Graph与静态Workspace
  Eager的相对基线差异摘要相同，首个差异位于Full Attention的decoder.3；验证路径更早
  在GDN偏离，观察到的argmax翻转均涉及相差0.125或并列的top2候选。
- 诊断器新增`decode_padded_table`，保持普通Eager，仅扩大页表宽度，不增加请求/token，
  不改变有效历史长度/页ID；直接比较其hidden/logits与Graph。该对照仍需A800运行，
  不能把旧报告的相同误差摘要当成逐元素相同或声称根因已修复。
- 本地9项诊断器测试通过，新增元数据padding保留有效页/长度/slot、异常恢复prepare_decode
  和GDN checkpoint等检查；正常推理、CUDA kernel和接受/回滚规则未改。

## 混合模型Decode统一页表容量（2026-10-16）

- A800单因素对照回传：input=26/89处，普通Eager仅把页表扩宽，即与Graph的hidden和
  logits逐元素相同，input=89的argmax翻转也复现。确认了这些位置的容量敏感路径。
- 混合模型普通Decode按max_model_len/block_size向上取整固定表宽，和已有Graph一致；
  Eager-only及非bucket回退也采用相同规则。保留真实长度、页ID与slot mapping；
  不增加KV页，不补虚假请求/token。Prefill/投机验证、非混合Decode保持旧布局。
- 新增7项实际ModelRunner元数据方法的小测试（仅替换pinned分配/CUDA传输），覆盖
  Eager开关、跨页、mRoPE delta、请求重排/结束、非bucket B=3、向上取整、拒绝截断有效页
  以及Prefill/非混合兼容；7项通过。原9项诊断器测试也通过。
- 服务器套件加入元数据测试；诊断器记录实际表宽与逻辑页数，供A800复核修正后的普通
  Eager/Graph对齐。该修正尚无完整27B运行结果，未宣称投机一致性问题已全部解决。

## 普通Decode A800复核与GDN子步骤诊断（2026-10-16）

- 用户回传修正后的input=26/89：Graph、页表对照、静态Eager均相对普通Eager
  hidden_max=0、argmax_diff=0；verify_one仍有hidden差异，input=89的verify_block有1个
  argmax差异。仅确认这两个固定历史检查点的普通Decode对齐。
- opt-in诊断增加首个GDN的11处子步骤观察，保留数学操作；记录真实执行顺序下的首个
  不同位置，区分输入投影、Conv激活、递推输出与门控RMSNorm/输出投影。
- 本地13项诊断器测试通过，新增Decode/packed/flattened-head对齐、观察不改变输出、
  注入Conv输出误差定位到conv.output、异常后恢复既有方法覆盖与清理hooks。
  这些是诊断器的小测试，不是完整GDN/27B的精度通过。
