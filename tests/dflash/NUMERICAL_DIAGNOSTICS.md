# Fixed-history numerical diagnostics

本入口仅定位，不输出性能或正确性 PASS，不修改普通生成算法。需要已经下载的主模型与
之前的 `cases/summary.json`，不加载草稿模型。当前仅支持内置短 prompt、单 KV 页请求。

```bash
CUDA_VISIBLE_DEVICES=0,1 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=2 \
python tests/dflash/diagnose_numerics.py \
  --model /root/autodl-tmp/models/Qwen3.8-27B \
  --results artifacts/dflash-a800/20261016_003713/cases/summary.json \
  --batch 16 --trial 0 --drafts 3 --probe-at 0 5 11 26 39 89 100 \
  --output artifacts/dflash-a800/numerics-b16.json
```

小 batch 可另跑 `--batch 1 --drafts 3 --probe-at 0 26`，使用另一个输出文件。
不要使用 torchrun；引擎负责 TP worker。probe 编号表示正在计算的 completion token 的
零基下标；该步预测下一个下标。例如 probe=26 会检查输出 token 下标27的 logits。

五条路径从同一个 GDN checkpoint 和历史 KV 开始：

- `decode_eager`：普通直接寻址 Decode，作为基线。
- `decode_graph`：现有捕获的 Decode Graph。
- `decode_static_eager`：使用同一静态 Workspace/metadata，但通过 Python 直接执行模型。
- `verify_one`：无草稿的单token功能式验证。
- `verify_block`：强制使用普通基线的后续token作为候选；比较anchor输出，并把GDN状态重放至
  同一个anchor边界，避免误把整段验证最终状态和单token状态比较。

记录每层输出差异、最终hidden/logits、top2间隔、argmax变化、每层Conv/Recurrent差异，
以及两个rank的本地状态结果。Graph本体没有重新捕获，所以 `decode_graph` 无逐层hook；
它记录最终hidden/logits与状态。静态Eager路径用于补充定位metadata或kernel选择差异。

非probe位置仅执行普通Eager；所有后续输入强制使用保存的普通基线token。
完成每次比较后只保留普通Eager的一步GDN更新，移除hook并恢复方法/执行选项。
候选只写当前页中历史边界之后的KV槽位，下次会覆盖；不写历史KV前缀。

## 已确认的代码差异与待验证假设

1. 普通Eager的block table按当前页数构建，这批请求为1列；普通Graph按1024最大上下文
   分配4列。FA2.8.3的KV-cache入口以表宽×页大小作为容量，自动SplitKV heuristic使用该容量。
   在108SM、局部Q=12/KV=2、D=256条件下，估计B=16分别为4/6个split、每split64/192个K token；
   B=1/4虽然split数不同，但分段宽度均为64。与已有输出模式相符，仍需实际同状态对照。
   报告中的split数量是官方算法的估计，不是Nsight或运行库回报值。
2. 原始GDN Split-K与短序列验证的归约顺序不同；原CUDA kernel启用FMA，验证扩展使用
   `--fmad=false`，Conv也分别使用显式fmaf与普通乘加。已经确认实现不同，尚未确认哪部分
   导致完整模型argmax翻转；不能仅凭这一点排除状态错误。
3. 若静态Eager更接近Graph且从首个Full Attention层开始偏离普通Eager，应重点验证
   表宽与SplitKV选择；若verify_one在首个GDN层即偏离，应重点检查Conv/Recurrent数值。
   若verify_block的重放anchor状态与verify_one在同层不一致，需要继续查捕获与重放边界。

参考：[FlashAttention v2.8.3 C++入口](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/csrc/flash_attn/flash_api.cpp)。

本地小张量测试只验证诊断器指标、路径接线和状态恢复，不能替代A800/27B的定位结果。
