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

六条路径从同一个 GDN checkpoint 和历史 KV 开始：

- `decode_eager`：普通直接寻址 Decode，作为基线。
- `decode_graph`：现有捕获的 Decode Graph。
- `decode_padded_table`：普通Eager仅把Block Table补到静态表宽；不改变真实请求数、
  token数、context length、有效页ID或GDN状态。另记录与Graph的hidden/logits直接比较，
  用于区分表宽和其他静态Workspace因素。这是页表容量对照，不是batch padding实现。
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

## 已回传A800结果与下一步（2026-10-16）

`numerics-b16.json`完成7个固定历史位置、每处B=16的对照。实际GPU为108SM，动态/静态
表宽1/4。前三个GDN层的Graph状态与普通Eager一致；出现偏离时静态Eager的第一个差异
位于decoder.3（第一个Full Attention），后面的GDN状态才开始不同。Graph与静态Eager
相对基线的全部hidden/logits/state差异摘要相同；旧报告没有两者直接的逐元素比较。

Graph在预测index=90、row14处，top2从28.25/28.125变成28.125/28.125而改变argmax。
verify_block在预测index=40的row6/15、index=101的row3/12处也改变argmax，基线top2
分别27/27、21.5/21.5，验证路径变成相差0.125。BF16在[16,32)间的间隔正是0.125。
这说明观察到的翻转与近邻/并列候选相关，不等于完整模型或回滚已经验证正确。

验证路径第0层Conv状态与普通Decode完全一致，Recurrent状态最大绝对差约1.9e-6～5.8e-6，
相对L2约3e-8～5e-8；最终hidden相对L2约1%～1.8%。仍需分离Conv激活、Recurrent输出、
不同token行数的线性投影与归约/FMA路径。原报告不能单独证明回滚错误或排除所有状态问题。

先用新增`decode_padded_table`只复查`--probe-at 26 89`，输出另一个JSON；
若只改变表宽就与Graph逐元素一致，才能进一步锁定Attention表容量/内部计算选择。
该检查没有改变正常推理算法、采样或CUDA kernel。

## 页表容量对照结果及修正

A800回传`numerics-b16-table.json`的终端输出显示，在input=26/89两处，普通Eager
仅增加页表列数，即与Graph的hidden、logits逐元素相同；input=89的argmax翻转也复现。
因此这些检查点的普通Decode分歧可由页表容量变化复现。实际SplitKV数量仍是源码估计，
没有用设备跟踪直接测量；该结果也不代表投机验证和回滚已全部对齐。

混合模型`prepare_decode`现在统一使用`ceil(max_model_len / block_size)`列，与已有
Graph Workspace相同，覆盖Eager-only、Graph和非bucket回退。有效context length、
物理页ID、slot mapping仍按真实请求构造，多余列为-1，不分配额外KV页或虚假token。
Prefill/投机验证仍使用原来的动态页表；普通非混合Qwen3的Decode保持原有布局。
这是统一普通Decode路径的数值条件，不是非标准batch的Graph padding，也不改变GDN内核。

诊断报告现在区分`logical_table_columns`（真实页数）与`dynamic_table_columns`（普通
Eager实际传入的列数）。修正后本例应为logical=1、dynamic=4、static=4；重新运行
`--probe-at 26 89`检查Graph相对新普通Eager的差异。预期hidden/logits逐元素相同，仍需
A800验证。旧性能数据不能当作修正后的性能结果，验证路径的GDN差异仍待定位。

## 普通Decode复核与首个GDN细分观察

修正后用户回传：input=26/89的`decode_graph`、`decode_padded_table`和
`decode_static_eager`均为hidden_max=0、argmax_diff=0。该两个位置的普通Decode一致，
不能外推为完整生成、所有batch或投机验证通过。input=89的verify_block仍有1个argmax差异。

诊断版本`first_gdn_stages_v3`只对第一个GDN增加11处观察点，保留原计算：

- `input`：GDN输入，可区分更前面的Embedding/Input RMSNorm与GDN内部差异。
- `in_proj_qkv`、`in_proj_z`、`in_proj_b`、`in_proj_a`：输入线性投影。
- `conv.input`、`conv.output`：Conv原始输入和卷积+激活后的输出，区别于历史Conv状态。
- `recurrent.output`：进入门控RMSNorm前的递推输出；该误差也可能从Conv/投影传入。
- `norm.gate`、`norm.output`：归一化的门控输入和最终结果。
- `out_proj.output`：GDN输出投影及其TP聚合后的结果。

记录`first_gdn_stages`与`first_nonidentical_gdn_stage`。每处仅取同一请求的anchor，
将Decode的[B,1,C]、验证的[1,B*L,C]、归一化的[tokens*heads,D]对齐到[B,C]。
观察点按实际执行顺序记录；现有Graph不会执行Python hooks，因此Graph的该列表为空。
非probe步骤不安装这些观察点，成功/异常退出都会清理hooks和临时Conv方法包装。

下次仍使用原诊断命令，建议`--probe-at 0 26 89`，输出新文件`numerics-b16-gdn.json`。
不要因为first_gdn=`recurrent.output`就直接认定Recurrent算子错误；需要先确认前面的
input、QKV投影、conv.output以及g/beta投影一致。当前只定位，不更改浮点运算或接受规则。
