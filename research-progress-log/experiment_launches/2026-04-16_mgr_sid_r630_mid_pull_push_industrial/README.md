# 2026-04-16 `R630` Mid-Only Pull/Push（仅中层拉近/推远） Industrial

Status（状态）: `snapshot（快照）`

## Scope（范围）

这是 selective separation（选择性分离）方向在 `S000` 之后的第一组正式重启实验。

这次不再沿用 diagnostics gate（诊断门）推进逻辑，而是直接把方法收敛到一个更简单的 objective（目标）：

- `L_base`：原始 `RQ-VAE` 重建与量化目标
- `L_pull`：只在 `L2`（第 2 层）做 `G_mid`（中尺度图）拉近
- `L_push`：只在 `L2`（第 2 层）做 selective separation（选择性分离）推远

## Why This Stage Exists（为什么做这个阶段）

当前主线的关键缺口已经比较明确：

- base `v2`（基础 `v2`）能表达 `who should be close`（谁应该靠近）
- 但还不能显式表达 `who should not stay too close`（谁不该继续过近）

同时，上一轮 `R610a` 也暴露了另一个问题：

- 如果把 selective separation（选择性分离）直接叠在完整 `v2` 多项损失上，很难判断到底是哪一项在起作用

所以这次阶段的目标不是继续堆 loss（损失），而是把方法压缩成一个 clean comparison（干净对比）：

- `R630a`：只有 pull（拉近）
- `R630b`：只有 push（推远）
- `R630c`：pull + push（拉近 + 推远）

## Method Definition（方法定义）

### 1. Pull Graph（拉近图）

只使用 `G_mid = fagsp_mid_base`（中尺度图）：

- 不再对 `L1 / L3`（第 1 / 第 3 层）施加图约束
- 不再附加 semantic retention（语义保持）辅助损失
- ambiguity-aware weighting（歧义感知加权）只保留在 pull（拉近）项上

### 2. Push Pair Source（推远物品对来源）

物品对只从 `semantic-near + mid-graph-weak`（语义接近 + 中图弱连接）中构造：

- 先取 `semantic kNN`（语义近邻图）候选对
- 再用 `fagsp_mid_base`（中尺度图）筛出弱连接对
- 不再混入 `G_coarse / G_local`（粗粒度图 / 局部图）
- 不再使用 user overlap（用户重叠）启发式

具体定义：

- `A_mid = sym(keep_topk(fagsp_mid_base))`
- 若 `0 < A_mid(i,j) <= tau_q`，则 `(i,j)` 进入 weak pair（弱连接物品对）
- 其中 `tau_q` 是 `A_mid` 正边权重的 `q = 0.25` 分位数
- pair reliability（物品对可靠性）定义为：
  - `r_ij = s_ij * (1 - min(A_mid(i,j) / tau_q, 1))`
  - 其中 `s_ij` 是语义候选边权重

### 3. Separation Scope（分离作用范围）

- selective separation（选择性分离）只作用在 `L2`（第 2 层）
- 不做 `L3`（第 3 层）推远
- push（推远）项不再使用 ambiguity scaling（歧义缩放）

## Run Matrix（运行矩阵）

- `R630a`
  - objective（目标）: `L_base + lambda_pull * L_pull`
  - active terms（生效项）: `mid graph pull`（中图拉近）
  - purpose（目的）: 看只有 graph pull（图拉近）时，`L2` 单层干预能到什么程度
- `R630b`
  - objective（目标）: `L_base + lambda_push * L_push`
  - active terms（生效项）: `mid selective push`（中层选择性推远）
  - purpose（目的）: 单独测试 selective separation（选择性分离）是否本身就有价值
- `R630c`
  - objective（目标）: `L_base + lambda_pull * L_pull + lambda_push * L_push`
  - active terms（生效项）: `mid pull + mid push`（中层拉近 + 中层推远）
  - purpose（目的）: 检查 pull / push（拉近 / 推远）是否互补

## Files（文件）

- pair source builder（物品对来源脚本）:
  - [experiment_mgr_sid_mid_pull_push_pair_source.py](/home/leejt/OneRec/scripts/experiment_mgr_sid_mid_pull_push_pair_source.py)
- pair source outputs（物品对来源产物）:
  - [R630_pair_source_summary.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_r630_mid_pull_push_industrial/R630_pair_source_summary.json)
  - [R630_all_mid_graph_weak_pairs.csv](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_r630_mid_pull_push_industrial/R630_all_mid_graph_weak_pairs.csv)
- configs（配置）:
  - [sid_train_industrial_mgr_sid_r630a_mid_pull_only.yaml](/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_r630a_mid_pull_only.yaml)
  - [sid_train_industrial_mgr_sid_r630b_mid_push_only.yaml](/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_r630b_mid_push_only.yaml)
  - [sid_train_industrial_mgr_sid_r630c_mid_pull_push.yaml](/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_r630c_mid_pull_push.yaml)
- launchers（启动脚本）:
  - [experiment_mgr_sid_r630a_mid_pull_only_train.sh](/home/leejt/OneRec/scripts/experiment_mgr_sid_r630a_mid_pull_only_train.sh)
  - [experiment_mgr_sid_r630b_mid_push_only_train.sh](/home/leejt/OneRec/scripts/experiment_mgr_sid_r630b_mid_push_only_train.sh)
  - [experiment_mgr_sid_r630c_mid_pull_push_train.sh](/home/leejt/OneRec/scripts/experiment_mgr_sid_r630c_mid_pull_push_train.sh)
  - [launch_mgr_sid_r630_mid_pull_push_tmux.sh](/home/leejt/OneRec/scripts/launch_mgr_sid_r630_mid_pull_push_tmux.sh)

## Runtime（运行时）

- checkpoints（检查点）:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_mid_pull_push_20260416/industrial_r630a_mid_pull_only`
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_mid_pull_push_20260416/industrial_r630b_mid_push_only`
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_mid_pull_push_20260416/industrial_r630c_mid_pull_push`
- generated indices（生成索引）:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_mid_pull_push_20260416/generated_indices/Industrial_and_Scientific.r630a_mid_pull_only.index.json`
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_mid_pull_push_20260416/generated_indices/Industrial_and_Scientific.r630b_mid_push_only.index.json`
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_mid_pull_push_20260416/generated_indices/Industrial_and_Scientific.r630c_mid_pull_push.index.json`
- logs（日志）:
  - `/home/leejt/OneRec/logs/experiment_mgr_sid_r630a_mid_pull_only_20260416.log`
  - `/home/leejt/OneRec/logs/experiment_mgr_sid_r630b_mid_push_only_20260416.log`
  - `/home/leejt/OneRec/logs/experiment_mgr_sid_r630c_mid_pull_push_20260416.log`
- generate summaries（生成摘要）:
  - [R630a_generate_summary.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_r630_mid_pull_push_industrial/R630a_generate_summary.json)
  - [R630b_generate_summary.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_r630_mid_pull_push_industrial/R630b_generate_summary.json)
  - [R630c_generate_summary.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_r630_mid_pull_push_industrial/R630c_generate_summary.json)
- tmux（终端复用）:
  - `mgr_r630a_mid_pull_only`
  - `mgr_r630b_mid_push_only`
  - `mgr_r630c_mid_pull_push`

## Result Snapshot（结果快照）

三条 tokenizer（分词器）训练与 `sid-generate`（SID 生成）都已完成。

### Raw Numbers（原始数字）

- `R630a`
  - best train collision rate（最佳训练冲突率）: `0.15057`
  - generated collision（生成后冲突）: `16 / 3686 = 0.0043407488`
- `R630b`
  - best train collision rate（最佳训练冲突率）: `0.16468`
  - generated collision（生成后冲突）: `15 / 3686 = 0.0040694520`
- `R630c`
  - best train collision rate（最佳训练冲突率）: `0.12344`
  - generated collision（生成后冲突）: `11 / 3686 = 0.0029842648`

### Comparison To Existing Lines（与现有主线对比）

- current `v2`（当前 `v2`）: `13 / 3686 = 0.0035268584`
- `R610a`（上一轮 `L3` selective separation）: `12 / 3686 = 0.0032555616`
- `R510`（`TAGCF` 属性图替换）: `11 / 3686 = 0.0029842648`

### Reading（解读）

- `R630a` 是明确负结果：
  - pull-only（仅拉近）把 generated collision（生成后冲突）退回到了 `16 / 3686`
  - 这基本等于 original semantic baseline（原始语义基线）的水平
- `R630b` 也不是可行主线：
  - push-only（仅推远）比 `R630a` 略好
  - 但仍然弱于 current `v2`（当前 `v2`）和 `R610a`
- `R630c` 是这组三路里唯一真正站出来的 tokenizer candidate（分词器候选）：
  - 它同时在 best train collision（最佳训练冲突率）和 generated collision（生成后冲突）上都明显优于 `R630a / R630b`
  - 也优于 current `v2`（`11 vs 13`）和 `R610a`（`11 vs 12`）

但这次结果也有一个必须保留的限制：

> `R630c` 只是 tokenizer-side（分词器侧）最强，不等于 downstream-ready（可直接下游推进）已经被证明。

原因很直接：

- 它目前只是**匹配**了 `R510` 的 `11 / 3686`
- 而 `R510` 已经做过完整 `SFT -> evaluate`（监督微调到评测），结论是负的

所以这次最准确的结论不是“pull+push 已经赢了”，而是：

> **在当前这组三路 clean attribution（干净归因）实验里，只有 `R630c` 证明了 pull（拉近）和 push（推远）需要联合出现；但它的真实项目价值仍然必须由 downstream `SFT -> evaluate`（下游监督微调到评测）裁决。**

### Later Downstream Verdict（后续下游裁决）

后续 `Stage U` 已经完成：

- [2026-04-16_mgr_sid_r630c_sft_eval_industrial/README.md](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_r630c_sft_eval_industrial/README.md)

补充结论是：

- `R630c` 虽然是这组三路里最强的 tokenizer candidate（分词器候选）
- 但它的 downstream `SFT -> evaluate`（下游监督微调到评测）结果为负
- 因此这次 `mid-only pull + push`（仅中层拉近加推远）线不能进入 strongest line（最强主线）

## Promotion Rule（推进规则）

这次阶段的推进规则只有一条：

> 不再使用任何 retired prior diagnostic（已退役前验诊断）做 promotion gate（推进门槛）；三条 tokenizer（分词器）线先完整训练，再决定是否进入最小 `SFT -> evaluate`（监督微调到评测）裁决。

如果后续需要下游裁决，优先仍然使用当前 `v2` 最强 recipe（配方）：

- `title_history2sid_on + desc_align_p05`

因为这条 recipe（配方）是目前 graph-aware SID（图感知 SID）最稳定的已验证入口。

## Next Action（下一步）

- 冻结 `R630a / R630b`
  - 它们已经完成了这次 stage（阶段）里各自的归因任务，不值得继续下游化
- 只推进 `R630c`
  - 作为这次 stage（阶段）唯一的 downstream candidate（下游候选）
- 但推进口径必须保持克制：
  - 这是一次 minimal downstream adjudication（最小下游裁决）
  - 不是因为 generated collision（生成后冲突）已经足够可信，而是因为 `R630c` 在同组实验里唯一没有明显退化
