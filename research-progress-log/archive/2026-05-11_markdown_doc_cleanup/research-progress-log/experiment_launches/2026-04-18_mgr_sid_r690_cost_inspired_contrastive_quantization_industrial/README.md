# Stage AD: `R690` CoST-inspired contrastive quantization（受 CoST 启发的对比式量化）

Status（状态）: `partial_sft_evaluated（部分完成 SFT 评测）`

Last updated（更新日期）: `2026-04-19`

## Goal（目标）

把 `CoST`（基于对比量化的语义分词）思路和我们当前的 graph-structured collaborative signal（图结构协同信号）结合起来，验证：

- `L2`（第二层）是否更适合承担 contrastive discrimination（对比分辨）角色，而不是继续只做 graph smoothness（图平滑）
- `L1/L3`（第一层/第三层）是否只需要轻量保护，而不需要重新回到多项 full constraint（全套约束）

## Shared Pair Source（共享负样本来源）

- script（脚本）:
  - `scripts/experiment_mgr_sid_mid_pull_push_pair_source.py`
- tag（标签）:
  - `R690`
- mid view（中层图）:
  - `fagsp_mid_base`
- output files（输出文件）:
  - `R690_all_mid_graph_weak_pairs.csv`
  - `R690_top_mid_graph_weak_pairs.csv`
  - `R690_pair_source_summary.json`

该文件为 `L2 InfoNCE`（第二层对比学习损失）提供 semantic-near + mid-weak negatives（语义近但中图弱连接的负样本）。

## Variants（变体）

### `R690a`: pure `L2` graph-guided InfoNCE（纯第二层图引导 InfoNCE）

- purpose（目的）:
  - 做最干净的 `CoST-inspired`（受 CoST 启发）机制验证
  - 不加 `L1/L3`（第一层/第三层）保护
- config（配置）:
  - `config/experiments/sid_train_industrial_mgr_sid_r690a_l2_graph_infonce.yaml`
- loss sketch（损失骨架）:
  - `RQ-VAE`（残差量化变分自编码器） base loss（基础损失）
  - `L2` graph-guided InfoNCE（第二层图引导对比损失）
- key settings（关键设置）:
  - `mid_view_name = fagsp_mid_base`
  - `l2_contrastive_mode = graph_infonce`
  - `hierarchy_stopgrad_previous_levels = false`
- launch target（启动目标）:
  - `tmux`（终端复用器） session: `mgr_r690a_l2_graph_infonce`

### `R690b`: hierarchical protected CoST-inspired branch（带层级保护的 CoST 启发分支）

- purpose（目的）:
  - 在 `R690a` 基础上加回轻量层级分工保护
  - 测试 `L1`（第一层）语义入口和 `L3`（第三层）局部细化是否能稳住前缀空间
- config（配置）:
  - `config/experiments/sid_train_industrial_mgr_sid_r690b_hier_cost_guided.yaml`
- loss sketch（损失骨架）:
  - `RQ-VAE`（残差量化变分自编码器） base loss（基础损失）
  - `L1` semantic pairwise pull（第一层语义成对拉近）
  - `L2` graph-guided InfoNCE（第二层图引导对比损失）
  - `L3` local pairwise pull（第三层局部成对拉近）
- key settings（关键设置）:
  - `l1_contrastive_pull_weight = 0.03`
  - `l2_contrastive_pull_weight = 0.10`
  - `l3_contrastive_pull_weight = 0.02`
  - `hierarchy_stopgrad_previous_levels = true`
- launch target（启动目标）:
  - `tmux`（终端复用器） session: `mgr_r690b_hier_cost_guided`

## Runtime Plan（运行计划）

- launcher（启动脚本）:
  - `scripts/launch_mgr_sid_r690_cost_inspired_tmux.sh`
- planned GPUs（计划显卡）:
  - `R690a -> GPU 3`
  - `R690b -> GPU 4`
- train+generate script（训练加生成脚本）:
  - `scripts/experiment_mgr_sid_r690a_l2_graph_infonce_train_generate.sh`
  - `scripts/experiment_mgr_sid_r690b_hier_cost_guided_train_generate.sh`

## Final Tokenizer Status（最终分词器状态）

- pair source（共享负样本来源）已生成:
  - `semantic_pair_count = 82596`
  - `weak_pair_count = 1211`
  - `weak_pair_item_coverage_rate = 0.2797`
  - `weak_threshold = 0.0016112356`
  - `reliability_mean = 0.0079141`
- `tmux`（终端复用器） sessions（会话）均已结束
- launcher（启动脚本）:
  - `scripts/launch_mgr_sid_r690_cost_inspired_tmux.sh`

### `R690a` final result（最终结果）

- run dir（运行目录）:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r690_cost_inspired_20260418/industrial_r690a_l2_graph_infonce/Apr-18-2026_03-16-17`
- best train collision（训练最佳冲突率）:
  - `0.0887140532`
- best epoch（最佳轮次）:
  - `9899`
- final eval collision（最终评估冲突率）:
  - `0.103907`
- generated collision（生成后冲突）:
  - `11 / 3686 = 0.0029842648`
- max conflict（最大冲突簇）:
  - `2`
- active L1（活跃第一层码）:
  - `118`
- unique L2 pairs（唯一第二层前缀数）:
  - `2527`
- generated SID index（生成语义标识索引）:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r690_cost_inspired_20260418/generated_indices/Industrial_and_Scientific.r690a_l2_graph_infonce.index.json`

### `R690b` final result（最终结果）

- run dir（运行目录）:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r690_cost_inspired_20260418/industrial_r690b_hier_cost_guided/Apr-18-2026_03-16-17`
- best train collision（训练最佳冲突率）:
  - `0.1120455779`
- best epoch（最佳轮次）:
  - `9999`
- final eval collision（最终评估冲突率）:
  - `0.124525`
- generated collision（生成后冲突）:
  - `14 / 3686 = 0.0037981552`
- max conflict（最大冲突簇）:
  - `2`
- active L1（活跃第一层码）:
  - `33`
- unique L2 pairs（唯一第二层前缀数）:
  - `1989`
- generated SID index（生成语义标识索引）:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r690_cost_inspired_20260418/generated_indices/Industrial_and_Scientific.r690b_hier_cost_guided.index.json`

### `R690b` downstream `SFT/evaluate`（监督微调/评测） result（结果）

- SFT config（监督微调配置）:
  - `config/experiments/sft_industrial_mgr_r690b_title_on_desc_p05.yaml`
- evaluate config（评测配置）:
  - `config/experiments/evaluate_industrial_mgr_r690b_title_on_desc_p05.yaml`
- SFT output（监督微调输出）:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r690b_sft_eval_20260418/title_on_desc_p05/sft/final_checkpoint`
- result json（结果文件）:
  - `./results/experiments/mgr_sid_r690b_sft_eval_20260418/final_result_sft_mgr_r690b_title_on_desc_p05_Industrial_and_Scientific.json`
- core metrics（核心指标）:
  - `NDCG@1 = 0.06706`
  - `NDCG@3 = 0.08149`
  - `NDCG@5 = 0.08859`
  - `NDCG@10 = 0.09719`
  - `HR@1 = 0.06706`
  - `HR@3 = 0.09221`
  - `HR@5 = 0.10942`
  - `HR@10 = 0.13611`
- decoding diagnosis（解码诊断）:
  - `root_branch_count = 33`
  - `constraint_invalid_total = 0`
  - `num_beams = 50 > first-step valid branches（首步有效分支数） = 33`
- verdict（结论）:
  - 结果低于当前 `v2_on_p05` 和严格配方对齐原版基线，不推进到 `RL`（强化学习）
  - 由于 `constraint_invalid_total = 0`，问题更像是 `code space compression`（码本空间压缩）/ 前缀空间过紧，而不是非法约束解码

## Mid-Graph Clarification（中图澄清）

这两个实验的 `mid graph`（中图）都**不是** `local_multihop`（局部多跳图）。

- `R690a`:
  - `mid_view_name = fagsp_mid_base`
- `R690b`:
  - `mid_view_name = fagsp_mid_base`

共享负样本文件 `R690_all_mid_graph_weak_pairs.csv` 也是基于这个 `fagsp_mid_base`（基础中层图）构建的，不是基于 `multihop graph`（多跳图）。

配置里虽然还保留了 `local_multihop_alpha`、`local_multihop_max_hop` 这些 inherited hyperparameters（继承超参数），但在 `R690` 这条线里它们并没有作为 `mid graph` 被真正启用。

## Current Reading（当前判读）

- `R690b` 的 tokenizer（分词器）结果本身并没有塌缩，但下游 `SFT/evaluate`（监督微调/评测）已经给出负结论。
  - `NDCG@10 = 0.09719`，低于当前 `v2_on_p05 = 0.10271`
  - 也低于严格配方对齐原版基线 `0.09870`
  - `root_branch_count = 33` 且 `constraint_invalid_total = 0`，说明问题更像是前缀空间压得太紧，而不是解码约束本身坏掉
- `R690a` 仍然保留为 tokenizer-only（仅分词器侧）候选。
  - 它没有像 `R690b` 那样出现这么强的前缀压缩
  - 如果后面要继续回看 `CoST-inspired`（受 CoST 启发）这条支线，`R690a` 仍然比 `R690b` 更值得优先讨论

## Decision Rule（决策规则）

- 当前结果已经说明：
  - pure `L2 InfoNCE`（纯第二层对比损失）不一定会自动塌缩
  - 轻量层级保护 + stop-gradient prefix（前缀停梯度）并不天然更稳，`R690b` 反而更容易把前缀空间压得过紧
- 当前阶段结论是：
  - `R690b` 不再推进
  - 这条 `CoST-inspired`（受 CoST 启发）支线若要继续，只应把 `R690a` 作为可能的回看对象，而不是继续围绕 `R690b` 做下游推进
