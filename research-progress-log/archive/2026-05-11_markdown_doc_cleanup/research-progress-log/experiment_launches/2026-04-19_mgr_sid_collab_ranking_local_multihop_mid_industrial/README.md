# R720b Local-Multihop Mid Graph SID

Status（状态）: `sft_evaluated（监督微调已评测）`

Date（日期）: `2026-04-19`

## Purpose（目的）

`R720b` 是对当前 collab-ranking mainline（协同排序主线）做的一次干净结构替换：

- 只把 `L2`（第二层）的 positive graph（正样本图）从 `fagsp_mid_base`（基础中层图）换成 `local_multihop`（局部多跳图）
- ranking negatives（排序负样本）同步按 `local_multihop` 重新构造
- 其余 loss（损失）、weight（权重）、stop-gradient（停梯度）和 `L1/L3` 图角色全部不动

所以这次实验的核心问题非常单纯：

> 如果主线逻辑不变，只更换 `mid graph`（中层图）载体，SID（语义标识）空间和下游 `SFT`（监督微调）会不会变好？

## Method Delta（方法改动）

相对 `R720a`，唯一结构改动是：

- `mid_view_name: fagsp_mid_base -> local_multihop`
- `l2_ranking_negative_pair_csv` 改为基于 `local_multihop` 重新生成的 `semantic_near_mid_weak`（语义近但中图弱连接）负样本对

总目标保持不变：

$$
\mathcal L
=
\mathcal L_{\mathrm{rec}}
+
\mathcal L_{\mathrm{rq}}
+
0.05\,\mathcal L_{\mathrm{pull}}^{(1)}
+
0.03\,\mathcal L_{\mathrm{rank}}^{(2)}
+
0.03\,\mathcal L_{\mathrm{pull}}^{(3)}.
$$

## Artifacts（产物）

- config（配置）:
  - `/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_collab_ranking_local_multihop_mid.yaml`
- train/generate chain（训练生成链路）:
  - `/home/leejt/OneRec/scripts/experiment_mgr_sid_collab_ranking_local_multihop_mid_train_generate.sh`
- tmux launcher（终端复用器启动脚本）:
  - `/home/leejt/OneRec/scripts/launch_mgr_sid_collab_ranking_local_multihop_mid_tmux.sh`
- tokenizer log（分词器日志）:
  - `/home/leejt/OneRec/logs/experiment_mgr_sid_collab_ranking_local_multihop_mid_20260419.log`
- pair source summary（样本对摘要）:
  - `R720b_ranking_pair_source_summary.json`
- generated SID index（生成 SID 索引）:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_collab_ranking_local_multihop_mid_20260419/generated_indices/Industrial_and_Scientific.r720b_local_multihop_mid.index.json`

## Tokenizer Result（分词器结果）

- run dir（运行目录）:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_collab_ranking_local_multihop_mid_20260419/industrial_r720b_local_multihop_mid/Apr-19-2026_02-02-45`
- best train collision（训练最佳冲突）:
  - `0.0949538795`
- best epoch（最佳轮次）:
  - `9549`
- final eval collision（最终评估冲突）:
  - `0.1006511123`
- generated collision（生成冲突）:
  - `11 / 3686 = 0.0029842648`
- negative pair count（困难负样本对数）:
  - `157425`
- negative item coverage（负样本覆盖率）:
  - `1.0`
- active L1（活跃第一层码）:
  - `247`
- unique L2 pairs（唯一第二层前缀数）:
  - `2889`
- unique leaf codes（唯一叶子码）:
  - `3675`

## Structural Reading（结构判读）

- 相比 `R720a`，`R720b` 的最重要变化是：
  - `L2`（第二层）从 `1558` 个前缀对明显展开到 `2889`
  - generated collision（生成冲突）从 `14` 降到 `11`
- 但代价也很明显：
  - `active L1`（活跃第一层码）从 `88` 涨到 `247`
  - 出现了更多小桶和更细的 `L1` 组织
- 因而它的结构结论不是“全面更稳”，而是：
  - `L2` 更展开
  - `L1` 更碎，但重点家族上出现了更多合理粗入口和功能邻域

## Downstream SFT Result（下游监督微调结果）

- SFT config（监督微调配置）:
  - `/home/leejt/OneRec/config/experiments/sft_industrial_mgr_collab_ranking_local_multihop_mid_title_on_desc_p05.yaml`
- evaluate config（评测配置）:
  - `/home/leejt/OneRec/config/experiments/evaluate_industrial_mgr_collab_ranking_local_multihop_mid_title_on_desc_p05.yaml`
- SFT log（监督微调日志）:
  - `/home/leejt/OneRec/logs/experiment_mgr_sid_collab_ranking_local_multihop_mid_sft_20260419.log`
- eval log（评测日志）:
  - `/home/leejt/OneRec/logs/experiment_mgr_sid_collab_ranking_local_multihop_mid_eval_20260419.log`
- output dir（输出目录）:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_collab_ranking_local_multihop_mid_sft_eval_20260419/title_on_desc_p05/sft`
- best checkpoint（最佳检查点）:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_collab_ranking_local_multihop_mid_sft_eval_20260419/title_on_desc_p05/sft/checkpoint-342`
- final model path（最终模型路径）:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_collab_ranking_local_multihop_mid_sft_eval_20260419/title_on_desc_p05/sft/final_checkpoint`
- result JSON（结果文件）:
  - `/home/leejt/OneRec/results/experiments/mgr_sid_collab_ranking_local_multihop_mid_sft_eval_20260419/final_result_sft_mgr_collab_ranking_local_multihop_mid_title_on_desc_p05_Industrial_and_Scientific.json`
- W&B（实验追踪）:
  - `sft_mgr_collab_ranking_local_multihop_mid_title_on_desc_p05_industrial`
  - `5ipt2ius`
- final eval loss（最终验证损失）:
  - `1.5633758307`
- best eval loss（最佳验证损失）:
  - `1.5207175016`
- final train loss（最终训练损失）:
  - `0.5220191915`
- stop epoch（停止轮次）:
  - `4.5`
- root branch count（根分支数）:
  - `247`
- constraint invalid total（约束失配总数）:
  - `0`
- `NDCG@1/3/5/10`:
  - `0.06926980 / 0.08363504 / 0.08997465 / 0.09940224`
- `HR@1/3/5/10`:
  - `0.06926980 / 0.09441871 / 0.11008162 / 0.13942202`

## Comparison（对比）

相对 `R720a`：

- `NDCG@1/3/5/10` 分别提升：
  - `+0.00684 / +0.00609 / +0.00577 / +0.00706`
- `HR@1/3/5/10` 分别提升：
  - `+0.00684 / +0.00574 / +0.00507 / +0.00927`

相对当前 `v2_on_p05`：

- `NDCG@10` 仍低 `0.00331`
- `HR@10` 仍低 `0.00684`

相对严格 recipe-aligned original baseline（严格配方对齐原版基线）：

- `NDCG@10` 低 `0.00243`
- `HR@10` 低 `0.00684`

相对 strongest original SFT（原版最强 SFT）：

- `NDCG@10` 低 `0.00432`
- `HR@10` 低 `0.01147`

## Verdict（裁决）

- `R720b` 是一次有信息量的正向推进。
  - 它明确优于 `R720a`
  - 说明当前主线里，“把 `mid graph` 换成 `local_multihop`”是有效方向
- 但它还没有超过当前 `v2_on_p05` 或 strongest original SFT（原版最强 SFT）
  - 因此不能直接晋级为新的 strongest validated line（最强已验证线）
  - 也不应直接推 `RL`（强化学习）
- 当前最合理的定位是：
  - `R720b` 是当前 collab-ranking（协同排序）主线里的 screening winner（筛选胜出版本）
  - 后续应围绕它做小范围微调，而不是回到 `R720a`

## Strongest Recipe Check（最强配方验证）

- SFT config（监督微调配置）:
  - `/home/leejt/OneRec/config/experiments/sft_industrial_mgr_collab_ranking_local_multihop_mid_title_off_desc_p05.yaml`
- evaluate config（评测配置）:
  - `/home/leejt/OneRec/config/experiments/evaluate_industrial_mgr_collab_ranking_local_multihop_mid_title_off_desc_p05.yaml`
- SFT log（监督微调日志）:
  - `/home/leejt/OneRec/logs/experiment_mgr_sid_collab_ranking_local_multihop_mid_title_off_desc_p05_sft_20260419.log`
- eval log（评测日志）:
  - `/home/leejt/OneRec/logs/experiment_mgr_sid_collab_ranking_local_multihop_mid_title_off_desc_p05_eval_20260419.log`
- final model path（最终模型路径）:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_collab_ranking_local_multihop_mid_sft_eval_20260419/title_off_desc_p05/sft/final_checkpoint`
- result JSON（结果文件）:
  - `/home/leejt/OneRec/results/experiments/mgr_sid_collab_ranking_local_multihop_mid_sft_eval_20260419/final_result_sft_mgr_collab_ranking_local_multihop_mid_title_off_desc_p05_Industrial_and_Scientific.json`
- W&B（实验追踪）:
  - `sft_mgr_collab_ranking_local_multihop_mid_title_off_desc_p05_industrial`
  - `umah3b27`
- final eval loss（最终验证损失）:
  - `1.5350477695`
- final train loss（最终训练损失）:
  - `0.3457109191`
- stop epoch（停止轮次）:
  - `7.5`
- root branch count（根分支数）:
  - `247`
- constraint invalid total（约束失配总数）:
  - `0`
- `NDCG@1/3/5/10`:
  - `0.06596073 / 0.08221681 / 0.08868210 / 0.09806915`
- `HR@1/3/5/10`:
  - `0.06596073 / 0.09419810 / 0.10986102 / 0.13920141`

### Strongest Recipe Comparison（最强配方对比）

相对 `R720b` 的 `title_history2sid_on + desc_align_p05`：

- `NDCG@10 -0.00133`
- `HR@10 -0.00022`

相对 strongest original SFT（原版最强监督微调）`title_history2sid_off + desc_align_p05`：

- `NDCG@10 -0.00565`
- `HR@10 -0.01169`

### Strongest Recipe Verdict（最强配方裁决）

- strongest recipe（最强配方）没有把 `R720b` 拉回到 stronger baseline（更强基线）之上。
- 它甚至略低于 `R720b` 自己的 `title_history2sid_on + desc_align_p05` 版本。
- 这说明当前差距不能主要归因于 recipe mismatch（配方错配）。
- 当前更应把注意力放回 tokenizer（分词器）侧，尤其是 `L1`（第一层）入口和 `L2`（第二层）分叉质量，而不是继续优先扫 recipe（配方）。
