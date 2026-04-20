# 2026-04-19 MGR-SID R720e Local-Multihop Mid + L1 Inverse Ambiguity（局部多跳中图 + 第一层逆歧义）

Status（状态）: `finalized tokenizer + SFT result（已定稿分词器与监督微调结果）`

## Purpose（目的）

在 `R720b`（`local_multihop` 作为 `L2` 载体）的基座上，只做一个控制变量修改：

- 不改 graph source（图来源）
- 不改 loss form（损失形式）
- 不加新 loss（损失）
- 只把 `L1 coarse pull`（第一层粗图拉近）的 item weighting（样本权重）从正向 ambiguity（正向歧义）改成 inverse ambiguity（逆歧义）

也就是：让低歧义、稳定、大类清楚的 item（物品）在 `L1` 上拥有更高权重；`L2/L3` 维持 `R720b` 设定不变。

## Core Config（核心配置）

- config（配置）:
  - `/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_collab_ranking_local_multihop_mid_l1_inverse_ambiguity.yaml`
- launch（启动脚本）:
  - `/home/leejt/OneRec/scripts/launch_mgr_sid_collab_ranking_local_multihop_mid_l1_inverse_ambiguity_tmux.sh`
- train/generate chain（训练/生成链路）:
  - `/home/leejt/OneRec/scripts/experiment_mgr_sid_collab_ranking_local_multihop_mid_l1_inverse_ambiguity_train_generate.sh`
- log（日志）:
  - `/home/leejt/OneRec/logs/experiment_mgr_sid_collab_ranking_local_multihop_mid_l1_inverse_ambiguity_20260419.log`

## Result（结果）

- train summary（训练汇总）:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_collab_ranking_local_multihop_mid_l1_inverse_ambiguity_20260419/industrial_r720e_local_multihop_mid_l1_inverse_ambiguity/Apr-19-2026_21-39-32/summary.json`
- generated SID index（生成 SID 索引）:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_collab_ranking_local_multihop_mid_l1_inverse_ambiguity_20260419/generated_indices/Industrial_and_Scientific.r720e_local_multihop_mid_l1_inverse_ambiguity.index.json`

Tokenizer metrics（分词器指标）:

- best train collision（训练最佳冲突）: `0.09278`
- final eval collision（最终评估冲突）: `0.11069`
- generated collision（生成冲突）: `13 / 3686 = 0.0035268584`
- `active L1`（活跃第一层码）: `190`
- `unique L2 pairs`（唯一第二层前缀数）: `2695`
- `unique leaf codes`（唯一叶子码）: `3673`
- `L1 buckets <= 10`（第一层小桶数）: `17`
- `L1 median bucket size`（第一层中位桶大小）: `18`

## Interpretation（解释）

相对 `R720b / R720c`，`R720e` 给出了一个清晰的 tokenizer-side（分词器侧）正信号：

- `L1`（第一层）明显收缩，但没有像 `R720d` 那样塌缩
- 小桶数量从 `70 / 50` 降到 `17`
- `active L1` 从 `247 / 240` 降到 `190`
- `unique L2 pairs` 从 `2889 / 2788` 降到 `2695`

重点家族上，`same-L1 pair share`（同第一层配对占比）相对 `R720b` 有这些变化：

- `gauge_meter`（仪表/测量器）: `0.0276 -> 0.0314`
- `connector_fitting`（连接件）: `0.0521 -> 0.0759`
- `tape`（胶带）: `0.0610 -> 0.0799`
- `adhesive_epoxy`（胶粘剂/环氧）: `0.0806 -> 0.0732`
- `3d_filament`（3D 打印耗材）: `0.1373 -> 0.1050`

当前判断：

- `L1 inverse ambiguity`（第一层逆歧义）是一个值得继续保留和下游验证的正向改动
- 它显著改善了 `L1` 可学习性（learnability，可学习性）相关的结构代理
- 但 `3d_filament` 这类大同族仍未回到 `R720b` 的最佳聚合状态
- 是否真正带动下游，需要 `SFT/evaluate`（监督微调/评测）验证

## Verdict（裁决）

## Downstream SFT/evaluate（下游监督微调/评测）

Recipe（配方）: `title_history2sid_on + desc_align_p05`

- SFT config（监督微调配置）:
  - `/home/leejt/OneRec/config/experiments/sft_industrial_mgr_collab_ranking_local_multihop_mid_l1_inverse_ambiguity_title_on_desc_p05.yaml`
- evaluate config（评测配置）:
  - `/home/leejt/OneRec/config/experiments/evaluate_industrial_mgr_collab_ranking_local_multihop_mid_l1_inverse_ambiguity_title_on_desc_p05.yaml`
- result JSON（结果文件）:
  - `/home/leejt/OneRec/results/experiments/mgr_sid_collab_ranking_local_multihop_mid_l1_inverse_ambiguity_sft_eval_20260419/final_result_sft_mgr_collab_ranking_local_multihop_mid_l1_inverse_ambiguity_title_on_desc_p05_Industrial_and_Scientific.json`
- SFT checkpoint（监督微调检查点）:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_collab_ranking_local_multihop_mid_l1_inverse_ambiguity_sft_eval_20260419/title_on_desc_p05/sft/final_checkpoint`

Metrics（指标）:

- `NDCG@1/3/5/10`: `0.06398 / 0.08397 / 0.09185 / 0.10094`
- `HR@1/3/5/10`: `0.06398 / 0.09883 / 0.11802 / 0.14604`
- `constraint_invalid_total`（约束失配总数）: `0`
- `root_branch_count`（根分支数）: `190`

Comparison（对比）:

- vs `R720b title_on_desc_p05`: `NDCG@10 +0.00154`, `HR@10 +0.00662`
- vs `R720a title_on_desc_p05`: `NDCG@10 +0.00860`, `HR@10 +0.01588`
- vs `v2_on_p05`: `NDCG@10 -0.00176`, `HR@10 -0.00022`
- vs strict recipe-aligned original baseline（严格配方对齐原版基线）: `NDCG@10 -0.00088`, `HR@10 -0.00022`
- vs strongest original SFT（原版最强监督微调）: `NDCG@10 -0.00278`, `HR@10 -0.00485`

Interpretation（解释）:

- `L1 inverse ambiguity`（第一层逆歧义）不仅改善 tokenizer-side（分词器侧）结构代理，也转化成了 downstream SFT（下游监督微调）正收益。
- `R720e` 是当前 collab-ranking（协同排序）方法族内最强的 SFT 版本，明显优于 `R720a/R720b`。
- 但它仍未超过 `v2_on_p05`、strict recipe-aligned original baseline（严格配方对齐原版基线）和 strongest original SFT（原版最强监督微调），因此还不能晋级 RL（强化学习）。

## Verdict（裁决）

`R720e` 是当前 collab-ranking（协同排序）主线里最强的已验证 SFT 候选，证明 `L1 inverse ambiguity`（第一层逆歧义）是正向改动；但它还不是 strongest validated line（最强已验证线），暂不推进 RL（强化学习）。
