# Experiment Tracker（实验跟踪表）

Status（状态）: `reference（参考）`
Last updated（更新日期）: `2026-04-17`

| Run ID | Milestone（里程碑） | Purpose（目的） | System / Variant（系统 / 变体） | Split（划分） | Metrics（指标） | Priority（优先级） | Status（状态） | Notes（备注） |
|---|---|---|---|---|---|---|---|---|
| D640 | M0 | graph audit（图审计） | `Seq2Graph-lite` context / reliability / mask graph summary | Industrial | connectivity, rescue-edge ratio, hotspot visibility | MUST | COMPLETED | `rel`（可靠性版） gives the best hotspot visibility（热点可见性）; `rel_masked`（可靠性加掩码版） gives the highest rescue purity（补盲纯度）; all three variants pass engineering sanity（工程合理性） and can enter tokenizer screen（分词器筛选） |
| R640a | M1 | tokenizer screen（分词器筛选） | `L1 <- coarse_seq2g_ctx_only`, `L2 <- fagsp_mid_seq2g_ctx_only`, `L3 <- local_purified` | Industrial | catastrophic generate check（灾难性生成检查）, collision | MUST | TODO | naive `Seq2Graph-lite` baseline（朴素 `Seq2Graph-lite` 基线） |
| R640b | M1 | tokenizer screen（分词器筛选） | `L1 <- coarse_seq2g_rel`, `L2 <- fagsp_mid_seq2g_rel`, `L3 <- local_purified` | Industrial | catastrophic generate check（灾难性生成检查）, collision | MUST | COMPLETED | negative（负）; first eval collision（首次评估冲突率） `0.9997`, best train collision（训练最佳冲突率） `0.9284`, generated collision（生成后冲突率） `0.4121`; reliability-only rescue（仅可靠性感知补盲） keeps too many direct-strong family edges（直接强连接家族边） |
| R640c | M1 | tokenizer screen（分词器筛选） | `L1 <- coarse_seq2g_rel_masked`, `L2 <- fagsp_mid_seq2g_rel_masked`, `L3 <- local_purified` | Industrial | catastrophic generate check（灾难性生成检查）, collision | MUST | COMPLETED | positive screen（正向筛选）; best train collision（训练最佳冲突率） `0.1243`, generated collision（生成后冲突率） `12 / 3686 = 0.0032556`, promoted（已推进） to `R645` |
| R645 | M2 | downstream verdict（下游裁决） | best non-catastrophic `R640*` -> `title_history2sid_on + desc_align_p05` | Industrial | `NDCG@10`, `HR@10`, output diagnosis | MUST | RUNNING | `R640c` promoted（已推进）; `SFT -> evaluate`（监督微调到评测） launched in `tmux`（终端复用器） session `mgr_r640c_sft_eval` |
| R641 | M3 | appendix variant（附录变体） | dynamic neighbor sampling（动态邻居采样） on top of best static `R640*` | Industrial | tokenizer screen + optional downstream | NICE | TODO | only if static `Seq2Graph-lite` shows real signal |
| R646 | M4 | RL promotion（`RL` 推进） | best promoted `R640*` -> `RL` | Industrial | `NDCG@10`, `HR@10` | NICE | TODO | only if `R645` is positive versus current `v2_on_p05` |
