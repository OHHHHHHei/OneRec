# Experiment Launch Index（实验启动索引）

Status（状态）: `stage-index（阶段索引）`

This directory keeps the experiment-by-experiment record for the current MGR-SID line.

The goal of this index is simple:

- one stage = one canonical entry
- detailed raw artifacts stay inside the stage folder
- old flat notes are archived after they are merged into a clearer stage summary

Every folder here is a historical stage record by default.
The latest stage is the current active execution path; earlier stages should be
read as provenance, not as the current optimization target.

This file is not the canonical current-state summary（权威当前状态摘要） anymore（不再承担该角色）.

If you want the live project status first, read:

1. `/home/leejt/OneRec/research-progress-log/CURRENT_STATE.md`
2. `/home/leejt/OneRec/experiment_results.csv`

Diagnostic-only stages（仅诊断阶段） that failed the `S000` audit have been retired from
active reading and decision use（活跃阅读与决策用途）.

Archive pointer（归档指针）:

- `/home/leejt/OneRec/research-progress-log/archive/2026-04-16_retired_prior_diagnostics/README.md`

## Active Reading Order（活跃阅读顺序）

1. `/home/leejt/OneRec/research-progress-log/CURRENT_STATE.md`
2. `2026-04-18_mgr_sid_r720_l2_ranking_contrastive_industrial/README.md`
3. `/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/19_mgr_sid_current_method_code_aligned_formulas.md`

All earlier stages（更早阶段） below are archived provenance（归档追溯材料）. They are kept for traceability（可追溯性）, not as active optimization targets（活跃优化目标）.

## Historical Stage Map（历史阶段地图）

### Stage A: Pipeline provenance and upstream-aligned reproduction

- canonical doc:
  `2026-04-09_pipeline_alignment_and_reproduction/README.md`
- covers:
  - default pipeline reproduction failure
  - upstream-style RQ-VAE launch/result
  - aligned training sanity
  - why final SID should be judged after `sid-generate`

### Stage B: Upstream-aligned `MGR-SID v1`

- canonical docs:
  - `2026-04-09_mgr_sid_v1_upstream/README.md`
  - `2026-04-09_mgr_sid_v1_upstream/RESULTS.md`
  - `2026-04-09_mgr_sid_v1_upstream/LOCAL_AMBIGUITY_BASELINE_VS_HIERARCHY.md`
- covers:
  - first positive tokenizer-side evidence
  - `hierarchy_reg` beating the semantic baseline on final SID structure

### Stage C: First downstream transfer and diagnostics

- canonical docs:
  - `2026-04-10_mgr_sid_data_experiment_convert/README.md`
  - `2026-04-10_mgr_sid_sft_eval_industrial/RESULTS.md`
  - `2026-04-10_mgr_sid_sft_eval_industrial/TOPK_STRUCTURAL_ANALYSIS.md`
- covers:
  - first `baseline vs hierarchy` SFT transfer
  - discovery that downstream gains are local and structure-sensitive

### Stage D: `v2` tokenizer construction

- canonical docs:
  - `2026-04-11_mgr_sid_v2_proxy_sanity/README.md`
  - `2026-04-11_mgr_sid_tokenizer_v2_r005/RESULTS.md`
  - `2026-04-11_mgr_sid_tokenizer_v2_r005/STRUCTURE_COMPARISON.md`
- covers:
  - ambiguity proxy sanity
  - first `v2` tokenizer run
  - tokenizer-side structural gains over both baseline and `v1`

### Stage E: `v2` downstream recipe isolation

- canonical docs:
  - `2026-04-11_mgr_sid_v2_recipe_isolation_industrial/RESULTS.md`
  - `2026-04-11_mgr_sid_v2_recipe_isolation_industrial/TOPK_ERROR_DISTRIBUTION_COMPARISON.md`
- covers:
  - four-cell recipe isolation
  - discovery that `title_history2sid_on + desc_align_p05` is the best current `v2` downstream recipe
  - strongest original recipe mismatch is mainly caused by `title_history2sid_off`

### Stage F: `v2_on_p05 -> RL`

- canonical docs:
  - `2026-04-12_mgr_sid_v2_rl_on_p05_industrial/RESULTS.md`
  - `2026-04-12_mgr_sid_v2_rl_on_p05_industrial/EVAL_ANALYSIS.md`
- covers:
  - end-to-end `v2` RL confirmation
  - current gap to strongest original MiniOneRec RL
  - the final remaining issue: mid-beam retention

### Stage G: Stage-2 retention-targeted tokenizer refinement

- canonical docs:
  - `2026-04-13_mgr_sid_stage2_stopgrad_industrial/README.md`
  - `2026-04-13_mgr_sid_stage2_semantic_retention_industrial/RESULTS.md`
  - `2026-04-13_mgr_sid_stage2_r202a_sft_eval_industrial/RESULTS.md`
  - `../../idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/refine-logs/archive/2026-04-14_stage3_scope_cleanup/EXPERIMENT_PLAN_STAGE2_RETENTION.md`
- covers:
  - launch of the first retention-targeted tokenizer refinements
  - `R202a` as the best Block-2 structural branch
  - failure of the first semantic-retention KL implementation (`R205`)
  - first downstream screen of `R202a`, which shows that structural gains alone did not yet beat current `v2_on_p05`

### Stage H: Stage-2 interface diagnostics

- canonical docs:
  - `2026-04-13_mgr_sid_stage2_interface_diagnostics_industrial/README.md`
- covers:
  - explicit diagnosis that tokenizer-side structural cleanup did not
    automatically create a better downstream SID space
  - measurement of prefix rearrangement, code polysemy, and learnability
    probes
- current status:
  - retired from active decision workflow（活跃决策工作流） after `S000`

### Stage I: Stage-3 codebook-space search

- canonical docs:
  - `2026-04-14_mgr_sid_stage3_prefix_retained_industrial/README.md`
- covers:
  - current stage-3 tokenizer search
  - `R401b` and `R401d` as candidate SID codebook spaces
  - shift from “stay near baseline” to “find a better downstream SID space”
  - tokenizer-side diagnostics before downstream adjudication

### Stage J: Stage-3 downstream adjudication

- canonical docs:
  - `2026-04-14_mgr_sid_stage3_sft_eval_industrial/README.md`
- covers:
  - full downstream `SFT -> evaluate` for `R401b` and `R401d`
  - the negative verdict that structurally stronger stage-3 SID spaces did not beat current `v2_on_p05`
  - why stage-3 no longer remains the active execution stage

### Stage K: Learnability reinterpretation

- canonical docs:
  - `2026-04-14_mgr_sid_learnability_probe_baseline_check/README.md`
- covers:
  - baseline learnability probe for original semantic SID
  - evidence that original semantic may win on easier first-step routing while graph-informed SID spaces remain stronger on deeper conditional prediction
- current status:
  - retired from active decision workflow（活跃决策工作流） after `S000`

### Stage L: Graph-carrier upgrade exploration

- canonical docs:
  - `2026-04-15_mgr_sid_tagcf_m0_attribute_graphs/README.md`
  - `2026-04-15_mgr_sid_tagcf_r510_attr_mid/README.md`
  - `2026-04-15_mgr_sid_tagcf_r511_mix_mid/README.md`
  - `2026-04-15_mgr_sid_fagsp_r520_mid_cascade/README.md`
- covers:
  - the current shift from retention/codebook-space refinements to graph-carrier upgrades
  - `TAGCF`-inspired `semantics -> topology` attribute-mid exploration
  - deeper `FaGSP`-style `item-side cascade` `G_mid` exploration
  - this stage showed that repeated `G_mid` replacement still had no positive evidence

### Stage M: Coarse / Local diagnostics gate

- canonical docs:
  - `2026-04-15_mgr_sid_coarse_local_graph_diagnostics_industrial/README.md`
- covers:
  - the first explicit offline diagnostics gate for under-tested `G_coarse / G_local`
  - evidence that current `user-segment` and `CIR` coarse candidates remain too close to baseline in their first formulations
  - evidence that multi-hop `G_local` really changes the graph, but does not by itself prove a baseline coverage failure
  - this stage decided that the first promoted tokenizer candidate should come from the local branch
- current status:
  - retired from active decision workflow（活跃决策工作流） after `S000`

### Stage N: First coarse/local tokenizer screen

- canonical docs:
  - `2026-04-15_mgr_sid_r530_local_multihop_industrial/README.md`
- covers:
  - the first tokenizer-side promotion after the coarse/local diagnostics gate
  - a shallow multi-hop `G_local`（浅层多跳局部图） screen that changes only `L3`
  - a clear negative verdict: changing only `L3` with the current local multi-hop formulation did not produce a viable tokenizer candidate
  - this stage re-centered the next exploration step on `G_coarse`（粗粒度图） reconstruction rather than deeper local expansion

### Stage O: `MGDCF` coarse reconstruction

- canonical docs:
  - `2026-04-16_mgr_sid_mgdcf_coarse_diagnostics_industrial/README.md`
  - `2026-04-16_mgr_sid_r542_mgdcf_coarse_industrial/README.md`
- covers:
  - the first true coarse-mother-graph reconstruction branch
  - diagnostics showing that `MGDCF` is not a near-baseline coarse tweak
  - the first tokenizer promotion that changes both `L1` and the mother graph of `G_mid`（中尺度图）
  - a mixed but ultimately negative tokenizer verdict: better than `R530a`, but still clearly weaker than current `v2` and stage-3 candidates
- current status:
  - the tokenizer run remains historical evidence（历史证据）, but its diagnostics-driven promotion path（诊断驱动推进路径） is retired after `S000`

### Stage P: `MGDCF` coarse-only isolation

- canonical docs:
  - `2026-04-16_mgr_sid_mgdcf_coarse_isolation_industrial/README.md`
- covers:
  - the direct follow-up to `R542a`
  - isolation of whether the `MGDCF` coarse branch itself is promising when `L2` is restored to `fagsp_mid_base`
  - parallel sensitivity check on `mgdcf_keep_ratio = 0.10 / 0.20`

### Stage Q: Selective-Separation pair diagnostics

- canonical docs:
  - `2026-04-16_mgr_sid_selective_separation_pair_diagnostics_industrial/README.md`
- covers:
  - the first explicit diagnostics gate for `semantic-close but collaboratively inconsistent`（语义接近但协同不一致） pair construction
  - evidence that `semantic-near + graph-non-neighbor`（语义接近 + 图上无邻接） is too broad as a first training rule
  - the first narrowed recommendation that phase-1 selective separation should start from `semantic-near + graph-weak`（语义接近 + 图弱连接）
- current status:
  - retired from active decision workflow（活跃决策工作流） after `S000`

### Stage R: Selective-Separation tokenizer screen

- canonical docs:
  - `2026-04-16_mgr_sid_r610_selective_separation_industrial/README.md`
- covers:
  - the first real training-time test of `reliability-aware selective separation`（可靠性感知选择性分离）
  - a minimal `L3`-only intervention on top of the current base `v2` tokenizer（分词器） backbone
  - promotion of `semantic-near + graph-weak`（语义接近 + 图弱连接） from diagnostics gate to formal tokenizer run
- current status:
  - preserved as a frozen tokenizer snapshot（冻结分词器快照）
  - not an active candidate（活跃候选）, because its supporting prior diagnostics（支撑前验诊断） were retired and no downstream verdict exists

### Stage S: Diagnostic audit

- canonical docs:
  - `2026-04-16_mgr_sid_diagnostic_audit_industrial/SUMMARY.md`
- covers:
  - the first retrospective audit（回顾性审计） of prior diagnostics（前验诊断） against historical downstream comparisons（历史下游对比）
  - a negative but important verdict: no current prior diagnostic（前验诊断） is strong enough to serve as a tokenizer promotion gate（分词器推进门槛）
  - retirement of generated collision（生成后冲突率）, local ambiguity（局部歧义）, and prefix collaborative consistency（前缀协同一致性） from the active decision workflow（活跃决策工作流）

### Stage T: Mid-only pull/push relaunch

- canonical docs:
  - `2026-04-16_mgr_sid_r630_mid_pull_push_industrial/README.md`
- covers:
  - the first selective-separation（选择性分离） relaunch after `S000`
  - compression of the objective（目标） into `pull-only / push-only / pull+push`（仅拉近 / 仅推远 / 拉近加推远）
  - restriction of auxiliary intervention（辅助干预） to `L2`（第 2 层） only
  - replacement of the old pair source（物品对来源） with `semantic-near + mid-graph-weak`（语义接近 + 中图弱连接）
- current status:
  - tokenizer stage（分词器阶段） completed（已完成）
  - `R630c` emerged as the only downstream candidate（唯一值得下游推进的候选）
  - later downstream adjudication（后续下游裁决） was negative（为负）, so this stage is now historical evidence（历史证据）, not an active branch（活跃分支）

### Stage U: `R630c` downstream adjudication

- canonical docs:
  - `2026-04-16_mgr_sid_r630c_sft_eval_industrial/README.md`
- covers:
  - direct promotion of `R630c` into the current strongest graph-aware recipe（当前最强图感知配方）
  - the first downstream verdict（下游裁决） for the simplified `mid-only pull + push`（仅中层拉近加推远） line
- current status:
  - completed（已完成）
  - final judgment（最终裁决） is negative（为负）
  - the simplified `mid-only pull + push`（仅中层拉近加推远） line did not survive downstream `SFT -> evaluate`（监督微调到评测）

### Stage V: `Seq2Graph-lite` high-order carrier audit

- canonical docs:
  - `2026-04-16_mgr_sid_d640_seq2graph_lite_graph_audit_industrial/README.md`
  - `2026-04-16_mgr_sid_d640_seq2graph_lite_graph_audit_industrial/SUMMARY.md`
- covers:
  - the first explicit offline audit（离线审计） for the new `Seq2Graph-lite`（轻量 `Seq2Graph`） high-order rescue carrier（高阶补盲载体）
  - confirmation that the new coarse variants（粗图变体） are not near-baseline tweaks（近基线微调）, but materially change hotspot neighborhoods（热点邻域）
  - separation of two useful regimes（有效模式）:
    - `coarse_seq2g_rel`（可靠性感知粗图） for broader hotspot visibility（更广热点可见性）
    - `coarse_seq2g_rel_masked`（可靠性感知加掩码粗图） for purer direct-zero rescue（更纯的直接零连接补盲）
- current status:
  - completed（已完成） as an engineering filter（工程过滤）
  - supports direct promotion into `R640a / R640b / R640c` tokenizer screen（分词器筛选）

### Stage W: `Seq2Graph-lite` tokenizer screen

- canonical docs:
  - `2026-04-16_mgr_sid_r640_seq2graph_lite_industrial/README.md`
- covers:
  - the first actual tokenizer training launch（分词器训练启动） after `D640`
  - direct comparison between a broader reliability-oriented（偏可靠性的） branch and a purer masked-rescue（带掩码补盲） branch
  - postponement of `R640a`（仅上下文版） as a lower-priority appendix-style（附录式） baseline
- current status:
  - completed（已完成）
  - `R640b` is a catastrophic failure（灾难性失败）:
    - first visible eval（首次可见评估） already collapsed（塌缩） at `collision = 0.9997`
    - final generated collision（最终生成冲突率） is `0.4121` with `max_conflict = 310`
    - the failure is now traced to reliability-only rescue（仅可靠性感知补盲） over-injecting direct-strong family edges（直接强连接家族边）
  - `R640c` is the only viable candidate（唯一可继续候选）:
    - final generated collision（最终生成冲突率） `12 / 3686 = 0.0032556`
    - promoted（已推进） to downstream `R645`

### Stage X: `R640c` downstream adjudication

- canonical docs:
  - `2026-04-17_mgr_sid_r640c_sft_eval_industrial/README.md`
- covers:
  - direct promotion（直接推进） of the only non-catastrophic `Seq2Graph-lite`（轻量 `Seq2Graph`） tokenizer candidate
  - the first formal downstream verdict（正式下游裁决） for the high-order carrier（高阶载体） branch
  - reuse of the current strongest graph-aware recipe（当前最强图感知配方） `title_history2sid_on + desc_align_p05`
- current status:
  - finished negative（已完成，负结果）
- result:
  - `NDCG@10 = 0.09305728`
  - `HR@10 = 0.13125965`
- conclusion:
  - carrier-only smoothness（仅图载体加平滑监督） is not promotable（不可推进） for `RL`（强化学习）
  - this does not reject high-order carrier + explicit push-pull（高阶载体 + 显式推远拉近） as a later method stage

### Stage Y: `R650a` Seq2Graph push-pull

- canonical docs:
  - `2026-04-17_mgr_sid_r650_seq2graph_push_pull_industrial/README.md`
- covers:
  - the first direct experiment that places the `R640c` `Seq2Graph-lite rel_masked`（轻量 `Seq2Graph` 可靠性感知加掩码版） carrier inside mid-only `push-pull`（仅中层推远拉近）
  - `pull`（拉近） from `fagsp_mid_seq2g_rel_masked`
  - `push`（推远） pairs rebuilt from semantic-near + `fagsp_mid_seq2g_rel_masked` weak（语义近 + Seq2Graph 中图弱连接）
- current status:
  - tokenizer/generate finished（分词器训练与生成已完成）
  - `tmux`（终端复用器） session: `mgr_r650a_seq2graph_push_pull`，已退出
- pair source summary（物品对来源摘要）:
  - `semantic_pair_count = 82596`
  - `weak_pair_count = 1190`
  - `weak_pair_item_coverage_rate = 0.2990`
- tokenizer result（分词器结果）:
  - train best collision（训练最佳冲突率）: `0.1142159523`
  - generated collision（生成后冲突）: `11 / 3686 = 0.0029842648`
  - max conflict（最大冲突簇）: `2`
- current conclusion（当前结论）:
  - non-catastrophic tokenizer candidate（非灾难性分词器候选）
  - needs `title_history2sid_on + desc_align_p05` downstream verdict（下游裁决）

### Stage Z: `R650a` SFT downstream adjudication

- canonical docs:
  - `2026-04-17_mgr_sid_r650a_sft_eval_industrial/README.md`
- covers:
  - downstream SFT（下游监督微调） for `R650a` under `title_history2sid_on + desc_align_p05`
  - prepared data root（已准备数据根目录）: `data_experiment/Amazon/r650a_seq2graph_mid_pull_push`
  - output root（输出根目录）: `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r650a_sft_eval_20260417/title_on_desc_p05/sft`
- current status:
  - completed negative（已完成，负结果）
- result（结果）:
  - `NDCG@1/3/5/10 = 0.06530 / 0.08132 / 0.08778 / 0.09518`
  - `HR@1/3/5/10 = 0.06530 / 0.09354 / 0.10920 / 0.13236`
- conclusion（结论）:
  - `R650a` does not beat current `v2_on_p05`（当前 v2_on_p05）
  - it should not be promoted to `RL`（强化学习）

### Stage AA: `R660a` constraint restoration

- canonical docs:
  - `2026-04-17_mgr_sid_r660_constraint_restoration_industrial/README.md`
- covers:
  - direct follow-up to the negative `R650a` downstream verdict（下游裁决）
  - testing whether the loss of `L1/L3/semantic`（第一层/第三层/语义） constraints caused `R650a` to lose L1 organization quality（第一层组织质量）
  - keeping the `R650a` `Seq2Graph-lite rel_masked + mid-only push-pull`（轻量 Seq2Graph 加掩码版 + 仅中层推远拉近） carrier fixed while restoring v2-style constraints（v2 风格约束）
- current status:
  - tokenizer/generate finished（分词器训练与生成已完成）
  - `tmux`（终端复用器） session: `mgr_r660a_constraint_restoration`，已退出
  - GPU（显卡）: `7`
- tokenizer result（分词器结果）:
  - train best collision（训练最佳冲突率）: `0.1323928378`
  - generated collision（生成后冲突）: `12 / 3686 = 0.0032555616`
  - max conflict（最大冲突簇）: `2`
- decision target（决策目标）:
  - because retired prior diagnostics（已退役前验诊断） cannot act as promotion gate（推进门槛）, this branch still needs `title_history2sid_on + desc_align_p05` SFT/evaluate（监督微调/评测） for downstream verdict（下游裁决）

### Stage AB: `R670a` clean L1 semantic + L2 push-pull

- canonical docs:
  - `2026-04-18_mgr_sid_r670_clean_l1_semantic_l2_push_pull_industrial/README.md`
- covers:
  - the first clean hierarchy（干净层级分工） tokenizer（分词器） experiment after the `R650a/R660a` negative signals
  - replacing full `v2` constraints（全套约束） with a narrower objective:
    - `L1` high-confidence semantic pull（第一层高置信语义拉近）
    - `L2` base collaborative pull（第二层基础协同拉近）
    - `L2` selective separation（第二层选择性分离）
    - stop-gradient prefix（前缀停梯度）
- current status:
  - completed negative（已完成，负结果）
  - `tmux`（终端复用器） session: `mgr_r670a_clean_l1_semantic_l2_push_pull`，已退出
  - final generated collision（最终生成冲突）: `162 / 3686 = 0.0439500814`
  - max conflict（最大冲突簇）: `35`
  - active L1（活跃第一层码）: `19`
  - unique L2 pairs（唯一第二层前缀数）: `375`
- decision target（决策目标）:
  - branch closed at tokenizer stage（在分词器阶段终止）
  - do not promote（不要推进） to downstream `SFT -> evaluate`（监督微调到评测）

### Stage AC: `R680a` clean L2 contrastive multihop

- canonical docs:
  - `2026-04-18_mgr_sid_r680_l1_smooth_l2_contrastive_multihop_industrial/README.md`
- covers:
  - the first clean `L2` interface test（第二层接口测试） after the `R670a` collapse（塌缩）
  - restoring `L1/L3`（第一层/第三层） to stable graph smoothness（图平滑） while changing only the `L2` supervision interface（第二层监督接口）
  - replacing `L2` graph smoothness（第二层图平滑） with:
    - `local_multihop`（局部多跳图） pairwise pull（成对拉近）
    - semantic-near + multihop-weak（语义近 + 多跳弱连接） selective push（选择性推远）
    - stop-gradient prefix（前缀停梯度）
- current status:
  - tokenizer_generated（分词器已生成）
  - `tmux`（终端复用器） session: `mgr_r680a_l1_smooth_l2_contrastive_multihop`，已结束
  - GPU（显卡）: `7`
  - pair source（物品对来源） already generated（已生成）:
    - `weak_pair_count = 1738`
    - `weak_pair_item_coverage_rate = 0.4881`
    - `weak_threshold = 0.0028070429`
  - tokenizer result（分词器结果）:
    - train best collision（训练最佳冲突率）: `0.0984807379`
    - generated collision（生成后冲突）: `11 / 3686 = 0.0029842648`
    - max conflict（最大冲突簇）: `2`
    - active L1（活跃第一层码）: `226`
    - unique L2 pairs（唯一第二层前缀数）: `2833`
- decision target（决策目标）:
  - tokenizer/generate（分词器训练与生成） is already non-catastrophic（已确认非灾难性）, so this branch is eligible（可进入） for downstream `SFT -> evaluate`（监督微调到评测）
  - final method judgment（最终方法判断） still depends on downstream result（仍取决于下游结果）

### Stage AD: `R690` CoST-inspired contrastive quantization

- canonical docs:
  - `2026-04-18_mgr_sid_r690_cost_inspired_contrastive_quantization_industrial/README.md`
- covers:
  - the first formal branch that combines `CoST`（基于对比量化的语义分词） style contrastive tokenization（对比式分词） with our graph-structured collaborative signal（图结构协同信号）
  - using `fagsp_mid_base`（基础中层图） as `L2` positive carrier（第二层正样本载体）
  - using semantic-near + mid-weak（语义近但中图弱连接） pairs as `L2 InfoNCE`（第二层对比学习损失） negatives
  - comparing:
    - `R690a` pure `L2` graph-guided `InfoNCE`（纯第二层图引导对比损失）
    - `R690b` hierarchical protected variant（带层级保护的变体）
- current status:
  - tokenizer_generated（分词器已生成）
  - both `tmux`（终端复用器） sessions have ended（均已结束）
  - GPUs（显卡）:
    - `R690a -> 3`
    - `R690b -> 4`
  - shared pair source（共享物品对来源） already generated（已生成）:
    - `weak_pair_count = 1211`
    - `weak_pair_item_coverage_rate = 0.2797`
    - `weak_threshold = 0.0016112356`
- decision target（决策目标）:
  - current evidence（当前证据）:
    - both branches use `fagsp_mid_base`（基础中层图） as `mid graph`（中图）, not `local_multihop`（局部多跳图）
    - `R690a` is the better tokenizer candidate（更好的分词器候选）:
      - generated collision（生成后冲突） `11 / 3686`
      - active L1（活跃第一层码） `118`
    - `R690b` shows stronger prefix compression（更强的前缀压缩） risk:
      - generated collision（生成后冲突） `14 / 3686`
      - active L1（活跃第一层码） `33`
  - next step（下一步）:
    - prioritize `R690a -> SFT -> evaluate`（监督微调到评测） before deciding whether `R690b` is worth downstream compute（下游算力）

### Stage AE: `R680a` downstream SFT adjudication

- canonical docs:
  - `2026-04-18_mgr_sid_r680a_sft_eval_industrial/README.md`
- covers:
  - direct promotion（直接推进） of `R680a` into the current strongest graph-aware recipe（当前最强图感知配方） `title_history2sid_on + desc_align_p05`
  - a controlled `2`-GPU `SFT`（监督微调） run under effective-batch alignment（有效批大小对齐）
  - full downstream `SFT -> evaluate`（监督微调到评测） verdict for the clean `L2` contrastive interface（第二层对比式接口） line
- current status:
  - completed_negative（已完成，负结果）
  - GPUs（显卡）: `5,7`
  - effective batch（有效批大小） kept aligned（已对齐） with prior `4`-GPU runs（4 卡运行）
  - result（结果）:
    - `NDCG@10 = 0.09863899`
    - `HR@10 = 0.13567174`
- decision target（决策目标）:
  - verdict obtained（裁决已得到）:
    - stronger than recent negative branches（强于近期多个负分支） such as `R640c / R650a`
    - but still below current `v2_on_p05`（当前 `v2_on_p05`）, so not promotable（不可推进） to `RL`（强化学习）

### Stage AF: `R720a` main-method L2 ranking contrastive SID

- canonical docs:
  - `2026-04-18_mgr_sid_r720_l2_ranking_contrastive_industrial/README.md`
- covers:
  - current active method candidate（当前活跃方法候选） after convergence away from broad branching（横向发散）
  - `L1/L3`（第一层/第三层） light graph pull（轻量图拉近）
  - `L2`（第二层） ranking contrastive loss（排序对比损失） over collaborative-positive vs semantic-near mid-weak hard negatives（协同正样本与语义近但中图弱连接困难负样本）
- current status:
  - implemented（已实现）
  - pair source generated（物品对来源已生成）
  - one-epoch smoke run passed（单轮冒烟运行已通过）
- decision target（决策目标）:
  - eligible for full tokenizer train -> generate（可进入完整分词器训练到生成）
  - final verdict still requires downstream `SFT -> evaluate`（监督微调到评测）

## Artifact Policy Inside Run Folders

Inside each run folder:

- `README.md`:
  launch context and runtime metadata
- `RESULTS.md`:
  the canonical result summary
- extra `TOPK_*`, `EVAL_*`, or `*_diagnostics.*`:
  supporting analysis artifacts only; do not treat retired diagnostics（已退役诊断） as decision inputs
- raw `.json` / `.csv`:
  keep them for traceability; they do not need their own prose note unless they support a claim directly
