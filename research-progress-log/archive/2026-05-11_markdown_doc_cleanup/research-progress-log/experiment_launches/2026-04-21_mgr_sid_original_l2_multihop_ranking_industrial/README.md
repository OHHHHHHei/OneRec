# Original L2 Multihop Ranking（原版第二层多跳排序）

Status（状态）: `completed_sft_evaluated（已完成监督微调评测）`

Date（日期）: `2026-04-21`

## Goal（目标）

Test whether collaborative signal（协同信号） is more useful as a low-disturbance edit（低扰动编辑） to the original RQ-VAE（原版残差量化变分自编码器） instead of a full three-level graph-supervised SID（图监督 SID） rewrite.

This branch keeps the original reconstruction/quantization objective（重建/量化目标） and injects collaboration only at `L2`（第二层） through `local_multihop`（局部多跳图） ranking contrastive loss（排序对比损失）.

## Tokenizer Design（分词器设计）

- Config（配置）: `/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_original_l2_multihop_ranking.yaml`
- Base（基座）: original RQ-VAE（原版残差量化变分自编码器）
- Active collaborative term（启用的协同项）: `l2_ranking_contrastive_weight = 0.03`
- Positive carrier（正样本载体）: `local_multihop`（局部多跳图）
- Negative pairs（负样本对）: semantic-near but mid-weak（语义近但中层图弱连接）
- Stop-gradient（停梯度）: `hierarchy_stopgrad_previous_levels = true`
- Disabled terms（关闭项）: `coarse_weight = 0.0`, `mid_weight = 0.0`, `local_weight = 0.0`, semantic retention（语义保持） all off

Artifacts（产物）:

- Train summary（训练摘要）: `/data/leejt/OneRec/output_weights/experiments/mgr_sid_original_l2_multihop_ranking_20260421/industrial_original_l2_multihop_ranking/Apr-21-2026_00-05-05/summary.json`
- Best checkpoint（最佳检查点）: `/data/leejt/OneRec/output_weights/experiments/mgr_sid_original_l2_multihop_ranking_20260421/industrial_original_l2_multihop_ranking/Apr-21-2026_00-05-05/best_collision_model.pth`
- Generated SID（生成 SID）: `/data/leejt/OneRec/output_weights/experiments/mgr_sid_original_l2_multihop_ranking_20260421/generated_indices/Industrial_and_Scientific.original_l2_multihop_ranking.index.json`
- Generate summary（生成摘要）: `original_l2_multihop_ranking_generate_summary.json`
- Negative pair summary（负样本对摘要）: `original_l2_multihop_ranking_ranking_pair_source_summary.json`

Tokenizer result（分词器结果）:

- Generated collision（生成冲突）: `15 / 3686 = 0.0040694520`
- Max conflict（最大冲突簇）: `2`
- Active L1（活跃第一层码）: `88`
- Unique L2 pairs（唯一第二层前缀）: `2449`

## SFT / Evaluate（监督微调 / 评测）

- Recipe（训练配方）: `title_history2sid_on + desc_align_p05`
- SFT config（监督微调配置）: `/home/leejt/OneRec/config/experiments/sft_industrial_mgr_original_l2_multihop_ranking_title_on_desc_p05_4gpu.yaml`
- Evaluate config（评测配置）: `/home/leejt/OneRec/config/experiments/evaluate_industrial_mgr_original_l2_multihop_ranking_title_on_desc_p05_4gpu.yaml`
- Model（模型）: `/data/leejt/OneRec/output_weights/experiments/mgr_sid_original_l2_multihop_ranking_sft_eval_20260421/title_on_desc_p05_4gpu/sft/final_checkpoint`
- Result JSON（结果文件）: `./results/experiments/mgr_sid_original_l2_multihop_ranking_sft_eval_20260421/final_result_sft_mgr_original_l2_multihop_ranking_title_on_desc_p05_4gpu_Industrial_and_Scientific.json`
- W&B run（实验跟踪）: `sft_mgr_original_l2_multihop_ranking_title_on_desc_p05_4gpu_industrial`, id `4ckgns6u`
- Constraint invalid total（约束失配总数）: `0`
- Root branch count（根分支数）: `88`

Metrics（指标）:

| metric（指标） | @1 | @3 | @5 | @10 | @20 | @50 |
|---|---:|---:|---:|---:|---:|---:|
| NDCG | 0.06618134 | 0.08283132 | 0.09144148 | 0.10165136 | 0.11189154 | 0.12516853 |
| HR | 0.06618134 | 0.09485992 | 0.11581734 | 0.14736378 | 0.18817560 | 0.25612177 |

## Comparison（对比）

- Vs strongest original SFT（原版最强监督微调）: `NDCG@10 -0.00207`, `HR@10 -0.00353`
- Vs strongest original RL（原版最强强化学习）: `NDCG@10 -0.00561`, `HR@10 -0.00397`
- Vs `v2_on_p05` SFT（当前 v2_on_p05 监督微调）: `NDCG@10 -0.00106`, `HR@10 +0.00110`
- Vs `R720e` SFT（R720e 监督微调）: `NDCG@10 +0.00071`, `HR@10 +0.00132`
- Vs `original_l3_collab_local` SFT（原版第三层局部协同监督微调）: `NDCG@10 +0.00006`, `HR@10 +0.00132`

## Bootstrap / Proxy Diagnostics（自助法 / 代理诊断）

Diagnostic artifacts（诊断产物）:

- Report（报告）: `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-21_mgr_sid_original_l2_multihop_ranking_industrial/L2_PREFIX_BOOTSTRAP_PROXY_DIAGNOSTICS_20260421.md`
- JSON（结构化结果）: `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-21_mgr_sid_original_l2_multihop_ranking_industrial/l2_prefix_bootstrap_proxy_diagnostics_20260421.json`
- Primary-cutoff report（主要截断报告）: `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-21_mgr_sid_original_l2_multihop_ranking_industrial/L2_PREFIX_PRIMARY_CUTOFF_DIAGNOSTICS_20260421.md`
- Primary-cutoff JSON（主要截断结构化结果）: `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-21_mgr_sid_original_l2_multihop_ranking_industrial/l2_prefix_primary_cutoff_diagnostics_20260421.json`
- Script（脚本）: `/home/leejt/OneRec/scripts/experiment_mgr_sid_l2_prefix_diagnostics.py`

Paired bootstrap（配对自助法） conclusion（结论）:

- Vs recipe-aligned original（配方对齐原版）: `HR@50 +0.01368`, 95% CI（置信区间） `[0.00397, 0.02338]`, approximate p（近似 p 值） `0.0048`; this is only a secondary diagnostic signal（次级诊断信号）, not a primary success criterion（主要成功标准）.
- Vs recipe-aligned original（配方对齐原版）: `NDCG@10 -0.00018`, 95% CI（置信区间） `[-0.00472, 0.00436]`; not stable（不稳定）.
- Vs recipe-aligned original（配方对齐原版）: `HR@10 +0.00110`, 95% CI（置信区间） `[-0.00662, 0.00860]`; not stable（不稳定）.
- Since primary evaluation（主要评测） focuses on `@1/@3/@5/@10`, this run should not be read as a primary downstream win（主要下游胜利）.

Primary-cutoff rerun（主要截断重诊断） conclusion（结论）:

- Against recipe-aligned original（配方对齐原版）, no `NDCG/HR@1/@3/@5/@10` metric has a stable positive bootstrap interval（稳定正向自助法区间）.
- Against strongest original SFT（原版最强监督微调）, `original_l2` is raw-lower on all primary metrics（所有主要指标原始值更低）: `NDCG@1/3/5/10 = -0.00088 / -0.00218 / -0.00171 / -0.00207`; `HR@1/3/5/10 = -0.00088 / -0.00353 / -0.00243 / -0.00353`.
- Against `v2_on_p05`, `original_l2` is worse at `HR@1` with CI（置信区间） `[-0.00882, -0.00022]` and approximate p（近似 p 值） `0.0494`; other primary cutoffs（主要截断） are not stable.
- Therefore this branch is a no-go（停止） under the current primary objective（当前主要目标）, despite the secondary `HR@50`（命中率@50） clue.

Final Top50 proxy diagnostics（最终前 50 代理诊断）:

- `GT L2 covered@50`（真实目标第二层前缀覆盖率@50） improves to `0.32760`, higher than recipe-aligned original（配方对齐原版） `0.32120`.
- `GT L2 covered@10`（真实目标第二层前缀覆盖率@10） is `0.20671`, lower than recipe-aligned original（配方对齐原版） `0.20869`.
- Mean hit rank@50（前 50 命中平均排名） worsens to `13.73`, worse than recipe-aligned original（配方对齐原版） `12.77`.
- This supports tail candidate retention（尾部候选保留） rather than better top ranking / calibration（头部排序 / 校准）.

Final Top10 proxy diagnostics（最终前 10 代理诊断）:

- `GT L2 covered@1/3/5/10`（真实目标第二层前缀覆盖率@1/3/5/10） is `0.11052 / 0.16016 / 0.17935 / 0.20671`.
- Recipe-aligned original（配方对齐原版） is higher at every primary proxy cutoff（每个主要代理截断）: `0.13016 / 0.16391 / 0.18266 / 0.20869`.
- This means `original_l2` does not improve target L2 prefix survival（目标第二层前缀存活） in the actual top-10 region（前 10 区域）.

Tokenizer prefix diagnostics（分词器前缀诊断）:

- Graph-neighbor L2 overlap（图邻居第二层前缀重合率） is `0.01446`, lower than recipe-aligned original（配方对齐原版） `0.01653`.
- Semantic-neighbor L2 overlap（语义邻居第二层前缀重合率） is `0.02649`, also lower than recipe-aligned original（配方对齐原版） `0.03567`.
- Therefore the `HR@50` gain should not be read as global graph-neighbor prefix sharing（全局图邻居前缀共享） improvement.

Code-path audit（代码路径审查）:

- `hierarchy_stopgrad_previous_levels = true` is already enabled.
- The `L2` ranking representation（第二层排序表示） is already `detach(q1) + q2`, so the auxiliary ranking path（辅助排序路径） does not backpropagate（反向传播） into `q1`.
- A new `route_preserving_teacher_rank`（路由保持教师排序） that only repeats this stop-gradient（停梯度） mechanism would not be a meaningful new experiment（新实验）.

## Verdict（裁决）

This is the current strongest minimal-edit collaborative-injection（最小编辑协同注入） screen.

It does not beat strongest original SFT（原版最强监督微调） or `v2_on_p05` SFT（当前 v2_on_p05 监督微调） on `NDCG@10`, so it is not directly RL-promotable（不可直接推进强化学习）.

The important signal is now narrower: `L2` ranking（第二层排序） on the original base（原版基座） gives a stable secondary `HR@50` tail-retention（尾部保留） clue, but does not show stable primary `@1/@3/@5/@10`（主要评测截断） gains or global graph-neighbor prefix sharing（全局图邻居前缀共享） improvement.

Next reasonable action（下一步合理行动）: do not reimplement（重新实现） an equivalent route-preserving（路由保持） method, and do not promote（晋级） this branch based on `HR@50`. If continuing this line at all, it should be a small L2-only repair（仅第二层小修复） targeting top-10 prefix survival（前 10 前缀存活） and `NDCG/HR@1/@3/@5/@10`, not a larger method expansion（更大方法扩展）.
