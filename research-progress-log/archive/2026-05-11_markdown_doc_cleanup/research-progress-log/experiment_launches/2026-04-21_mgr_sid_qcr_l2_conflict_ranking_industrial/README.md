# QCR-L2 Conflict Ranking（量化冲突感知第二层排序）

Status（状态）: `finalized tokenizer-side candidate（分词器侧定稿候选）`

## Purpose（目的）

This experiment tests a minimal conflict-targeted interface（最小冲突定向接口）:

- keep original RQ-VAE backbone（原版残差量化变分自编码器主干）
- disable global graph propagation（关闭全局图传播）
- disable ordinary L2 ranking（关闭普通第二层排序）
- activate L2 repulsion（第二层推开） only when semantic-near graph-weak negatives（语义近但图弱负样本） currently share the L2 prefix（第二层前缀）

## Artifacts（产物）

- Config（配置）: `/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_qcr_l2_conflict_ranking.yaml`
- Launch script（启动脚本）: `/home/leejt/OneRec/scripts/experiment_mgr_sid_qcr_l2_conflict_ranking_train_generate.sh`
- Log（日志）: `/home/leejt/OneRec/logs/experiment_mgr_sid_qcr_l2_conflict_ranking_20260421.log`
- Checkpoint（检查点）: `/data/leejt/OneRec/output_weights/experiments/mgr_sid_qcr_l2_conflict_ranking_20260421/industrial_qcr_l2_conflict_ranking/Apr-21-2026_15-27-34/best_collision_model.pth`
- Train summary（训练摘要）: `/data/leejt/OneRec/output_weights/experiments/mgr_sid_qcr_l2_conflict_ranking_20260421/industrial_qcr_l2_conflict_ranking/Apr-21-2026_15-27-34/summary.json`
- Generated index（生成索引）: `/data/leejt/OneRec/output_weights/experiments/mgr_sid_qcr_l2_conflict_ranking_20260421/generated_indices/Industrial_and_Scientific.qcr_l2_conflict_ranking.index.json`
- Generate summary（生成摘要）: `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-21_mgr_sid_qcr_l2_conflict_ranking_industrial/qcr_l2_conflict_ranking_generate_summary.json`

## Result（结果）

| tokenizer（分词器） | generated collision（生成冲突） | max conflict（最大冲突簇） | active L1（活跃第一层码） | unique L2（唯一第二层前缀） |
| --- | ---: | ---: | ---: | ---: |
| original semantic（原版语义） | `16 / 3686 = 0.00434` | `3` | `48` | `2295` |
| v2 original（原始 v2） | `13 / 3686 = 0.00353` | `2` | `203` | `2680` |
| original_l2_multihop_ranking（原版第二层多跳排序） | `15 / 3686 = 0.00407` | `2` | `88` | `2449` |
| qcr_l2_conflict_ranking（QCR 第二层冲突排序） | `11 / 3686 = 0.00298` | `2` | `117` | `2632` |

Training best collision rate（训练最佳冲突率） is `0.0846446012`, and generation reaches `11` duplicate-excess collisions（重复冗余冲突） after collision repair（碰撞修复）.

## QCR Pair Diagnostic（QCR 样本对诊断）

On the semantic-near graph-weak negative pair set（语义近但图弱负样本对集合）:

| tokenizer（分词器） | same L1（同第一层） | same L2（同第二层） |
| --- | ---: | ---: |
| original semantic（原版语义） | `0.46085` | `0.01767` |
| v2 original（原始 v2） | `0.16299` | `0.01025` |
| original_l3_collab_local（原版第三层局部协同） | `0.34270` | `0.01025` |
| original_l2_multihop_ranking（原版第二层多跳排序） | `0.33276` | `0.01217` |
| qcr_l2_conflict_ranking（QCR 第二层冲突排序） | `0.30012` | `0.01119` |

QCR reduces same-L2 conflicts（同第二层冲突） relative to `original_l2_multihop_ranking`, but does not beat v2/original_l3 on this proxy（代理指标）.

## Verdict（裁决）

This is a healthy tokenizer-side candidate（健康的分词器侧候选） and is worth SFT/evaluate screening（监督微调/评测筛选）.

It is not yet downstream-validated（下游已验证） and should not be claimed as better than strongest original MiniOneRec（最强原版 MiniOneRec） or v2 before SFT/evaluate（监督微调/评测） finishes.
