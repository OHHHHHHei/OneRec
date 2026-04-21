# R740a: v2 L1 Capacity Cap 128（v2 第一层容量限制 128）

Status（状态）: `finalized tokenizer-side no-go（分词器侧定稿停止）`

## Purpose（目的）

This experiment tests whether v2's weak downstream retention（下游候选保留） comes from excessive L1 fragmentation（第一层过度碎片化）.

Compared with original v2, only one variable changes:

- `num_emb_list`（码本大小列表）: `[256, 256, 256] -> [128, 256, 256]`

All original v2 ambiguity-aware graph supervision（歧义感知图监督） and semantic retention（语义保持） settings are kept unchanged.

## Artifacts（产物）

- Config（配置）: `/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_v2_l1cap128.yaml`
- Launch script（启动脚本）: `/home/leejt/OneRec/scripts/experiment_mgr_sid_v2_l1cap128_train_generate.sh`
- Log（日志）: `/home/leejt/OneRec/logs/experiment_mgr_sid_v2_l1cap128_20260421.log`
- Checkpoint（检查点）: `/data/leejt/OneRec/output_weights/experiments/mgr_sid_v2_l1cap128_20260421/industrial_v2_l1cap128/Apr-21-2026_12-40-43/best_collision_model.pth`
- Train summary（训练摘要）: `/data/leejt/OneRec/output_weights/experiments/mgr_sid_v2_l1cap128_20260421/industrial_v2_l1cap128/Apr-21-2026_12-40-43/summary.json`
- Generated index（生成索引）: `/data/leejt/OneRec/output_weights/experiments/mgr_sid_v2_l1cap128_20260421/generated_indices/Industrial_and_Scientific.v2_l1cap128.index.json`
- Generate summary（生成摘要）: `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-21_mgr_sid_v2_l1cap128_industrial/v2_l1cap128_generate_summary.json`

## Result（结果）

| tokenizer（分词器） | generated collision（生成冲突） | max conflict（最大冲突簇） | active L1（活跃第一层码） | unique L2（唯一第二层前缀） |
| --- | ---: | ---: | ---: | ---: |
| original semantic（原版语义） | `16 / 3686 = 0.00434` | `3` | `48` | `2295` |
| v2 original（原始 v2） | `13 / 3686 = 0.00353` | `2` | `203` | `2680` |
| v2_l1cap128（v2 第一层限制 128） | `114 / 3686 = 0.03093` | `21` | `15` | `452` |

Training best collision rate（训练最佳冲突率） is `0.5919696148`, and generation still has `114` duplicate-excess collisions（重复冗余冲突） after `20` collision-repair rounds（碰撞修复轮次）.

## Interpretation（解释）

The intended effect was to reduce v2's L1 fragmentation（第一层碎片化） from `203` active L1 codes（活跃第一层码） toward a moderate range.

The actual effect is severe over-compression（严重过度压缩）:

- active L1（活跃第一层码） collapses to `15`, not to a moderate `~128` range.
- unique L2 pairs（唯一第二层前缀） drops from v2's `2680` to `452`.
- generated collision（生成冲突） rises from v2's `13` to `114`.
- the largest full-SID conflict（完整 SID 冲突） contains `21` items（物品）.

This means hard L1 capacity capping（硬性第一层容量限制） is not a safe routeability repair（可路由性修复） for v2.

## Verdict（裁决）

Do not push this tokenizer（分词器） to SFT（监督微调）.

This experiment supports a narrower conclusion:

> v2's L1 over-fragmentation（第一层过度碎片化） is a real concern, but directly reducing `K1`（第一层码本大小） to `128` causes global SID over-compression（全局 SID 过度压缩） rather than a controlled routing repair（受控路由修复）.
