# Original L2 Ambiguity-Aware（原版第二层歧义感知）

Status（状态）: `completed_tokenizer_hold（分词器已完成但暂缓）`

Date（日期）: `2026-04-21`

## Goal（目标）

Run a paired control（成对对照） for `original_l3_ambiguity_aware`（原版第三层歧义感知）:

- no `L1`（第一层） graph loss（图损失）
- only `L2`（第二层） graph smoothness（图平滑）
- no `L3`（第三层） graph loss（图损失）
- item-wise ambiguity-aware weighting（逐物品歧义感知加权）

This is not a continuation of `original_l2_multihop_ranking`（原版第二层多跳排序） because that ranking interface（排序接口） was diagnosed as no-go（停止） under primary cutoffs（主要截断）. This run tests a cleaner L2 smoothness interface（更干净的第二层平滑接口）.

## Design（设计）

New config（新配置）:

- `/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_original_l2_ambiguity_aware.yaml`

Key settings（关键设置）:

- `coarse_weight = 0.0`
- `mid_weight = 0.05`
- `local_weight = 0.0`
- `mid_view_name = local_multihop`
- `graph_scale_min/max = 0.5 / 1.5`
- `hierarchy_stopgrad_previous_levels = true`

Effective graph term（有效图项）:

$$
\mathcal L
=
\mathcal L_{\mathrm{rec}}
+
\mathcal L_{\mathrm{rq}}
+
0.05 \cdot
\mathcal L_{\mathrm{mid}}^{(2)}.
$$

The L2 representation（第二层表示） under stop-gradient（停梯度） is:

$$
p_i^{(2)}
=
\mathrm{sg}(q_i^{(1)}) + q_i^{(2)}.
$$

Item-wise scale（逐物品缩放）:

$$
s_i
=
0.5 + (1.5 - 0.5) a_i,
$$

where \(a_i\) is the offline ambiguity prior（离线歧义先验） from `offline_combined`.

## Artifacts（产物）

- Launch script（启动脚本）: `/home/leejt/OneRec/scripts/launch_mgr_sid_original_l2_ambiguity_aware_tmux.sh`
- Train/generate script（训练/生成脚本）: `/home/leejt/OneRec/scripts/experiment_mgr_sid_original_l2_ambiguity_aware_train_generate.sh`
- Log（日志）: `/home/leejt/OneRec/logs/experiment_mgr_sid_original_l2_ambiguity_aware_20260421.log`
- Checkpoint root（检查点目录）: `/data/leejt/OneRec/output_weights/experiments/mgr_sid_original_l2_ambiguity_aware_20260421/industrial_original_l2_ambiguity_aware`
- Generated SID（生成语义标识）: `/data/leejt/OneRec/output_weights/experiments/mgr_sid_original_l2_ambiguity_aware_20260421/generated_indices/Industrial_and_Scientific.original_l2_ambiguity_aware.index.json`
- Generate summary（生成摘要）: `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-21_mgr_sid_original_l2_ambiguity_aware_industrial/original_l2_ambiguity_aware_generate_summary.json`

## Tokenizer Result（分词器结果）

Training/generate（训练/生成） completed at `2026-04-21 11:06:43 CST`.

Raw result（原始结果）:

- Train best collision（训练最佳冲突率）: `0.1527400977`
- Generated collision（生成冲突）: `13 / 3686 = 0.0035268584`
- Max conflict（最大冲突簇）: `2`
- Collision rounds used（冲突修复轮数）: `20`
- Active L1（活跃第一层码）: `50`
- Unique L2 pairs（唯一第二层前缀）: `1693`
- Unique full SID（唯一完整语义标识）: `3673`

Comparison（对比）:

| tokenizer（分词器） | collision（冲突） | max conflict（最大冲突簇） | active L1（活跃第一层码） | unique L2（唯一第二层前缀） |
|---|---:|---:|---:|---:|
| `v2` | `13 / 3686 = 0.00353` | `2` | `203` | `2680` |
| `original_l3_collab_local`（原版第三层局部协同） | `13 / 3686 = 0.00353` | `2` | `95` | `2632` |
| `original_l2_multihop_ranking`（原版第二层多跳排序） | `15 / 3686 = 0.00407` | `2` | `88` | `2449` |
| `original_l2_ambiguity_aware`（原版第二层歧义感知） | `13 / 3686 = 0.00353` | `2` | `50` | `1693` |
| `original_l3_ambiguity_aware`（原版第三层歧义感知） | `657 / 3686 = 0.17824` | `72` | `18` | `256` |

Concentration diagnostic（集中度诊断）:

- Top-5 `L1` bucket sizes（前 5 个第一层桶大小）: `291 / 149 / 145 / 112 / 111`
- Top-5 `L1` total mass（前 5 个第一层桶总覆盖）: `808`
- As a comparison（对比）:
  - `original_l2_multihop_ranking`（原版第二层多跳排序）: top-5 total `516`
  - `original_l3_collab_local`（原版第三层局部协同）: top-5 total `594`
  - `v2` : top-5 total `282`

This means generated collision（生成冲突） is low, but the prefix space（前缀空间） is substantially more concentrated than other non-catastrophic candidates（其他非灾难性候选）.

Ambiguity bucket diagnostic（歧义分桶诊断）:

- There is no global collapse（全局塌缩） like `original_l3_ambiguity_aware`（原版第三层歧义感知）.
- Collided item rate（冲突物品率） stays low across buckets（分桶）: about `0.14%` to `1.08%`.
- However, the highest-ambiguity bucket（最高歧义分桶） only uses `36` active `L1` codes（活跃第一层码）, while the full tokenizer（分词器） uses `50`, which is consistent with stronger prefix concentration（更强前缀集中） under ambiguity-aware scaling（歧义感知缩放）.

## Verdict（裁决）

This tokenizer（分词器） passes the non-catastrophic generate screen（非灾难性生成筛查）, but it is not a strong positive result（强正结果）.

The key pattern is:

- ambiguity-aware scaling（歧义感知缩放） at `L2`（第二层） is much safer than at `L3`（第三层）
- but compared with `original_l2_multihop_ranking`（原版第二层多跳排序） and `original_l3_collab_local`（原版第三层局部协同）, it compresses the `L1/L2` prefix space（第一/第二层前缀空间） much more aggressively
- therefore it is not the preferred next SFT candidate（首选下一步监督微调候选）

Current recommendation（当前建议）:

- do not prioritize this branch for SFT（监督微调）
- use it as a paired control（成对对照） showing that `L2` ambiguity-aware smoothing（第二层歧义感知平滑） is structurally safer than `L3` ambiguity-aware smoothing（第三层歧义感知平滑）
- keep `original_l2_ranking_ambiguity_aware`（原版第二层排序歧义感知） as the more aligned next check（更贴近核心问题的下一检验）

## Success Gate（成功门槛）

If this tokenizer（分词器） is pushed to SFT（监督微调）, the real comparison target（真实对比目标） is strongest original MiniOneRec SFT（最强原版 MiniOneRec 监督微调）:

- `NDCG@10 > 0.10372025`
- `HR@10 > 0.15089345`
- `NDCG@1/3/5` should not broadly regress（不应整体退化）
