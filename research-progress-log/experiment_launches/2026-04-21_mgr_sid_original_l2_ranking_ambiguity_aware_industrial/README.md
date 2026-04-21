# Original L2 Ranking Ambiguity-Aware（原版第二层排序歧义感知）

Status（状态）: `completed_tokenizer_hold（分词器已完成但暂缓）`

Date（日期）: `2026-04-21`

## Goal（目标）

Directly test the current core motivation（核心动机）:

> Among semantic-near items（语义相近物品）, collaborative-positive items（协同正样本） should be closer than semantic-near but collaborative-weak negatives（语义近但协同弱负样本）.

This is the push-pull version（推拉版本） of the ambiguity-aware L2 test（歧义感知第二层测试）. Unlike pure graph smoothness（纯图平滑）, it includes explicit negative pressure（显式负样本推开压力）.

## Design（设计）

Base control（基座对照）:

- `original_l2_multihop_ranking`（原版第二层多跳排序）

New config（新配置）:

- `/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_original_l2_ranking_ambiguity_aware.yaml`

Only intended change（唯一意图变化）:

- keep `l2_ranking_contrastive_weight = 0.03`
- keep `l2_ranking_margin = 0.1`
- keep positives（正样本） from `local_multihop`（局部多跳图）
- keep negatives（负样本） from `semantic_near_mid_weak`（语义近但中层弱连接）
- change `graph_scale_min/max`（图缩放范围） from `1.0 / 1.0` to `0.5 / 1.5`

Effective objective（有效目标）:

$$
\mathcal L
=
\mathcal L_{\mathrm{rec}}
+
\mathcal L_{\mathrm{rq}}
+
0.03 \cdot
\mathcal L_{\mathrm{rank}}^{(2)}.
$$

The L2 representation（第二层表示） uses stop-gradient（停梯度） on L1（第一层）:

$$
p_i^{(2)}
=
\mathrm{sg}(q_i^{(1)}) + q_i^{(2)}.
$$

Ranking loss（排序损失）:

$$
\mathcal L_{\mathrm{rank}}^{(2)}
=
\sum_{i,p,n}
w_{ipn}
\max(0, m + s_{in}^{(2)} - s_{ip}^{(2)}),
$$

where \(p\) is a collaborative-positive item（协同正样本） and \(n\) is a semantic-near collaborative-weak item（语义近但协同弱物品）.

Ambiguity-aware pair scale（歧义感知样本对缩放）:

$$
s_i = 0.5 + (1.5 - 0.5)a_i,
\qquad
w_{ipn} \propto \sqrt{s_i s_p}\sqrt{s_i s_n}.
$$

## Artifacts（产物）

- Launch script（启动脚本）: `/home/leejt/OneRec/scripts/launch_mgr_sid_original_l2_ranking_ambiguity_aware_tmux.sh`
- Train/generate script（训练/生成脚本）: `/home/leejt/OneRec/scripts/experiment_mgr_sid_original_l2_ranking_ambiguity_aware_train_generate.sh`
- Log（日志）: `/home/leejt/OneRec/logs/experiment_mgr_sid_original_l2_ranking_ambiguity_aware_20260421.log`
- Checkpoint root（检查点目录）: `/data/leejt/OneRec/output_weights/experiments/mgr_sid_original_l2_ranking_ambiguity_aware_20260421/industrial_original_l2_ranking_ambiguity_aware`
- Generated SID（生成语义标识）: `/data/leejt/OneRec/output_weights/experiments/mgr_sid_original_l2_ranking_ambiguity_aware_20260421/generated_indices/Industrial_and_Scientific.original_l2_ranking_ambiguity_aware.index.json`
- Generate summary（生成摘要）: `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-21_mgr_sid_original_l2_ranking_ambiguity_aware_industrial/original_l2_ranking_ambiguity_aware_generate_summary.json`

## Tokenizer Result（分词器结果）

Training/generate（训练/生成） completed at `2026-04-21 11:10:35 CST`.

Raw result（原始结果）:

- Train best collision（训练最佳冲突率）: `0.1855670103`
- Generated collision（生成冲突）: `15 / 3686 = 0.0040694520`
- Max conflict（最大冲突簇）: `2`
- Collision rounds used（冲突修复轮数）: `20`
- Active L1（活跃第一层码）: `77`
- Unique L2 pairs（唯一第二层前缀）: `1649`
- Unique full SID（唯一完整语义标识）: `3671`

Comparison（对比）:

| tokenizer（分词器） | collision（冲突） | active L1（活跃第一层码） | unique L2（唯一第二层前缀） | L1 max bucket（第一层最大桶） | top-5 L1 total（前 5 个第一层桶总覆盖） |
|---|---:|---:|---:|---:|---:|
| strongest original（最强原版） | `16 / 3686 = 0.00434` | `48` | `2295` | `247` | `833` |
| `original_l2_multihop_ranking`（原版第二层多跳排序） | `15 / 3686 = 0.00407` | `88` | `2449` | `166` | `516` |
| `original_l2_ambiguity_aware`（原版第二层歧义感知平滑） | `13 / 3686 = 0.00353` | `50` | `1693` | `291` | `808` |
| `original_l2_ranking_ambiguity_aware`（原版第二层排序歧义感知） | `15 / 3686 = 0.00407` | `77` | `1649` | `236` | `743` |
| `original_l3_collab_local`（原版第三层局部协同） | `13 / 3686 = 0.00353` | `95` | `2632` | `212` | `594` |
| `v2` | `13 / 3686 = 0.00353` | `203` | `2680` | `97` | `282` |

Interpretation（解读）:

- Relative to `original_l2_multihop_ranking`（原版第二层多跳排序）, ambiguity-aware ranking（歧义感知排序） keeps the same generated collision（生成冲突） but compresses the prefix space（前缀空间） much more:
  - active `L1`（第一层）: `88 -> 77`
  - unique `L2`（第二层前缀）: `2449 -> 1649`
  - top-5 `L1` total（前 5 个第一层桶总覆盖）: `516 -> 743`
- Relative to `original_l2_ambiguity_aware`（原版第二层歧义感知平滑）, the ranking version（排序版本） is less compressed（更少压缩） at `L1`（第一层） but still heavily compressed（明显压缩） at `L2`（第二层）:
  - active `L1`: `50 -> 77`
  - unique `L2`: `1693 -> 1649`
  - largest `L1` bucket（第一层最大桶）: `291 -> 236`
- Relative to strongest original（最强原版）, this branch spreads `L1`（第一层） more but coarsens `L2`（第二层） sharply: `2295 -> 1649`

Distribution detail（分布细节）:

- The largest bucket（最大桶） is a heterogeneous mixed industrial bucket（异质工业混合桶） of size `236`, containing switches（开关）, tapes（胶带）, rods（杆件）, tweezers（镊子）, bottles（瓶子）, filters（滤芯）, and adhesives（粘接用品）.
- The second bucket（第二大桶） of size `196` is a coherent 3D-printing cluster（3D 打印簇） with HATCHBOX / eSUN / 3D Solutech filament（耗材） and related accessories（配件）.
- The third bucket（第三大桶） of size `132` is a hardware fastener bucket（五金紧固件桶） with Hillman / Small Parts / Kreg screws（螺丝） and washers（垫圈）.

Ambiguity bucket diagnostic（歧义分桶诊断）:

- There is no catastrophic global collapse（灾难性全局塌缩）.
- Collided item rate（冲突物品率） stays low across buckets（分桶）: about `0.41%` to `1.22%`.
- But active `L1`（活跃第一层码） drops from `76` in `q1_low`（低歧义分桶） to `47` in `q5_high`（高歧义分桶）, which is consistent with stronger compression（更强压缩） on high-ambiguity items（高歧义物品）.

## Verdict（裁决）

This tokenizer（分词器） is not catastrophic（灾难性失败）, but it is not a strong positive result（强正结果） either.

The direct push-pull motivation（推拉动机） is logically closer to the project’s core question（核心问题） than pure smoothness（纯平滑）, but the current ambiguity-aware scaling（歧义感知缩放） still over-compresses the mid-level structure（中层结构）:

- tokenizer-side（分词器侧） it is structurally worse than `original_l2_multihop_ranking`（原版第二层多跳排序）
- it is somewhat safer than `original_l2_ambiguity_aware`（原版第二层歧义感知平滑） at `L1`（第一层）, but not enough to justify SFT（监督微调） priority
- therefore it should remain a tokenizer-side control（分词器侧对照）, not the next promoted candidate（下一晋级候选）

## Success Gate（成功门槛）

Tokenizer-side（分词器侧） structure is only a screening signal（筛选信号）.

If this tokenizer（分词器） is pushed to SFT（监督微调）, the real comparison target（真实对比目标） is strongest original MiniOneRec SFT（最强原版 MiniOneRec 监督微调）:

- `NDCG@10 > 0.10372025`
- `HR@10 > 0.15089345`
- `NDCG@1/3/5` should not broadly regress（不应整体退化）
