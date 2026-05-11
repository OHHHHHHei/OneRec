# Original L3 Ambiguity-Aware（原版第三层歧义感知）

Status（状态）: `completed_tokenizer_no_go（分词器已完成但停止）`

Date（日期）: `2026-04-21`

## Goal（目标）

Test a minimal RQ-VAE（残差量化变分自编码器） graph-loss variant（图损失变体） that keeps the original SID routeability（原版语义标识可路由性） as much as possible:

- no `L1`（第一层） graph loss（图损失）
- no `L2`（第二层） graph loss（图损失）
- only `L3`（第三层） local collaborative smoothing（局部协同平滑）
- ambiguity-aware weighting（歧义感知加权）: higher ambiguity（更高歧义） gets stronger collaborative supervision（协同监督）

This is a narrow sanity experiment（窄合理性实验）, not a new broad graph carrier（图载体） direction.

## Design（设计）

Base config（基座配置）:

- `/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_original_l3_collab_local.yaml`

New config（新配置）:

- `/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_original_l3_ambiguity_aware.yaml`

Only intended change（唯一意图变化）:

- keep `local_weight = 0.05`
- change `graph_scale_min/max`（图缩放范围） from `1.0 / 1.0` to `0.5 / 1.5`
- because only `local_weight`（局部图权重） is active, this makes the `L3` graph loss（第三层图损失） item-wise ambiguity-aware（逐物品歧义感知）

Effective graph term（有效图项）:

$$
\mathcal L
=
\mathcal L_{\mathrm{rec}}
+
\mathcal L_{\mathrm{rq}}
+
0.05 \cdot
\mathcal L_{\mathrm{local}}^{(3)}.
$$

with item-wise scale（逐物品缩放）:

$$
s_i
=
0.5 + (1.5 - 0.5) a_i,
$$

where \(a_i\) is the offline ambiguity prior（离线歧义先验） from `offline_combined`.

## Artifacts（产物）

- Launch script（启动脚本）: `/home/leejt/OneRec/scripts/launch_mgr_sid_original_l3_ambiguity_aware_tmux.sh`
- Train/generate script（训练/生成脚本）: `/home/leejt/OneRec/scripts/experiment_mgr_sid_original_l3_ambiguity_aware_train_generate.sh`
- Log（日志）: `/home/leejt/OneRec/logs/experiment_mgr_sid_original_l3_ambiguity_aware_20260421.log`
- Checkpoint root（检查点目录）: `/data/leejt/OneRec/output_weights/experiments/mgr_sid_original_l3_ambiguity_aware_20260421/industrial_original_l3_ambiguity_aware`
- Generated SID（生成语义标识）: `/data/leejt/OneRec/output_weights/experiments/mgr_sid_original_l3_ambiguity_aware_20260421/generated_indices/Industrial_and_Scientific.original_l3_ambiguity_aware.index.json`
- Generate summary（生成摘要）: `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-21_mgr_sid_original_l3_ambiguity_aware_industrial/original_l3_ambiguity_aware_generate_summary.json`

## Tokenizer Result（分词器结果）

Training/generate（训练/生成） completed at `2026-04-21 11:03:22 CST`.

Raw result（原始结果）:

- Train best collision（训练最佳冲突率）: `0.8052088985`
- Generated collision（生成冲突）: `657 / 3686 = 0.1782419967`
- Max conflict（最大冲突簇）: `72`
- Collision rounds used（冲突修复轮数）: `20`
- Active L1（活跃第一层码）: `18`
- Unique L2 pairs（唯一第二层前缀）: `256`
- Unique full SID（唯一完整语义标识）: `3029`

Comparison（对比）:

| tokenizer（分词器） | collision（冲突） | max conflict（最大冲突簇） | active L1（活跃第一层码） | unique L2（唯一第二层前缀） |
|---|---:|---:|---:|---:|
| original semantic（原版语义） | `16 / 3686 = 0.00434` | `3` | `48` | `2295` |
| v2_on_p05 tokenizer（v2_on_p05 分词器） | `13 / 3686 = 0.00353` | `2` | `203` | `2680` |
| original_l3_collab_local（原版第三层局部协同） | `13 / 3686 = 0.00353` | `2` | `95` | `2632` |
| original_l3_ambiguity_aware（原版第三层歧义感知） | `657 / 3686 = 0.17824` | `72` | `18` | `256` |

Largest conflict groups（最大冲突组）:

- `<a_29><b_165><c_0>`: `72` items（物品）
- `<a_112><b_44><c_93>`: `27` items（物品）
- `<a_112><b_44><c_162>`: `27` items（物品）
- `<a_112><b_153><c_93>`: `24` items（物品）

Ambiguity bucket diagnostic（歧义分桶诊断）:

- The collapse is not confined to high-ambiguity items（高歧义物品）.
- Collided item rate（冲突物品率） is high in all ambiguity buckets（歧义分桶）: roughly `21.8%` to `26.4%`.
- Active L1（活跃第一层码） stays around only `17-18` in every bucket, which indicates global routing collapse（全局路由塌缩）.

## Verdict（裁决）

This tokenizer（分词器） is a clear no-go（停止）.

Although the design intended to touch only `L3`（第三层） through stop-gradient previous levels（前层停梯度）, the ambiguity-aware scaling（歧义感知缩放） destabilized the learned SID space（语义标识空间） and caused severe over-compression（严重过度压缩）:

- do not prepare data_experiment（实验数据转换）
- do not push to SFT（监督微调）
- do not use this result as evidence that L3 local pull（第三层局部拉近） is useful

The useful comparison is instead negative: fixed-weight `original_l3_collab_local`（固定权重原版第三层局部协同） was structurally safe, while ambiguity-aware L3 scaling（歧义感知第三层缩放） collapsed. Therefore, this exact ambiguity-aware scaling interface（歧义感知缩放接口） is not viable at L3（第三层）.

## Success Gate（成功门槛）

Tokenizer-side（分词器侧） structure is only a screening signal（筛选信号）.

If this tokenizer is pushed to SFT（监督微调）, the real comparison target（真实对比目标） is strongest original MiniOneRec SFT（最强原版 MiniOneRec 监督微调）, not recipe-aligned original（配方对齐原版）:

- `NDCG@10 > 0.10372025`
- `HR@10 > 0.15089345`
- `NDCG@1/3/5` should not broadly regress（不应整体退化）
