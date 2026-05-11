# 2026-04-20 Original L3 Collaborative Local（原版第三层局部协同）

Status（状态）: `finalized（已定稿）`

## Goal（目标）

This stage tests a minimal collaborative edit（最小协同编辑） to the original MiniOneRec RQ-VAE（原版 MiniOneRec 残差量化变分自编码器） SID tokenizer（语义 ID 分词器）.

The design keeps the original reconstruction / quantization path（重建/量化路径） as the base and injects collaborative signal（协同信号） only at `L3`（第三层） through local graph pull（局部图拉近） with previous-level stop-gradient（前层停梯度）.

## Tokenizer（分词器）

- Config（配置）: `/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_original_l3_collab_local.yaml`
- Train checkpoint（训练检查点）: `/data/leejt/OneRec/output_weights/experiments/mgr_sid_original_l3_collab_local_20260420/industrial_original_l3_collab_local/Apr-20-2026_22-56-39/best_collision_model.pth`
- Train summary（训练摘要）: `/data/leejt/OneRec/output_weights/experiments/mgr_sid_original_l3_collab_local_20260420/industrial_original_l3_collab_local/Apr-20-2026_22-56-39/summary.json`
- Generated index（生成索引）: `/data/leejt/OneRec/output_weights/experiments/mgr_sid_original_l3_collab_local_20260420/generated_indices/Industrial_and_Scientific.original_l3_collab_local.index.json`
- Data root（数据根目录）: `/home/leejt/OneRec/data_experiment/Amazon/original_l3_collab_local`

Tokenizer result（分词器结果）:

| Metric（指标） | Value（数值） |
| --- | ---: |
| generated collision count（生成冲突数） | `13 / 3686` |
| generated collision rate（生成冲突率） | `0.0035268584` |
| max conflict（最大冲突簇） | `2` |
| best train collision（训练最佳冲突） | `0.0908844276` |
| best epoch（最佳轮次） | `9949` |

## SFT / Evaluate（监督微调/评测）

- Recipe（配方）: `title_history2sid_on + desc_align_p05`
- SFT config（监督微调配置）: `/home/leejt/OneRec/config/experiments/sft_industrial_mgr_original_l3_collab_local_title_on_desc_p05_4gpu.yaml`
- Evaluate config（评测配置）: `/home/leejt/OneRec/config/experiments/evaluate_industrial_mgr_original_l3_collab_local_title_on_desc_p05_4gpu.yaml`
- SFT model（监督微调模型）: `/data/leejt/OneRec/output_weights/experiments/mgr_sid_original_l3_collab_local_sft_eval_20260421/title_on_desc_p05_4gpu/sft/final_checkpoint`
- Result JSON（结果文件）: `/home/leejt/OneRec/results/experiments/mgr_sid_original_l3_collab_local_sft_eval_20260421/final_result_sft_mgr_original_l3_collab_local_title_on_desc_p05_4gpu_Industrial_and_Scientific.json`
- W&B run（实验追踪）: `r4cip28c`
- Evaluation examples（评测样本数）: `4533`
- Constraint invalid total（约束失配总数）: `0`

| Metric（指标） | @1 | @3 | @5 | @10 | @20 | @50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NDCG（归一化折损累计增益） | `0.06684315` | `0.08445174` | `0.09226315` | `0.10159264` | `0.11176226` | `0.12392398` |
| HR（命中率） | `0.06684315` | `0.09772777` | `0.11692036` | `0.14604015` | `0.18663137` | `0.24795941` |

## Baseline Comparison（基线对比）

Compared with strongest original SFT（原版最强监督微调）:

- `NDCG@10`: `0.10159264` vs `0.10372025`, delta（差值） `-0.00212761`
- `HR@10`: `0.14604015` vs `0.15089345`, delta（差值） `-0.00485330`

Compared with strongest original RL（原版最强强化学习）:

- `NDCG@10`: `0.10159264` vs `0.10726345`, delta（差值） `-0.00567081`
- `HR@10`: `0.14604015` vs `0.15133466`, delta（差值） `-0.00529451`

Compared with `v2_on_p05` SFT（当前 v2_on_p05 监督微调）:

- `NDCG@10`: `0.10159264` vs `0.10270767`, delta（差值） `-0.00111503`
- `HR@10`: `0.14604015` vs `0.14626075`, delta（差值） `-0.00022060`

Compared with `R720e` SFT（R720e 监督微调）:

- `NDCG@10`: `0.10159264` vs `0.10094471`, delta（差值） `+0.00064793`
- `HR@10`: `0.14604015` vs `0.14604015`, delta（差值） `+0.00000000`

## Verdict（裁决）

`original_l3_collab_local` is not RL-promotable（不可推进强化学习） yet because it does not beat strongest original SFT（原版最强监督微调） or `v2_on_p05` SFT（当前 v2_on_p05 监督微调）.

Its value is diagnostic（诊断性） and directional（方向性）: a low-disturbance collaborative injection（低扰动协同注入） on the original SID base（原版 SID 基座） outperforms several heavier collab-ranking（协同排序） branches and beats `R720e` on `NDCG@10`. This supports continuing the minimal-edit line（最小编辑路线）, especially the running `original_l2_multihop_ranking`（原版第二层多跳排序） screen.
