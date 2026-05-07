# Hierarchy-Aware AttnRes Readout for SID Generation（层级感知注意力残差读出）

Status（状态）: `active-experiment（活跃实验）`

Last updated（更新日期）: `2026-05-06`

## Goal（目标）

This branch tests whether Attention Residuals-inspired depth-wise aggregation（受注意力残差启发的深度方向聚合） can improve downstream SID generation（下游语义标识生成） without changing the tokenizer（分词器）.

The key hypothesis（关键假设） is:

> Hierarchical SID tokens（层级语义标识 token） may require different LLM hidden depths（大语言模型隐藏层深度）. A single final-layer readout（最后层读出） may underuse the hierarchy constructed by SID tokenization（语义标识分词）.

## Phase 1（第一阶段）

We do not rerun the original baseline（原版基线）. We reuse the strongest recorded OneRec SFT baseline（已记录最强 OneRec 监督微调基线） as the comparison point:

- `sft_industrial_title_history2sid_off__desc_align_p05_20260325_192249`
- `NDCG@10=0.10372025`
- `HR@10=0.15089345`

Phase 1 uses original OneRec SID（原版 OneRec 语义标识） and the same strong SFT recipe（强监督微调配方）:

- `title_history2sid_off + desc_align_p05`
- `batch_size=1024`
- `micro_batch_size=2`
- `world_size=4`
- `num_epochs=10`
- `learning_rate=0.0003`
- `eval_num_beams=50`

## Runs（运行）

| Run ID（运行编号） | Mode（模式） | Description（说明） |
|---|---|---|
| `attnres_h1_global` | `global` | Apply AttnRes readout to all supervised token positions（对所有监督 token 位置应用读出）, testing depth-wise mixing（深度混合） itself. |
| `attnres_h2_sid_only` | `sid_only` | Apply AttnRes readout only when predicting SID tokens（只在预测语义标识 token 时应用读出）. |
| `attnres_h3_level_aware` | `level_aware` | Use separate readout routes for `<a_*>`, `<b_*>`, and `<c_*>`（为三层语义标识使用不同读出路径）. |

## Files（文件）

- `configs/sft_attnres_h2_sid_only_title_off_desc_p05_4gpu.yaml`
- `configs/eval_attnres_h2_sid_only_title_off_desc_p05_4gpu.yaml`
- `configs/sft_attnres_h1_global_title_off_desc_p05_4gpu.yaml`
- `configs/eval_attnres_h1_global_title_off_desc_p05_4gpu.yaml`
- `configs/sft_attnres_h3_level_aware_title_off_desc_p05_4gpu.yaml`
- `configs/eval_attnres_h3_level_aware_title_off_desc_p05_4gpu.yaml`
- `scripts/experiment_attnres_phase1_sft_eval_chain.sh`
- `scripts/launch_attnres_phase1_sft_eval_tmux.sh`
- `refine-logs/attnres_sft_registry.csv`

Large SFT checkpoints（大监督微调权重） are written under:

- `/data/leejt/OneRec/output_weights/`
