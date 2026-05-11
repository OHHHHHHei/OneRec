# L2 Square Candidate（第二层平方图候选）

Status（状态）: `evaluated-negative / registry-backfill-needed（已评测负向 / 需补总账）`

Last updated（更新日期）: `2026-05-11`

This candidate tests whether L2（第二层） should use an A-local squared graph（局部图平方） instead of the current local graph（当前局部图） view.

## Variants（变体）

- `r690b_lmh_l2_square_dominant_b025`: L2 graph（第二层图） uses `RowNorm(0.25 * A_local + A_local^2)`.
- `r690b_lmh_l2_square_only`: L2 graph（第二层图） uses `RowNorm(A_local^2)`.

## Tokenizer Diagnosis（分词器诊断）

`square_dominant_b025`:

- generated collision rate（生成冲突率）: `0.002984`
- max conflict（最大冲突簇）: `2`
- structural profile（结构画像）: `separating-but-risky（拆分强但有风险）`
- interpretation（解释）: stronger S-near C-far separation（语义近协同远拆分更强）, but weaker S-near C-near preservation（语义近协同近保持更弱）.

`square_only`:

- generated collision rate（生成冲突率）: `0.032013`
- max conflict（最大冲突簇）: `41`
- interpretation（解释）: too risky for immediate SFT（监督微调）.

## SFT Result（监督微调结果）

For `square_dominant_b025`, the result JSON（结果文件） exists and metrics were recomputed from the per-sample evaluate output（逐样本评测输出）:

| Metric（指标） | @1 | @3 | @5 | @10 |
| --- | ---: | ---: | ---: | ---: |
| NDCG（归一化折损累计增益） | 0.062652 | 0.078404 | 0.085286 | 0.093804 |
| HR（命中率） | 0.062652 | 0.090227 | 0.106993 | 0.133686 |

Result path（结果路径）:

- [final_result_sft_mgr_r690b_lmh_l2_square_dominant_b025_title_on_desc_p05_4gpu_Industrial_and_Scientific.json](/home/leejt/OneRec/results/experiments/mgr_sid_l2_square_sft_eval_20260509/final_result_sft_mgr_r690b_lmh_l2_square_dominant_b025_title_on_desc_p05_4gpu_Industrial_and_Scientific.json)

Verdict（裁决）:

- Negative versus current mainline（相对当前主线负向）.
- Negative versus strongest original SFT baseline（相对原版最强监督微调基线负向）.
- This is an example where stronger-looking structural separation（结构拆分） did not translate into downstream learnability（下游可学习性）.

Maintenance note（维护备注）:

- If this result is used in a formal comparison（正式对比）, backfill it into `sft_registry.csv` and `downstream_scoreboard.csv` first.
