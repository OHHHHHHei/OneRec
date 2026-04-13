# 2026-04-11 MGR-SID v2 on Strongest Original MiniOneRec Recipe (Industrial)

## Result Files

- SFT log:
  `/home/leejt/OneRec/logs/experiment_mgr_sid_v2_sft_desc_align_p05_industrial_20260411.log`
- evaluate log:
  `/home/leejt/OneRec/logs/experiment_mgr_sid_v2_eval_desc_align_p05_industrial_20260411.log`
- final result json:
  `/home/leejt/OneRec/results/experiments/mgr_sid_v2_sft_desc_align_p05_industrial_20260411/final_result_sft_mgr_tokenizer_v2_desc_align_p05_Industrial_and_Scientific.json`

## Task Recipe

This run uses the strongest original MiniOneRec SFT recipe:

- `title_history2sid = off`
- `SID-description alignment = on`
- `description_task_probability = 0.5`

The tokenizer source is still:

- `mgr_tokenizer_v2_offline`

## Main Metrics

| Run | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | HR@1 | HR@3 | HR@5 | HR@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v2 + strongest-recipe SFT` | 0.05912 | 0.07366 | 0.08019 | 0.08993 | 0.05912 | 0.08427 | 0.10015 | 0.13082 |
| strongest original MiniOneRec SFT | 0.06706 | 0.08501 | 0.09315 | 0.10372 | 0.06706 | 0.09839 | 0.11824 | 0.15089 |
| strongest original MiniOneRec RL | 0.07324 | 0.08903 | 0.09704 | 0.10726 | 0.07324 | 0.10038 | 0.11979 | 0.15133 |
| recipe-aligned original MiniOneRec (`title_history2sid_on + desc_align_off`) | 0.06243 | 0.07982 | 0.08688 | 0.09872 | 0.06243 | 0.09287 | 0.11008 | 0.14714 |

## Direct Comparison

Relative to the strongest original MiniOneRec SFT:

- `NDCG@1`: `-0.00794`
- `NDCG@3`: `-0.01135`
- `NDCG@5`: `-0.01296`
- `NDCG@10`: `-0.01379`
- `HR@1`: `-0.00794`
- `HR@3`: `-0.01412`
- `HR@5`: `-0.01809`
- `HR@10`: `-0.02008`

Relative to the strongest original MiniOneRec RL:

- `NDCG@10`: `-0.01733`
- `HR@10`: `-0.02052`

Relative to the recipe-aligned original MiniOneRec (`title_history2sid_on + desc_align_off`):

- `NDCG@1`: `-0.00331`
- `NDCG@3`: `-0.00616`
- `NDCG@5`: `-0.00669`
- `NDCG@10`: `-0.00879`
- `HR@1`: `-0.00331`
- `HR@3`: `-0.00860`
- `HR@5`: `-0.00993`
- `HR@10`: `-0.01632`

## Training Notes

- training stopped at `epoch = 6.5`
- final tracked eval loss: `3.1991`
- final tracked train loss: `0.3959`
- wandb run id: `d5fhjtu1`

## Current Interpretation

This is a negative result.

The current `v2 tokenizer` does **not** transfer well when directly plugged into the strongest original MiniOneRec SFT recipe (`title_history2sid_off + desc_align_p05`).

The most important implication is:

> the current `v2` gains are not recipe-invariant.

In other words:

- `v2` works under the recipe-aligned noalign task setting
- but direct insertion into the strongest original MiniOneRec recipe causes a clear drop

This suggests that the current tokenizer improvement and the strongest original task recipe may be mismatched, rather than simply additive.
