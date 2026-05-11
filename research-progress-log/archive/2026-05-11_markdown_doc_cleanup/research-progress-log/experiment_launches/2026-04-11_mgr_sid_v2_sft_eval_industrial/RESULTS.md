# 2026-04-11 MGR-SID v2 SFT + Evaluate Results (Industrial)

## Result Files

- `v2` result:
  `/home/leejt/OneRec/results/experiments/mgr_sid_v2_sft_eval_industrial_20260411/final_result_sft_mgr_tokenizer_v2_offline_Industrial_and_Scientific.json`
- internal control result:
  `/home/leejt/OneRec/results/experiments/mgr_sid_sft_eval_industrial_20260410/final_result_sft_mgr_upstream_baseline_Industrial_and_Scientific.json`
- recipe-aligned original MiniOneRec:
  `record_id = sft_industrial_noalign_20260323_235623`
- strongest original MiniOneRec SFT:
  `record_id = sft_industrial_title_history2sid_off__desc_align_p05_20260325_192249`
- strongest original MiniOneRec RL:
  `record_id = rl_industrial_title_history2sid_off__desc_align_p05_batch256_20260329_152417`

## Task Recipe

Current `v2` uses:

- `title_history2sid = on`
- `SID-description alignment = off`

So the fairest original MiniOneRec comparison is:

- `title_history2sid_on__desc_align_off`

## Main Metrics

| Run | Recipe | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | HR@1 | HR@3 | HR@5 | HR@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v2` | `title_history2sid_on + desc_align_off` | 0.07037 | 0.08393 | 0.09053 | 0.10082 | 0.07037 | 0.09420 | 0.11030 | 0.14251 |
| recipe-aligned original MiniOneRec | `title_history2sid_on + desc_align_off` | 0.06243 | 0.07982 | 0.08688 | 0.09872 | 0.06243 | 0.09287 | 0.11008 | 0.14714 |
| strongest original MiniOneRec SFT | `title_history2sid_off + desc_align_p05` | 0.06706 | 0.08501 | 0.09315 | 0.10372 | 0.06706 | 0.09839 | 0.11824 | 0.15089 |
| strongest original MiniOneRec RL | `title_history2sid_off + desc_align_p05_batch256` | 0.07324 | 0.08903 | 0.09704 | 0.10726 | 0.07324 | 0.10038 | 0.11979 | 0.15133 |

## Comparison 1: Internal Control

Relative to the internal control `mgr_upstream_baseline`, `v2` is better on all reported cutoffs:

- `NDCG@1`: `0.06309 -> 0.07037` (`+0.00728`)
- `NDCG@3`: `0.07772 -> 0.08393` (`+0.00621`)
- `NDCG@5`: `0.08561 -> 0.09053` (`+0.00492`)
- `NDCG@10`: `0.09430 -> 0.10082` (`+0.00652`)
- `HR@1`: `0.06309 -> 0.07037` (`+0.00728`)
- `HR@3`: `0.08824 -> 0.09420` (`+0.00596`)
- `HR@5`: `0.10743 -> 0.11030` (`+0.00287`)
- `HR@10`: `0.13435 -> 0.14251` (`+0.00816`)

This shows that `v2` is not just cleaner at the tokenizer level; it already improves downstream ranking under the current controlled pipeline.

## Comparison 2: Recipe-Aligned Original MiniOneRec

Under the same task recipe (`title_history2sid_on + desc_align_off`), `v2` improves:

- `NDCG@1`: `+0.00794`
- `NDCG@3`: `+0.00411`
- `NDCG@5`: `+0.00365`
- `NDCG@10`: `+0.00210`
- `HR@1`: `+0.00794`
- `HR@3`: `+0.00132`
- `HR@5`: `+0.00022`

But it still trails slightly on:

- `HR@10`: `0.14714 -> 0.14251` (`-0.00463`)

Interpretation:

- `v2` is already better than the original MiniOneRec under a matched task recipe.
- The gain is most visible in ranking quality (`NDCG`) and shorter cutoffs.
- The remaining weakness is deeper candidate retention (`HR@10`).

## Comparison 3: Strongest Original MiniOneRec

`v2` does **not** yet surpass the strongest original MiniOneRec recipe.

Relative to the strongest original MiniOneRec SFT:

- `NDCG@1`: `+0.00331`
- `NDCG@3`: `-0.00108`
- `NDCG@5`: `-0.00262`
- `NDCG@10`: `-0.00290`
- `HR@1`: `+0.00331`
- `HR@3`: `-0.00419`
- `HR@5`: `-0.00794`
- `HR@10`: `-0.00838`

Relative to the strongest original MiniOneRec RL:

- `NDCG@10`: `0.10082 vs 0.10726`
- `HR@10`: `0.14251 vs 0.15133`

So the current position is:

- `v2` has passed the internal control.
- `v2` has passed the recipe-aligned original MiniOneRec on most key metrics.
- `v2` has **not** yet passed the strongest original MiniOneRec system.

## Current Conclusion

The most accurate conclusion is:

> `v2 tokenizer` is already a valid positive direction, not only in internal control experiments but also against the recipe-aligned original MiniOneRec baseline. However, the next decisive test is to plug `v2 tokenizer` into the strongest original MiniOneRec recipe (`title_history2sid_off + desc_align_p05`) rather than staying on the current no-alignment task setting.
