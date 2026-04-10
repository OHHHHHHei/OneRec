# 2026-04-10 MGR-SID SFT + Evaluate Results (Industrial)

## Status

Both parallel chains completed successfully:

- `mgr_upstream_baseline`
- `mgr_upstream_hierarchy`

Final checkpoints exist:

- `/home/leejt/OneRec/output/experiments/mgr_sid_sft_eval_industrial_20260410/mgr_upstream_baseline/sft/final_checkpoint`
- `/home/leejt/OneRec/output/experiments/mgr_sid_sft_eval_industrial_20260410/mgr_upstream_hierarchy/sft/final_checkpoint`

Final evaluate results exist:

- `/home/leejt/OneRec/results/experiments/mgr_sid_sft_eval_industrial_20260410/final_result_sft_mgr_upstream_baseline_Industrial_and_Scientific.json`
- `/home/leejt/OneRec/results/experiments/mgr_sid_sft_eval_industrial_20260410/final_result_sft_mgr_upstream_hierarchy_Industrial_and_Scientific.json`

## Raw Metrics

### Baseline

- `NDCG@[1,3,5,10,20,50] = [0.06309287, 0.07771807, 0.08560683, 0.09430022, 0.10419997, 0.11775708]`
- `HR@[1,3,5,10,20,50] = [0.06309287, 0.08824178, 0.10743437, 0.13434811, 0.17361571, 0.24244430]`

### Hierarchy

- `NDCG@[1,3,5,10,20,50] = [0.06265167, 0.08024707, 0.08641766, 0.09359572, 0.10289594, 0.11644559]`
- `HR@[1,3,5,10,20,50] = [0.06265167, 0.09287448, 0.10787558, 0.13037723, 0.16743878, 0.23604677]`

## Delta (Hierarchy - Baseline)

- `NDCG@1 = -0.00044120`
- `NDCG@3 = +0.00252900`
- `NDCG@5 = +0.00081083`
- `NDCG@10 = -0.00070450`
- `NDCG@20 = -0.00130403`
- `NDCG@50 = -0.00131149`

- `HR@1 = -0.00044120`
- `HR@3 = +0.00463270`
- `HR@5 = +0.00044121`
- `HR@10 = -0.00397088`
- `HR@20 = -0.00617693`
- `HR@50 = -0.00639753`

## Reading

Current tokenizer improvement does not cleanly transfer as a broad SFT gain.

- `hierarchy` is better on short-range ranking:
  - stronger `NDCG@3`
  - stronger `HR@3`
  - slightly stronger `NDCG@5 / HR@5`
- `baseline` remains better on:
  - `HR@1 / NDCG@1`
  - `HR/NDCG@10+`

So the current evidence is:

> hierarchy-aware SID helps local candidate quality, but the present SFT pipeline does not yet convert that into a consistent end-to-end recommendation improvement.
