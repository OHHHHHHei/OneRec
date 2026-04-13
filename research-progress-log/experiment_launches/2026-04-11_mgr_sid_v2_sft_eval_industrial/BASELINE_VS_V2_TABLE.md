# MiniOneRec Baseline vs MGR-SID v2 (Industrial, SFT + Evaluate)

这里的 `baseline` 默认指当前已经对齐并复现的 **MiniOneRec baseline**：

- baseline result:
  `/home/leejt/OneRec/results/experiments/mgr_sid_sft_eval_industrial_20260410/final_result_sft_mgr_upstream_baseline_Industrial_and_Scientific.json`
- v2 result:
  `/home/leejt/OneRec/results/experiments/mgr_sid_v2_sft_eval_industrial_20260411/final_result_sft_mgr_tokenizer_v2_offline_Industrial_and_Scientific.json`

## Main Table

| Metric | MiniOneRec baseline | MGR-SID v2 | Delta (v2 - baseline) |
|---|---:|---:|---:|
| NDCG@1 | 0.06309 | 0.07037 | +0.00728 |
| NDCG@3 | 0.07772 | 0.08393 | +0.00621 |
| NDCG@5 | 0.08561 | 0.09053 | +0.00492 |
| NDCG@10 | 0.09430 | 0.10082 | +0.00652 |
| NDCG@20 | 0.10420 | 0.11035 | +0.00615 |
| NDCG@50 | 0.11776 | 0.12273 | +0.00497 |
| HR@1 | 0.06309 | 0.07037 | +0.00728 |
| HR@3 | 0.08824 | 0.09420 | +0.00596 |
| HR@5 | 0.10743 | 0.11030 | +0.00287 |
| HR@10 | 0.13435 | 0.14251 | +0.00816 |
| HR@20 | 0.17362 | 0.18045 | +0.00684 |
| HR@50 | 0.24244 | 0.24289 | +0.00044 |

## Takeaway

- `v2` 在所有报告 cutoff 上都优于当前复现的 `MiniOneRec baseline`。
- 最显著的提升出现在：
  - `NDCG@1`: `+0.00728`
  - `HR@1`: `+0.00728`
  - `NDCG@10`: `+0.00652`
  - `HR@10`: `+0.00816`
- 这说明 `v2 offline_combined` 不只是改善了局部结构，也已经把收益传递到了下游 `SFT/evaluate`。
