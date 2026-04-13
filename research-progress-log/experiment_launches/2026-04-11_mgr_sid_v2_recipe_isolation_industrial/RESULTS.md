# Results: MGR-SID V2 Recipe Isolation (Industrial)

## Four-cell matrix

| Run | title_history2sid | desc_align | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | HR@1 | HR@3 | HR@5 | HR@10 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v2_on_off` | on | off | 0.07037 | 0.08393 | 0.09053 | 0.10082 | 0.07037 | 0.09420 | 0.11030 | 0.14251 |
| `v2_on_p05` | on | p05 | 0.07059 | 0.08451 | 0.09253 | 0.10271 | 0.07059 | 0.09508 | 0.11471 | 0.14626 |
| `v2_off_off` | off | off | 0.05890 | 0.07413 | 0.08106 | 0.09125 | 0.05890 | 0.08537 | 0.10236 | 0.13391 |
| `v2_off_p05` | off | p05 | 0.05912 | 0.07366 | 0.08019 | 0.08993 | 0.05912 | 0.08427 | 0.10015 | 0.13082 |

## Main takeaways

1. The best current downstream recipe for `v2` is `title_history2sid_on + desc_align_p05`.
2. `desc_align_p05` is beneficial for `v2` when `title_history2sid` stays on.
3. The main negative factor in the previous strongest-recipe mismatch is `title_history2sid_off`.
4. The current best `v2_on_p05` is already very close to the strongest original MiniOneRec SFT:
   - `NDCG@10`: `0.10271` vs `0.10372`
   - `HR@10`: `0.14626` vs `0.15089`

## Strongest-original comparison

| Run | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | HR@1 | HR@3 | HR@5 | HR@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| strongest original MiniOneRec SFT | 0.06706 | 0.08501 | 0.09315 | 0.10372 | 0.06706 | 0.09839 | 0.11824 | 0.15089 |
| `v2_on_p05` | 0.07059 | 0.08451 | 0.09253 | 0.10271 | 0.07059 | 0.09508 | 0.11471 | 0.14626 |

Current interpretation:

- `v2_on_p05` is already stronger at `top1`.
- the remaining gap is mainly a `top3/top5/top10` mid-beam neighborhood-retention gap.
- this is a downstream recipe interaction issue, not evidence that the tokenizer direction failed.
