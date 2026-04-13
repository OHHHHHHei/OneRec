# Top-k and Error Distribution Comparison for V2 Recipe Isolation

## Scope

This note summarizes two focused comparisons around the current `v2` tokenizer on Industrial:

1. `strongest original MiniOneRec SFT (title_history2sid_off + desc_align_p05)` vs `v2_on_p05`
2. `v2_on_off` vs `v2_on_p05`

The goal is to understand:

- how far the current best `v2` run is from the strongest original MiniOneRec SFT,
- and what exactly `desc_align_p05` changes when `title_history2sid` stays on.

Supporting artifacts:

- [TOPK_STRONGEST_ORIG_VS_V2_ON_P05.md](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-11_mgr_sid_v2_recipe_isolation_industrial/TOPK_STRONGEST_ORIG_VS_V2_ON_P05.md)
- [TOPK_V2_ON_OFF_VS_V2_ON_P05.md](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-11_mgr_sid_v2_recipe_isolation_industrial/TOPK_V2_ON_OFF_VS_V2_ON_P05.md)
- [strongest_orig_sid_diagnostics.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-11_mgr_sid_v2_recipe_isolation_industrial/strongest_orig_sid_diagnostics.json)
- [v2_on_p05_sid_diagnostics.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-11_mgr_sid_v2_recipe_isolation_industrial/v2_on_p05_sid_diagnostics.json)
- [v2_on_off_sid_diagnostics.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-11_mgr_sid_v2_recipe_isolation_industrial/v2_on_off_sid_diagnostics.json)

## Raw Table

| Run | title_history2sid | desc_align | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | HR@1 | HR@3 | HR@5 | HR@10 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| strongest original SFT | off | p05 | 0.06706 | 0.08501 | 0.09315 | 0.10372 | 0.06706 | 0.09839 | 0.11824 | 0.15089 |
| `v2_on_p05` | on | p05 | 0.07059 | 0.08451 | 0.09253 | 0.10271 | 0.07059 | 0.09508 | 0.11471 | 0.14626 |
| `v2_on_off` | on | off | 0.07037 | 0.08393 | 0.09053 | 0.10082 | 0.07037 | 0.09420 | 0.11030 | 0.14251 |

## Comparison A: strongest original SFT vs `v2_on_p05`

### Observation

`v2_on_p05` is now very close to the strongest original MiniOneRec SFT, but it still does not exceed it overall.

- `NDCG@1`: `0.06706 -> 0.07059`, `+0.00353`
- `NDCG@10`: `0.10372 -> 0.10271`, `-0.00101`
- `HR@10`: `0.15089 -> 0.14626`, `-0.00463`
- `HR@50`: `0.24531 -> 0.24818`, `+0.00287`

This means the current best `v2` already wins at the very top of the ranking and slightly wins again at very deep beam coverage, but still loses in the middle cutoffs that matter most for the main table.

### Top-k Structure

From [TOPK_STRONGEST_ORIG_VS_V2_ON_P05.md](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-11_mgr_sid_v2_recipe_isolation_industrial/TOPK_STRONGEST_ORIG_VS_V2_ON_P05.md):

- `top1`: `+0.00353`
- `top3`: `-0.00331`
- `top5`: `-0.00353`
- `top10`: `-0.00463`
- `top50`: `+0.00287`

The shape is now very clear:

- `v2_on_p05` is better at putting the correct item at rank 1.
- strongest original SFT is still better at keeping the target inside the short-to-mid beam (`top3/top5/top10`).
- `v2_on_p05` recovers some ground again at `top50`.

This is not a generic collapse. It is a very specific re-ranking pattern.

### Same-prefix / Error Distribution

From the diagnostics JSONs:

- `beam_contains_same_l1_rate`: `0.63402 -> 0.49327`, `-0.14075`
- `beam_contains_same_l2_rate`: `0.31480 -> 0.31149`, `-0.00331`
- `top1_error_same_l1_rate`: `0.21447 -> 0.08830`, `-0.12617`
- `top1_error_same_l2_rate`: `0.06952 -> 0.03276`, `-0.03676`

Interpretation:

- strongest original SFT keeps more same-`l1` neighbors in the beam.
- `v2_on_p05` makes far fewer same-prefix top1 mistakes when it is wrong.

So the current `v2` is more selective and less locally confused, but it also keeps a narrower semantic/prefix neighborhood in the beam.

### Hard-case Behavior

Fanout bucket analysis shows the current gap is **not** because `v2_on_p05` dominates the hardest crowded cases.

On the strongest-original baseline's `l2>=4` bucket:

- `top3`: `0.12357 -> 0.11286`, `-0.01071`
- `top10`: `0.23571 -> 0.21500`, `-0.02071`

Also, the `worsened` sets are highly concentrated on examples that the strongest original baseline already kept in the correct neighborhood:

- for `top10`, worsened examples have baseline same-`l1` rate = `1.00000`
- for `top10`, worsened examples have baseline same-`l2` rate = `1.00000`

This says the strongest original baseline is still better at stable neighborhood retention on many hard examples.

### Takeaway

The remaining gap to the strongest original MiniOneRec SFT is now:

- **not** a tokenizer-wide failure,
- **not** a top1 failure,
- but mainly a **mid-beam neighborhood-retention gap**.

Current `v2_on_p05` is sharper at the head and cleaner in same-prefix error behavior, while strongest original SFT is still stronger at keeping the correct target in `top3/top5/top10`.

## Comparison B: `v2_on_off` vs `v2_on_p05`

### Observation

When `title_history2sid` stays on, adding `desc_align_p05` is a net positive for the current `v2`.

- `NDCG@10`: `0.10082 -> 0.10271`, `+0.00189`
- `HR@10`: `0.14251 -> 0.14626`, `+0.00375`
- `HR@5`: `0.11030 -> 0.11471`, `+0.00441`
- `HR@50`: `0.24289 -> 0.24818`, `+0.00529`

So `desc_align_p05` is not the source of the earlier strongest-recipe failure. Under `title_history2sid_on`, it helps.

### Top-k Structure

From [TOPK_V2_ON_OFF_VS_V2_ON_P05.md](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-11_mgr_sid_v2_recipe_isolation_industrial/TOPK_V2_ON_OFF_VS_V2_ON_P05.md):

- `top1`: `+0.00022`
- `top3`: `+0.00088`
- `top5`: `+0.00441`
- `top10`: `+0.00375`
- `top50`: `+0.00529`

This is a broad positive shift, not a narrow metric artifact.

### Error Distribution

Diagnostics say:

- `beam_contains_same_l2_rate`: `0.30333 -> 0.31149`, `+0.00816`
- `top1_error_same_l2_rate`: `0.03607 -> 0.03276`, `-0.00331`
- `top1_hit_rate_for_collided_targets`: `0.62977 -> 0.65649`, `+0.02672`

Interpretation:

- `desc_align_p05` helps the current `v2` more on collided or locally ambiguous targets.
- It slightly improves same-`l2` retention and reduces same-`l2` top1 confusion.

### Fanout Pattern

The gain is not uniform across buckets.

For `l2>=4` under the `v2_on_off` baseline:

- `top3`: `0.09901 -> 0.08614`, `-0.01287`
- `top5`: `0.12475 -> 0.13762`, `+0.01287`
- `top10`: `0.18911 -> 0.20891`, `+0.01980`

This suggests:

- `desc_align_p05` does not necessarily help immediate short-range recovery inside the hardest crowded subtrees,
- but it improves medium-depth beam placement and lets more hard examples stay inside `top5/top10`.

### Takeaway

Under `title_history2sid_on`, `desc_align_p05` is a valid improvement for the current `v2` recipe.  
The strongest-recipe failure came mainly from turning `title_history2sid` off, not from adding description alignment.

## Main Conclusions

1. The current best `v2` downstream recipe is now clearly `title_history2sid_on + desc_align_p05`.
2. Compared with strongest original MiniOneRec SFT, `v2_on_p05` already wins on `top1` and slightly wins again at `top50`, but still loses on the crucial `top3/top5/top10` middle beam.
3. The remaining gap is best described as a **mid-beam neighborhood-retention gap**, not a head-ranking or tokenizer-collapse problem.
4. `desc_align_p05` helps `v2` when `title_history2sid` stays on; it is not the culprit.
5. The main negative factor in the previous failed strongest-recipe run was `title_history2sid_off`.

## Implication for Next Step

The current evidence supports this next move:

- treat `v2_on_p05` as the best current SFT recipe for `v2`,
- and, before changing the tokenizer again, seriously consider whether the next stage should be built on `v2_on_p05` rather than on the original `title_history2sid_off + desc_align_p05` recipe.

