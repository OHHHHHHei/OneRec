# 2026-04-12 V2 RL Evaluate Analysis

## Scope

This note analyzes the final `v2_on_p05 -> RL -> evaluate` result from three angles:

1. `v2_on_p05 RL` vs `v2_on_p05 SFT`
2. `v2_on_p05 RL` vs strongest original MiniOneRec RL
3. error-structure changes from SID diagnostics and top-k structural analysis

Relevant artifacts:

- `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-12_mgr_sid_v2_rl_on_p05_industrial/RESULTS.md`
- `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-12_mgr_sid_v2_rl_on_p05_industrial/TOPK_V2_SFT_VS_V2_RL.md`
- `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-12_mgr_sid_v2_rl_on_p05_industrial/TOPK_STRONGEST_ORIG_RL_VS_V2_RL.md`
- `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-12_mgr_sid_v2_rl_on_p05_industrial/strongest_orig_rl_sid_diagnostics.json`
- `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-12_mgr_sid_v2_rl_on_p05_industrial/v2_rl_sid_diagnostics.json`

## Headline

The final picture is not “RL failed” and also not “RL fully won”.

The more accurate reading is:

> `v2_on_p05` survives into RL and becomes stronger at the head of the ranking, but the remaining weakness is still mid-beam retention, especially around `top5/top10`.

## Raw Metric Pattern

### `v2_on_p05 RL` vs `v2_on_p05 SFT`

- `NDCG@1/3/5/10`: all improve
- `HR@1/3/5`: improve
- `HR@10/20/50`: decrease

This means RL makes the model more decisive and better at early ranking, but it does not preserve as many correct targets in the mid/deep beam.

### `v2_on_p05 RL` vs strongest original MiniOneRec RL

- better at `@1/@3`
- worse at `@5/@10/@20`
- better again at `@50`

This is a very specific U-shaped difference:

- stronger exact/top-short ranking
- weaker middle-range retention
- stronger large-beam recovery at `top50`

So the gap is no longer “overall weaker everywhere”.
It is now a very structured redistribution of rank mass.

## What RL Changed Relative to `v2_on_p05 SFT`

From `/TOPK_V2_SFT_VS_V2_RL.md`:

- `top1`: `+0.00375`
- `top3`: `+0.00772`
- `top5`: `+0.00221`
- `top10`: `-0.00441`
- `top50`: `-0.01081`

This is the clearest signal that RL sharpened the head but narrowed the beam.

### Fanout-sensitive effect

The gain is highly concentrated on hard crowded local cases:

- on `l2>=4`, `top10` improves by `+0.04752`
- on `l2<=2`, `top10` drops by `-0.01560`
- on `l2=3`, `top10` drops by `-0.04974`

Interpretation:

> RL amplifies the main `v2` strength on hard local ambiguity, but it also over-compresses many already-stable easier examples.

### Improved vs worsened sets

For `top10`:

- improved count: `153`
- worsened count: `173`

But the two groups are very different:

- improved samples come from higher-fanout and less-stable local structures
- worsened samples are almost always cases where the baseline beam already kept the correct local neighborhood

This is exactly the same mechanism we had already seen at SFT time, but RL makes the “head sharpening vs retention loss” tradeoff stronger.

## What Still Separates `v2_on_p05 RL` from Strongest Original RL

From `/TOPK_STRONGEST_ORIG_RL_VS_V2_RL.md`:

- `top1`: `+0.00110`
- `top3`: `+0.00243`
- `top5`: `-0.00287`
- `top10`: `-0.00949`
- `top20`: `-0.00154`
- `top50`: `+0.01743`

This means strongest original RL is not winning because of top1 precision.
It is winning because it is better at keeping the target inside the useful mid-range beam.

### Same-prefix retention vs same-prefix confusion

From SID diagnostics:

strongest original RL:

- `beam_contains_same_l1_rate = 0.41805`
- `beam_contains_same_l2_rate = 0.25347`
- `top1_error_same_l1_rate = 0.21519`
- `top1_error_same_l2_rate = 0.07712`

`v2_on_p05 RL`:

- `beam_contains_same_l1_rate = 0.39157`
- `beam_contains_same_l2_rate = 0.26473`
- `top1_error_same_l1_rate = 0.11130`
- `top1_error_same_l2_rate = 0.05863`

This is very informative:

- strongest original RL keeps more same-`l1` neighborhood candidates in the beam
- `v2_on_p05 RL` makes far fewer same-prefix top1 mistakes

So the remaining tradeoff is:

> original RL has better neighborhood retention,
> while `v2_on_p05 RL` has cleaner local disambiguation.

### Collided-target behavior

Another strong signal:

- strongest original RL top1 hit on collided targets: `61.51%`
- `v2_on_p05 RL` top1 hit on collided targets: `74.43%`

This is a major gain.
It strongly suggests the `v2` tokenizer is still doing exactly what we designed it to do, even after RL:
it is much better at resolving collision-heavy / ambiguity-heavy targets.

## Final Reading

The final evaluate result supports the following interpretation:

1. `v2` is not an SFT-only effect. Its main benefit survives into RL.
2. RL further strengthens `v2` on hard ambiguity-heavy cases and on head ranking.
3. The remaining gap to strongest original MiniOneRec RL is no longer a general quality gap.
4. The remaining gap is primarily a **mid-beam retention gap**:
   - original RL keeps more useful candidates in `top5/top10`
   - `v2 RL` is more decisive, cleaner on same-prefix confusion, and better on collided targets
   - but still loses too many already-salvageable middle-rank cases

## Practical Conclusion

At this point, the most useful summary is:

> `v2_on_p05 RL` has already crossed the “method survives end-to-end training” threshold.
> It beats the strongest original MiniOneRec SFT on `NDCG@10`, and approaches the strongest original RL from a different tradeoff profile.
> The next improvement target should focus on retention in the `top5/top10` band, not on rethinking the tokenizer from scratch.
