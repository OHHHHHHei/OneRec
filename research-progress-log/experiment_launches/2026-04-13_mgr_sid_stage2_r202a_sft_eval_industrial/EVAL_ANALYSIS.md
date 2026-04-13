# 2026-04-13 Stage-2 `R202a` -> SFT/Evaluate Analysis

## Scope

This note analyzes the most recent SFT result in stage-2:

- `R208 = R202a tokenizer -> title_history2sid_on + desc_align_p05 -> SFT -> evaluate`

It compares `R208` from two angles:

1. vs current best `v2_on_p05 SFT`
2. vs strongest original MiniOneRec SFT

It also uses top-k structural analysis to answer the main question:

> what exactly changed after `R202a` was pushed into downstream SFT?

## Relevant Artifacts

- `RESULTS.md`
- `TOPK_V2_ON_P05_SFT_VS_R208.md`
- `TOPK_STRONGEST_ORIG_SFT_VS_R208.md`

## Raw Metric Table

| System | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | HR@1 | HR@3 | HR@5 | HR@10 | HR@20 | HR@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current best `v2_on_p05 SFT` | 0.07059 | 0.08451 | 0.09253 | 0.10271 | 0.11173 | 0.07059 | 0.09508 | 0.11471 | 0.14626 | 0.18244 | 0.24818 |
| `R208` | 0.06552 | 0.08360 | 0.09045 | 0.09974 | 0.10912 | 0.06552 | 0.09729 | 0.11383 | 0.14251 | 0.18001 | 0.23737 |
| strongest original MiniOneRec SFT | 0.06706 | 0.08501 | 0.09315 | 0.10372 | 0.11358 | 0.06706 | 0.09839 | 0.11824 | 0.15089 | 0.18972 | 0.24531 |

## Delta Table

### `R208 - current best v2_on_p05 SFT`

| Metric | Delta |
|---|---:|
| NDCG@1 | -0.00507 |
| NDCG@3 | -0.00091 |
| NDCG@5 | -0.00208 |
| NDCG@10 | -0.00297 |
| NDCG@20 | -0.00260 |
| HR@1 | -0.00507 |
| HR@3 | +0.00221 |
| HR@5 | -0.00088 |
| HR@10 | -0.00375 |
| HR@20 | -0.00243 |
| HR@50 | -0.01081 |

### `R208 - strongest original MiniOneRec SFT`

| Metric | Delta |
|---|---:|
| NDCG@1 | -0.00154 |
| NDCG@3 | -0.00141 |
| NDCG@5 | -0.00270 |
| NDCG@10 | -0.00398 |
| NDCG@20 | -0.00446 |
| HR@1 | -0.00154 |
| HR@3 | -0.00110 |
| HR@5 | -0.00441 |
| HR@10 | -0.00838 |
| HR@20 | -0.00971 |
| HR@50 | -0.00794 |

## Main Findings

### 1. There was no prior dedicated evaluate analysis for `R208`

Before this note, the run had:

- `README.md`
- `RESULTS.md`

but no dedicated evaluate-side interpretation. This file is the first explicit
analysis of the `R208` ranking profile.

### 2. `R208` is not a uniform regression, but it is still a clear downstream loss overall

Relative to the current best `v2_on_p05 SFT`:

- `HR@3` improves slightly: `+0.00221`
- but `NDCG@1`, `HR@1`, `NDCG@10`, `HR@10`, `NDCG@20`, `HR@20`, and `HR@50` all regress

So the right reading is **not**:

> `R202a` destroyed the downstream model everywhere.

The more accurate reading is:

> `R202a` shifts some examples into the `top2-3` region, but loses too much at
> `top1` and in the wider beam.

### 3. Against the current best `v2_on_p05 SFT`, the main pattern is:

- weaker `top1`
- slightly stronger `top3`
- weaker `top5/top10/top20/top50`

This is visible directly in the top-k structural table:

- `top1`: `0.07059 -> 0.06552`
- `top3`: `0.09508 -> 0.09729`
- `top5`: `0.11471 -> 0.11383`
- `top10`: `0.14626 -> 0.14251`
- `top20`: `0.18244 -> 0.18001`
- `top50`: `0.24818 -> 0.23737`

Interpretation:

> `R202a` can move some hard examples into the short beam, but it does not keep
> enough correct targets alive in the middle and deeper beam.

### 4. The gain is concentrated in hard crowded local cases, and the loss is concentrated on already-stable examples

From `TOPK_V2_ON_P05_SFT_VS_R208.md`:

On baseline fanout buckets:

- `l2<=2`
  - `top1`: `-0.00764`
  - `top3`: `-0.00732`
  - `top5`: `-0.00732`
  - `top10`: `-0.00637`

- `l2=3`
  - mixed, mostly small

- `l2>=4`
  - `top1`: `+0.01188`
  - `top3`: `+0.03564`
  - `top5`: `+0.01980`
  - `top10`: `+0.00198`

This is the clearest diagnostic in the whole run:

> `R202a` helps exactly where tokenizer-side hard local ambiguity is worst,
> but it hurts a larger pool of easier/stabler examples that the previous
> `v2_on_p05` model already handled reasonably well.

### 5. The `top3` improvement is real, but it is not enough to compensate for the broader retention loss

The rank transition matrix versus current `v2_on_p05 SFT` shows:

- many `>50` cases move into `11-20` / `21-50`
- some `6-10` or `11-20` cases also move upward
- but many baseline `top1` examples drop to `2-3`, `4-5`, or deeper

This is why we observe:

- `HR@3` improving
- while `NDCG@1` and `HR@1` worsen

And more importantly:

- `HR@50` falls by `0.01081`

So the move is not a net win in useful beam retention.

### 6. Against strongest original MiniOneRec SFT, `R208` is weaker at every reported cutoff

There is no cutoff among `@1/@3/@5/@10/@20` where `R208` wins against strongest original SFT.

The strongest deficits are:

- `HR@10`: `-0.00838`
- `HR@20`: `-0.00971`
- `HR@50`: `-0.00794`

This reinforces the same story:

> the main weakness is not head-side collapse, but insufficient retention in
> the useful beam.

### 7. Same-prefix beam retention drops noticeably

Compared with strongest original SFT, `R208` has much lower same-prefix
presence in the beam:

- top-10 same `l1`: `0.43062 -> 0.31171`
- top-10 same `l2`: `0.21090 -> 0.18906`
- top-20 same `l1`: `0.51335 -> 0.39135`
- top-20 same `l2`: `0.24796 -> 0.22943`

This does **not** mean stronger local structure is useless.
It means:

> the tokenizer-side cleanup achieved by `R202a` did not transfer into a
> downstream beam that preserves the right local neighborhood well enough.

## Interpretation

`R208` gives us a very specific message:

1. `R202a` is a real tokenizer-side structural refinement.
2. But once converted into a new SID space and pushed through SFT, its gain is
   mostly visible on hard crowded local cases.
3. The change also disrupts a larger set of easier examples, especially those
   where the old `v2_on_p05` beam already kept the target or the right local
   neighborhood.
4. So the current bottleneck is not:
   - “does stop-grad work?”
5. The current bottleneck is:
   - “why does tokenizer-side structural cleanup fail to preserve downstream
     beam retention?”

## Practical Conclusion

The final reading should be:

> `R202a` is a meaningful tokenizer-side result, but it is not a downstream
> winner. Its benefit is real but too localized, while the broader beam
> retention cost is too large.

This supports the current stage-2 conclusion:

> no first-round stage-2 refinement has surpassed the existing
> `v2_on_p05` downstream mainline.
