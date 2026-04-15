# Stage-2 Review Summary

**Date**: 2026-04-13  
**Project**: MGR-SID / ambiguity-aware tokenizer refinement for generative recommendation  
**Dataset**: Amazon Industrial and Scientific  
**Purpose**: reviewer-facing summary of the recent experiment cycle, from motivation to design to results

## 1. Motivation

The project already established a viable `v2` tokenizer line:

- graph-structured collaborative information is injected into SID generation
- the effect survives downstream `SFT` and `RL`
- the strongest current `v2` mainline is:
  - tokenizer: `v2`
  - recipe: `title_history2sid_on + desc_align_p05`
  - downstream chain: `SFT -> RL -> evaluate`

However, the current best `v2` result still does **not** fully surpass the
strongest reproduced original MiniOneRec result.

### Strongest Original vs Current Best `v2`

| System | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | HR@1 | HR@3 | HR@5 | HR@10 | HR@20 | HR@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| strongest original SFT | 0.06706 | 0.08501 | 0.09315 | 0.10372 | 0.11358 | 0.06706 | 0.09839 | 0.11824 | 0.15089 | 0.18972 | 0.24531 |
| strongest original RL | 0.07324 | 0.08903 | 0.09704 | 0.10726 | 0.11366 | 0.07324 | 0.10038 | 0.11979 | 0.15133 | 0.17670 | 0.21994 |
| current best `v2` SFT (`v2_on_p05`) | 0.07059 | 0.08451 | 0.09253 | 0.10271 | 0.11173 | 0.07059 | 0.09508 | 0.11471 | 0.14626 | 0.18244 | 0.24818 |
| current best `v2` RL (`v2_on_p05`) | 0.07434 | 0.09054 | 0.09630 | 0.10432 | 0.11269 | 0.07434 | 0.10280 | 0.11692 | 0.14185 | 0.17516 | 0.23737 |

### What this meant before Stage-2

The strongest `v2` line already gave us a clear message:

- `v2` is end-to-end valid
- `v2` improves head ranking and hard ambiguity cases
- the remaining gap is **not** a total quality gap
- the remaining gap is a **retention gap**, especially around `top5/top10/top20`

This motivated a focused Stage-2:

> can a small tokenizer-side refinement improve local ambiguity structure in a
> way that finally transfers into stronger downstream beam retention?

## 2. Stage-2 Hypothesis

Stage-2 was built around one specific working diagnosis:

> the current `v2` tokenizer may still suffer from cross-level gradient
> interference, and a cleaner hierarchy-aware training interface may improve
> local ambiguity structure and ultimately help downstream retention.

Two tokenizer-side refinement directions were tested:

1. **Block 2: stop-gradient hierarchy isolation**
   - isolate graph loss so each level mainly updates its own codebook

2. **Block 3: stronger semantic retention**
   - replace semantic smoothness with a more structure-aware semantic retention term

Then we tested whether any tokenizer-side gain actually transfers downstream.

## 3. Stage-2 Experimental Design

### Block 2: Stop-Gradient Variants

- `R202a`: stop-gradient hierarchy isolation
- `R202b`: stop-gradient + stronger level-1 coarse compensation (`coarse_weight = 0.10`)
- `R202b-r075`: stop-gradient + smaller level-1 compensation (`coarse_weight = 0.075`)

### Block 3: Semantic-Retention Variant

- `R205`: `R202a`-style stop-grad + batch-local semantic neighborhood KL

### Downstream Screen

- `R208`: `R202a -> title_history2sid_on + desc_align_p05 -> SFT -> evaluate`

### Interface Diagnostics

After stage-2 revealed that structure gain did not automatically transfer, we
added a diagnostic package:

- `R301`: prefix stability / SID rearrangement
- `R302`: code polysemy / semantic consistency
- `R303`: structure-to-downstream transfer attribution
- `R304`: lightweight SID learnability probe

## 4. Tokenizer-Side Results

### Structural Metrics Summary

| Variant | generated collision | target-weighted mean l2 leaf | fraction in deep crowded `l2>=4` | target-weighted `H(l3|l1,l2)` | multi-leaf `same_l2` |
|---|---:|---:|---:|---:|---:|
| current `v2` | 13 / 3686 | 4.3422 | 0.2228 | 1.1001 | 0.4873 |
| `R202a` | 13 / 3686 | 3.6148 | 0.1994 | 1.0308 | 0.4988 |
| `R202b-r075` | 12 / 3686 | 4.1266 | 0.2585 | 1.2128 | 0.5831 |
| `R205` | 12 / 3686 | 4.9572 | 0.2621 | 1.2623 | 0.5449 |

### Interpretation

#### `R202a`

`R202a` is the only clean tokenizer-side structural winner:

- mean `l2` fanout drops strongly
- deep crowded buckets shrink
- conditional entropy drops

This supports the claim that cross-level interference was a real issue.

But `R202a` is not a perfect win:

- final generated collision stays at `13 / 3686`
- multi-leaf `same_l2` rises slightly

So the right reading is:

> `R202a` improves hard local ambiguity structure, but does not fully solve the
> broader downstream transfer problem.

#### `R202b-r075`

`R202b-r075` improves final collision count from `13` to `12`, but local
structure regresses relative to both current `v2` and `R202a`.

So it is a tradeoff branch, not a clean stage-2 winner.

#### `R205`

`R205` also reaches `12 / 3686` generated collision, but its local ambiguity
structure becomes clearly worse:

- larger `l2` fanout
- more deep crowded targets
- higher conditional entropy

So the first `batch_local_kl` implementation is a negative result for the
retention-oriented stage-2 objective.

## 5. Downstream Screen: `R208`

`R208` tested whether the strongest tokenizer-side stage-2 branch (`R202a`)
could beat the existing `v2_on_p05` downstream mainline.

### `R208` vs current best `v2_on_p05 SFT`

| System | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | NDCG@20 | HR@1 | HR@3 | HR@5 | HR@10 | HR@20 | HR@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current best `v2_on_p05 SFT` | 0.07059 | 0.08451 | 0.09253 | 0.10271 | 0.11173 | 0.07059 | 0.09508 | 0.11471 | 0.14626 | 0.18244 | 0.24818 |
| `R208` | 0.06552 | 0.08360 | 0.09045 | 0.09974 | 0.10912 | 0.06552 | 0.09729 | 0.11383 | 0.14251 | 0.18001 | 0.23737 |
| delta (`R208 - v2_on_p05`) | -0.00507 | -0.00091 | -0.00208 | -0.00297 | -0.00260 | -0.00507 | +0.00221 | -0.00088 | -0.00375 | -0.00243 | -0.01081 |

### What `R208` tells us

`R208` is not a uniform regression, but it is still a downstream loss overall.

Its profile is:

- slightly better at `HR@3`
- worse at `@1`
- worse at `@5/@10/@20/@50`

The fanout-stratified analysis is the key:

- on hard crowded examples (`l2>=4`), `R208` improves
- on easier / already-stable examples (`l2<=2`), `R208` regresses across the board

This means:

> the tokenizer-side stop-grad refinement helps exactly the hard local cases it
> was designed to help, but it hurts a larger pool of easier examples that the
> previous `v2_on_p05` space already handled well.

So `R202a` is a meaningful tokenizer-side result, but not a downstream winner.

## 6. Interface Diagnostics: Why Didn’t `R202a` Transfer?

### R301: Prefix Stability / SID Rearrangement

Relative to current `v2`, `R202a` shows:

- changed `l1` rate: `0.9965`
- changed `l2` rate: `1.0000`
- changed full SID rate: `1.0000`

So `R202a` is a **near-full SID remapping**.

But it is not random:

- `41.4%` of same-`l1` baseline pairs remain same-`l1`
- `61.2%` of same-`l2` baseline pairs remain same-`l2`
- mean `l2` neighbor Jaccard remains `0.589`

Meaning:

> `R202a` preserves a moderate amount of local prefix structure, but changes the
> global SID interface almost completely.

### R302: Code Polysemy / Semantic Consistency

This was one of the most important surprises.

Current `v2` and `R202a` are almost identical on:

- `b` token semantic spread
- `c` token semantic spread
- prefix-conditioned `b` drift
- prefix-conditioned `c` drift

So the downstream loss is **not** well explained by:

> “the codes became much more semantically overloaded”

Instead, the more plausible issue is:

> the SID routing / reuse pattern changed a lot, while token-level semantic
> consistency stayed roughly the same.

### R303: Structure-to-Downstream Transfer Attribution

This block joins SID changes with `R208` improved/worsened examples.

The main pattern is:

- improved examples are harder
  - higher baseline `l2` fanout
- worsened examples are easier / more stable

Also:

- improved examples tend to have higher `l1` neighbor Jaccard but lower `l2` Jaccard
- worsened examples tend to have lower `l1` Jaccard but higher `l2` Jaccard

This gives us a sharper diagnosis:

> `R202a` helps when hard local `l2` neighborhoods need to be rewritten, but it
> hurts when broader `l1` routing for easier/stable examples is disturbed too much.

### R304: SID Learnability Probe

We trained a lightweight linear probe to predict:

- `a`
- `b_given_a`
- `c_given_ab`

Result:

- `R202a` makes `a` slightly easier to predict
- but makes `b_given_a` and `c_given_ab` harder to predict
- and this degradation is strongest on hard examples

This helps reconcile the stage-2 puzzle:

> `R202a` improved tokenizer structure, but likely made deeper conditional SID
> decisions harder for downstream models to learn.

## 7. Consolidated Conclusion

The most important finding of this cycle is:

> **tokenizer-side structural improvement does not automatically transfer into
> downstream ranking improvement.**

More specifically:

1. `R202a` proves that cross-level gradient interference was real and that
   tokenizer-side hard local ambiguity can be improved.
2. But `R202a` also triggers near-full SID remapping.
3. That remapping does not seem to worsen token semantic polysemy much.
4. Instead, it appears to:
   - preserve some local structure
   - but alter global routing enough to hurt stable examples
   - and make deeper-level SID decisions harder to learn

So the stage-2 outcome is not:

- “the tokenizer direction failed”

It is:

- “the transfer/interface problem is now more clearly exposed”

## 8. Current Project Position

As of now:

- the strongest tokenizer-side stage-2 branch is still `R202a`
- but no stage-2 variant has replaced the current mainline
- the strongest end-to-end `v2` branch remains:
  - `v2_on_p05 -> RL`

That line still has the following profile:

- stronger head ranking and hard ambiguity handling
- stronger collision-heavy target resolution
- stronger long-beam recovery at `top50`
- but still weaker than strongest original MiniOneRec RL in the `top5/top10/@20` retention band

## 9. Practical Takeaway for the Next Step

This experiment cycle suggests a very concrete lesson:

> if we continue tokenizer-side refinement, we should avoid full-SID remapping
> and prefer more conservative changes that preserve the learned interface.

That is why the most plausible next tokenizer-side direction is now:

- keep `l1/l2` more stable
- if refining again, consider **leaf-only / `l3`-only** changes

At the same time, because the strongest current mainline is still
`v2_on_p05 -> RL`, any next decision should be made relative to that line, not
relative to stage-2 local structural metrics alone.

## 10. Reviewer Reading Order

If a reviewer wants the shortest path through the evidence:

1. [2026-04-12_mgr_sid_v2_rl_on_p05_industrial/EVAL_ANALYSIS.md](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-12_mgr_sid_v2_rl_on_p05_industrial/EVAL_ANALYSIS.md)
2. [2026-04-13_mgr_sid_stage2_stopgrad_industrial/README.md](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_stopgrad_industrial/README.md)
3. [2026-04-13_mgr_sid_stage2_r202a_sft_eval_industrial/EVAL_ANALYSIS.md](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_r202a_sft_eval_industrial/EVAL_ANALYSIS.md)
4. [2026-04-13_mgr_sid_stage2_interface_diagnostics_industrial/README.md](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_interface_diagnostics_industrial/README.md)
5. [experiment_results.csv](/home/leejt/OneRec/experiment_results.csv)
