# Evaluate Error Distribution Analysis

## Scope

This note analyzes the final `evaluate` outputs of:

- `mgr_upstream_baseline`
- `mgr_upstream_hierarchy`

using the diagnostic artifacts:

- [baseline_sid_diagnostics.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-10_mgr_sid_sft_eval_industrial/baseline_sid_diagnostics.json)
- [hierarchy_sid_diagnostics.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-10_mgr_sid_sft_eval_industrial/hierarchy_sid_diagnostics.json)
- [baseline_sid_diagnostics.csv](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-10_mgr_sid_sft_eval_industrial/baseline_sid_diagnostics.csv)
- [hierarchy_sid_diagnostics.csv](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-10_mgr_sid_sft_eval_industrial/hierarchy_sid_diagnostics.csv)

The goal is to answer one question:

> after switching to the hierarchy-aware SID, what kinds of evaluate errors are reduced, and what kinds of errors become worse?

## Headline

The current `hierarchy` SID helps on the exact kind of cases we expected:

- crowded local `same_l2` ambiguity
- high-fanout targets
- short-range reranking around difficult local neighborhoods

But it also introduces a second effect:

- some examples that the baseline already placed in the correct local neighborhood are pushed away from that neighborhood
- this hurts `top1` and part of the `@10+` metrics

So the present picture is not "hierarchy is uniformly better" but:

> hierarchy reduces local leaf ambiguity, yet the current tokenizer-to-SFT transfer still sacrifices some already-correct local routing.

## Global Diagnostic Comparison

### Catalog-level

| metric | baseline | hierarchy | delta |
|---|---:|---:|---:|
| unique SID count | 3673 | 3674 | +1 |
| collision rate | 0.3527% | 0.3256% | -0.0271% |
| weighted `H(level3 | level1, level2)` | 1.1120 | 0.8664 | -0.2456 |

Interpretation:

- `hierarchy` produces a slightly cleaner catalog
- the largest structural change is lower `l3` uncertainty under the same `l1,l2` prefix
- this is consistent with reduced local leaf ambiguity

### Evaluate-level

| metric | baseline | hierarchy | delta |
|---|---:|---:|---:|
| top1 hit | 0.06309 | 0.06265 | -0.00044 |
| top3 hit | 0.08824 | 0.09287 | +0.00463 |
| top5 hit | 0.10743 | 0.10788 | +0.00044 |
| top10 hit | 0.13435 | 0.13038 | -0.00397 |
| beam contains same `l1` | 0.67240 | 0.52482 | -0.14758 |
| beam contains same `l2` | 0.32716 | 0.30774 | -0.01941 |
| top1 error same `l1` | 0.18154 | 0.12356 | -0.05798 |
| top1 error same `l2` | 0.05251 | 0.03695 | -0.01556 |
| avg top1 LCP | 0.40856 | 0.33841 | -0.07015 |
| avg best LCP in beam | 1.24200 | 1.06861 | -0.17340 |

Interpretation:

- `hierarchy` clearly reduces same-prefix errors among top1 misses
- especially:
  - `same_l1` miss rate drops a lot
  - `same_l2` miss rate also drops materially
- but `hierarchy` also lowers prefix overlap in the beam
- this means the beam is less likely to stay near the baseline's local semantic neighborhood

This already explains the metric pattern:

- local reranking improves enough to help `@3`
- but some previously retained local candidates disappear from the beam, which hurts `@1` and `@10+`

## Per-example Transition Analysis

Using aligned row-wise comparison between the two diagnostics CSVs, the 4533 evaluation examples split into:

| case | count |
|---|---:|
| both hit | 221 |
| fixed by hierarchy | 63 |
| broken by hierarchy | 65 |
| both miss | 4184 |

So the current hierarchy model is close to neutral at top1 overall:

- `63` examples are fixed
- `65` examples are broken

The key is that these two groups are not the same kind of examples.

## What Hierarchy Fixes

### Fixed cases: baseline miss -> hierarchy hit

| metric | baseline-side value | hierarchy-side value |
|---|---:|---:|
| count | 63 | 63 |
| same `l1` rate | 0.4444 | 1.0000 |
| same `l2` rate | 0.1905 | 1.0000 |
| top3 hit | 0.3810 | 1.0000 |
| top5 hit | 0.5873 | 1.0000 |
| avg target `l2` fanout | 7.3175 | 6.0476 |
| collided target fraction | 0.2381 | 0.2381 |

Reading:

- these are hard cases
- baseline often already had the target in `top3/top5`, but not at top1
- their average `l2` fanout is high: `7.32`
- collided targets are also overrepresented
- after hierarchy, these cases become exact hits, so they naturally fall into the same `l1/l2` neighborhood with full rate

This is exactly the pattern we wanted:

> hierarchy is helping when the baseline is confused inside a crowded local neighborhood and needs a stronger tie-break.

## What Hierarchy Breaks

### Broken cases: baseline hit -> hierarchy miss

| metric | baseline-side value | hierarchy-side value |
|---|---:|---:|
| count | 65 | 65 |
| same `l1` rate | 1.0000 | 0.1692 |
| same `l2` rate | 1.0000 | 0.1385 |
| top3 hit | 1.0000 | 0.4615 |
| top5 hit | 1.0000 | 0.5692 |
| avg target `l2` fanout | 4.8000 | 5.3538 |
| collided target fraction | 0.1538 | 0.1538 |

Reading:

- these are not the most crowded cases
- baseline was already correct at top1
- after replacing the SID, the new system often fails to keep the target within top3/top5
- hierarchy is not merely "slightly reordering" these cases; it is sometimes moving away from the right neighborhood

So the current weakness is:

> hierarchy helps on difficult crowded cases, but it can over-correct and disrupt some examples that the baseline already handled correctly.

## Both-miss Region

For the 4184 examples that both systems still miss:

| metric | baseline | hierarchy | delta |
|---|---:|---:|---:|
| same `l1` miss rate | 0.1272 | 0.0875 | -0.0397 |
| same `l2` miss rate | 0.0504 | 0.0354 | -0.0151 |
| top3 hit | 0.0215 | 0.0256 | +0.0041 |
| top5 hit | 0.0392 | 0.0402 | +0.0010 |
| avg target `l2` fanout | 4.8420 | 4.5387 | -0.3033 |

This is another positive sign:

- even where hierarchy still does not solve the example
- it makes the residual error distribution less locally ambiguous
- and slightly improves short-range retrieval

## Bucketed Analysis by Local Crowding

We bucket examples by baseline target `l2` fanout:

| bucket | count | baseline top1 | hierarchy top1 | delta |
|---|---:|---:|---:|---:|
| `l2 <= 2` | 2259 | 0.02435 | 0.02302 | -0.00133 |
| `l2 = 3` | 778 | 0.20566 | 0.19794 | -0.00771 |
| `l2 >= 4` | 1496 | 0.04746 | 0.05214 | +0.00468 |

This is one of the clearest pieces of evidence in the whole analysis:

- hierarchy is worse on sparse / easier local structures
- hierarchy is better on crowded `l2 >= 4` structures

That matches the method motivation almost exactly.

## Recovery Rate by Baseline Error Type

Now bucket baseline top1 errors by their local relation to the target:

| baseline error type | count | hierarchy recovery rate |
|---|---:|---:|
| same `l2` error | 223 | 0.05381 |
| same `l1`-only error | 548 | 0.02920 |
| cross-prefix error | 3476 | 0.01007 |

Interpretation:

- hierarchy recovers `same_l2` errors much more often than cross-prefix errors
- recovery is almost `5.3x` higher than the cross-prefix bucket
- this strongly supports the original claim:

> the tokenizer change is not acting like a generic global improvement; it is specifically targeting local leaf ambiguity.

## Top-k Improvement vs Degradation

### `top3`

| type | count | baseline same `l1` | baseline same `l2` | baseline avg `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 119 | 0.5294 | 0.2353 | 10.3277 |
| worsened by hierarchy | 98 | 0.7959 | 0.4898 | 8.2755 |

### `top5`

| type | count | baseline same `l1` | baseline same `l2` | baseline avg `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 125 | 0.4560 | 0.2000 | 9.1680 |
| worsened by hierarchy | 123 | 0.7642 | 0.4472 | 8.5854 |

### `top10`

| type | count | baseline same `l1` | baseline same `l2` | baseline avg `l2` fanout |
|---|---:|---:|---:|---:|
| improved by hierarchy | 154 | 0.3766 | 0.1429 | 8.0844 |
| worsened by hierarchy | 172 | 0.6279 | 0.3547 | 8.8256 |

Interpretation:

- top-k gains still come from difficult, high-fanout cases
- but the worsened cases are disproportionately examples where baseline already stayed in the correct local neighborhood
- this again points to the same problem:
  - hierarchy is cleaning ambiguous local structures
  - but current SFT does not fully exploit that cleaner structure without losing some prefix-local retention

## Main Takeaways

### Confirmed

- `hierarchy` really does reduce local ambiguity at evaluate time
- this is visible both in the final SID structure and in the error distribution
- the strongest positive evidence is on:
  - crowded `same_l2` cases
  - high-fanout targets
  - short-range recovery from baseline local errors

### Not yet solved

- the gain is not yet a uniform end-to-end gain
- some already-correct baseline cases are broken
- the current SFT stack still appears to lose part of the baseline's local neighborhood retention

## Practical Reading for Next Step

The present evidence supports the following interpretation:

> tokenizer-side hierarchy-aware graph regularization is doing the intended structural job, but the current SFT adaptation is not yet aligned with that new SID well enough to turn local ambiguity reduction into a broad ranking gain.

So if we continue from here, the most reasonable next questions are:

1. can the SFT stage be made more robust to the new SID distribution?
2. can we preserve the baseline's coarse-prefix retention while keeping hierarchy's local leaf cleanup?
3. can training or decoding place more emphasis on the crowded `same_l2` buckets where hierarchy is already helping?
