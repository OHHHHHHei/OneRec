# 2026-04-11 Tokenizer v2 Structure Comparison

## Inputs

- MiniOneRec baseline:
  `/home/leejt/OneRec/output/experiments/mgr_sid_v1_upstream/generated_indices/Industrial_and_Scientific.mgr_upstream_baseline.index.json`
- `v1 hierarchy_reg`:
  `/home/leejt/OneRec/output/experiments/mgr_sid_v1_upstream/generated_indices/Industrial_and_Scientific.mgr_upstream_hierarchy.index.json`
- `v2 offline_combined`:
  `/home/leejt/OneRec/output/experiments/mgr_sid_tokenizer_v2/generated_indices/Industrial_and_Scientific.mgr_tokenizer_v2_offline.index.json`

Pairwise reports:

- baseline vs v2:
  `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-11_mgr_sid_tokenizer_v2_r005/LOCAL_AMBIGUITY_BASELINE_VS_V2.md`
- `v1` vs v2:
  `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-11_mgr_sid_tokenizer_v2_r005/LOCAL_AMBIGUITY_HIERARCHY_VS_V2.md`

## Main Finding

`v2 offline_combined` is structurally cleaner than both the MiniOneRec baseline and `v1 hierarchy_reg`, even though its final generate-stage collision only ties the MiniOneRec baseline and does not beat `v1`.

This means the current `v2` issue is **not** that it failed to improve local SID structure. The issue is that the structural gain is not yet converted into a better final tokenizer artifact under the current collision-resolution interface.

## Baseline vs v2

### Test-weighted local ambiguity

| Metric | MiniOneRec baseline | `v2 offline_combined` | Delta |
|---|---:|---:|---:|
| Mean target `l2` leaf count | `4.7999` | `4.3422` | `-0.4578` |
| Fraction targets in multi-leaf `same_l2` | `0.6894` | `0.4873` | `-0.2021` |
| Fraction targets in `l2` with `>=4` leaves | `0.3283` | `0.2228` | `-0.1054` |
| Mean target `l3` entropy under `l2` | `1.4533` | `1.1001` | `-0.3532` |

### Movement summary

- targets moved out of multi-leaf `same_l2`: `26.74%`
- targets moved into multi-leaf `same_l2`: `6.53%`
- targets with reduced `l2` leaf count: `44.21%`
- targets with increased `l2` leaf count: `19.32%`

## v1 vs v2

### Test-weighted local ambiguity

| Metric | `v1 hierarchy_reg` | `v2 offline_combined` | Delta |
|---|---:|---:|---:|
| Mean target `l2` leaf count | `4.4498` | `4.3422` | `-0.1077` |
| Fraction targets in multi-leaf `same_l2` | `0.6131` | `0.4873` | `-0.1257` |
| Fraction targets in `l2` with `>=4` leaves | `0.2828` | `0.2228` | `-0.0600` |
| Mean target `l3` entropy under `l2` | `1.2935` | `1.1001` | `-0.1934` |

### Movement summary

- targets moved out of multi-leaf `same_l2`: `23.32%`
- targets moved into multi-leaf `same_l2`: `10.74%`
- targets with reduced `l2` leaf count: `38.52%`
- targets with increased `l2` leaf count: `21.69%`

## Interpretation

There is now a clear split between:

1. **Structure-side behavior**
   - `v2` reduces local ambiguity more strongly than both references.

2. **Final generate-stage collision**
   - `v2` only ties the MiniOneRec baseline and remains worse than `v1`.

The most plausible current reading is:

- `v2` is not failing because the ambiguity-aware prior is useless.
- `v2` is producing a cleaner local hierarchy.
- But the current train-to-generate interface is not translating that cleaner local hierarchy into a better final collision outcome.

In other words, the bottleneck has moved from **whether the ambiguity-aware structure is meaningful** to **how that structure interacts with the current `sid-generate` collision-repair process**.
