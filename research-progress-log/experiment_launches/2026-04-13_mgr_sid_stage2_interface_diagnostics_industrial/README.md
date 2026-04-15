# 2026-04-13 Stage-2 Interface Diagnostics on Industrial

## Scope

This run executes the interface-diagnostics plan proposed after the stage-2
tokenizer refinements:

- `R301`: prefix stability / SID rearrangement audit
- `R302`: code polysemy / semantic consistency
- `R303`: structure-to-downstream transfer attribution
- `R304`: lightweight SID learnability probe

The central question is:

> why did `R202a` improve tokenizer-side structure, yet fail to beat the
> current `v2_on_p05` downstream mainline?

## Inputs

- baseline tokenizer:
  - `current v2`
- compared tokenizer variants:
  - `R202a`
  - `R202b-r075`
  - `R205`
  - strongest original SID (reference only where useful)
- downstream comparison:
  - `current v2_on_p05 SFT` vs `R208`

## Artifacts

### Raw outputs

- [R301_prefix_stability.csv](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_interface_diagnostics_industrial/R301_prefix_stability.csv)
- [R301_prefix_stability.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_interface_diagnostics_industrial/R301_prefix_stability.json)
- [R302_code_polysemy.csv](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_interface_diagnostics_industrial/R302_code_polysemy.csv)
- [R302_code_polysemy.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_interface_diagnostics_industrial/R302_code_polysemy.json)
- [R303_transfer_attribution.csv](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_interface_diagnostics_industrial/R303_transfer_attribution.csv)
- [R303_transfer_attribution.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_interface_diagnostics_industrial/R303_transfer_attribution.json)
- [R304_learnability_probe.csv](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_interface_diagnostics_industrial/R304_learnability_probe.csv)
- [R304_learnability_probe.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_interface_diagnostics_industrial/R304_learnability_probe.json)

### Upstream context

- [R204_v2_vs_r202a_local_ambiguity.md](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_stopgrad_industrial/R204_v2_vs_r202a_local_ambiguity.md)
- [EVAL_ANALYSIS.md](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_r202a_sft_eval_industrial/EVAL_ANALYSIS.md)
- [TOPK_V2_ON_P05_SFT_VS_R208.md](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_r202a_sft_eval_industrial/TOPK_V2_ON_P05_SFT_VS_R208.md)

## R301: Prefix Stability / SID Rearrangement

### Raw summary

| Variant vs `current v2` | Changed l1 | Changed l2 | Changed full SID | l1 pair retention | l2 pair retention | mean l1 Jaccard | mean l2 Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|
| `R202a` | 0.9965 | 1.0000 | 1.0000 | 0.4137 | 0.6122 | 0.2401 | 0.5890 |
| `R202b-r075` | 0.9883 | 1.0000 | 1.0000 | 0.3797 | 0.6333 | 0.2039 | 0.5292 |
| `R205` | 0.9984 | 1.0000 | 1.0000 | 0.5137 | 0.6966 | 0.1378 | 0.5362 |
| strongest original | 0.9992 | 1.0000 | 1.0000 | 0.6620 | 0.7138 | 0.1345 | 0.5050 |

### Reading

1. `R202a` is indeed a **near-full SID rearrangement** relative to current `v2`.
   - `100%` of items change their full SID.
   - `100%` of items change their `(l1,l2)` prefix.
   - `99.65%` of items even change `l1`.

2. But the rearrangement is **not random**.
   - `41.4%` of baseline same-`l1` item pairs remain same-`l1`.
   - `61.2%` of baseline same-`l2` item pairs remain same-`l2`.
   - mean per-item `l2` neighbor Jaccard is still `0.589`.

3. This gives us a more precise version of the “global rearrangement cost” claim:
   - `R202a` does not preserve token identity
   - but it does preserve a moderate amount of local prefix structure
   - so the downstream penalty is likely caused by **large-scale remapping of the SID interface**, not by total destruction of neighborhood structure

4. `R205` and strongest original preserve more baseline pair structure than `R202a`,
   but they are not structurally better tokenizers overall.
   - prefix stability alone is therefore **not** enough to define tokenizer quality
   - it is one axis of interface compatibility, not the whole story

## R302: Code Polysemy / Semantic Consistency

### Raw summary

| Variant | Level | token count | weighted mean spread / drift | p90 |
|---|---|---:|---:|---:|
| `current v2` | `a` | 203 | 0.0620 | 0.0903 |
| `R202a` | `a` | 157 | 0.0631 | 0.0925 |
| `current v2` | `b` | 256 | 0.1068 | 0.1270 |
| `R202a` | `b` | 256 | 0.1057 | 0.1242 |
| `current v2` | `c` | 256 | 0.1202 | 0.1321 |
| `R202a` | `c` | 256 | 0.1199 | 0.1312 |
| `current v2` | `b` prefix drift | 256 | 0.2265 | 0.2605 |
| `R202a` | `b` prefix drift | 254 | 0.2282 | 0.2622 |
| `current v2` | `c` prefix drift | 256 | 0.2426 | 0.2677 |
| `R202a` | `c` prefix drift | 255 | 0.2421 | 0.2653 |

### Reading

1. The big surprise is that **`current v2` and `R202a` are almost identical on code polysemy statistics**.
   - `b`-level spread: `0.1068 -> 0.1057`
   - `c`-level spread: `0.1202 -> 0.1199`
   - `b` prefix-conditioned drift: `0.2265 -> 0.2282`
   - `c` prefix-conditioned drift: `0.2426 -> 0.2421`

2. This strongly suggests:
   - **token semantic overload is probably not the main reason why `R202a` lost downstream**
   - at least not in the simple sense of “the codes became much more polysemous”

3. The most visible change is at level `a`:
   - token count shrinks from `203 -> 157`
   - mean reuse count rises from `18.16 -> 23.48`
   - semantic spread rises slightly from `0.0620 -> 0.0631`

4. So the semantic story is not:
   - "`R202a` made code semantics much messier"
   but rather:
   - "`R202a` changed the routing and reuse pattern of the SID space, while keeping token-level semantic consistency roughly similar"

## R303: Structure-to-Downstream Transfer Attribution

This block joins:

- SID change features from `current v2 -> R202a`
- per-example top-k migration from `current v2_on_p05 SFT -> R208`

### Raw summary

| Cutoff | Group | Count | changed l1 | changed l2 | mean l1 Jaccard | mean l2 Jaccard | mean baseline l2 fanout |
|---|---|---:|---:|---:|---:|---:|---:|
| `@1` | improved | 48 | 1.000 | 1.000 | 0.3739 | 0.5932 | 10.42 |
| `@1` | worsened | 71 | 1.000 | 1.000 | 0.1490 | 0.6299 | 4.58 |
| `@3` | improved | 102 | 1.000 | 1.000 | 0.3474 | 0.4969 | 11.72 |
| `@3` | worsened | 92 | 0.989 | 1.000 | 0.2423 | 0.6995 | 6.93 |
| `@5` | improved | 111 | 1.000 | 1.000 | 0.3214 | 0.5004 | 9.95 |
| `@5` | worsened | 115 | 1.000 | 1.000 | 0.2339 | 0.6840 | 7.17 |
| `@10` | improved | 130 | 1.000 | 1.000 | 0.3000 | 0.4870 | 7.72 |
| `@10` | worsened | 147 | 1.000 | 1.000 | 0.2579 | 0.6528 | 6.98 |
| `@20` | improved | 176 | 1.000 | 1.000 | 0.2994 | 0.4897 | 5.91 |
| `@20` | worsened | 187 | 1.000 | 1.000 | 0.2540 | 0.5885 | 6.57 |

### Reading

1. The coarse-grained “changed prefix” indicators are not discriminative enough.
   - almost every example changes `l1`
   - every example changes `l2`
   - so the useful information lies in **how neighborhoods are rearranged**, not whether they changed at all

2. Improved examples are consistently **harder** than worsened ones.
   - at `@3`, improved examples have baseline fanout `11.72` vs worsened `6.93`
   - at `@10`, improved examples have baseline fanout `7.72` vs worsened `6.98`

3. Improved examples tend to have:
   - **higher l1 neighbor Jaccard**
   - **lower l2 neighbor Jaccard**

4. Worsened examples tend to have:
   - **lower l1 neighbor Jaccard**
   - **higher l2 neighbor Jaccard**

The pattern is consistent from `@3` to `@20`.

### Interpretation

This is a useful refinement of the previous “global rearrangement cost” story:

- `R202a` helps difficult examples when it **aggressively rewrites local `l2` neighborhoods**
- but it hurts easier/stabler examples when it **disturbs the broader `l1` routing too much**

So the real transfer problem is not:
- “any prefix change is bad”

It is closer to:
- “for easy/stable examples, preserving broader prefix routing matters more than rewriting local leaf competition”

## R304: SID Learnability Probe

We trained a lightweight linear probe on the Industrial train split and evaluated
on the validation split.

Targets:

- predict `a`
- predict `b` given gold `a`
- predict `c` given gold `a,b`

### Raw summary

| Variant | Probe target | overall acc | hard acc (`l2>=4`) | stable acc (`l2<=2`) |
|---|---|---:|---:|---:|
| `current v2` | `a` | 0.0902 | 0.2230 | 0.0593 |
| `R202a` | `a` | 0.0997 | 0.2231 | 0.0676 |
| `current v2` | `b_given_a` | 0.2392 | 0.4085 | 0.1914 |
| `R202a` | `b_given_a` | 0.2134 | 0.3665 | 0.1794 |
| `current v2` | `c_given_ab` | 0.4365 | 0.2218 | 0.4973 |
| `R202a` | `c_given_ab` | 0.4159 | 0.1806 | 0.4784 |

### Reading

1. The learnability signal is **mixed but very informative**.

2. `R202a` makes level `a` slightly easier to predict.
   - overall: `0.0902 -> 0.0997`
   - stable bucket: `0.0593 -> 0.0676`

3. But `R202a` makes deeper levels harder to predict.
   - `b_given_a`: `0.2392 -> 0.2134`
   - `c_given_ab`: `0.4365 -> 0.4159`
   - the hard bucket suffers most:
     - `b_given_a`: `0.4085 -> 0.3665`
     - `c_given_ab`: `0.2218 -> 0.1806`

### Interpretation

This helps reconcile the stage-2 puzzle:

- tokenizer-side structure said `R202a` was better
- downstream SFT said `R208` was worse

The probe suggests the following:

> `R202a` may have improved coarse routing at level `a`, but it made the
> deeper conditional decisions (`b`, `c`) harder to learn from history.

That aligns well with the observed downstream profile:

- some hard crowded examples improve
- but broader beam retention does not improve

## Consolidated Takeaways

### 1. The interface problem is real

Stage-2 failure is not well explained by:
- “tokenizer structure did not improve”

because `R202a` clearly improved structure.

It is better explained by:
- **SID-space interface / transfer issues**

### 2. The main interface issue is not simple token polysemy

`R202a` and current `v2` are extremely close on token semantic spread and
prefix-conditioned drift.  
So the downstream loss does **not** look like:

- “the code vocabulary became much more semantically overloaded”

### 3. The more plausible mechanism is:

- near-global SID remapping
- moderate preservation of local structure
- but altered coarse routing and reduced deeper-level learnability

### 4. The best current explanation for `R208` is now:

> `R202a` helps when hard local `l2` neighborhoods need to be rewritten, but it
> hurts already-stable examples because the broader SID interface changes too
> much and the deeper conditional code decisions become harder to learn.

## Action Implication

This diagnosis supports a more conservative next tokenizer move:

- avoid full-SID rearrangement
- preserve `l1/l2` routing as much as possible
- if we refine tokenizer again, prefer **leaf-only / `l3`-only** changes over
  whole-space remapping

It also suggests that if we stay on the current mainline, the strongest
end-to-end branch is still:

- `v2_on_p05 -> RL`

while interface-aware tokenizer refinement should prioritize:

- prefix stability
- deeper-level learnability
- not just local ambiguity cleanup
