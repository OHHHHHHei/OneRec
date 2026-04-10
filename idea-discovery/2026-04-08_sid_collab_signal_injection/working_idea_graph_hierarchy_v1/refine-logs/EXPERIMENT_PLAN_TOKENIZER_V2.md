# Experiment Plan

**Problem**: current `MGR-SID v1 hierarchy_reg` already improves final SID quality over the reproduced MiniOneRec baseline, but the tokenizer still applies graph supervision too uniformly. We now have a usable train-time ambiguity proxy (`R001` offline combined prior), and the next question is whether a tokenizer-only `v2` can beat the MiniOneRec baseline before we spend more time on downstream stages.  
**Method Thesis**: `MGR-SID tokenizer v2` should improve over the reproduced MiniOneRec baseline, and ideally over `v1 hierarchy_reg`, by using an **offline ambiguity-aware graph weighting** plus **semantic-structure retention**, while keeping the graph bank and level roles fixed.  
**Date**: 2026-04-11

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|-----------------|-----------------------------|---------------|
| `C1` Primary: tokenizer `v2` beats the reproduced MiniOneRec baseline on final SID quality | This is the immediate decision gate. If `v2` cannot win tokenizer-side, there is no reason to continue to new SFT runs. | On Industrial, final generated SID from `v2` has lower collision and better local ambiguity structure than the MiniOneRec baseline. | `B1`, `B2` |
| `C2` Supporting: ambiguity-aware weighting is a better front-end integration mechanism than uniform `v1 hierarchy_reg` | This sharpens the story from “graph helps” to “graph should act selectively based on ambiguity.” | `v2` matches or beats `v1` on hard local ambiguity while avoiding more over-correction on easy stable regions. | `B2`, `B3` |

### Anti-claims to rule out

- The gain only comes from another round of hyperparameter luck.
- The gain only comes from weakening graph influence globally.
- The offline ambiguity prior is decorative and does not change the tokenizer behavior in meaningful regions.

### Minimum convincing evidence

- `v2` final generated SID improves over the reproduced MiniOneRec baseline.
- `v2` keeps or improves hard-case local ambiguity metrics relative to `v1`.
- `v2` changes the SID structure in the intended direction, not just globally shrinking or perturbing codes.

## Paper Storyline

- Main paper must prove:
  - ambiguity-aware graph supervision is a better tokenizer-side mechanism than fixed uniform graph supervision
  - the first practical step is already possible with an offline ambiguity prior only
  - the MiniOneRec baseline remains the main anchor, and `v2` should be judged against it first

- Appendix can support:
  - detailed proxy construction
  - alternative ambiguity priors
  - online uncertainty variants
  - ablations after the tokenizer-side win is established

- Experiments intentionally cut for now:
  - any new SFT / RL runs
  - Office generalization
  - full ablation matrix
  - fully free learned gate

## Experiment Blocks

### Block 1: Proxy Freeze for First v2 Run

- Claim tested:
  - we have a usable ambiguity proxy for a first tokenizer `v2` run
- Why this block exists:
  - we should not train `v2` blindly, but we also do not need a full proxy paper before the first result
- Dataset / split / task:
  - Industrial
  - train-only / analysis-only
- Compared systems:
  - `P0`: reuse `R001` offline combined ambiguity prior
  - `P1`: reuse `R002` offline + online ambiguity
- Metrics:
  - hard/easy separation
  - hard-item enrichment in high-proxy buckets
  - concentration of improved vs worsened cases
- Setup details:
  - no retraining
  - reuse existing proxy sanity outputs
- Success criterion:
  - choose exactly one practical proxy recipe for `v2`
- Failure interpretation:
  - if no proxy is usable, pause tokenizer `v2`
- Table / figure target:
  - appendix note only
- Priority:
  - `MUST-RUN` but already mostly completed

### Block 2: Tokenizer Main Result

- Claim tested:
  - tokenizer `v2` improves final generated SID over the reproduced MiniOneRec baseline
- Why this block exists:
  - this is the cheapest real test of whether `v2` is worth continuing
- Dataset / split / task:
  - Industrial
  - `sid-train -> sid-generate`
- Compared systems:
  - `T0`: reproduced MiniOneRec baseline
  - `T1`: current `v1 hierarchy_reg`
  - `T2`: `v2 full` using **offline combined ambiguity prior only**
- Metrics:
  - decisive:
    - final generated SID collision
    - weighted `H(level3 | level1, level2)`
    - target-weighted mean `l2` leaf count
  - secondary:
    - fraction of items in multi-leaf `same_l2`
    - fraction of test targets in crowded `l2>=4` buckets
    - changed-SID fraction vs baseline
- Setup details:
  - keep graph bank fixed:
    - `Level 1 <- G_coarse`
    - `Level 2 <- G_mid`
    - `Level 3 <- G_local`
  - keep upstream-aligned MiniOneRec tokenizer training setup
  - first run one seed only
  - do **not** include the current online uncertainty term in the first `v2`
- Success criterion:
  - `T2 > T0` on final SID quality
  - ideally `T2 >= T1`, but the hard gate is first beating `T0`
- Failure interpretation:
  - if `T2` fails to beat `T0`, stop and revisit the weighting design before any downstream work
- Table / figure target:
  - Main tokenizer-side table
- Priority:
  - `MUST-RUN`

### Block 3: Tokenizer Structural Diagnosis

- Claim tested:
  - `v2` improves the *right* part of the SID structure rather than introducing arbitrary changes
- Why this block exists:
  - a tokenizer-side win should still be diagnosed in terms of local ambiguity, not only aggregate collision
- Dataset / split / task:
  - Industrial
  - final generated SID comparison
- Compared systems:
  - `T0`: reproduced MiniOneRec baseline
  - `T1`: current `v1 hierarchy_reg`
  - `T2`: `v2 full`
- Metrics:
  - decisive:
    - target-weighted `l2` leaf count
    - fraction of targets in multi-leaf `same_l2`
    - weighted `H(level3 | level1, level2)`
  - secondary:
    - moved out of crowded `l2>=4`
    - changed-SID fraction
    - representative improved / worsened item families
- Setup details:
  - reuse the current local ambiguity analysis tooling
  - keep the diagnosis tokenizer-only
- Success criterion:
  - `v2` improves or preserves the local ambiguity gains already observed in `v1`
- Failure interpretation:
  - if `v2` only improves collision but worsens local ambiguity structure, the method is not aligned with the current research anchor
- Table / figure target:
  - Main or appendix tokenizer diagnosis table
- Priority:
  - `MUST-RUN`

### Block 4: Deferred Tokenizer Ablations

- Claim tested:
  - ambiguity-aware weighting and semantic-structure retention each matter
- Why this block exists:
  - needed for the paper later, but not for the first tokenizer decision gate
- Dataset / split / task:
  - Industrial tokenizer only
- Compared systems:
  - `A1`: `v2 full`
  - `A2`: no semantic-structure retention
  - `A3`: ambiguity-aware weighting replaced by uniform weighting
  - `A4`: alternative ambiguity prior variants
- Metrics:
  - same as Blocks 2 and 3
- Setup details:
  - only run after `T2` first beats `T0`
- Success criterion:
  - ablations sharpen the story rather than decide whether the direction is alive
- Failure interpretation:
  - if the ablations collapse together, the `v2` story is not yet sharp
- Table / figure target:
  - appendix or later main ablation table
- Priority:
  - `DEFERRED`

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| `M0` | freeze first proxy recipe | `P0/P1` | use offline combined prior only | very low | proxy choice may still be crude |
| `M1` | tokenizer main run | `T0/T1/T2` | `T2 > T0` tokenizer-side | medium | weighting may be too weak or too rigid |
| `M2` | tokenizer diagnosis | local ambiguity comparison | `T2` improves the right structure | low | gain may be collision-only |
| `M3` | deferred ablations | `A1-A4` | only if `M1` is positive | medium | too early complexity |

## Compute and Data Budget

- Total estimated GPU-hours:
  - `M0`: none beyond analysis reuse
  - `M1`: one new upstream-aligned tokenizer run plus one generate pass
  - `M2`: CPU-side analysis only
- Data preparation needs:
  - no new downstream data conversion yet
- Human evaluation needs:
  - none
- Biggest bottleneck:
  - tokenizer training turnaround and `sid-generate` verification

## Risks and Mitigations

- **Risk**: the offline proxy is only moderately aligned and overfocuses on dense product families
  - **Mitigation**: keep `v2` first run conservative; do not amplify graph weights too aggressively

- **Risk**: `v2` improves collision but not the local ambiguity measures we care about
  - **Mitigation**: make tokenizer structural diagnosis mandatory before any downstream work

- **Risk**: `v2` still loses to `v1`
  - **Mitigation**: treat `T0` as the hard gate and `T1` as a supporting comparator, not the only success criterion

## Final Checklist

- [ ] First tokenizer `v2` proxy recipe is frozen
- [ ] `v2` tokenizer result is compared to the MiniOneRec baseline
- [ ] `v2` tokenizer result is compared to current `v1`
- [ ] Local ambiguity diagnosis is included
- [ ] Downstream stages are intentionally deferred until tokenizer `v2` first wins
