# Experiment Plan

**Problem**: current hierarchy-aware graph regularization already improves final SID quality and reduces local `same_l2` ambiguity, but the downstream SFT gain is only partial because graph supervision is still too rigid and can over-correct stable semantic neighborhoods.  
**Method Thesis**: `MGR-SID v2` should improve over the reproduced MiniOneRec baseline and current `hierarchy_reg` by making graph supervision **ambiguity-aware** and adding **semantic-structure retention**, so graph acts strongly on hard local ambiguity while remaining conservative on already-stable semantic regions.  
**Date**: 2026-04-10

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|-----------------|-----------------------------|---------------|
| `C1` Primary: `MGR-SID v2` can beat the reproduced MiniOneRec baseline by making graph supervision ambiguity-aware and semantically conservative | Right now the project has a good tokenizer story but not a decisive downstream win. The next step must answer the only question that matters first: can `v2` beat the MiniOneRec baseline? | On Industrial, `v2` beats the reproduced MiniOneRec baseline on tokenizer-side final SID quality and then produces a positive downstream `SFT -> evaluate` result, ideally on both short-range and overall metrics. | `B1`, `B2`, `B3` |
| `C2` Deferred supporting claim: semantic-structure retention explains why `v2` beats `v1` | This is important for the paper, but not the first decision gate. | After `v2` is positive, ablations show removing semantic-structure retention increases over-correction. | `B4` |

### Anti-claims to rule out

- The gain only comes from another round of graph tuning rather than ambiguity-awareness.
- The gain only comes from globally weakening graph influence.
- The result only improves tokenizer diagnostics but not downstream recommendation behavior.

### Minimum convincing evidence

- `v2` improves final generated SID quality over the MiniOneRec baseline.
- `v2` at least matches the current hard-case `same_l2` / crowded-bucket gains of `v1`.
- `v2` produces a downstream `SFT -> evaluate` result that beats the MiniOneRec baseline, not just `v1`.

## Paper Storyline

- Main paper must prove:
  - graph-structured collaborative supervision must be **ambiguity-aware**
  - `v2` beats the MiniOneRec baseline rather than only producing partial tokenizer-side gains
  - `v2` better solves hard local ambiguity without excessive disruption to already-stable semantic structure

- Appendix can support:
  - exact proxy definitions and implementation details
  - proxy correlation tables
  - more proxy combinations
  - extra seeds
  - mechanism ablations
  - Office experiments after Industrial is positive

- Experiments intentionally cut:
  - reopening the full graph-bank search
  - full free learned gate over all graph views
  - RL-stage validation before SFT transfer is fixed
  - full ablation matrix before `v2` first beats the MiniOneRec baseline

## Experiment Blocks

### Block 1: Ambiguity Proxy Validation

- Claim tested:
  - the proposed ambiguity proxies are good enough to justify a first `v2` training run
- Why this block exists:
  - we should avoid wiring a completely useless ambiguity prior into training, but we do not need a paper-ready proxy analysis before the first result
- Dataset / split / task:
  - Industrial
  - train-only proxy construction
  - lightweight analysis against current tokenizer outputs
- Compared systems:
  - `P1`: offline combined ambiguity prior
  - `P2`: offline + online ambiguity
- Metrics:
  - decisive:
    - separation of hard local ambiguity vs easy stable items
    - rank concentration on crowded `l2` buckets
- Setup details:
  - no downstream retraining
  - keep this block lightweight and decision-oriented
- Success criterion:
  - at least one practical proxy setting is good enough to try in `v2`
- Failure interpretation:
  - if no proxy has signal, pause and redesign `v2` before training
- Table / figure target:
  - appendix only for now
- Priority:
  - `MUST-RUN`

### Block 2: Tokenizer-Only Main Result

- Claim tested:
  - `v2` improves the front-end tokenizer enough to justify downstream SFT
- Why this block exists:
  - this is the cheapest real decision gate before SFT
- Dataset / split / task:
  - Industrial
  - `sid-train -> sid-generate`
- Compared systems:
  - `T0`: reproduced MiniOneRec baseline
  - `T1`: current `v1 hierarchy_reg`
  - `T2`: `v2 full` (ambiguity-aware + semantic-structure retention)
- Metrics:
  - decisive:
    - final generated SID collision
    - weighted `H(level3 | level1, level2)`
    - target-weighted `l2` leaf count
  - secondary:
    - changed-SID fraction
    - fraction moved out of multi-leaf `same_l2`
    - code usage health
- Setup details:
  - keep graph bank fixed to current best v1 setting
  - first run one seed only
  - minimal `v2` only; no ablation variants yet
- Success criterion:
  - `T2` improves final SID structure over `T0`
  - ideally `T2` also beats `T1`, but the first hard gate is `T2 > T0`
- Failure interpretation:
  - if `T2` cannot beat `T0`, do not proceed to SFT
- Table / figure target:
  - Main Table 1 (tokenizer-side)
- Priority:
  - `MUST-RUN`

### Block 3: Downstream SFT Transfer Result

- Claim tested:
  - `v2` improves tokenizer-to-SFT transfer by preserving hard-case gains while reducing over-correction
- Why this block exists:
  - this is the real bottleneck exposed by current experiments
- Dataset / split / task:
  - Industrial
  - `convert -> sft -> evaluate`
- Compared systems:
  - `S0`: reproduced MiniOneRec baseline
  - `S1`: `v2 full`
- Metrics:
  - decisive:
    - `NDCG@3`
    - `NDCG@10`
    - `HR@3`
    - `HR@10`
  - mechanism:
    - top-k structural analysis
    - broken vs fixed counts
    - same-prefix rates inside `top-k`
- Setup details:
  - use the same SFT config across both
  - evaluate with the same diagnostics scripts
  - start with one seed
- Success criterion:
  - `S1` beats the MiniOneRec baseline on the primary reported metrics
  - if aggregate metrics are mixed, `S1` should still show a stronger overall top-k story than current `v1`
- Failure interpretation:
  - if tokenizer improves but `S1` still cannot beat baseline downstream, the transfer issue remains the dominant bottleneck
- Table / figure target:
  - Main Table 2
  - Main/appendix figure: rank-transition and broken/fixed comparison
- Priority:
  - `MUST-RUN`

### Block 4: Mechanism Isolation

- Claim tested:
  - ambiguity-awareness and semantic-structure retention each matter
- Why this block exists:
  - this block matters for the paper, but it does not decide whether `v2` is worth continuing
- Dataset / split / task:
  - Industrial tokenizer + SFT
- Compared systems:
  - `A1`: `v2 full`
  - `A2`: no online quantization uncertainty
  - `A3`: no offline ambiguity prior
  - `A4`: no semantic-structure retention
  - `A5`: ambiguity-aware weighting with uniform graph regularization only
- Metrics:
  - decisive:
    - final SID ambiguity metrics
    - `NDCG@10`
    - `HR@10`
  - mechanism:
    - top-k same-prefix retention
    - broken / fixed counts
- Setup details:
  - only launch after `v2` already beats the MiniOneRec baseline
  - tokenizer-only first
- Success criterion:
  - removing the anchor increases broken cases
  - removing ambiguity-aware weighting reduces hard-case benefits
- Failure interpretation:
  - if ablations are too close, the method story is not yet sharp enough
- Table / figure target:
  - Main Table 3 or appendix ablation table
- Priority:
  - `DEFERRED`

### Block 5: Generalization and Appendix Support

- Claim tested:
  - the direction is not Industrial-only
- Why this block exists:
  - only worth doing after Industrial is clearly positive
- Dataset / split / task:
  - Office
  - tokenizer first, then SFT if tokenizer is positive
- Compared systems:
  - `T0/S0`: MiniOneRec baseline
  - `T1/S1`: `v1 hierarchy_reg`
  - `T2/S2`: `v2 full`
- Metrics:
  - same as Blocks 2 and 3
- Setup details:
  - one seed first
- Success criterion:
  - directionally similar gains
- Failure interpretation:
  - if Office diverges strongly, paper story should remain Industrial-centered
- Table / figure target:
  - appendix generalization table
- Priority:
  - `DEFERRED`

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| `M0` | minimal proxy sanity | `P1-P2` | one usable ambiguity configuration exists | low | ambiguity proxy may be too noisy |
| `M1` | tokenizer validation | `T0/T1/T2` | `T2 > T0`; ideally `T2 > T1` | medium | front-end weighting may be too weak |
| `M2` | SFT transfer | `S0/S1` | `S1` beats baseline or clearly improves the overall top-k story | high | tokenizer gain may still not transfer |
| `M3` | mechanism isolation | `A1-A5` | only after `M2` is positive | medium-high | too many interacting knobs |
| `M4` | generalization | Office runs | only after `M2` is clearly positive | high | compute cost and weaker signal |

## Compute and Data Budget

- Total estimated GPU-hours:
  - `M0`: mostly CPU + tiny tokenizer instrumentation
  - `M1`: 1 tokenizer family with one new `v2` run
  - `M2`: 1 new Industrial SFT run against the MiniOneRec baseline
  - `M3+`: deferred
- Data preparation needs:
  - reuse current `data_experiment` structure
  - create a new `v2` variant folder after tokenizer generation
- Human evaluation needs:
  - none
- Biggest bottleneck:
  - SFT turnaround time

## Risks and Mitigations

- **Risk**: ambiguity proxies do not align well with hard local ambiguity
  - **Mitigation**: start with offline + online combination, not a single proxy

- **Risk**: semantic-structure retention becomes too strong and cancels graph benefits
  - **Mitigation**: stage the validation as `T2` vs `T3`

- **Risk**: tokenizer-side gains again fail to transfer to SFT
  - **Mitigation**: use top-k structural analysis as the main decision signal, not only aggregate metrics

- **Risk**: the method becomes too complex too early
  - **Mitigation**: keep `v2` focused on one proxy recipe and one full method variant first

## Final Checklist

- [ ] Minimal proxy sanity is covered
- [ ] `v2` tokenizer result is compared to the MiniOneRec baseline
- [ ] `v2` SFT result is compared to the MiniOneRec baseline
- [ ] Post-hoc patching is explicitly avoided
- [ ] Ablations are deferred until after `v2` first beats the baseline
