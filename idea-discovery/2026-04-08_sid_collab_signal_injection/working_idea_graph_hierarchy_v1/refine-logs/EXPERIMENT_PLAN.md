# Experiment Plan

**Problem**: existing collaborative SID tokenizers mostly inject one collaborative signal globally, while our repo evidence suggests that hierarchical SID learning may need graph-structured collaborative supervision with different resolutions and different denoising strategies.  
**Method Thesis**: `MGR-SID` should improve SID-based generative recommendation by learning item codes under **level-specific supervision from a denoised multiplex graph bank**, instead of relying on one globally fused collaborative representation.  
**Date**: 2026-04-09

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|-----------------|-----------------------------|---------------|
| `C1` Primary: graph-regularized hierarchical SID beats graph feature fusion | Without this, the method loses its main novelty and becomes another multi-view collaborative tokenizer. | `MGR-SID` beats semantic baseline and parameter-matched graph-fusion baselines on Industrial. | `B2`, `B3` |
| `C2` Primary: hierarchy-aware graph allocation matters | Without this, the work reduces to generic graph regularization. | `MGR-SID` beats uniform graph regularization and swapped allocation controls; learned allocations differ by level. | `B3`, `B4` |
| `C3` Supporting: denoising must be view-specific | This explains why naive global front-end fusion failed and why local graph signals can be used safely. | purified graph bank beats raw graph bank with better SID health and recommendation quality. | `B1`, `B4` |

## Run Philosophy

Do not launch full-scale training first.

The right order is:

1. settle graph construction
2. settle denoising
3. settle whether graph regularization is better than graph fusion
4. only then run the full main comparison

## Experiment Blocks

### Block 0: Mid-Scale Graph Search

- Goal:
  - find a credible `G_mid`
- Why:
  - this is the most open design choice in the whole method
- Candidate `G_mid` constructions:
  - diffusion residual
  - community graph
  - band-pass spectral graph
  - optional PPR-difference graph
- Evaluation:
  - target-better rate on `same_l1` and `same_l2`
  - coverage
  - graph sparsity / density
  - edge confidence statistics
- Datasets:
  - Industrial first
  - Office if needed for tie-breaking
- Success criterion:
  - pick one `G_mid` that improves deep ambiguity signal without looking too noisy or too trivial
- Priority:
  - `MUST-RUN`

### Block 1: Graph Purification Sanity

- Goal:
  - verify that graph denoising is necessary and not decorative
- Compared systems:
  - raw `G_coarse / G_mid / G_local`
  - purified `G_coarse / G_mid / G_local`
- Evaluation:
  - graph diagnostics:
    - degree skew
    - edge retention
    - neighbor agreement
  - SID health proxy:
    - collision rate
    - code usage balance
  - cheap compatibility probe:
    - target-better rates by bucket
- Success criterion:
  - purification reduces pathological graph structure or improves probe quality
- Priority:
  - `MUST-RUN`

### Block 2: Integration Sanity on Industrial

- Goal:
  - determine whether graph regularization is worth full continuation
- Compared systems:
  - `S0`: semantic-only tokenizer
  - `S1`: single global graph feature fusion
  - `S2`: multi-graph feature fusion with no graph regularization
  - `S3`: uniform graph regularization across all SID levels
  - `S4`: `MGR-SID`
- Metrics:
  - `HR@1`, `NDCG@10`
  - tokenizer collision
  - code usage balance
- Setup:
  - Industrial only
  - one seed first
  - fixed 3-level SID
- Success criterion:
  - `S4` beats `S2` and `S3` without collapse
- Priority:
  - `MUST-RUN`

### Block 3: Novelty Isolation

- Goal:
  - prove the gain is from hierarchy-aware graph supervision, not generic graph help
- Compared systems:
  - `S2`: graph feature fusion
  - `S3`: uniform graph regularization
  - `S4`: `MGR-SID`
  - `S5`: swapped allocation control
  - `S6`: no-denoising `MGR-SID`
- Metrics:
  - `HR@1`, `NDCG@10`
  - same-`l2` error reduction
  - learned level-wise graph weights
- Success criterion:
  - `S4 > S2`
  - `S4 > S3`
  - `S5 < S4`
  - `S6 < S4`
- Priority:
  - `MUST-RUN`

### Block 4: Bucketed Mechanism Analysis

- Goal:
  - verify that different graph views matter in different ambiguity regions
- Compared systems:
  - `S0`
  - `S3`
  - `S4`
- Metrics:
  - all / same-`l1` / same-`l2`
  - activation and gain per activated sample
  - per-level graph allocation weights
- Success criterion:
  - deeper levels should rely more on non-global graph information, and gains should be visible on local ambiguity buckets
- Priority:
  - `MUST-RUN`

### Block 5: Office Generalization

- Goal:
  - show the direction is not Industrial-only
- Compared systems:
  - `S0`
  - `S3`
  - `S4`
- Metrics:
  - `HR@1`, `NDCG@10`
  - same-`l2` improvement
  - learned graph allocations
- Priority:
  - `MUST-RUN` after Industrial is positive

## First Practical Milestones

### `M0`

Pick `G_mid`.

Do not continue to full training before this is settled.

### `M1`

Implement graph bank construction and purification as standalone cached artifacts.

Outputs should be reusable across all training runs.

### `M2`

Implement two minimal integration baselines:

- graph feature fusion
- uniform graph regularization

These must exist before `MGR-SID`, otherwise the novelty claim cannot be tested.

### `M3`

Run Industrial one-seed anchor comparison:

- `S0`
- `S2`
- `S3`
- `S4`

### `M4`

Only after positive signal:

- run novelty controls
- run Office
- add seeds

## Decision Gates

| Milestone | Continue if | Stop or pivot if |
|-----------|-------------|------------------|
| `M0` | one `G_mid` clearly looks reasonable | all `G_mid` candidates are noisy or redundant |
| `M1` | graph bank is stable and cacheable | graph construction is too brittle or too expensive |
| `M3` | `S4` beats `S2` or at least matches it with clearly better analysis evidence | `S4` is not better than graph fusion or causes unstable tokenizer behavior |
| `M4` | novelty controls support level-aware supervision | gains come only from generic graph priors |

## What to Cut First if Compute Is Tight

Cut in this order:

1. Office extra seeds
2. optional additional `G_mid` variants after one winner is clear
3. overbuilt extensions like confidence-routed dynamic graph allocation

Do not cut:

1. graph-fusion baseline
2. uniform graph-regularization baseline
3. swapped allocation control
4. no-denoising control
