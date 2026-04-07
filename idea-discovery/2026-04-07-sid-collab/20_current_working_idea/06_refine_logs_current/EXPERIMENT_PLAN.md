# Experiment Plan

**Problem**: MiniOneRec still makes many local same-prefix mistakes after reaching the right semantic neighborhood.
**Method Thesis**: Add a train-only collaborative residual only for ambiguous `(a,b)` prefixes so third-token SID prediction becomes behavior-aware exactly where text-only SID remains fragile.
**Date**: 2026-04-06

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1: ACLR improves recommendation by fixing local leaf ambiguity | This is the paper's core mechanism claim | HR/NDCG gain plus reduced same-prefix local errors | B1, B4 |
| C2: Selective local activation is better than global collaborative injection | This defends the elegance of the method | ACLR beats global activation and heuristic-only baselines at similar complexity | B2, B3 |

## Anti-Claim To Rule Out

- the gain only comes from heuristic reranking
- the gain only comes from adding extra score terms everywhere
- the gain only comes from more training complexity, not from ambiguity-aware local repair

## Paper Storyline

- **Main paper must prove**:
  - ACLR improves MiniOneRec on the real bottleneck: local same-prefix leaf confusion
  - local selective activation is the right level of intervention
- **Appendix can support**:
  - stronger collaborative residual variants
  - more detailed subgroup analyses
- **Experiments intentionally cut**:
  - full tokenizer rebuild ablations
  - large graph-model comparisons unless the simple residual fails

## Experiment Blocks

### Block 1: Reproduce The Local Signal

- **Claim tested**: there is actionable collaborative signal inside ambiguous local prefixes
- **Why this block exists**: it is the empirical bridge from diagnosis to method
- **Dataset / split / task**: Industrial and Office test sets using existing RL outputs
- **Compared systems**:
  - current top-1 beam choice
  - global collaborative rerank
  - top-`l1` local collaborative rerank
  - top-`l2` local collaborative rerank
- **Metrics**:
  - top-1 hit rate
  - hit rate inside `top1_same_l2`
  - hit rate inside `beam_has_same_l2`
- **Setup details**: reuse current result JSONs and train-only pair statistics
- **Success criterion**: top-`l2` rerank beats baseline on at least Industrial and does not regress materially on Office
- **Failure interpretation**: if there is no rerank headroom, local collaborative repair is probably too weak
- **Table / figure target**: main paper motivation table or early method figure
- **Priority**: MUST-RUN

### Block 2: Main Anchor Result

- **Claim tested**: ACLR improves the actual recommendation task
- **Why this block exists**: it is the main paper result
- **Dataset / split / task**: Industrial first, then Office; SFT stage first, RL stage after the method stabilizes
- **Compared systems**:
  - repo-faithful baseline
  - current best MiniOneRec configuration
  - heuristic top-`l2` collaborative rerank only
  - ACLR training-only
  - ACLR full training + inference residual
- **Metrics**:
  - HR@3/5/10
  - NDCG@3/5/10
  - same-prefix top-1 error rates
  - beam-local rescue rate
- **Setup details**:
  - same tokenizer
  - same backbone
  - same data splits
  - only ACLR-specific modules change
- **Success criterion**: ACLR full beats the current best configuration by a stable margin and lowers same-prefix error rates
- **Failure interpretation**: if heuristic rerank works but ACLR training does not, the learning interface is wrong even if the signal exists
- **Table / figure target**: main result table
- **Priority**: MUST-RUN

### Block 3: Novelty Isolation

- **Claim tested**: the key gain comes from ambiguity-aware local repair, not from generic collaborative bias
- **Why this block exists**: it isolates the paper contribution
- **Dataset / split / task**: Industrial primary, Office confirmatory
- **Compared systems**:
  - ACLR with ambiguity gate on
  - same residual with ambiguity gate off
  - ACLR without inference-time residual
  - ACLR without training-time leaf loss
- **Metrics**:
  - HR/NDCG
  - same-prefix local error rates
  - ambiguous-prefix vs non-ambiguous-prefix subgroup results
- **Setup details**: fixed backbone and fixed residual dimensionality
- **Success criterion**: selective activation outperforms global activation or matches it with lower side effects
- **Failure interpretation**: if global activation is just as good, the ambiguity-gating claim weakens
- **Table / figure target**: ablation table
- **Priority**: MUST-RUN

### Block 4: Simplicity Check

- **Claim tested**: a simple train-only collaborative residual is enough
- **Why this block exists**: it defends the paper's elegance
- **Dataset / split / task**: Industrial
- **Compared systems**:
  - simple co-occurrence residual
  - stronger residual variant such as LightGCN or a richer embedding, only if available
- **Metrics**:
  - HR/NDCG
  - same-prefix local rescue rate
  - compute cost
- **Setup details**: keep ACLR otherwise fixed
- **Success criterion**: the simple residual is competitive enough that a heavy graph stack is not necessary for the core paper
- **Failure interpretation**: if the heavy variant dominates by a lot, the simplicity story weakens
- **Table / figure target**: simplicity/deletion study
- **Priority**: NICE-TO-HAVE

### Block 5: Failure Analysis

- **Claim tested**: ACLR helps where the target is already in the right neighborhood, and fails when that neighborhood is missing
- **Why this block exists**: it makes the mechanism interpretable and reviewer-friendly
- **Dataset / split / task**: Industrial and Office
- **Compared systems**: baseline vs ACLR full
- **Metrics**:
  - gain by `beam_has_same_l1`
  - gain by `beam_has_same_l2`
  - gain by prefix ambiguity bucket
  - gain by item popularity bucket
- **Setup details**: reuse diagnostics scripts and add subgroup exports
- **Success criterion**: gains concentrate in ambiguous local buckets, matching the method thesis
- **Failure interpretation**: diffuse gains suggest the method story is incomplete
- **Table / figure target**: qualitative/failure figure
- **Priority**: MUST-RUN

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | Reproduce the offline local-signal pilot | R001-R002 | Top-`l2` rerank shows positive headroom on Industrial | Low CPU | Script mismatch with stored outputs |
| M1 | Integrate static ambiguity profiler and local logit bias | R003-R004 | In-pipeline eval reproduces offline direction | Low to medium | Leakage or scoring bugs |
| M2 | Train ACLR on Industrial SFT | R005 | Beats current best SFT or clearly improves same-prefix diagnostics | Medium | Training hook may not learn the local signal |
| M3 | Confirm on Office and then RL-best path | R006-R007 | Improvement transfers at least partially across dataset and stage | Medium to high | Signal may be dataset-specific |
| M4 | Run novelty and simplicity ablations | R008-R010 | Selective local story survives ablation | Medium | Paper story may collapse into heuristic reranking |

## Compute and Data Budget

- **Total estimated GPU-hours**: about `40-60` GPU-hours for the must-run suite on the current small-scale backbone, excluding optional heavy residual baselines
- **Data preparation needs**:
  - train-only transition statistics
  - prefix ambiguity table
  - leaf prototype export
- **Human evaluation needs**: none
- **Biggest bottleneck**: integrating the local leaf loss cleanly into the current training path

## Risks and Mitigations

- **Risk**: the correct candidate is not in the beam or subtree
  - **Mitigation**: explicitly report beam coverage and do not overclaim beyond local repair
- **Risk**: Office gains are smaller than Industrial gains
  - **Mitigation**: make Industrial the anchor dataset and treat Office as transfer confirmation
- **Risk**: inference-only rerank helps, but training-time ACLR does not
  - **Mitigation**: keep heuristic-only and training-only ablations separate
- **Risk**: the method looks like reranking rather than modeling
  - **Mitigation**: show the training-time leaf loss matters and that the gain concentrates in the intended ambiguity buckets

## First Three Runs To Launch

1. Reproduce the top-`l2` collaborative rerank pilot on Industrial and Office and save the outputs as versioned artifacts.
2. Implement the ambiguity profiler plus static leaf-logit bias in the evaluation path and verify the in-pipeline effect matches the offline pilot.
3. Add the leaf-level auxiliary loss for ambiguous prefixes and train the Industrial SFT variant first.

## Final Checklist

- [ ] Main paper tables are covered
- [ ] Novelty is isolated
- [ ] Simplicity is defended
- [ ] No frontier component is being claimed unnecessarily
- [ ] Nice-to-have runs are separated from must-run runs
