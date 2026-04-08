# Experiment Plan

**Problem**: Existing SID tokenizers increasingly use collaborative information, but they usually fuse it uniformly. In this repo, the remaining bottleneck is local leaf ambiguity under semantically correct prefixes, while naive global front-end fusion causes structural collapse.  
**Method Thesis**: `MRC-SID` should improve generative recommendation by learning **level-wise allocation over purified collaborative views with different resolutions**, instead of using one uniform collaborative representation at every SID level.  
**Date**: 2026-04-09

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|-----------------|-----------------------------|---------------|
| `C1` Primary: learned level-wise allocation beats uniform fusion | This is the core novelty claim. Without it, the work collapses into ordinary multi-view fusion. | `MRC-SID` improves HR/NDCG over semantic-only baseline and over parameter-matched uniform all-view fusion on Industrial; learned gates are measurably non-uniform. | `B1`, `B2`, `B4` |
| `C2` Supporting: purified multi-resolution views are necessary for stable and useful tokenizer fusion | This protects the story against “just add more CF features” and explains why naive front-end fusion failed. | `MRC-SID` beats no-purification and naive single-view global fusion while maintaining healthy tokenizer statistics. | `B1`, `B3`, `B4` |

### Anti-claims to rule out

- The gain only comes from using more collaborative features or more parameters.
- The gain only comes from one strong collaborative view, not from hierarchy-aware allocation.
- The gain only comes from a hand-designed heuristic mapping.
- The gain comes with tokenizer collapse or unusable SID statistics.

## Paper Storyline

- Main paper must prove:
  - `MRC-SID` solves the anchored bottleneck better than semantic-only and uniform-fusion baselines.
  - The dominant contribution is **level-wise allocation**, not just more views.
  - Purification matters because naive collaborative injection is unstable.
- Appendix can support:
  - deeper gate interpretation
  - additional Office runs
  - alternative purification variants
  - robustness across seeds / view-construction choices
- Experiments intentionally cut:
  - fully personalized or dynamic tokenization in the first paper version
  - too many view-bank variants
  - broad baseline lists that do not test the main claim directly

## Experiment Blocks

### Block 1: Sanity and Pilot Revalidation

- Claim tested:
  - The multi-resolution premise is real in this repo before we spend full training budget.
- Why this block exists:
  - We need to re-confirm that different collaborative views help different buckets, otherwise the main idea is under-motivated.
- Dataset / split / task:
  - Industrial and Office
  - train/test splits already used in current diagnostics
  - train-only collaborative compatibility probing
- Compared systems:
  - `coarse10`
  - `mid3`
  - `local2`
  - `fine1`
- Metrics:
  - target-better rate on all errors
  - target-better rate on same-`l1`
  - target-better rate on same-`l2`
  - nonzero coverage / sparsity
- Setup details:
  - no model retraining
  - reuse current result files and diagnostics scripts
  - CPU or lightweight GPU only
- Success criterion:
  - at least one non-global view is relatively strongest on deeper local ambiguity buckets, while coarse view remains strongest or near-strongest globally
- Failure interpretation:
  - If one single view dominates every bucket, the hierarchy-allocation claim weakens significantly
- Table / figure target:
  - Figure 2 or an analysis table in main paper / appendix
- Priority:
  - `MUST-RUN`

### Block 2: Main Anchor Result

- Claim tested:
  - `MRC-SID` improves actual recommendation quality on the anchored bottleneck
- Why this block exists:
  - Without this block, the work remains an analysis insight rather than a method paper
- Dataset / split / task:
  - Industrial first
  - full tokenizer -> index -> downstream generative recommendation pipeline
- Compared systems:
  - `S0` semantic-only baseline
  - `S1` naive single-view global collaborative fusion
  - `S2` uniform all-view fusion
  - `S3` fixed level assignment (`coarse/mid/local`)
  - `S4` `MRC-SID`
- Metrics:
  - decisive:
    - `HR@1`, `NDCG@10`
  - secondary:
    - `HR@10`, `NDCG@1`, `HR@50`
    - same-`l1` / same-`l2` error rates
- Setup details:
  - Use the same 3-level SID structure as the current repo
  - Keep the first implementation minimal:
    - exactly three views
    - one purification step per view
    - one learned gate vector per SID level
  - Start with 1 seed for rapid decision; upgrade to 3 seeds only after signal is positive
- Success criterion:
  - `S4` beats `S0` and `S2` on Industrial with no catastrophic tokenizer degradation
- Failure interpretation:
  - If `S4` does not beat `S2`, the novelty claim weakens to “purified multi-view fusion” rather than hierarchy-aware allocation
- Table / figure target:
  - Main Table 1
- Priority:
  - `MUST-RUN`

### Block 3: Novelty Isolation

- Claim tested:
  - The gain comes from learned level-wise allocation itself, not from extra features or a convenient heuristic
- Why this block exists:
  - This is the block that changes reviewer belief about novelty
- Dataset / split / task:
  - Industrial full pipeline first
- Compared systems:
  - `S2` uniform all-view fusion
  - `S3` fixed level assignment
  - `S4` `MRC-SID`
  - `S5` swapped assignment control
- Metrics:
  - `HR@1`, `NDCG@10`
  - same-`l2` improvement
  - learned gate weights per level
- Setup details:
  - Parameter-match `S2` and `S4` as closely as possible
  - `S5` examples:
    - local-heavy Level 1
    - coarse-heavy Level 3
  - Keep all other downstream settings fixed
- Success criterion:
  - `S4 > S2`, `S4 > S3`, and `S5` underperforms `S4`
  - learned gate distributions are visibly non-uniform
- Failure interpretation:
  - If `S3 ≈ S4`, then learning the allocation may be unnecessary
  - If `S2 ≈ S4`, then hierarchy-aware allocation may not be the main source of gain
- Table / figure target:
  - Main Table 2 + gate visualization figure
- Priority:
  - `MUST-RUN`

### Block 4: Simplicity and Purification Check

- Claim tested:
  - The minimal `MRC-SID` is enough; extra complexity is not required, but purification is useful
- Why this block exists:
  - We need to defend elegance and explain why we are not overbuilding the method
- Dataset / split / task:
  - Industrial first
- Compared systems:
  - `S4` `MRC-SID`
  - `S6` no-purification `MRC-SID`
  - optional overbuilt variant:
    - `S7` ambiguity-aware or dynamic gate extension
- Metrics:
  - `HR@1`, `NDCG@10`
  - collision rate
  - code usage balance
  - gate entropy / interpretability
- Setup details:
  - `S6` keeps the same architecture, only removes view-specific purification
  - `S7` only if the core method is already positive; otherwise cut
- Success criterion:
  - `S4 > S6`
  - `S7` is unnecessary or only marginally better, supporting the simple core method
- Failure interpretation:
  - If `S6 ≈ S4`, denoising should be downgraded to a minor detail
  - If `S7 >> S4`, the paper may need to reconsider whether the minimal method is sufficient
- Table / figure target:
  - Appendix ablation table, with one main-paper sentence if needed
- Priority:
  - `MUST-RUN` for `S6`, `NICE-TO-HAVE` for `S7`

### Block 5: Generalization and Failure Analysis

- Claim tested:
  - The method is not Industrial-only and actually targets the claimed failure mode
- Why this block exists:
  - A second dataset and bucketed analysis are important for credibility
- Dataset / split / task:
  - Office full pipeline
  - plus bucketed error analysis on both datasets
- Compared systems:
  - `S0`
  - `S2`
  - `S3`
  - `S4`
- Metrics:
  - `HR@1`, `NDCG@10`
  - same-`l1` / same-`l2` target-better and error reduction
  - per-level gate weights
- Setup details:
  - Only run after Industrial shows positive signal
  - If compute is tight, run 1 seed first, then 3 seeds for the strongest pair only
- Success criterion:
  - Directionally similar gain on Office and consistent improvement on deeper ambiguity buckets
- Failure interpretation:
  - If the gain only appears on Industrial, the paper must narrow its scope and explain dataset dependence
- Table / figure target:
  - Main Table 3 or appendix generalization table + failure case figure
- Priority:
  - `MUST-RUN` once Industrial clears the decision gate

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| `M0` | Revalidate premise cheaply | Re-run Block 1 probe and log current bucket behavior | Continue only if coarse and mid/local views show differentiated utility | `~0-1 GPU-h` or CPU-only | The premise may collapse if one view dominates all buckets |
| `M1` | Lock implementation spec | Implement 3-view builder + purification + gate module and validate tensor shapes / SID generation | Continue only if tokenizer generation is stable and no immediate collapse appears | `~1-3 GPU-h` | Engineering errors or unstable view scaling |
| `M2` | Anchor on Industrial | Run `S0`, `S2`, `S3`, `S4` on Industrial with 1 seed | Continue only if `S4` beats `S2` or at least matches it with cleaner gate evidence | `~8-16 GPU-h` | The gain may come only from extra views or from heuristic mapping |
| `M3` | Isolate novelty | Run `S5` and `S6` on Industrial; inspect learned gates | Continue only if allocation is non-uniform and swapped / no-purification controls are worse | `~6-12 GPU-h` | The learned allocation may collapse to near-uniform |
| `M4` | Generalize and polish | Run `S2`, `S3`, `S4` on Office; add seeds / appendix extras | Paper-ready only if Office trend is consistent and stats remain healthy | `~8-16 GPU-h` | Gains may be dataset-specific or unstable across seeds |

## Must-Run vs Nice-to-Have

### Must-run

- Block 1 full
- Block 2 on Industrial
- Block 3 on Industrial
- Block 4 with `S6`
- Block 5 on Office with `S2`, `S3`, `S4`

### Nice-to-have

- Block 4 with overbuilt `S7`
- 3 seeds for every ablation once the main signal is already clear
- richer gate visualizations and additional purification variants

## Compute and Data Budget

- Total estimated GPU-hours:
  - minimal decision path (`M0`-`M3`): `~15-32 GPU-h`
  - paper-ready must-run path (`M0`-`M4`): `~23-48 GPU-h`
  - plus optional nice-to-have runs: `+8-20 GPU-h`
- Data preparation needs:
  - build and cache `coarse`, `mid`, `local` collaborative views for Industrial and Office
  - view purification scripts
  - parameter-matched uniform-fusion baseline implementation
- Human evaluation needs:
  - none
- Biggest bottleneck:
  - not raw model size, but disciplined baseline matching and end-to-end pipeline reproducibility

## Risks and Mitigations

- Risk:
  - `MRC-SID` beats semantic-only but not uniform all-view fusion
  - Mitigation:
    - downgrade the claim to purified multi-view fusion and avoid overstating hierarchy novelty

- Risk:
  - learned gates become nearly uniform
  - Mitigation:
    - add gate analysis early and inspect whether view scaling or purification makes the views too similar

- Risk:
  - tokenizer quality collapses before downstream evaluation
  - Mitigation:
    - add `M1` tokenizer-only health checks before any expensive runs

- Risk:
  - fixed mapping is as strong as learned allocation
  - Mitigation:
    - simplify the method and shift the paper claim toward “resolution-matched design principle” rather than learned gating

- Risk:
  - Office does not reproduce the Industrial trend
  - Mitigation:
    - narrow the paper scope and emphasize when the hierarchy benefit appears strongest

## Final Checklist

- [ ] Main paper tables are covered
- [ ] Novelty is isolated
- [ ] Simplicity is defended
- [ ] Frontier contribution is justified or explicitly not claimed
- [ ] Nice-to-have runs are separated from must-run runs

