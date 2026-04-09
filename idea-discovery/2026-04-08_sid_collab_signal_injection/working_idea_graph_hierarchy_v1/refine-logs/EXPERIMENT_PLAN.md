# Experiment Plan

**Problem**: existing collaborative SID tokenizers mostly inject one collaborative signal globally, while our current repo evidence suggests that hierarchical SID learning needs graph-structured collaborative supervision with different scales and different noise controls.  
**Method Thesis**: `MGR-SID` should improve MiniOneRec-style generative recommendation by learning item codes under **level-specific supervision from a denoised multiplex graph bank**, instead of relying on one globally fused collaborative representation.  
**Date**: 2026-04-09

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|-----------------|-----------------------------|---------------|
| `C1` Primary: level-aware graph supervision is the real contribution | Without this, the work collapses into ordinary graph feature fusion or generic graph regularization. | On Industrial, `MGR-SID` beats semantic-only, graph feature fusion, and uniform graph regularization; learned allocations are non-uniform and useful. | `B2`, `B3` |
| `C2` Supporting: graph bank design and view-specific purification are necessary for stable SID enhancement | This explains why naive front-end fusion failed and why graph information can help without collapse. | A chosen `G_mid` beats naive mid-scale proxies; purified graphs produce better bucketed probes and healthier tokenizer statistics than raw graph views. | `B1`, `B4`, `B5` |

### Anti-claims to rule out

- The gain only comes from adding more graph information or more parameters.
- The gain only comes from one strong graph prior, not from level-aware usage.
- The gain is an artifact of a lucky `G_mid` choice rather than a stable design.
- The gain comes with tokenizer collapse, severe code imbalance, or semantic hierarchy damage.
- The graph component is just another feature-fusion trick in disguise.

## Paper Storyline

- Main paper must prove:
  - `MGR-SID` improves recommendation quality on the actual bottleneck, not just graph proxy metrics.
  - The dominant contribution is **level-aware graph supervision**, not generic graph enhancement.
  - The graph bank must be multi-resolution and denoised; naive graph injection is not enough.
  - The semantic SID backbone remains stable and usable.

- Appendix can support:
  - full `G_mid` search table
  - raw vs purified graph diagnostics
  - extra seeds
  - optional overbuilt variants such as local-graph augmentation
  - additional bucketed error analyses and gate heatmaps

- Experiments intentionally cut:
  - full personalized dynamic tokenization
  - graph encoder competitions as a main story
  - large baseline zoos unrelated to the core claim
  - too many graph views beyond `coarse / mid / local`

- Frontier necessity:
  - explicitly **not claimed**
  - this paper is not about LLM / VLM / RL-era primitives; skip a frontier-necessity block

## Baseline Families

To keep the paper focused, use three baseline families only:

1. **Semantic SID baseline**
   - current MiniOneRec semantic-only tokenizer and downstream recommender

2. **Graph integration baselines**
   - graph feature fusion
   - uniform graph regularization

3. **Mechanism ablations**
   - fixed level assignment
   - swapped assignment
   - no-denoising
   - optional overbuilt variant

## Experiment Blocks

### Block 1: Graph Bank Search and Purification

- Claim tested:
  - `C2`: the graph carrier and denoising design are meaningful and not arbitrary
- Why this block exists:
  - `G_mid` is the most open design choice in the whole method
  - we need to settle graph construction before expensive full-pipeline runs
- Dataset / split / task:
  - Amazon Industrial first
  - Office only as a tie-breaker if Industrial is inconclusive
  - train-only graph construction and compatibility probes
- Compared systems:
  - `M0a`: raw `G_coarse + G_mid(diffusion residual) + G_local`
  - `M0b`: raw `G_coarse + G_mid(band-pass) + G_local`
  - `M0c`: raw `G_coarse + G_mid(community-aware) + G_local`
  - `M1a/M1b/M1c`: purified versions of the above
  - optional backup:
    - `G_mid(PPR-difference)`
- Metrics:
  - decisive:
    - target-better rate on `same_l2`
    - target-better rate on `same_l1`
  - secondary:
    - all-error target-better rate
    - coverage
    - graph sparsity / density
    - popularity correlation of edge weights
    - degree skew before and after purification
- Setup details:
  - no tokenizer retraining
  - graph bank is built from train split only
  - evaluate current result files and SID buckets with reusable probe scripts
- Success criterion:
  - choose one `G_mid` whose purified version improves deep ambiguity support without obvious instability or triviality
- Failure interpretation:
  - if all `G_mid` candidates are weak, the method story is too underdetermined and should pause before full implementation
- Table / figure target:
  - appendix table: graph candidate comparison
  - appendix figure: coverage vs ambiguity gain scatter
- Priority:
  - `MUST-RUN`

### Block 2: Main Anchor Result

- Claim tested:
  - `C1`: level-aware graph supervision improves the actual recommendation pipeline
- Why this block exists:
  - without this block, the work remains a diagnosis plus graph probe story
- Dataset / split / task:
  - Industrial
  - full tokenizer -> SID index -> downstream generative recommendation pipeline
- Compared systems:
  - `S0`: semantic-only MiniOneRec baseline
  - `S1`: graph feature fusion
  - `S2`: uniform graph regularization
  - `S3`: `MGR-SID`
- Metrics:
  - decisive:
    - `HR@1`
    - `NDCG@10`
  - secondary:
    - `HR@10`
    - `HR@50`
    - `NDCG@1`
    - same-`l1` and same-`l2` error reduction
    - collision rate
    - active-code ratio / code usage balance
- Setup details:
  - fixed 3-level SID
  - same semantic base embeddings for all systems
  - same downstream recommender training budget
  - start with 1 seed; move to 3 seeds only after a positive signal
  - `S1` and `S2` must be parameter-matched to `S3` as closely as practical
- Success criterion:
  - `S3 > S1` and `S3 > S2` on Industrial, with no catastrophic tokenizer degradation
- Failure interpretation:
  - if `S3` cannot beat `S1`, the work weakens to graph fusion with extra machinery
  - if `S3` cannot beat `S2`, the hierarchy-aware claim weakens to generic graph regularization
- Table / figure target:
  - Main Table 1
- Priority:
  - `MUST-RUN`

### Block 3: Novelty Isolation and Mechanism

- Claim tested:
  - `C1`: the gain comes from level-aware graph supervision, not from arbitrary graph bias
- Why this block exists:
  - this is the block that changes reviewer belief about novelty
- Dataset / split / task:
  - Industrial full pipeline
- Compared systems:
  - `S2`: uniform graph regularization
  - `S3`: `MGR-SID`
  - `S4`: fixed level assignment (`coarse / mid / local`)
  - `S5`: swapped assignment control
  - optional:
    - `S6`: single-best-graph regularization only
- Metrics:
  - decisive:
    - `HR@1`
    - `NDCG@10`
  - mechanism:
    - same-`l2` improvement
    - learned level-wise weights
    - per-level graph usage pattern
- Setup details:
  - same selected graph bank from Block 1
  - same backbone and training budget as Block 2
  - 1 seed first; 3 seeds for the strongest two systems
- Success criterion:
  - `S3 > S2`, `S3 > S4`, and `S5 < S3`
  - learned graph usage is clearly non-uniform and interpretable
- Failure interpretation:
  - if `S4 ≈ S3`, learned allocation may be unnecessary
  - if `S2 ≈ S3`, the claimed contribution is too weak
- Table / figure target:
  - Main Table 2
  - Main or appendix figure: level-wise allocation heatmap
- Priority:
  - `MUST-RUN`

### Block 4: Fusion Safety and Simplicity Check

- Claim tested:
  - `C2`: graph information must be purified and structurally integrated
  - the minimal method is sufficient; extra complexity is not required for the first paper
- Why this block exists:
  - we must defend both stability and elegance
- Dataset / split / task:
  - Industrial full pipeline
- Compared systems:
  - `S3`: core `MGR-SID`
  - `S7`: no-denoising `MGR-SID`
  - `S8`: weak semantic anchor / anchor-ablated `MGR-SID`
  - optional overbuilt variant:
    - `S9`: `MGR-SID +` local-graph augmentation or confidence-routed allocation
- Metrics:
  - decisive:
    - `HR@1`
    - `NDCG@10`
  - stability:
    - collision rate
    - code usage entropy / active-code ratio
    - graph allocation entropy
    - training failures or divergence
- Setup details:
  - keep graph bank fixed
  - only one factor changes at a time
  - overbuilt `S9` is appendix-only and runs only if `S3` is already positive
- Success criterion:
  - `S3 > S7`
  - `S3 > S8`
  - `S9` is unnecessary or only marginally better relative to its complexity
- Failure interpretation:
  - if `S7 ≈ S3`, denoising is not a major claim
  - if `S8 ≈ S3`, semantic anchor is weaker than expected
  - if `S9 >> S3`, the simple core method may be underbuilt
- Table / figure target:
  - appendix ablation table
  - one main-paper sentence defending simplicity
- Priority:
  - `MUST-RUN` for `S7`
  - `MUST-RUN` for `S8` if cheap
  - `NICE-TO-HAVE` for `S9`

### Block 5: Generalization and Failure Analysis

- Claim tested:
  - the direction is not Industrial-only and the gains really target the claimed bottleneck
- Why this block exists:
  - a second dataset and bucketed evidence make the paper much more believable
- Dataset / split / task:
  - Office full pipeline
  - plus bucketed analyses on both Industrial and Office
- Compared systems:
  - `S0`: semantic-only baseline
  - `S2`: uniform graph regularization
  - `S3`: `MGR-SID`
- Metrics:
  - decisive:
    - `HR@1`
    - `NDCG@10`
  - mechanism:
    - same-`l1` and same-`l2` improvements
    - gain per activated sample
    - per-level graph allocations
    - representative failure cases
- Setup details:
  - run only after Industrial clears the decision gate
  - 1 seed first, 3 seeds for finalists if signal remains positive
- Success criterion:
  - directionally similar gains on Office
  - improvements remain concentrated in local ambiguity buckets
- Failure interpretation:
  - if gains vanish on Office, the method may be dataset-specific or graph-construction-specific
- Table / figure target:
  - Main Table 3 or appendix generalization table
  - appendix failure-analysis figure
- Priority:
  - `MUST-RUN` once Industrial is positive

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| `M0` | settle `G_mid` and purification choices | `B1` graph-candidate probes on Industrial | continue only if one purified `G_mid` looks clearly nontrivial and useful | CPU-heavy + `~0-4 GPU-h` | mid-scale graph may remain vague |
| `M1` | build reusable graph bank cache and sanity-check metrics | graph cache generation, probe reproducibility, tokenizer health scripts | continue only if graph bank artifacts are stable and reproducible | `~2-6 GPU-h` or CPU-only | engineering drift between graph cache and training pipeline |
| `M2` | establish Industrial anchor | `S0`, `S1`, `S2`, `S3` with 1 seed on Industrial | continue only if `S3` beats `S1` or at least matches it with clear same-`l2` benefit and healthy tokenizer stats | `~20-36 GPU-h` | graph regularization may not beat simpler graph fusion |
| `M3` | isolate novelty and defend simplicity | `S4`, `S5`, `S7`, optional `S8`, plus extra seeds for finalists | continue only if level-awareness is supported and no-denoising is worse | `~18-34 GPU-h` | gain may collapse to generic graph prior |
| `M4` | generalize and polish | Office finalists, bucketed analysis, appendix extras | paper-ready only if Office trend is directionally consistent and bucketed gains match the claim | `~20-40 GPU-h` | gains may be dataset-specific |

## Must-Run vs Nice-to-Have

### Must-run

- Block 1 full
- Block 2 full on Industrial
- Block 3 core controls on Industrial
- Block 4 no-denoising and anchor ablations
- Block 5 Office one-seed plus bucketed analysis

### Nice-to-have

- additional `G_mid` variants after one winner is clear
- 3-seed Office confirmation
- overbuilt `S9`
- local-graph augmentation if `G_local` is too sparse
- richer qualitative graphs in the appendix

## Compute and Data Budget

- Total estimated GPU-hours:
  - one-seed must-run path: `~40-70 GPU-h`
  - paper-ready path with selective 3-seed confirmation: `~70-120 GPU-h`
- Data preparation needs:
  - train-only graph bank construction for `G_coarse / G_mid / G_local`
  - cached purified graph artifacts
  - reusable bucketed evaluation scripts for same-`l1` / same-`l2`
- Human evaluation needs:
  - none
- Biggest bottleneck:
  - stable integration of graph regularization into SID learning without collapsing the semantic hierarchy

## Risks and Mitigations

- **Risk**: `G_mid` remains vague or weak.
  - **Mitigation**: do not launch full training until Block 1 chooses a specific `G_mid`.

- **Risk**: graph regularization destabilizes SID construction.
  - **Mitigation**: keep semantic anchor strong; start with conservative graph-loss weights; monitor collision, active-code ratio, and code usage entropy every run.

- **Risk**: gains come from extra graph information rather than level-aware design.
  - **Mitigation**: implement `graph feature fusion` and `uniform graph regularization` before claiming success.

- **Risk**: `G_local` is too sparse to matter.
  - **Mitigation**: start with temporal weighting and pruning; only if needed, add Seq2Graph-style augmentation as an appendix variant.

- **Risk**: compute expands before the story is validated.
  - **Mitigation**: use strict decision gates; one seed before three; Industrial before Office.

## Final Checklist

- [ ] Main paper tables are covered
- [ ] Novelty is isolated
- [ ] Simplicity is defended
- [ ] Frontier contribution is explicitly not claimed
- [ ] Nice-to-have runs are separated from must-run runs
- [ ] `G_mid` is chosen before full training
- [ ] Fusion baselines exist before claiming hierarchy-aware gain
