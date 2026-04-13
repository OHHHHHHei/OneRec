# Experiment Plan

**Problem**: the current strongest line `v2_on_p05 -> RL` already proves that graph-structured collaborative supervision survives end-to-end training, but it still trails the strongest original MiniOneRec RL mainly on `HR@5/10` and `NDCG@10`. Existing top-k and error analyses show that the remaining gap is a **mid-beam retention gap**, not a tokenizer-collapse or top1-ranking failure.  
**Method Thesis**: the next stage should target retention directly with **small tokenizer-side refinements**, not a full redesign. The two highest-value candidates are:

- **hierarchy loss isolation** via stop-gradient on earlier levels
- **stronger semantic-structure retention** that preserves neighborhood structure more faithfully than the current semantic smoothness loss

**Date**: 2026-04-12

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|-----------------|-----------------------------|---------------|
| `C1` Primary: the remaining gap to strongest original MiniOneRec RL can be reduced by **retention-targeted tokenizer refinements** rather than by replacing the graph bank or changing the recipe again | This keeps the paper story compact and avoids reopening the tokenizer design space | A small tokenizer refinement improves `HR@5/10` and `NDCG@10` under the fixed `v2_on_p05` downstream recipe, while keeping the current top1 / ambiguity advantages | `B1`, `B2`, `B3`, `B4` |
| `C2` Supporting: the current bottleneck is mainly **cross-layer interference and weak semantic neighborhood preservation**, not weak `G_mid` or missing new graph families | This tells us what to fix next and what not to overbuild | `stop-grad` isolation and/or stronger semantic retention close the retention gap more effectively than recipe churn or graph replacement | `B2`, `B3`, `B4` |

### Anti-claims to rule out

- the gap only exists because `v2` still needs a completely different downstream recipe
- the gap only exists because `G_mid` is the wrong signal source
- the only way forward is a much more complex tokenizer (`gate`, end-to-end joint training, new graph encoder)

### Minimum convincing evidence

- at least one retention-targeted refinement beats current `v2_on_p05 SFT` on both `HR@10` and `NDCG@10`
- the same candidate keeps or improves:
  - `NDCG@1/@3`
  - same-prefix top1 error cleanup
  - collided-target top1 hit
- best candidate is strong enough to justify one aligned `RL` confirmation run

## Response to the External Review

We take the external review seriously, but we do not adopt it wholesale.

### What we agree with

- The review correctly identifies that the current hierarchy loss is applied on cumulative representations, so later-level graph losses can backpropagate into earlier levels. This is a real implementation fact in the current code and a plausible source of cross-level interference.
- The review is also right that the current `semantic retention` term is still another smoothness-style regularizer, not a fully faithful semantic-structure preservation objective.
- The critique that current ambiguity weighting is item-level rather than truly level-wise is conceptually valid.
- The follow-up warning about `stop-gradient` changing the effective level-1 regularization strength is also valid. In the current `v2`, level 1 effectively receives graph-related pressure from multiple later losses; once we isolate gradients, that pressure drops sharply and should not be ignored.

### What we do not treat as immediate conclusions

- We do **not** yet accept that `G_mid` being derived from `G_coarse` is the dominant bottleneck. Our current evidence still says the present spectral `mid` view is the strongest `G_mid` candidate we have.
- We do **not** elevate online uncertainty to immediate top priority. Our own proxy sanity results showed that the current offline+online version was weaker than the offline proxy alone.
- We do **not** think this is the right moment for a large redesign such as fully free gates, end-to-end tokenizer/downstream joint optimization, or new graph-encoder families.

### Why the current stage-2 plan is the right response

The current evidence already tells us something very specific:

- `v2` works end-to-end.
- `v2_on_p05 -> RL` is already competitive and preserves the intended ambiguity-cleanup effect.
- The remaining gap is mainly a `top5/top10` retention gap, not a top1 failure and not a tokenizer-collapse problem.

So the rational next step is **not** to reopen the whole tokenizer design space.
Instead, we adopt the parts of the external review that most directly match the observed gap:

1. isolate hierarchy losses across levels with `stop-gradient`
2. replace weak semantic smoothness with stronger semantic-structure retention

This keeps the next stage:

- focused on the actual bottleneck,
- cheap enough to test quickly,
- and clean enough that success or failure will still be interpretable.

In short, this stage-2 plan is a **selective response** to the review:
it absorbs the highest-value critiques, rejects premature redesign, and targets the exact failure mode supported by the current experiments.

## Paper Storyline

- Main paper must prove:
  - graph-aware tokenizer improvement is real and survives SFT/RL
  - the remaining gap is retention-specific
  - a small, principled tokenizer refinement can further close that gap without reopening the full design space

- Appendix can support:
  - detailed retention diagnostics
  - tokenizer-only structural comparisons
  - more speculative alternatives such as level-wise online ambiguity or new graph families

- Experiments intentionally cut for now:
  - new `G_mid` independent-source construction
  - fully free learned gate
  - end-to-end tokenizer/downstream joint optimization
  - STOSA-style distributional quantization

## Experiment Blocks

### Block 1: Retention Hypothesis Freeze

- Claim tested:
  - the current main bottleneck is mid-beam retention, not head ranking or recipe mismatch
- Why this block exists:
  - we already have enough evidence; this block freezes the next-stage target so we do not drift back into broad tokenizer redesign
- Dataset / split / task:
  - Industrial
  - reuse completed `v2_on_p05` SFT/RL analyses
- Compared systems:
  - strongest original MiniOneRec SFT
  - strongest original MiniOneRec RL
  - `v2_on_p05 SFT`
  - `v2_on_p05 RL`
- Metrics:
  - `NDCG@1/3/5/10`
  - `HR@1/3/5/10`
  - same-prefix error rates
  - `beam_contains_same_l1/l2`
  - collided-target top1 hit
- Setup details:
  - no new training
  - use existing reports as the frozen diagnosis base
- Success criterion:
  - the team agrees that the stage-2 target is retention
- Failure interpretation:
  - if this cannot be stated clearly, do not start implementation
- Table / figure target:
  - not a paper table; this is an internal planning checkpoint
- Priority:
  - `MUST-RUN`, but already completed by current analysis

### Block 2: Tokenizer Micro-Refinement A — Stop-Gradient Hierarchy Isolation

- Claim tested:
  - part of the retention gap is caused by cross-layer interference from applying graph losses to cumulative representations without gradient isolation
- Why this block exists:
  - this is the lowest-cost, highest-clarity tokenizer change suggested by both the external review and our own code reading
- Dataset / split / task:
  - Industrial
  - tokenizer-only: `sid-train -> sid-generate -> local ambiguity diagnosis`
- Compared systems:
  - `T0`: current `v2`
  - `T1a`: `v2 + stop-grad isolation` with unchanged weights
  - `T1b`: `v2 + stop-grad isolation` with a modest level-1 compensation (`coarse_weight` increased)
- Metrics:
  - tokenizer-side:
    - final generated collision
    - weighted `H(level3 | level1, level2)`
    - target-weighted mean `l2` leaf count
  - secondary:
    - changed-SID fraction
    - multi-leaf `same_l2`
- Setup details:
  - keep graph bank fixed
  - keep offline ambiguity prior fixed
  - only change:
    - for level `l`, build cumulative representation as
      `sg(sum_{t<l} q^(t)) + q^(l)` when computing that level's graph loss
  - because this isolation sharply reduces the graph-related pressure seen by level 1, test two variants:
    - `T1a`: unchanged weights, for a clean mechanism check
    - `T1b`: increased `coarse_weight`, to avoid a false negative caused only by weakened level-1 regularization
- Success criterion:
  - tokenizer structure is at least as clean as current `v2`, without obvious regression
  - at least one of `T1a/T1b` is worth carrying forward
- Failure interpretation:
  - if tokenizer structure regresses immediately, stop and do not push this variant downstream
- Table / figure target:
  - appendix tokenizer refinement table
- Priority:
  - `MUST-RUN`

### Block 3: Tokenizer Micro-Refinement B — Stronger Semantic-Structure Retention

- Claim tested:
  - the current semantic retention is too weakly aligned with the real retention bottleneck because it is only another graph smoothness term
- Why this block exists:
  - current evidence says we need better neighborhood preservation, not more graph strength
- Dataset / split / task:
  - Industrial
  - tokenizer-only: `sid-train -> sid-generate -> local ambiguity diagnosis`
- Compared systems:
  - `T0`: current `v2`
  - `T1`: Block-2 best candidate
  - `T2`: Block-2 best candidate + stronger semantic retention
  - `T2b`: fallback branch = current `v2` + stronger semantic retention without stop-grad, only if Block 2 regresses
- Metrics:
  - same tokenizer metrics as Block 2
  - plus semantic-neighborhood preservation diagnostics if easy to compute
- Setup details:
  - keep graph bank and ambiguity prior fixed
  - replace current semantic smoothness term with a **more structure-faithful retention term**
  - first implementation is fixed as:
    - **batch-local softmax neighborhood KL**
  - concrete form:
    - build a semantic-neighbor soft distribution from the original semantic space inside the batch
    - build the corresponding distribution from the tokenizer representation
    - minimize `KL(p_sem || q_tok)`
  - reasons for choosing this first:
    - more faithful than another smoothness term
    - lighter and more stable than a full-graph KL
    - more informative than a single sampled pairwise margin
- Success criterion:
  - tokenizer structure stays strong and downstream candidate looks promising for beam retention
- Failure interpretation:
  - if stronger retention destroys hard-case gains, this branch is not the right fix
- Table / figure target:
  - appendix or later method-refinement table
- Priority:
  - `MUST-RUN`

### Block 4: Fixed-Recipe SFT Screen

- Claim tested:
  - the tokenizer refinement closes the actual downstream gap under the **same best current recipe** rather than by introducing another recipe change
- Why this block exists:
  - the key question is not tokenizer beauty, but whether the change improves `top5/top10` under `title_history2sid_on + desc_align_p05`
- Dataset / split / task:
  - Industrial
  - fixed downstream recipe: `title_history2sid_on + desc_align_p05`
- Compared systems:
  - `S0`: current `v2_on_p05 SFT`
  - `S1`: strongest original MiniOneRec SFT
  - `S2`: Block-2 candidate under `on+p05`
  - `S3`: Block-3 candidate under `on+p05`
- Metrics:
  - decisive:
    - `HR@10`
    - `NDCG@10`
  - secondary:
    - `HR@5`
    - `NDCG@1/@3`
    - top-k / same-prefix diagnostics
- Setup details:
  - keep SFT hyperparameters and launcher aligned to the current successful `v2_on_p05` run
  - use 4 GPUs
  - run one seed first
  - if the best candidate only improves by a small margin, do one extra seed before RL
- Success criterion:
  - candidate beats current `v2_on_p05 SFT` on both `HR@10` and `NDCG@10`
  - ideally also beats strongest original MiniOneRec SFT on both
- Failure interpretation:
  - if no candidate improves `HR@10`, the tokenizer-side refinement did not address the real gap
- Table / figure target:
  - main paper comparison table candidate
- Priority:
  - `MUST-RUN`

### Block 5: RL Confirmation on the Best Retention Candidate

- Claim tested:
  - retention-targeted tokenizer refinement translates into better end-to-end RL performance, not only SFT improvements
- Why this block exists:
  - the stage goal is to surpass strongest original MiniOneRec RL, not only to polish SFT
- Dataset / split / task:
  - Industrial
  - fixed downstream recipe source checkpoint: best Block-4 candidate
  - RL hyperparameters aligned to the currently working `v2_on_p05 -> RL` run
- Compared systems:
  - `R0`: strongest original MiniOneRec RL
  - `R1`: current `v2_on_p05 RL`
  - `R2`: best stage-2 candidate -> RL
- Metrics:
  - decisive:
    - `HR@10`
    - `NDCG@10`
  - secondary:
    - `HR@3/@5`
    - collided-target top1 hit
    - top-k / same-prefix diagnostics
- Setup details:
  - keep RL recipe fixed and aligned
  - only the source SFT checkpoint changes
- Success criterion:
  - `R2` beats current `v2_on_p05 RL` on both `HR@10` and `NDCG@10`
  - target paper milestone:
    - beat strongest original MiniOneRec RL on the main cutoffs
- Failure interpretation:
  - if SFT improves but RL does not, the remaining issue is in RL-stage retention rather than tokenizer-side structure
- Table / figure target:
  - final main result table
- Priority:
  - `MUST-RUN`, but only after Block 4 picks a clear winner

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| `M0` | freeze retention target | reuse existing analyses | agree that the next-stage target is `top5/top10` retention | very low | drifting back into broad redesign |
| `M1` | test smallest tokenizer fix | `T1a/T1b` | tokenizer structure does not regress | medium | stop-grad may not help enough downstream |
| `M2` | test stronger retention fix | `T2/T2b` | at least one candidate is worth downstream screening | medium | semantic retention redesign may blunt hard-case gains |
| `M3` | fixed-recipe SFT screen | `S2/S3` | beat current `v2_on_p05 SFT` on both `HR@10` and `NDCG@10`; if gain is small, confirm with a second seed | high | candidate may improve head but not retention |
| `M4` | RL confirmation | `R2` | beat current `v2_on_p05 RL`; aim at strongest original RL | high | RL may reintroduce the same retention gap |

## Compute and Data Budget

- Total estimated GPU-hours:
  - tokenizer candidates: moderate
  - one or two SFT screens: moderate to high
  - one RL confirmation: high
- Data preparation needs:
  - no new recipe search
  - reuse current `data_experiment` style pipeline if tokenizer changes alter SID outputs
- Human evaluation needs:
  - none
- Biggest bottleneck:
  - each downstream run is expensive, so tokenizer candidates must be kept small and purposeful

## Risks and Mitigations

- **Risk**: stop-grad isolation improves retention but weakens ambiguity cleanup
  - **Mitigation**: make same-prefix error rates and collided-target top1 hit mandatory checks
  - **Mitigation 2**: include a compensated `T1b` variant so the branch is not rejected only because level-1 regularization became too weak

- **Risk**: stronger semantic retention simply drags the model back toward the original semantic tokenizer
  - **Mitigation**: require that head advantages and hard-case gains are preserved

- **Risk**: tokenizer-side improvement does not survive RL
  - **Mitigation**: keep RL recipe fixed so the failure can be localized to stage mismatch rather than recipe churn

- **Risk**: we reopen too many design axes at once
  - **Mitigation**: do not change graph bank, ambiguity prior source, or downstream recipe in this stage

- **Risk**: single-seed gains near the noise floor are mistaken for real progress
  - **Mitigation**: if the best SFT gain is only marginal, run a second seed before spending RL budget

## Final Checklist

- [ ] Stage-2 target is frozen as `mid-beam retention`
- [ ] `stop-grad` hierarchy isolation is implemented and tested
- [ ] stop-grad branch includes an uncompensated and a modestly compensated level-1 variant
- [ ] stronger semantic-structure retention is implemented and tested
- [ ] semantic retention first implementation is fixed as batch-local softmax neighborhood KL
- [ ] all downstream screens use the fixed `v2_on_p05` recipe
- [ ] best candidate is confirmed by RL before claiming a new strongest line
- [ ] speculative redesigns remain out of scope for this stage
