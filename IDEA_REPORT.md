# Idea Discovery Report

**Direction**: MiniOneRec idea discovery for collaborative-aware generative recommendation
**Date**: 2026-04-06
**Pipeline**: repo context + local papers + latest public literature + in-repo pilot + refinement + experiment planning

## Executive Summary

I generated 6 candidate directions, filtered them to 3 realistic ones, and ran 1 lightweight pilot on the strongest direction. The main conclusion is that the broad "rebuild SID with collaborative information" story is now crowded by recent 2025-2026 work, while this repo's own evidence points to a sharper bottleneck: MiniOneRec often reaches the correct semantic subtree but fails at local leaf disambiguation. The recommended idea is therefore **Ambiguity-Aware Collaborative Leaf Refinement (ACLR)**, a lightweight prefix-local collaborative residual that only activates inside ambiguous `(a,b)` subtrees instead of replacing the whole tokenizer.

## Literature Landscape

### 1. What the field already covers well

The recent literature is now dense around **global collaborative tokenization** and **end-to-end tokenizer-recommender co-training**:

- `LCRec`, `TokenRec`, `UTGRec`, and `ETEGRec` push collaborative or transferable tokenization into LLM-based recommendation.
- `DiscRec` and `PRORec` continue the line of collaborative-aware semantic ID construction.
- `HiD-VAE` focuses on hierarchical and disentangled semantic IDs with uniqueness regularization.
- `PRISM` and `ReSID` directly target collaborative denoising, hierarchy, and predictable quantization.
- `PIT` goes even further toward dynamic personalized item tokenization.
- `FusID` shows that even modality-fused semantic ID design is already becoming active in 2026.

### 2. What the field is starting to cover outside tokenizer design

There is also movement beyond pure tokenizer rebuilding:

- `LLaDA-Rec` explores diffusion-style parallel SID generation.
- `RASTP` studies token pruning and generation efficiency.
- `APAO` targets vulnerable semantic prefixes and adaptive prefix optimization.

### 3. Repo-grounded gap that still looks promising

The strongest local signal in this repo is not "the whole tokenizer is broken." It is:

- collision is low: about `0.43%` on both Industrial and Office
- prefix ambiguity is high: Industrial has `max_layer3_parent_count = 96`
- same-prefix confusion is real: Industrial top-1 errors share `l1` in `21.5%` of cases and `l2` in `7.7%`; Office shares `l1` in `10.2%` and `l2` in `1.9%`
- when the model is wrong, the text similarity is still extremely high, especially in same-prefix errors
- train-only collaborative evidence often still prefers the true target inside these local confusions

That combination suggests a narrower opening:

> do not rebuild the whole tokenizer; repair the local leaf decision where text-only SID is already near the right subtree but lacks behavior-aware separation.

## Ranked Ideas

### 1. Ambiguity-Aware Collaborative Leaf Refinement (ACLR) - RECOMMENDED

- **Core idea**: keep the current tokenizer and coarse semantic prefixes, but add a lightweight train-only collaborative residual only when decoding the third SID token inside ambiguous `(a,b)` subtrees.
- **Why it fits this repo**: the existing codebase already has SID diagnostics, collaborative-gap analysis, beam outputs, and a stable SFT/RL pipeline. ACLR only needs an ambiguity profiler, prefix-local leaf prototypes, and a small leaf-level training/inference hook.
- **Pilot**: POSITIVE
  - Current Industrial top-1 hit: `0.07324`
  - Train-only collaborative rerank inside the top predicted `(a,b)` subtree: `0.07743`
  - Absolute gain: `+0.00419`
  - Relative gain: about `+5.7%`
  - Current Office top-1 hit: `0.08570`
  - Same local rerank: `0.08672`
  - Absolute gain: `+0.00103`
  - Relative gain: about `+1.2%`
- **Online validation update**:
  - We further implemented an `ACLR-lite` online evaluator that only activates inside ambiguous `(a,b)` subtrees (`ambiguity_l2` mode).
  - On Industrial, the real constrained-generation evaluation improved from `HR@1 = 0.07324` to `0.07920`, with `NDCG@3/5/10 = 0.09459 / 0.10145 / 0.11091`.
  - The online result matched the offline `ambiguity_l2` proxy exactly and kept `constraint_invalid_total = 0`.
- **Online ablation update**:
  - `same_l2` was stronger than `ambiguity_l2` (`HR@1 = 0.08162`), while `global` was the strongest upper bound (`HR@1 = 0.08383`).
  - This suggests the core gain indeed comes from leaf-local collaborative repair, but the current ambiguity gate is still conservative rather than fully optimized.
- **Interpretation of the pilot**: there is real signal in local behavior-aware correction, and the gain appears without retokenizing all items or widening beam size.
- **Novelty check**:
  - Too close: full collaborative tokenizer rebuild (`ReSID`, `PRISM`, `PIT`, `DiscRec`, `PRORec`)
  - Closest but still distinct: `APAO`, which is prefix-aware but not a train-only collaborative leaf residual tied to ambiguous-prefix repair
  - Current novelty status: `partial-to-good`
- **Reviewer-style verdict**: `8.7/10`
- **Best paper story**: "MiniOneRec's remaining error is a local ambiguity problem, not a global tokenizer problem. A selective collaborative residual at the leaf step is enough."
- **Next step**: implement the static logit bias first, then add the learned leaf-level auxiliary loss.

### 2. Ambiguity-Aware Partial Retokenization (APR) - BACKUP

- **Core idea**: identify only the high-risk `(a,b)` subtrees and locally rebuild their leaf assignments with lightweight collaborative residuals, while leaving the rest of the catalog unchanged.
- **Why it is interesting**: it preserves the repo's existing tokenizer for most items while making the tokenizer story more explicit than ACLR.
- **Why it is not ranked first**:
  - no pilot signal yet
  - much higher engineering risk
  - harder data/versioning story because partial SID remapping affects `index.json`, `convert`, and token extension
- **Pilot**: needs manual pilot
- **Novelty status**: better than generic collaborative tokenizer fusion, but still close to the tokenizer-heavy literature
- **Reviewer-style verdict**: `7.7/10`

### 3. Same-Prefix Hard-Negative Distillation (SPHD) - BACKUP

- **Core idea**: mine same-prefix confusers from existing beams and use them as leaf-level hard negatives during SFT, optionally with a small contrastive or preference-style auxiliary loss.
- **Why it fits**: easiest code path after ACLR and fully compatible with current data contracts.
- **Why it is ranked third**:
  - likely useful
  - but the contribution may look too training-trick-like unless paired with a stronger mechanistic story
- **Pilot**: no direct pilot, but strongly supported by the same-prefix error statistics and beam diagnostics
- **Novelty status**: moderate
- **Reviewer-style verdict**: `7.3/10`

## Eliminated Ideas

- **Full FAMAE + GAOQ + uniqueness-loss tokenizer rewrite**
  - eliminated because the novelty overlap with `ReSID`, `PRISM`, `HiD-VAE`, and related collaborative-tokenizer papers is too high
- **End-to-end personalized tokenizer co-evolution**
  - eliminated because `PIT`, `ETEGRec`, and `DAS` already occupy this space
- **Large graph or RL redesign**
  - eliminated because it does not match the repo's current bottleneck and would sprawl beyond a clean paper
- **Global collaborative residual in `embed.py` with no local ambiguity logic**
  - eliminated because it is feasible but too close to existing collaborative-tokenizer fusion lines

## Refined Proposal

- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Tracker: `refine-logs/EXPERIMENT_TRACKER.md`
- Pipeline summary: `refine-logs/PIPELINE_SUMMARY.md`

## Next Steps

- [ ] Implement the ambiguity profiler and static `cf_top_l2` leaf bias in evaluation first
- [ ] Add the learned leaf-level auxiliary loss for ambiguous prefixes
- [ ] Run the claim-driven experiment blocks in `refine-logs/EXPERIMENT_PLAN.md`
- [ ] If the local repair saturates, revisit APR as a tokenizer-heavy backup

## References Used in This Pass

- ReSID: https://arxiv.org/abs/2602.02338
- PRISM: https://arxiv.org/abs/2601.16556
- PIT: https://arxiv.org/abs/2602.08530
- FusID: https://arxiv.org/abs/2601.08764
- DiscRec: https://arxiv.org/abs/2506.15576
- PRORec: https://arxiv.org/abs/2502.06269
- LLaDA-Rec: https://arxiv.org/abs/2511.06254
- APAO: https://arxiv.org/abs/2603.02730
- Local grounding files: `Idea.tex`, `RESEARCH_BRIEF.md`, `v0_5_experiment_plan.md`, `mini_onerec_reproduction_progress.md`, `results/sid_diagnostics/*.json`, `results/collaborative_diagnostics/*.json`
