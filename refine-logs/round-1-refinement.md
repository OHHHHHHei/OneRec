# Research Proposal: Ambiguity-Aware Collaborative Leaf Refinement (ACLR)

## Problem Anchor

- Bottom-line problem: MiniOneRec often reaches the correct semantic neighborhood but fails to separate behaviorally different yet textually similar items at the final SID leaf decision.
- Must-solve bottleneck: fix local same-prefix ambiguity without rebuilding the entire tokenizer.
- Non-goals: no full collaborative tokenizer rewrite, no new RL method, no graph-heavy end-to-end architecture.
- Constraints: must fit the current MiniOneRec pipeline, stay leakage-safe, and remain implementable on the current repo with modest code changes.
- Success condition: improve HR/NDCG and reduce same-prefix local errors, especially in ambiguous `(a,b)` subtrees, while preserving the existing tokenizer and training stack.

## Technical Gap

Recent work has already covered global collaborative tokenization well. The actual repo evidence points to a narrower gap: the coarse semantic prefix is often already usable, but the final leaf code is still overly text-driven inside ambiguous subtrees. None of the local grounding material suggests that a full tokenizer replacement is the smallest adequate fix.

## Method Thesis

Use a train-only collaborative residual only for ambiguous `(a,b)` prefixes so the model learns and applies behavior-aware leaf disambiguation exactly where MiniOneRec still fails.

## Contribution Focus

- Dominant contribution: ambiguity-aware collaborative residualization of the third SID token
- Optional supporting contribution: a diagnostic-driven ambiguity profiler that decides where the residual is active
- Explicit non-contributions: full retokenization, graph-heavy collaborative encoder, new RL algorithm

## Proposed Method

### Complexity Budget

- Frozen / reused backbone: current tokenizer, current `index.json`, current SFT/RL backbone, current constrained decoding
- New trainable components:
  - a small projection from decoder hidden state to a collaborative query space
  - offline collaborative leaf prototypes built from train-only interactions
- Tempting additions intentionally not used:
  - global tokenizer retraining
  - end-to-end personalized tokenizer co-evolution
  - new graph neural network by default

### System Overview

1. Build an **ambiguity profiler** from existing diagnostics.
2. For each ambiguous `(a,b)` prefix, build train-only collaborative leaf prototypes for the valid `c` tokens under that prefix.
3. During SFT, add a leaf-level auxiliary loss only when the target prefix is marked ambiguous.
4. During decoding, add a small collaborative residual to the valid third-level token logits under that prefix.
5. Leave all other prefixes untouched.

### Core Mechanism

- Let `p = (a,b)` denote a two-level prefix.
- Build an ambiguity score `alpha_p` from prefix entropy, same-prefix error rate, and collaborative-gap diagnostics.
- Build a collaborative residual vector `r_i` for each item from train-only item transitions or co-occurrence compression.
- Aggregate item residuals into prefix-local leaf prototypes `e_{p,c}`.
- At the third-token step, project the model hidden state `h_t` into a collaborative query `q_t = W h_t`.
- Apply an auxiliary loss over valid leaf tokens in the prefix:

  `L_leaf = -log exp(q_t . e_{p,c*} / tau) / sum_{c in C(p)} exp(q_t . e_{p,c} / tau)`

- Only activate this loss when `p` is ambiguous, weighted by `alpha_p`.
- At inference, adjust logits for valid leaf tokens:

  `z'_c = z_c + beta * alpha_p * (q_t . e_{p,c})`

This keeps the intervention local, behavior-aware, and easy to ablate.

## Minimal Claim-Driven Validation

- Claim 1: ACLR improves recommendation by fixing local leaf ambiguity rather than by rebuilding the tokenizer.
  - Evidence: improved HR/NDCG plus lower same-prefix miss rates
- Claim 2: selective local activation is better than global collaborative injection
  - Evidence: ACLR beats global activation and heuristic reranking-only baselines

## Remaining Risks

- The correct target may not always appear in the beam or valid subtree.
- Train-only collaborative statistics may be noisy for tail items.
- If improvements remain tiny, the method may be too small to carry a full paper alone.

## Verdict

This version is focused, better grounded in the repo's diagnostics, and materially cleaner than the original global collaborative tokenizer rewrite.
