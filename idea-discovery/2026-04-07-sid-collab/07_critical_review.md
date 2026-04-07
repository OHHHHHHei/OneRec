# Critical Review

## Reviewer-style verdict

Current score for the top idea:

- `6.5 / 10` as a paper direction
- `8 / 10` as an internal research direction

## Main strengths

1. The motivation is grounded in actual repo diagnostics rather than generic intuition.
2. The latest literature makes the bottleneck sharper instead of undermining it.
3. The method can start from existing code and lightweight pilots.
4. The story is much less crowded than a full tokenizer rewrite.

## Main weaknesses

### 1. Risk of looking like an evaluation hack

If the method is just:

- detect ambiguous prefix
- rerank with collaborative score

then reviewers may say it is not a real model contribution.

### 2. Need a principled trigger

The current `ambiguity_l2` heuristic already shows that a naive trigger is not enough. Office even regresses. So the real contribution cannot be a hard-coded leaf-count threshold.

### 3. Need to explain why global rerank is not the answer

The local pilot shows `global` rerank is strongest in raw hit@1. If the paper proposes selective local repair, it must explain:

- why local repair is preferable
- why it preserves semantic structure better
- why it is more robust or cleaner than full-list reranking

### 4. Need a training story or at least a consistency story

Pure post-hoc reranking is fragile as a paper. The paper needs one of:

- a learned ambiguity predictor
- a calibrated risk objective
- a consistency loss that exposes risky prefixes during training

## Minimum viable publishable version

The idea becomes much stronger if it is upgraded from:

- "heuristic ambiguity reranker"

to:

- "ambiguity-calibrated local repair framework with train-only risk estimation and restricted candidate refinement"

## Recommended scope guardrails

Do:

- keep the tokenizer fixed
- focus on local ambiguity after coarse semantic routing
- use train-only collaborative evidence
- compare against both local and global rerank baselines

Do not:

- drift into full tokenizer retraining
- claim to solve collision globally
- oversell the method as a universal replacement for SID learning

## Final review conclusion

This is a viable direction if framed narrowly and upgraded from heuristic to calibrated method. It is not yet a paper just because the motivation is good. The paper lives or dies on whether the selective trigger is principled and whether the method is more than a decoding patch.
