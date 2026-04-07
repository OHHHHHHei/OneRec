# Structured Process Log

This file records the working process in a structured form. It is intentionally written as reproducible notes and decision records rather than hidden raw chain-of-thought.

## 2026-04-07 Step 1: gather repo context

Actions:

- read the user's motivations
- inspected local notes and reports
- checked whether the repo already contains any collaborative repair utilities

Key findings:

- the repo already has `collaborative_rerank.py`
- the evaluation pipeline already has `aclr_lite` hooks
- previous local notes already suspected local same-prefix ambiguity

Decision:

- use the current repo as the evidence anchor, not just the literature

## Step 2: verify the dominant local bottleneck

Actions:

- reviewed local diagnostic summaries
- checked collision, same-prefix error, beam-local overlap, and entropy-style indicators

Key findings:

- collision exists but is small
- same-prefix confusion is much larger
- many failures already stay in semantically plausible local neighborhoods

Decision:

- treat local ambiguity as the primary practical bottleneck

## Step 3: search latest arXiv papers

Actions:

- searched recent arXiv papers around collaborative tokenization, prefix-aware modeling, and collision-aware SID learning
- prioritized 2025-2026 papers

Key findings:

- the literature is now crowded on global collaborative tokenizer redesign
- the literature is also crowded on prefix-aware training and collision-aware SID learning

Decision:

- avoid recommending a generic tokenizer rewrite as the top idea

## Step 4: run a lightweight local pilot

Actions:

- used the repo's existing collaborative rerank utilities
- compared baseline, global rerank, and local-prefix-constrained rerank modes on Industrial and Office

Key findings:

- collaborative information clearly helps
- global rerank is strongest in raw top-1 gain
- local same-l1 / same-l2 rerank also helps
- the current `ambiguity_l2` trigger is too crude and is not stable across datasets

Decision:

- the paper should not be "collaboration helps"
- the sharper question is "when and where should collaboration intervene"

## Step 5: synthesize the paper space

Actions:

- compared local evidence against recent papers
- ranked feasible directions by novelty, fit, and engineering cost

Key findings:

- full collaborative tokenizer rewrite is strongest in raw intuition but weakest in novelty
- selective local repair best matches both repo evidence and literature gaps

Decision:

- recommend `ACLR-Plus` as the top idea

## Step 6: refine the top idea

Actions:

- converted the idea into a method thesis and experiment plan

Key findings:

- the decisive method ingredient is not collaboration alone
- it is ambiguity calibration plus restricted local repair

Decision:

- write the final proposal around ambiguity-calibrated local collaborative repair on a fixed tokenizer
