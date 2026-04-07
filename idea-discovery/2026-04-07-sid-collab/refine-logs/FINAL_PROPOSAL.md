# FINAL PROPOSAL

Date: 2026-04-07

## Title

ACLR-Plus: Ambiguity-Calibrated Local Collaborative Repair for Generative Recommendation

## Problem Anchor

In MiniOneRec-style generative recommendation, the model often reaches a semantically plausible coarse SID subtree but fails to identify the correct leaf within that subtree. This failure is not best explained by raw full-SID collision alone. Instead, it is driven by local ambiguity: the semantic tokenizer is good enough to route into the right neighborhood, but not strong enough to resolve fine-grained item distinctions that are behaviorally important.

## Method Thesis

Keep the existing SID/tokenizer fixed. Detect when the current prefix path is locally unreliable, and inject collaborative evidence only within the risky local subtree to repair the final ranking.

## Why this is the right angle

### Supported by repo evidence

- collision exists but is relatively small
- same-prefix confusion is much larger
- collaborative reranking helps immediately
- naive ambiguity triggering is too weak, so calibration is the real missing ingredient

### Better aligned with the latest literature

Recent papers already cover:

- collaborative tokenizer redesign: ReSID, PIT, PRISM, UNGER
- prefix-aware optimization: APAO, TrieRec
- collision-aware SID learning: QuaSID, HiD-VAE

The cleaner gap is therefore:

- fixed-tokenizer
- diagnosis-driven
- ambiguity-triggered
- local collaborative repair

## Proposed Method

### 1. Ambiguity calibration module

For each prediction instance, compute a local risk score using train-only statistics and decoder signals such as:

- local subtree fan-out
- prefix-conditional uncertainty
- beam score dispersion
- same-prefix candidate density
- collaborative separation between candidate leaves

Output:

- a calibrated ambiguity score for the current prefix or candidate set

### 2. Restricted local candidate set

When ambiguity is low:

- keep the original ranking

When ambiguity is high:

- rerank only candidates inside a local subtree or local prefix-consistent set

This preserves the semantic routing effect of the tokenizer while avoiding unnecessary global reranking.

### 3. Collaborative repair score

Use train-only collaborative evidence derived from user histories to score the local candidate leaves. The collaborative score should be lightweight and reproducible, for example:

- recency-weighted item co-occurrence
- short-history transition preference
- optional normalized pairwise statistics

### 4. Optional consistency training

If the pure inference-time version looks promising, add a lightweight training objective that teaches the model to expose or preserve risky prefixes instead of pruning them too early.

This is optional for the first phase and should only be added if the simpler version already validates the core thesis.

## Key claim

A fixed semantic tokenizer is already sufficient for coarse routing in MiniOneRec, but not for fine-grained local disambiguation. Ambiguity-calibrated local collaborative repair improves recommendation quality by intervening only where semantic SID prefixes become unreliable.

## What not to claim

Do not claim:

- universal superiority over all tokenizer redesigns
- full resolution of SID collision
- a new tokenizer learning framework

The paper should stay narrow:

- local ambiguity
- selective repair
- fixed tokenizer

## Success conditions

The method is successful if it shows:

1. consistent gains over current MiniOneRec
2. gains over simple local rerank baselines
3. clear improvements concentrated on high-risk local ambiguity cases
4. limited or no damage on easy low-risk cases
