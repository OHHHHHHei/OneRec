# EXPERIMENT PLAN

Date: 2026-04-07

## Objective

Validate whether ambiguity-calibrated local collaborative repair is a real method direction or only a weak heuristic.

## Datasets

### Phase 1

- Industrial only

Reason:

- stronger local pilot gains
- cleaner place to test the thesis first

### Phase 2

- Office for transfer / robustness check

## Core baselines

1. current MiniOneRec baseline
2. `global` collaborative rerank
3. `same_l1` rerank
4. `same_l2` rerank
5. current `ambiguity_l2` heuristic

These baselines are mandatory. Without them, the calibrated method cannot be evaluated fairly.

## Proposed experiment blocks

### E0. Freeze current evidence

Goal:

- lock in the current diagnostic profile and pilot numbers

Outputs:

- baseline metrics
- same-prefix error profile
- beam-local overlap profile

### E1. Better ambiguity scoring without changing the model

Goal:

- replace the current crude `leaf_count >= threshold` trigger

Candidate features:

- leaf count under `(l1, l2)`
- prefix conditional entropy
- beam margin or top-k score flatness
- local beam density under same prefix
- collaborative gap statistics from train data

Variants:

- simple linear risk score
- rule-based calibrated score
- logistic classifier trained on train or validation ambiguity labels

Success criterion:

- beats current `ambiguity_l2`
- at least matches or exceeds `same_l2` on Industrial

### E2. Restricted local candidate refinement

Goal:

- test multiple local candidate set definitions

Candidate sets:

- same predicted `l1`
- same predicted `l2`
- union of top risky beam-local branches
- dynamic local set selected by ambiguity score

Success criterion:

- calibrated local repair outperforms the best fixed local subset baseline

### E3. Score design ablation

Goal:

- understand which collaborative score is actually needed

Variants:

- raw recency-weighted co-occurrence
- normalized transition score
- best-item score over collided SID items
- average-item score over collided SID items

Success criterion:

- identify the minimal effective score

### E4. Optional lightweight training consistency

Only run this if E1-E3 are positive.

Goal:

- make the method more paper-like and less post-hoc

Possible direction:

- auxiliary loss that teaches the decoder not to prune risky local branches too early

Stop rule:

- skip this block if the inference-only version is already weak or unstable

## Metrics

### Primary metrics

- HR@10
- NDCG@10
- top1 hit rate on saved beam lists

### Diagnostic metrics

- same_l1 error rate
- same_l2 error rate
- activation precision of the ambiguity trigger
- gain on risky subset
- damage on easy subset
- fraction of examples touched by repair

## Pre-registered decision rules

### Proceed to Office only if

- E1 or E2 beats current `same_l2` on Industrial
- and does not rely on near-global activation

### Proceed to training consistency only if

- calibrated local repair shows stable gains over both `same_l2` and current `ambiguity_l2`

### Stop this direction if

- the calibrated method cannot beat `same_l2`
- or gains only appear when activation becomes effectively global
- or diagnostic gains do not transfer to HR/NDCG

## First concrete runs

1. Build a new ambiguity score on top of existing saved predictions for Industrial.
2. Compare against `global`, `same_l1`, `same_l2`, and current `ambiguity_l2`.
3. If positive, port the best ambiguity trigger into the evaluation pipeline config.
4. Repeat on Office.

## Compute posture

This plan intentionally starts with cheap evaluation-side experiments. It should not launch expensive tokenizer retraining unless the local-repair thesis is already supported.
