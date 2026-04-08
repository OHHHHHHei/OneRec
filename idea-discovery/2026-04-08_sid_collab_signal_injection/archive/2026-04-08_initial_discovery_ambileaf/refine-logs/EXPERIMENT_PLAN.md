# Experiment Plan

## Goal

Validate whether **AmbiLeaf** can improve recommendation quality by repairing only ambiguous leaf regions while preserving the global semantic prefix structure.

## Main evaluation principle

Every experiment must answer one of these three questions:

1. Does local leaf retokenization actually improve recommendation?
2. Is the gain coming from real collaborative signal rather than arbitrary local perturbation?
3. Does the method preserve tokenizer stability better than global front-end fusion?

## Datasets

- Industrial
- Office

Use the same repo pipeline and evaluation protocol as the current baseline whenever possible.

## Block A: Prefix Ambiguity Mining

### E1. Build train-time ambiguity statistics

- within-prefix item count
- within-prefix text similarity
- within-prefix collaborative disagreement
- optional validation same-prefix miss proxy

**Output**

- ranked prefix list
- coverage statistics
- ambiguity thresholds

### E2. Sanity-check ambiguity concentration

Measure:

- what fraction of errors land in top-`M` ambiguous prefixes
- how much catalog coverage they represent

**Pass condition**

- ambiguous prefixes cover a meaningful portion of same-prefix errors

## Block B: Minimal AmbiLeaf tokenizer construction

### E3. Semantic-only local retokenization control

Within selected ambiguous prefixes:

- keep prefix fixed
- relearn only the last token using semantic residual features

Purpose:

- isolate the effect of "local retokenization" from "collaborative retokenization"

### E4. AmbiLeaf with purified collaborative residuals

Within selected ambiguous prefixes:

- keep prefix fixed
- relearn only the last token using semantic residual + purified CF residual

### E5. Shuffled-CF local control

Same as E4, but replace CF residuals with shuffled CF residuals.

Purpose:

- show the gain is not due to arbitrary local code perturbation

## Block C: Downstream recommendation evaluation

### E6. Train/evaluate Industrial

Compare:

- baseline semantic tokenizer
- naive global `text + cf` tokenizer if still runnable as a failed reference
- semantic-only local retokenization
- AmbiLeaf
- shuffled-CF AmbiLeaf
- ACLR-lite `global`
- ACLR-lite `same_l2`
- ACLR-lite `ambiguity_l2`

### E7. Train/evaluate Office

Same comparison as Industrial.

## Block D: Analysis metrics

### Recommendation metrics

- HR@1/3/5/10/20/50
- NDCG@1/3/5/10/20/50

### Tokenizer diagnostics

- full-SID collision rate
- code usage distribution
- selected-prefix leaf entropy
- prefix preservation rate

### Error localization metrics

- top1 error same-`l1` rate
- top1 error same-`l2` rate
- improvement inside selected ambiguous prefixes
- improvement outside selected ambiguous prefixes

## Block E: Ablations

### E8. How many prefixes to retokenize?

- top-8
- top-16
- top-32
- top-64

Question:

- is the benefit concentrated in a small set of hot ambiguous subtrees?

### E9. Which collaborative purification works best?

- raw CF residual
- low-rank purified CF residual
- confidence-weighted CF residual

### E10. Replace leaf token vs append micro-leaf token

- replace last token
- append one extra local token

Question:

- does the method need a longer SID, or is better leaf partitioning enough?

### E11. Leaf-only vs deeper retokenization

- leaf-only
- last two levels

Question:

- is the benefit really leaf-local?

## Minimal success criteria

AmbiLeaf is worth continuing only if it satisfies all of the following:

1. beats the baseline on Industrial and at least one of the two datasets clearly
2. beats semantic-only local retokenization
3. beats shuffled-CF local retokenization
4. does not trigger global collision explosion
5. shows stronger improvement inside ambiguous prefixes than outside them

## Failure interpretation

### If AmbiLeaf fails but tokenizer stays stable

Interpretation:

- collaboration may help mostly at inference time, not at tokenizer construction time

Fallback:

- pivot to `Coarse2Fine Dual Signal`

### If AmbiLeaf improves only when too many prefixes are edited

Interpretation:

- the bottleneck may be broader than initially believed

Fallback:

- revisit `PurifyThenQuantize`

### If shuffled-CF performs similarly to real-CF

Interpretation:

- local retokenization itself may be acting as regularization, not collaborative repair

Fallback:

- weaken the collaborative claim and reposition as local structural refinement

