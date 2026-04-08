# Final Proposal

## Title

**AmbiLeaf: Prefix-Preserved Collaborative Leaf Retokenization for Generative Recommendation**

## Problem Anchor

The current MiniOneRec-style pipeline is text-first in SID construction. In this repo, that design does not mainly fail through widespread full-SID collision. Instead, the dominant residual failure is:

- semantic routing is often approximately correct
- but leaf-level item discrimination inside the routed subtree remains weak

At the same time, naive front-end collaborative fusion is unstable and can collapse the tokenizer. Therefore, the real challenge is not whether collaboration should be added in general, but:

> how to inject collaborative information into SID construction without destabilizing the semantic prefix structure.

## Method Thesis

Collaborative information should be used to **repair ambiguous leaf regions**, not to globally overwrite the whole semantic SID tree.

## Proposed Method

### 1. Ambiguous Prefix Mining

Identify ambiguous prefixes using only train or validation statistics:

- large within-prefix population
- high within-prefix semantic similarity
- high collaborative disagreement among sibling leaves
- optional high same-prefix error proxy on validation

This defines a set of prefixes where the leaf mapping is likely underdetermined by text semantics alone.

### 2. Prefix Preservation

Keep the existing semantic prefix unchanged for all items. This preserves:

- global semantic organization
- compatibility with the current decoding logic
- low global collision behavior of the baseline tokenizer

### 3. Purified Collaborative Leaf Retokenization

For items inside ambiguous prefixes only:

- compute semantic residual features
- compute purified collaborative residual features
- relearn the final token using the fused residual representation

The key principle is:

- semantic information dominates routing
- collaborative information sharpens leaf separation

### 4. Optional Leaf Margin Objective

Add a local contrastive or margin-style objective that separates:

- target item leaf
- confusable sibling leaves under the same prefix

This is optional for the first version. The minimal version can work with local retokenization alone.

## Why This Is the Right Scope

This method is deliberately narrow:

- it is more structural than reranking
- less crowded than full global collaborative tokenizer redesign
- more faithful to the measured bottleneck than collision-first or fully global stories

## Expected Contributions

### Main contribution

A hierarchy-local tokenizer intervention showing that collaborative signal is most effective when used to repair ambiguous leaves under stable semantic prefixes.

### Supporting contribution

A concrete empirical argument that the bottleneck in this repo is not full collision or full tokenizer failure, but local leaf ambiguity under text-driven SID construction.

## Falsifiable Claims

1. AmbiLeaf improves HR/NDCG over the current semantic-only baseline.
2. AmbiLeaf reduces same-prefix miss rates more than semantic-only local retokenization.
3. AmbiLeaf avoids the global collision explosion of naive `text + cf` front-end fusion.
4. Shuffled-CF local retokenization underperforms purified real-CF local retokenization, showing that the gain is not from arbitrary structural perturbation.

## Why It May Win Over Pure ACLR-lite

ACLR-lite proves that collaborative information helps at inference time, but it does not fix the identifier structure itself.

AmbiLeaf aims to push that information one stage earlier:

- not into the whole tokenizer
- but into the exact leaf regions where the structure is currently under-resolved

## Main Risks

- ambiguous-prefix coverage may be too small
- train-time ambiguity mining may be noisy
- gains may overlap with what local reranking already gives

## Risk Mitigation

- evaluate coverage of selected prefixes
- compare directly against ACLR-lite
- include semantic-only and shuffled-CF local controls
- keep the first version extremely simple

## Decision Rule

AmbiLeaf should remain the active line only if it shows:

- stable tokenizer statistics
- meaningful reduction in same-prefix local errors
- gains beyond what can be explained by local extra capacity alone

If not, the fallback is the hybrid `Coarse2Fine Dual Signal` direction.

