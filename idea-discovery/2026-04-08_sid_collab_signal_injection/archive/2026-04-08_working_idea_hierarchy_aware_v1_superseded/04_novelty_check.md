# Novelty Check

## Top idea under check

`MRC-SID`: Multi-Resolution Collaborative Allocation for Hierarchical Semantic IDs

## Closest prior art

### `PRISM`

- closest on denoising and hierarchical stabilization
- but still centered on purified collaborative tokenization at the global tokenizer level

### `ReSID`

- closest on prefix predictability and recommendation-native tokenization
- but does not explicitly allocate different collaborative views to different SID levels

### `PIT`

- closest on dynamic tokenization and collaborative volatility
- but its main axis is co-evolution/personalization, not hierarchy-level multi-resolution allocation

### `DiscRec`

- closest on semantic-collaborative disentanglement and gated usage
- but not explicitly tied to 3-level SID token allocation

### `Align3GR`

- closest on multi-level alignment language
- but its levels are not the internal SID levels of a hierarchical tokenizer

## What would make our idea non-novel

`MRC-SID` would lose novelty if it were implemented as:

- just concatenate three collaborative embeddings and let the network sort it out
- or just say “upper layers use coarse, lower layers use local” with no learned allocation and no empirical validation

That would look like a weak variant of existing fusion papers.

## What keeps it differentiated

It remains differentiated if all three points hold:

### 1. The central unit is the SID level

The main claim must be:

- collaborative signal utility is hierarchy-dependent

not merely:

- more collaborative features help

### 2. The method uses multiple collaborative resolutions, not one

The method must explicitly distinguish:

- coarse
- mid
- local

and treat them as different resources with different roles.

### 3. The paper validates non-uniform allocation

The experiments must explicitly show:

- uniform all-level fusion is weaker
- swapped or mismatched assignments are weaker
- learned per-level allocation is meaningful and non-trivial

## Novelty judgment

### Judgment

**Novel enough if implemented carefully.**

### Why

The field already contains:

- front-end collaborative tokenizers
- denoising collaborative tokenizers
- dynamic or personalized tokenizers
- gated semantic/collaborative modeling

But the specific thesis:

> different SID levels should consume different purified collaborative resolutions, and this should be learned and validated as a tokenizer principle

still looks sufficiently differentiated.

### Main novelty risk

If the final paper drifts into a large bundle of:

- denoising
- curriculum
- gating
- ambiguity
- multi-view fusion

then reviewers may say the work is broad but blurry.

So the safest novelty framing is:

> **resolution-matched collaborative allocation across SID levels**

