# Final Proposal

## Title

**MGR-SID: Multiplex Graph-Regularized Hierarchical Semantic IDs**

## Problem Anchor

Current SID-based generative recommenders increasingly recognize that collaborative information should enter item tokenization. However, existing designs still mainly treat collaboration as a single resource to be globally fused, globally aligned, or globally denoised once and then reused everywhere.

That framing is too coarse for hierarchical semantic IDs.

In this repo, the evidence already shows:

1. collision exists but is not the dominant bottleneck
2. the main remaining error mode is local leaf ambiguity under semantically correct prefixes
3. naive front-end collaborative fusion can severely collapse the SID structure

Therefore, the real problem is:

> how should different graph-structured collaborative signals be denoised and allocated across different SID levels?

## Method Thesis

A hierarchical SID tokenizer should not treat collaborative information as one global embedding.  
It should learn item codes under **level-specific supervision from a denoised multiplex graph bank**.

## Core Hypothesis

Different SID levels should preserve different collaborative graph structures:

- upper levels may rely more on broad and stable collaborative organization
- middle levels may rely more on community or band-pass relational structure
- lower levels may rely more on local transition-sensitive distinctions

This should be learned, not hard-coded, but the structure itself must be explicit.

## Method Overview

### 1. Multiplex Graph Bank

Construct three train-only item graphs:

- `G_coarse`
  - debiased undirected collaborative graph
  - captures stable broad compatibility
- `G_mid`
  - community-aware or band-pass graph view
  - captures medium-scale substructure that is neither purely global nor purely local
- `G_local`
  - denoised directed transition graph
  - captures short-range next-item preference

These three graphs are not interchangeable.  
They encode different collaborative resolutions and different noise types.

### 2. View-Specific Graph Purification

Each graph view is denoised differently:

- `G_coarse`
  - support thresholding
  - popularity debiasing
  - low-pass smoothing or low-rank cleanup
- `G_mid`
  - community normalization, spectral residual extraction, or diffusion residual cleanup
- `G_local`
  - recency weighting
  - support pruning
  - confidence filtering for unstable edges

This is necessary because the three graph views fail differently.

### 3. Level-Wise Graph Allocation

For each SID level `l`, learn a mixture over:

- semantic anchor representation
- `G_coarse`
- `G_mid`
- `G_local`

The point is not to enforce a rigid rule like:

- Level 1 = coarse
- Level 2 = mid
- Level 3 = local

The point is:

> the best graph allocation should be learned separately for each level because each level solves a different structural discrimination problem.

### 4. Graph-Regularized Quantization

This is the central step.

At each SID level, the quantizer is trained not only to preserve semantic structure, but also to preserve the relevant graph structure selected by the level-wise allocation.

In practical terms, this means:

- items connected in the graph mixture for a level should be easier to group consistently at that level
- items that should remain separable at deeper levels should not be collapsed too early

This turns graph structure into a training signal for SID learning itself, not merely an auxiliary embedding source.

### 5. Semantic Anti-Collapse Anchor

To avoid the known collapse risk from noisy collaborative signals, keep a strong semantic anchor:

- reconstruction or semantic consistency
- code usage health checks
- optional prefix-stability constraint

This preserves the strengths of semantic SID while allowing graph structure to repair its blind spots.

## Why This Is Stronger Than Simple Fusion

`MGR-SID` differs from ordinary graph-enhanced tokenization in three ways:

### A. Graph views are explicit

The method does not assume one generic collaborative embedding.

### B. Denoising is view-specific

It does not assume one universal graph cleanup step.

### C. Graph structure supervises quantization

The main contribution is not just better item features.  
It is better SID learning under level-specific graph constraints.

## Closest Alternatives and Why This One Wins

### Plain level-wise graph fusion

Too easy to dismiss as gated multi-view fusion.

### Full graph-defined hierarchical partitioning

More radical, but much riskier and harder to stabilize.

### Semantic backbone plus graph residual features

Feasible, but weaker novelty.

`MGR-SID` is the best middle path:

- stronger than feature fusion
- cleaner than post-hoc repair
- more feasible than replacing the whole tokenizer with graph partitioning

## Falsifiable Claims

1. `MGR-SID` beats semantic-only SID tokenization.
2. `MGR-SID` beats parameter-matched graph feature fusion.
3. `MGR-SID` beats uniform graph regularization across all SID levels.
4. Learned level-wise graph allocation is non-uniform.
5. Different graph views contribute differently to different ambiguity buckets.

## Minimal Serious Version

The first implementation should stay narrow:

- one coarse graph
- one chosen mid graph
- one local graph
- one level-wise allocation module
- one graph-regularized quantizer

No dynamic personalized graph routing in the core version.

## Key Open Design Choice

The main unresolved design decision is the construction of `G_mid`.

The most important cheap pre-training experiment is:

- compare candidate `mid-scale` graph views
- choose one that best supports same-prefix ambiguity without destabilizing the tokenizer

## Success Condition

`MGR-SID` should become the active paper mainline only if:

- it beats graph feature fusion and uniform graph regularization
- its learned level allocation is clearly non-uniform
- tokenizer health remains acceptable

If those do not hold, the idea should be downgraded to an analysis insight or simplified fallback.
