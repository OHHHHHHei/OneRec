# Idea Candidates

## Candidate 1: `MGR-SID` — RECOMMENDED

**Full name**: `Multiplex Graph-Regularized Hierarchical Semantic IDs`

### Core hypothesis

Collaborative information should not only be fused as features.  
It should supervise SID learning through **level-specific graph structure preservation**.

### Core design

Build a graph bank with three purified views:

- `G_coarse`
  - debiased undirected collaborative graph
- `G_mid`
  - community-aware or band-pass graph view
- `G_local`
  - denoised directed transition graph

Then for each SID level:

- learn how much each graph view matters
- use that mixture to regularize quantization at that level

### Why it is promising

- graph structure is more paper-grade than manual windows
- it is more tokenizer-native than simple feature fusion
- it naturally connects graph denoising and hierarchy-aware collaborative allocation

### Main risk

- if implemented carelessly, it may collapse back into “multi-view fusion with extra losses”

### How to protect the story

- make graph regularization the central mechanism
- keep feature fusion secondary or optional
- compare directly against parameter-matched graph-fusion baselines

## Candidate 2: `HCP-SID`

**Full name**: `Hierarchical Collaborative Partitioning for Semantic IDs`

### Core hypothesis

Instead of learning SID levels from semantic quantization first and collaborative correction later, the SID hierarchy itself should be derived from a multiplex collaborative graph through recursive graph coarsening or partitioning.

### Why it is exciting

- strongest tokenizer-native novelty
- graph structure directly defines the hierarchy

### Main risk

- very high engineering and optimization risk
- easy to lose semantic interpretability
- likely too much for the first serious push

### Current judgment

Strong backup idea, but not the best first mainline.

## Candidate 3: `GRQ-SID`

**Full name**: `Graph Residual Quantization for SID`

### Core hypothesis

Keep the semantic tokenizer as the backbone, but inject graph-derived residuals at each quantization stage instead of using global collaborative fusion.

### Why it is attractive

- easier to implement than full graph regularization
- still more structured than plain fusion

### Main risk

- may still be perceived as a feature-level trick
- novelty weaker than `MGR-SID`

### Current judgment

A good fallback if `MGR-SID` proves too hard.

## Candidate 4: `TrustGraph-SID`

**Full name**: `Confidence-Routed Graph Denoising for SID`

### Core hypothesis

The main failure is not only which graph view to use, but whether each graph view should be trusted for a given item or region of the hierarchy.

### Why it matters

- directly addresses noisy collaborative signals
- can explain why local transition signals help only sparsely

### Main risk

- best used as a submodule, not a complete main paper story

### Current judgment

Useful extension or appendix module, not the main direction.

## Ranking

1. `MGR-SID`
2. `HCP-SID`
3. `GRQ-SID`
4. `TrustGraph-SID`

## Why `MGR-SID` wins

`MGR-SID` currently gives the best balance of:

- novelty
- tokenizer-native structure
- connection to graph denoising
- implementation feasibility inside this repo
