# Final Proposal

## Title

**MRC-SID: Multi-Resolution Collaborative Allocation for Hierarchical Semantic IDs**

## Problem Anchor

Current SID-based generative recommenders increasingly inject collaborative information into the tokenizer. However, existing designs still tend to treat collaborative information as a single resource to be fused globally. This is problematic for two reasons:

1. different parts of the SID hierarchy play different roles
2. different collaborative signals carry different utility and different noise

In this repo, the strongest remaining error pattern is local leaf ambiguity under semantically correct prefixes, while naive global front-end fusion causes severe structural collapse. Therefore, the real problem is not simply whether collaborative information should enter tokenization, but:

> how collaborative signals of different resolutions should be allocated across different SID levels.

## Method Thesis

A hierarchical SID tokenizer should not consume one uniform collaborative representation. Instead, it should learn **level-wise allocation over purified collaborative views with different resolutions**.

## Method Overview

### 1. Multi-Resolution Collaborative View Bank

Construct three collaborative views from train-only interaction data:

- `coarse view`
  - broad, stable, long-window collaborative compatibility
- `mid view`
  - medium-range collaborative structure
- `local view`
  - short-range recent transition signal

These views are intentionally different in resolution and expected noise profile.

### 2. View-Specific Purification

Each collaborative view is denoised separately.

- coarse view:
  - debiasing
  - smoothing
  - optional low-rank compression
- mid view:
  - structure-aware filtering
  - optional community or neighborhood normalization
- local view:
  - confidence thresholding
  - recency-aware smoothing
  - support pruning

The purpose is to avoid the failure mode of naive global fusion.

### 3. Level-Wise Collaborative Allocation

For a 3-level SID, each level learns its own mixture over:

- semantic representation
- purified coarse view
- purified mid view
- purified local view

The allocation is learned, not manually fixed.

This means the method can discover:

- whether upper levels prefer more stable views
- whether lower levels prefer more local views
- whether the allocation is actually non-uniform

### 4. Hierarchy-Aware Tokenization

The collaborative mixture is injected into the hierarchical tokenizer so that:

- Level 1 receives a learned mixture suitable for coarse routing
- Level 2 receives a learned mixture suitable for subgroup organization
- Level 3 receives a learned mixture suitable for leaf discrimination

The central claim is not a fixed rule like “Level 1 must be coarse, Level 3 must be local.”  
The claim is stronger and cleaner:

> the allocation should be learned separately for each level because uniform fusion is structurally mismatched.

## Why This Proposal Is Different

`MRC-SID` is different from:

### A. Generic front-end collaborative tokenizers

Those methods mainly ask:

- how to inject collaboration into tokenization

This method asks:

- how collaboration should be **distributed across SID levels**

### B. Post-hoc local repair

This proposal is a tokenizer-time design principle, not a post-processing refinement.

### C. Personalized/dynamic tokenization

This proposal does not rely on making the entire tokenization process user-specific or dynamically mutable. It remains a clean item-tokenizer design, but with structured collaborative allocation.

## Falsifiable Claims

1. `MRC-SID` beats semantic-only tokenization.
2. `MRC-SID` beats uniform global fusion with the same collaborative view bank.
3. `MRC-SID` beats a fixed hand-designed coarse/mid/local level assignment.
4. The learned allocation is non-uniform across SID levels.
5. Deeper levels benefit more from non-global views than upper levels.

## Minimal version

The first serious implementation should stay simple:

- exactly three views
- one purification step per view
- one learned gate per SID level
- no ambiguity-aware dynamic gating in the core version

This keeps the method sharp and experimentally falsifiable.

## Optional extension after the core result works

An ambiguity-aware leaf boost can be added later as an extension, but it should not be part of the first paper claim.

## Why It Fits This Repo

This proposal matches all major observations already established here:

- collision exists but is not the central issue
- text-only tokenization is insufficient
- local leaf ambiguity is real
- naive global collaborative fusion collapses the tokenizer
- different collaborative granularities show different utility in the lightweight pilot

## Success condition

`MRC-SID` should become the active mainline only if experiments show:

- clear improvement over uniform fusion
- stable tokenizer behavior
- interpretable non-uniform allocation across levels

If these do not hold, the idea should be downgraded to a useful analysis insight rather than a full paper method.

