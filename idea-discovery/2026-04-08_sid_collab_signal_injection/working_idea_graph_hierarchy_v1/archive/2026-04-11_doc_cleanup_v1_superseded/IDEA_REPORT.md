# Idea Discovery Report

**Direction**: graph-structured, denoised, hierarchy-aware collaborative supervision for SID-based generative recommendation  
**Date**: 2026-04-09  
**Pipeline**: literature survey → graph pilot → idea generation → novelty check → review round 1 → revision → review round 2 → proposal refinement

## Executive Summary

This round pushes the previous hierarchy-aware fusion idea into a stronger and more paper-grade direction.

The key conclusion is:

> if collaborative information should enter hierarchical SID learning, it should likely enter as **level-specific graph structure supervision**, not merely as a globally fused collaborative embedding.

The recommended idea is:

## Recommended idea

**`MGR-SID`: Multiplex Graph-Regularized Hierarchical Semantic IDs**

Its central claim is:

- build a denoised multiplex graph bank
- let different SID levels learn different graph mixtures
- use those mixtures to regularize hierarchical quantization itself

## Why this round was necessary

The earlier `MRC-SID` idea was already better than naive global fusion, but it still had a weakness:

- it could still be read as “multi-view fusion with level-wise gates”

This round strengthens the story by moving graph structure into the tokenizer learning objective.

## Repo-grounded motivation

Current repo evidence still points to the same underlying bottleneck:

- SID collision exists but is not the main contradiction
- text-driven SID lacks direct collaborative structure
- many errors already have the right semantic prefix but the wrong leaf
- naive global front-end fusion can severely collapse the SID structure

So the real design question is no longer:

- should collaboration enter the tokenizer?

It is:

> what graph-structured collaborative information should each SID level preserve, and how should each graph view be denoised?

## Graph Pilot Summary

Using simple train-only graph proxies:

- `coarse_graph` is the strongest robust global signal
- `local_trans` is sparse but can be very strong in the deepest ambiguity bucket
- naive `mid_graph` as plain 2-hop diffusion is not good enough

This supports the idea that:

- collaborative structure is multi-scale
- the scales are better understood as graph views than as fixed time windows

## Ranked Ideas

### 1. `MGR-SID` — RECOMMENDED

- **Hypothesis**: hierarchical SID learning should preserve different denoised graph structures at different levels
- **Why it is strong**: graph-native, tokenizer-native, and tightly aligned with the repo's failure mode
- **Novelty verdict**: strong enough if implemented as graph-regularized quantization, not just graph feature fusion
- **Reviewer score after two rounds**: `8.7/10`
- **Status**: active new mainline candidate

### 2. `HCP-SID` — HIGH-NOVELTY BACKUP

- graph coarsening or partitioning directly defines the SID hierarchy
- exciting but significantly riskier

### 3. `GRQ-SID` — FEASIBLE FALLBACK

- semantic tokenizer plus graph residuals per level
- easier to implement, but weaker novelty

### 4. `TrustGraph-SID` — SUBMODULE

- confidence-aware graph denoising and routing
- useful, but should not be the main paper story

## What changed after review

### Round 1

Main criticism:

- the first graph idea still sounded too much like multi-view fusion

Action:

- promote graph structure preservation to the center of the method
- define the method as graph-regularized quantization

### Round 2

Main criticism:

- gains could come from generic graph priors rather than hierarchy-aware design

Action:

- require uniform graph regularization and graph-fusion baselines
- require swapped allocation and no-denoising controls

## Final judgment

The graph-centric direction is promising enough to continue.  
Among the candidate ideas, `MGR-SID` currently offers the best balance of:

- novelty
- elegance
- alignment with the repo's empirical motivation
- feasibility for a real implementation path

## Next step

Read:

- `refine-logs/FINAL_PROPOSAL.md`
- `refine-logs/EXPERIMENT_PLAN.md`

The immediate practical goal is not full-scale training.  
It is to first settle:

- the best `mid-scale` graph view
- whether graph regularization beats graph feature fusion
- whether level-aware allocation really matters
