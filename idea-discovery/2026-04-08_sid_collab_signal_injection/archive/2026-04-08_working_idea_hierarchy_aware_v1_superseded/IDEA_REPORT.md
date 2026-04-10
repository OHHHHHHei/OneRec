# Idea Discovery Report

**Direction**: hierarchy-aware collaborative signal fusion for SID-based generative recommendation  
**Date**: 2026-04-08  
**Pipeline**: literature survey → local pilot → idea generation → novelty check → review round 1 → review round 2 → proposal refinement → experiment planning

## Executive Summary

This round validates a stronger and cleaner direction than simple global collaborative fusion. The literature says front-end collaborative tokenization is important, but our repo evidence and the new lightweight pilot show that different collaborative granularities behave differently across error buckets:

- `coarse` views are strongest globally
- `mid` views are strongest in deep same-prefix ambiguity
- ultra-local signals are too sparse to use everywhere

This supports a new main idea:

## Recommended idea

**`MRC-SID`: Multi-Resolution Collaborative Allocation for Hierarchical Semantic IDs**

Instead of fusing one collaborative representation into the whole tokenizer, `MRC-SID` constructs multiple purified collaborative views and learns how much each SID level should use from each view.

## Literature Landscape

### What prior work already established

- front-end collaborative tokenization is valid and active
- denoising is necessary to prevent collapse
- semantic/collaborative disentanglement is useful
- different collaborative scales matter in recommendation more broadly

### What remains underexplored

The literature still does not cleanly answer:

- whether different SID levels should consume different collaborative resolutions
- how that allocation should be learned and validated

## Local Pilot

Using the existing repo data and result files, we built a train-only compatibility probe with four collaborative views.

### Key findings

- `coarse10` is strongest on the overall error pool
- `mid3` is strongest on `same_l2` in both Industrial and Office
- `fine1` is the sparsest and weakest view

This is exactly the kind of pattern that motivates hierarchy-aware allocation.

## Ranked Ideas

### 1. `MRC-SID` — RECOMMENDED

- **Hypothesis**: different SID levels should learn different mixtures of purified collaborative views
- **Pilot signal**: positive and directly supportive
- **Novelty**: good, if kept centered on SID-level allocation
- **Reviewer score after iteration**: `8.5/10`
- **Status**: active mainline

### 2. `AAG-SID` — BACKUP

- ambiguity-aware extension of `MRC-SID`
- promising but too complex for the first core method

### 3. `Prog-MRC` — SUPPORTING VARIANT

- progressive training for stability
- likely useful as an implementation option or ablation, but weaker as the main claim

## What was eliminated

- post-hoc local refinement as the main story
- another uniform global collaborative tokenizer
- a fully dynamic/personalized tokenizer as the first step

## Review Iterations

### Round 1

Main criticism:

- the initial coarse/mid/local-to-level mapping was too heuristic

Action:

- replace fixed mapping with learned per-level allocation
- simplify to three views + purification + level-wise gates

### Round 2

Main criticism:

- gains could come from having more views, not from hierarchy-aware allocation

Action:

- add parameter-matched uniform baseline
- add swap controls
- make learned gate analysis a required result

## Final Proposal

See:

- `refine-logs/FINAL_PROPOSAL.md`

## Experiment Plan

See:

- `refine-logs/EXPERIMENT_PLAN.md`

## Recommended Next Step

The next concrete step is **not** to build the full final model immediately. It is to first implement the minimum viable comparison set:

1. uniform all-view fusion  
2. fixed level assignment  
3. `MRC-SID` learned level allocation  

If `MRC-SID` clearly beats both on Industrial, the idea is strong enough to continue toward full implementation and paper positioning.

