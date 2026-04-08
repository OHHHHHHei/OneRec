# Idea Discovery Report

**Direction**: collaborative signal injection for SID-based generative recommendation under local leaf ambiguity  
**Date**: 2026-04-08  
**Mode**: isolated discovery run in a new folder  
**Pipeline**: literature survey → idea generation → novelty check → critical review → top-idea refinement

## Executive Summary

This discovery round supports a more precise thesis than "just add collaboration to SID". The repo evidence says:

- full collision exists but is not the dominant bottleneck
- naive front-end global fusion collapses the tokenizer
- collaborative information is still useful, especially inside same-prefix local confusion

So the best question is not simply `global vs selective`. It is:

> at what stage and granularity should collaborative information enter the SID-generative pipeline, and in what purified form?

After comparing front-end, local, and hybrid options, the recommended idea is:

## Recommended idea

**AmbiLeaf: Prefix-Preserved Collaborative Leaf Retokenization**

Reason:

- it keeps front-end structural intervention
- it directly attacks the repo's measured bottleneck
- it avoids repeating the most crowded "global collaborative tokenizer" story
- it appears more differentiated than a pure global fusion redesign

## Literature Landscape

### What recent work says

- Front-end collaborative tokenization is a real and active direction:
  - ETEGRec
  - PIT
  - LCRec
  - Align3GR
  - UniGRec
- But recent work also agrees that tokenization can become unstable when collaborative signals are noisy or injected too aggressively:
  - PRISM
  - PIT
  - DIGER
  - ReSID
  - QuaSID

### What this repo adds that the papers do not fully answer

- The main current failure is not full SID collapse.
- The main current failure is local leaf ambiguity under semantically correct prefixes.
- A bad collaborative injection recipe can collapse the whole SID tree even when the raw collaborative signal itself is not the only problem.

That combination creates a fresh opening for hierarchy-aware collaborative injection.

## Ranked Ideas

### 1. AmbiLeaf — RECOMMENDED

- **Hypothesis**: keep semantic prefixes fixed, and relearn only ambiguous leaf regions with purified collaborative signals.
- **Why it matches the repo**:
  - directly addresses `prefix correct, leaf wrong`
  - explains why local collaboration helps without requiring global raw fusion
  - preserves prefix stability
- **Evidence already in hand**:
  - Industrial baseline collision: `0.00434`
  - same-`l2` target-better rate:
    - Industrial: `0.2284`
    - Office: `0.4048`
  - ACLR-lite local activation is more efficient per activated sample than global activation
- **Novelty**: promising; no close match found for prefix-preserved local leaf retokenization
- **Reviewer score**: `8.3/10`
- **Status**: push this first

### 2. PurifyThenQuantize — BACKUP HIGH-CEILING IDEA

- **Hypothesis**: global front-end collaborative fusion can still work if collaborative signals are purified and injected in a level-wise curriculum.
- **Why it matters**:
  - takes the front-end route seriously
  - aligns with PRISM / PIT / ReSID / UniGRec style lessons
- **Main issue**: crowded novelty space
- **Reviewer score**: `7.4/10`
- **Status**: strong backup or competitor baseline family

### 3. Coarse2Fine Dual Signal — PRACTICAL BACKUP

- **Hypothesis**: weak global collaborative prior + strong local leaf repair is the safest performance-oriented design.
- **Strength**: probably practical
- **Weakness**: more system-like, less clean as a paper core
- **Reviewer score**: `7.1/10`
- **Status**: fallback if AmbiLeaf underperforms

### 4. Leaf Alias Tokens — SPECULATIVE

- Interesting, but overlaps more with dynamic/personalized tokenization papers such as Pctx and PIT.

### 5. Prefix Stability Distillation — LOW PRIORITY

- Useful as a component or ablation, but weak as the whole paper story.

## Eliminated Ideas

- **Collision-first tokenizer redesign**
  - repo evidence says collision is not the dominant current bottleneck
- **Pure global collaborative rerank as the whole paper**
  - useful baseline, weak central claim
- **Naive global `text + cf` fusion**
  - already falsified by repo evidence

## Refined Proposal

- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`

## Recommended Next Step

Move forward with a minimal AmbiLeaf implementation:

1. mine ambiguous prefixes from train-only signals  
2. freeze baseline prefixes  
3. retokenize only the leaf level inside ambiguous prefixes  
4. compare against baseline, naive front-end fusion, and ACLR-lite  

If AmbiLeaf shows clear gain, then decide whether to stay local-structural or extend toward a hybrid coarse-to-fine paper.

