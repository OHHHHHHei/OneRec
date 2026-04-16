# Research Direction

## Title

**Hierarchy-Aware Collaborative Signal Fusion for SID-based Generative Recommendation**

## Motivation

Semantic ID (SID) based generative recommendation has become a strong and increasingly active direction. A common pipeline is to first construct a hierarchical SID tree from item representations, and then train a generative recommender to autoregressively predict the target SID. In recent work, many methods have started to inject collaborative information into the tokenizer itself, rather than relying on text semantics alone.

This trend is reasonable, but our current evidence suggests that the problem is not simply whether collaborative information should be used. The more important question is **how collaborative information should enter a hierarchical SID structure**.

Our current observations point to four key facts:

1. Full SID collision exists, but it is not the dominant bottleneck.
2. The current SID construction process is still mainly text-driven, and lacks direct behavioral guidance.
3. Many generation errors have the form of `prefix correct, leaf wrong`, suggesting that the major remaining difficulty lies in **local leaf ambiguity** rather than global routing failure.
4. Naive front-end global fusion of collaborative features can easily destabilize the tokenizer, causing severe collision explosion and structural collapse.

Taken together, these observations suggest a tension:

- collaborative information is clearly useful
- but simply injecting it globally and uniformly is not enough, and may even be harmful

There is also a second tension that is now becoming increasingly central:

- some items are semantically very close
- but their collaborative roles are clearly different
- current attraction-only graph supervision can say which items should move closer
- but it still does not explicitly model which semantically similar items should be pulled apart

This matters because many real recommendation failures are not caused by global routing collapse, but by keeping the wrong semantically similar items inside the same local SID neighborhood.

This motivates a more structured view of collaborative integration.

## Core Hypothesis

In a hierarchical SID tree, different levels play different roles. Therefore, they should not receive the same collaborative signal in the same way.

Instead of treating collaborative information as a single feature to be globally fused into the whole tokenizer, we propose the following research direction:

> **Collaborative information should be fused into the SID hierarchy in a level-aware manner, where different SID levels receive different granularities and different forms of collaborative signals.**

This means the central problem is no longer:

- should we fuse collaborative information or not?

but rather:

- what kind of collaborative signal should be injected at each SID level?
- how should these signals be denoised before fusion?
- how should semantic and collaborative information be balanced across the hierarchy?
- how should we explicitly separate semantically similar but collaboratively inconsistent items?

## Intuition Under a 3-Level SID

Under a 3-level SID structure, the three levels do not need the same collaborative view.

### Level 1: Coarse Collaborative Structure

The first level is responsible for coarse routing. At this stage, the model should not rely on noisy, highly local behavioral patterns. Instead, it should use a **stable and coarse collaborative view**, such as long-term co-occurrence trends, global graph structure, or denoised group-level behavioral similarity.

The goal of this level is not fine discrimination, but ensuring that the top-level partition is not purely semantic and remains behavior-aware at a broad scale.

### Level 2: Mid-Level Collaborative Organization

The second level is responsible for forming meaningful sub-groups inside a coarse semantic region. This level may benefit from a **mid-granularity collaborative view**, such as community-level co-occurrence, meso-scale graph neighborhoods, or medium-range behavioral affinity.

At this stage, collaborative information can help reorganize items that are semantically related but behaviorally separable.

### Level 3: Fine-Grained Local Disambiguation

The final level is responsible for distinguishing between highly similar candidate items. This is where local ambiguity is most severe, and where the evidence in our current repo is strongest. Therefore, the last level should focus on **fine-grained collaborative information**, such as local transitions, recent co-click patterns, or short-range sequence-level behavioral cues.

The goal of this level is to resolve leaf ambiguity among items that are already close in semantic space.

This also suggests that the final level should not only receive local attraction signals. It may also need a way to explicitly distinguish items that are semantically close yet behaviorally separable.

## Why This Direction Is Different

This direction differs from existing front-end collaborative fusion methods in an important way.

Most current methods implicitly assume that collaborative information should be fused into tokenization in a more or less uniform manner. Even when they include denoising, alignment, or end-to-end coupling, the collaborative signal is still usually treated as a single resource to be injected globally.

Our direction instead argues that:

1. **Collaborative utility is hierarchy-dependent.**
2. **Collaborative noise is also hierarchy-dependent.**
3. **Collaborative separation is hierarchy-dependent as well.**
4. Therefore, the correct unit of design is not only the item representation, but also the **SID level** itself.

In other words, we are not merely asking how to build a collaborative-aware tokenizer. We are asking how to build a tokenizer whose collaborative awareness is **structured across levels**.

## Role of Denoising

Denoising is likely to be a necessary part of this direction.

Global collaborative signals can contain popularity bias, over-smoothing, and irrelevant long-range associations. Very local collaborative signals can be sparse, volatile, and overly sensitive to accidental transitions. These are different kinds of noise, and they may need different treatments before entering different SID levels.

This suggests that denoising should not be viewed as a single preprocessing trick. Instead, it may need to be designed together with the hierarchy:

- coarse signals may require stronger smoothing and debiasing
- mid-level signals may require structure-aware filtering
- fine-grained signals may require confidence control or recency-aware selection

Therefore, a promising direction is not just level-aware fusion, but **level-aware fusion with level-aware collaborative purification**.

And beyond purification, a complete solution may also require **level-aware collaborative separation**, especially for semantically crowded local neighborhoods.

## Proposed Research Question

Based on the above, the research direction can be summarized as:

> Can we improve SID-based generative recommendation by injecting collaborative information into different SID levels using different granularities and different denoising strategies, instead of using uniform global fusion?

An even sharper version of the same question is:

> Can we build a hierarchy-aware SID tokenizer that uses collaborative information not only to attract truly related items, but also to separate semantically similar yet collaboratively inconsistent items?

This question is closely aligned with our current motivation:

- it explains why text-driven SID is not fully sufficient
- it explains why local leaf ambiguity remains severe
- it also explains why naive global fusion may collapse the tokenizer

## Expected Value of This Direction

If this direction works, it could provide three meaningful contributions.

### 1. A better problem formulation

It reframes the problem from "whether collaboration should be used" to "how collaboration should be distributed across a hierarchical tokenizer".

### 2. A cleaner explanation of current failures

It explains why current systems can simultaneously show:

- useful collaborative signal at prediction time
- but poor results from naive front-end global fusion

because the issue is not the existence of collaborative information, but its allocation and purification.

It also explains why attraction-only collaborative supervision may still be insufficient: some of the hardest recommendation errors come from semantically plausible but behaviorally wrong alternatives that are never explicitly separated.

### 3. A more differentiated method space

It opens a direction that is different from both:

- purely text-driven tokenization
- uniform global collaborative fusion

and is more directly tied to the measured failure pattern of local leaf ambiguity.

## Current Status

At this stage, this document defines a **research direction**, not a finalized algorithm.

What is fixed now:

- the motivation
- the main hypothesis
- the high-level design principle

What remains open:

- how to construct coarse, mid, and fine collaborative views
- how to design the layer-wise fusion mechanism
- how to define the denoising strategy for each level
- how to define selective separation signals for semantically similar but collaboratively inconsistent items
- how to validate that each SID level is really using the appropriate signal

These details should be explored next, but the overall direction is now clear enough to serve as the foundation for the next round of method design.
