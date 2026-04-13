# arXiv Related Work by Question

## Purpose

This note collects arXiv papers that are directly relevant to the three core questions in:

- `CURRENT_TASK_ALIGNMENT.md`

The goal is not to build a broad recommender survey.  
The goal is to answer a narrower question:

> what prior work can we borrow from when building a graph-structured, hierarchy-aware collaborative fusion mechanism for semantic SID construction?

## Search scope

The arXiv search was organized around these themes:

- recommendation + graph signal
- recommendation + graph denoising
- recommendation + multi-scale / spectral / band-pass graph
- generative recommendation + tokenization
- recommendation + transition + collaboration

## Already relevant papers in local library

These papers were already present in `papers/` and remain central:

1. `PRISM`  
   arXiv: `2601.16556`  
   local: `papers/PRISM.pdf`

2. `ReSID`  
   arXiv: `2602.02338`  
   local: `papers/ReSID.pdf`

3. `PIT`  
   arXiv: `2602.08530`  
   local: `papers/PIT.pdf`

4. `Align^3GR`  
   arXiv: `2511.11255`  
   local: `papers/Unified Multi-Level Alignment for LLM-based Generative.pdf`

5. `DiscRec`  
   arXiv: `2506.15576`  
   local: not yet saved by ID, but already used in idea discussion

6. `ETEGRec`  
   arXiv: `2409.05546`  
   local: `papers/ETEGRec.pdf`

7. `CoST`  
   arXiv: `2404.14774`  
   local: not yet saved by ID, but relevant for tokenization quality and neighborhood preservation

## Newly downloaded papers

The following new papers were downloaded into the local library for direct reference:

- `papers/GSPRec - Temporal-Aware Graph Spectral Filtering for Recommendation.pdf`
- `papers/How Do Graph Signals Affect Recommendation Unveiling the Mystery of Low and High-Frequency Graph Signals.pdf`
- `papers/Collaboration and Transition Distilling Item Transitions into Multi-Query Self-Attention for Sequential Recommendation.pdf`
- `papers/Continuous-Time Sequential Recommendation with Temporal Graph Collaborative Transformer.pdf`
- `papers/Discovering Collaborative Signals for Next POI Recommendation with Iterative Seq2Graph Augmentation.pdf`
- `papers/Frequency-aware Graph Signal Processing for Collaborative Filtering.pdf`

## Question 1: What graph should carry collaborative information?

### 1. `GSPRec: Temporal-Aware Graph Spectral Filtering for Recommendation`

- **arXiv**: `2505.11552`
- **Date**: `2025-05-15`
- **Why it matters**:
  - explicitly combines collaborative graph modeling with temporal transition structure
  - separates low-pass global trend and band-pass user-specific pattern
- **Borrowable idea**:
  - our `G_coarse` and `G_mid` can be understood as different graph-frequency views
  - sequentially-informed graph construction is directly relevant to how we design `G_local`
- **Most useful for**:
  - defining `G_mid`
  - motivating that graph views should not all be low-pass

### 2. `How Do Graph Signals Affect Recommendation: Unveiling the Mystery of Low and High-Frequency Graph Signals`

- **arXiv**: `2512.15744`
- **Date**: `2025-12-10`
- **Why it matters**:
  - challenges the simplistic assumption that only low-frequency graph signals are useful
  - argues recommendation can benefit from different graph signal regimes
- **Borrowable idea**:
  - do not hard-code the belief that “global smooth graph” is always best
  - justify why deep SID levels may need different graph signal characteristics from upper levels
- **Most useful for**:
  - supporting the claim that graph structure should be multi-view and non-uniform

### 3. `Frequency-aware Graph Signal Processing for Collaborative Filtering`

- **arXiv**: `2402.08426`
- **Date**: `2024-02-13`
- **Why it matters**:
  - introduces cascaded and parallel filters to capture different graph frequencies and neighborhood hierarchies
- **Borrowable idea**:
  - very useful inspiration for constructing `G_mid`
  - suggests that “middle scale” may come from graph filtering design rather than from hand-picked hop count
- **Most useful for**:
  - building candidate `G_mid` operators
  - supporting hierarchy of neighborhoods

### 4. `Collaboration and Transition: Distilling Item Transitions into Multi-Query Self-Attention for Sequential Recommendation`

- **arXiv**: `2311.01056`
- **Date**: `2023-11-02`
- **Why it matters**:
  - clearly separates collaborative signals and transition signals
  - treats global item-to-item transition as something worth distilling into representations
- **Borrowable idea**:
  - our `G_local` should not be thought of as a weak afterthought
  - transition structure can act as a calibrator for collaborative information
- **Most useful for**:
  - defining and denoising `G_local`
  - motivating why transition and collaboration should coexist

### 5. `Continuous-Time Sequential Recommendation with Temporal Graph Collaborative Transformer`

- **arXiv**: `2108.06625`
- **Date**: `2021-08-14`
- **Why it matters**:
  - uses a temporal graph to jointly model sequential and collaborative signals
- **Borrowable idea**:
  - the graph carrier need not be static; temporal weighting can be part of graph construction
- **Most useful for**:
  - designing `G_local`
  - thinking about time-sensitive graph edges

### 6. `Discovering Collaborative Signals for Next POI Recommendation with Iterative Seq2Graph Augmentation`

- **arXiv**: `2106.15814`
- **Date**: `2021-06-30`
- **Why it matters**:
  - explicitly uses graph augmentation to discover collaborative signals that are not obvious in isolated sequences
- **Borrowable idea**:
  - graph is not just a storage format; it can densify sparse transition structure
  - useful when `G_local` is too sparse
- **Most useful for**:
  - augmenting local or sparse transition graphs
  - improving graph coverage without collapsing to pure global smoothing

## Question 2: How do we make the method hierarchy-aware?

### 1. `Align^3GR: Unified Multi-Level Alignment for LLM-based Generative Recommendation`

- **arXiv**: `2511.11255`
- **Date**: `2025-11-14`
- **Why it matters**:
  - not the same as SID-level graph design, but directly supports the broader intuition that different levels play different roles
- **Borrowable idea**:
  - multi-level structure can be a first-class design principle, not just a downstream artifact
- **What to borrow carefully**:
  - use it as conceptual support for “levels matter”
  - do not present it as evidence for our exact SID-level graph mechanism

### 2. `PRISM`

- **arXiv**: `2601.16556`
- **Date**: `2026-01-23`
- **Why it matters**:
  - explicitly mentions hierarchical semantic anchoring
  - treats denoising and structural stability as central to tokenization
- **Borrowable idea**:
  - hierarchy should be protected while collaborative information is introduced
  - supports our insistence on a semantic anti-collapse anchor
- **Most useful for**:
  - justifying why graph information must be injected carefully into hierarchical SID

### 3. `ReSID`

- **arXiv**: `2602.02338`
- **Date**: `2026-02-02`
- **Why it matters**:
  - emphasizes prefix-conditional uncertainty
  - directly connects tokenization quality to generative predictability
- **Borrowable idea**:
  - hierarchy-awareness should not be cosmetic
  - different SID levels matter because they affect predictability differently
- **Most useful for**:
  - motivating why level-specific graph supervision is meaningful

### 4. `FaGSP` and `GSPRec`

- **Why they matter together**:
  - both suggest graph signal roles differ by frequency / neighborhood scale
- **Borrowable idea**:
  - hierarchy-aware design can be grounded in graph scales or graph frequencies
  - `Level 1 / Level 2 / Level 3` may correspond more naturally to coarse / mid / local graph structure than to arbitrary hand rules

## Question 3: How do we fuse graph-structured collaboration with MiniOneRec’s semantic SID?

### 1. `PRISM`

- **Most direct value**:
  - collaborative denoising before tokenization
  - hierarchical semantic anchoring
- **Borrowable idea**:
  - graph-structured collaboration should enhance tokenization while preserving semantic hierarchy
- **Direct caution**:
  - collaborative signal is noisy enough to collapse tokenization if injected naively

### 2. `PIT`

- **arXiv**: `2602.08530`
- **Date**: `2026-02-09`
- **Why it matters**:
  - emphasizes volatility of collaborative signals in tokenization
- **Borrowable idea**:
  - even if we do not follow PIT’s dynamic personalized tokenizer route, we should inherit its caution:
    collaborative information is not a static clean feature
- **Most useful for**:
  - motivating view-specific denoising and anti-collapse design

### 3. `DiscRec`

- **arXiv**: `2506.15576`
- **Date**: `2025-06-18`
- **Why it matters**:
  - explicitly disentangles semantic and collaborative signals
  - fuses them flexibly rather than forcing them into one embedding space too early
- **Borrowable idea**:
  - semantic and collaborative signals should not be blindly unified
  - supports our choice to keep semantic SID as backbone and use graph collaboration as structured enhancement

### 4. `ETEGRec`

- **arXiv**: `2409.05546`
- **Date**: `2024-09-09`
- **Why it matters**:
  - argues tokenization and recommendation should be jointly aligned
- **Borrowable idea**:
  - graph information should not only live in a preprocessing artifact
  - the tokenizer learning objective should reflect recommendation needs

### 5. `CoST`

- **arXiv**: `2404.14774`
- **Date**: `2024-04-23`
- **Why it matters**:
  - explicitly says tokenization should preserve item neighborhood relationships
- **Borrowable idea**:
  - our graph regularization can be framed as a stronger, structured version of “preserve useful item relations during tokenization”

### 6. `LETTER / Learnable Item Tokenization for Generative Recommendation`

- **arXiv**: `2405.07314`
- **Date**: `2024-05-12`
- **Why it matters**:
  - addresses tokenization quality and code assignment bias in generative recommendation
- **Borrowable idea**:
  - reminds us that graph fusion cannot ignore code assignment behavior and tokenizer health

## What these papers jointly suggest

Across these papers, three consistent messages appear:

### Message 1

Collaborative information should enter tokenization or recommendation-aware representation learning.  
That part is no longer controversial.

### Message 2

Collaborative information is noisy, volatile, and structurally heterogeneous.  
Naive global fusion is risky.

### Message 3

Different graph structures and different graph frequencies carry different recommendation value.  
This strongly supports our current direction:

> graph-structured collaborative information should be fused into SID construction in a level-aware and view-specific way.

## What still looks underexplored

Even after this search, there is still no paper here that cleanly combines all three of our target properties:

1. graph as the collaborative carrier
2. SID-level hierarchy-aware allocation or supervision
3. fusion into semantic SID through structural constraint rather than plain feature concat

This is exactly why `MGR-SID` still looks like a meaningful direction.

## Recommended reading order for our current task

If we only read a small subset first, the best order is:

1. `PRISM`
2. `ReSID`
3. `PIT`
4. `DiscRec`
5. `GSPRec`
6. `Collaboration and Transition`
7. `FaGSP`

## Most directly borrowable design ideas

### For `G_coarse`

- debiased collaborative graph
- low-pass or broad compatibility view

### For `G_mid`

- band-pass graph view
- diffusion residual
- community-aware graph

`GSPRec` and `FaGSP` are the strongest references here.

### For `G_local`

- transition-aware graph
- temporal weighting
- graph augmentation when local graph is too sparse

`Collaboration and Transition`, `TGSRec`, and `Seq2Graph` are the strongest references here.

### For semantic integration

- semantic backbone + structured collaborative denoising
- semantic anchor to prevent collapse
- graph relation preservation during tokenization

`PRISM`, `DiscRec`, `ETEGRec`, and `CoST` are the strongest references here.
