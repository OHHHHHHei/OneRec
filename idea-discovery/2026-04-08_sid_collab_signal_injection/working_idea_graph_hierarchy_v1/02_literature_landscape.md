# Literature Landscape

## What prior work already established

### 1. Front-end collaborative tokenization is real and active

Recent generative recommendation work has already moved beyond pure semantic tokenization:

- `PRISM` argues that semantic quantization alone is insufficient and introduces collaborative denoising before quantization  
  URL: https://arxiv.org/abs/2601.16556
- `PIT` explicitly points out the volatility of collaborative signals in end-to-end tokenization  
  URL: https://arxiv.org/abs/2602.08530
- `ETEGRec` pushes end-to-end learnable tokenization and recommender training together  
  URL: https://arxiv.org/abs/2409.05546
- `Align3GR` uses multi-level alignment and dual tokenization with semantic and collaborative signals  
  URL: https://arxiv.org/abs/2511.11255

This means the research question is no longer:

- should collaborative information be used at all?

That part is already answered.

### 2. Denoising is not optional

The strongest recent papers repeatedly warn that collaborative signals are noisy:

- `PRISM` uses adaptive collaborative denoising to avoid impure tokenization
- `PIT` argues that volatile collaborative signals can destabilize tokenization

This aligns tightly with our repo evidence:

- naive global front-end fusion produced severe collapse in the current experiments

### 3. Recommendation already benefits from different graph scales

Outside strict SID tokenization, graph-based recommendation papers already show that collaborative structure is not single-scale:

- `GSPRec` separates low-pass and band-pass graph signals, using them for different recommendation roles  
  URL: https://openreview.net/pdf?id=ifgApKmXIQ
- `Collaboration and Transition` argues that long-range collaboration and short-range transition should both be modeled  
  URL: https://arxiv.org/pdf/2311.01056
- `DiscRec` separates semantic and collaborative factors with learned fusion  
  URL: https://arxiv.org/abs/2506.15576

These works support the intuition that:

- different collaborative structures matter
- different structures carry different noise profiles

## What remains underexplored

### Missing question 1

Even when prior work uses collaborative tokenization, it usually treats collaboration as something to:

- fuse globally
- align globally
- denoise once and reuse everywhere

But in a hierarchical SID, different levels do different jobs.

The missing question is:

> should different SID levels preserve different graph structures?

### Missing question 2

Recent work discusses denoising, but not usually in a level-aware way.

The unexplored space is:

- coarse graph view may need debiasing and low-pass purification
- mid graph view may need community or band-pass extraction
- local transition view may need confidence pruning and temporal smoothing

In other words:

> denoising itself may need to be graph-view-specific, not globally uniform.

### Missing question 3

`ReSID` makes prefix-conditional uncertainty central to SID learning:  
URL: https://arxiv.org/abs/2602.02338

But existing methods still do not answer:

- which graph structures reduce uncertainty at which SID levels
- whether level-specific graph supervision is better than one global collaborative embedding

## Why this matters for our repo

Our current repo evidence says:

- collision exists but is not the main bottleneck
- leaf ambiguity under correct prefixes is the main remaining error mode
- naive front-end fusion is unstable

The literature says:

- collaborative information should enter tokenization
- denoising is necessary
- different graph scales matter

The natural gap between them is:

> graph-structured, denoised, hierarchy-aware collaborative supervision for SID learning

That is the space this discovery round focuses on.
