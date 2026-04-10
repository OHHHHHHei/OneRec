# Ambiguity Proxy Literature Scan

**Date**: 2026-04-10  
**Scope**: literature scan for train-time ambiguity proxies that can be used **before final SID generation**.  
**Current method context**: `MiniOneRec baseline -> hierarchy-aware graph regularization -> final SID gains on local ambiguity, but imperfect tokenizer-to-SFT transfer`.

---

## Why This Scan

We currently have a strong diagnosis result:

- final `hierarchy_reg` SID is cleaner than the MiniOneRec baseline
- `same_l2` / local leaf ambiguity is reduced
- downstream SFT gains are real but concentrated in short-range ranking and crowded local neighborhoods

The natural next question is:

> before final SID is generated, can we estimate which items or prefixes are **high ambiguity** and which are **low ambiguity**, so that graph supervision is applied more intelligently?

This scan focuses on four candidate signal families:

1. **semantic density**
2. **semantic-collaborative disagreement**
3. **graph competition strength**
4. **quantization-time uncertainty**

The goal is not to copy an existing paper, but to identify **borrowable modules, definitions, and design principles**.

---

## Short Answer

The literature strongly supports the idea that ambiguity should be estimated through **proxies**, not through post-hoc final SID labels.

The most promising borrowable ingredients are:

- **semantic crowding / uncertainty** from local neighborhood structure
- **semantic-collaborative mismatch** from dual-space alignment or disentanglement
- **competition-aware graph statistics** from topology-aware collaborative modeling
- **quantization confidence / uncertainty** from code assignment margin, entropy, and residual difficulty

The strongest synthesis for our project is:

> use **offline structural ambiguity priors** (semantic density, semantic-collaborative disagreement, graph competition) together with **online quantizer uncertainty** (margin / entropy / residual) to modulate hierarchy-aware graph regularization.

---

## Group A: Semantic Density and Uncertainty

These papers support the idea that some items are intrinsically more ambiguous because they live in dense or uncertain local semantic neighborhoods.

### ReSID

- Local file: [ReSID.pdf](/home/leejt/OneRec/papers/ReSID.pdf)
- arXiv: https://arxiv.org/abs/2602.02338
- Core signal:
  - recommendation-oriented tokenization should reduce **prefix-conditional uncertainty**
  - quantization should be judged by predictability, not just semantic reconstruction
- Borrowable idea:
  - define ambiguity not only as “many neighbors nearby”, but as **difficulty of making the next discrete decision predictable**
- Relevance to us:
  - this is the closest conceptual support for saying our ambiguity proxy should reflect **future leaf uncertainty**, not only local density

### STOSA

- Local file: [STOSA - Sequential Recommendation via Stochastic Self-Attention.pdf](/home/leejt/OneRec/papers/STOSA%20-%20Sequential%20Recommendation%20via%20Stochastic%20Self-Attention.pdf)
- arXiv: https://arxiv.org/abs/2201.06035
- Core signal:
  - item representations can carry explicit **uncertainty** rather than only deterministic embeddings
  - uncertainty is useful in sequential recommendation
- Borrowable idea:
  - treat ambiguity as a distributional property rather than a binary label
  - use a scalar uncertainty score or covariance-derived confidence
- Relevance to us:
  - supports the idea that “semantic density” should probably be converted into a **soft uncertainty prior**, not a hard gate

### HiD-VAE

- Local file: [HiD-VAE.pdf](/home/leejt/OneRec/papers/HiD-VAE.pdf)
- arXiv: https://arxiv.org/abs/2508.04618
- Core signal:
  - hierarchical tokenization becomes unstable when representations are entangled
  - uniqueness and disentanglement matter
- Borrowable idea:
  - semantic ambiguity can be viewed as **latent overlap risk**
  - ambiguity score can include “how many semantically close items still overlap in representation”
- Relevance to us:
  - suggests that a semantic-density proxy should not only count neighbors, but also measure **neighbor indistinguishability**

### What We Can Borrow for Our Setting

- `semantic_density_score(i)`:
  - average similarity to top-`k` semantic neighbors
  - or local density estimated by the gap between `k`-NN distances
- `semantic_overlap_score(i)`:
  - number of highly similar semantic neighbors above a threshold
- Better framing:
  - **high semantic density** means “many semantically plausible competitors”
  - **high semantic overlap** means “semantic representation alone may under-resolve this item”

---

## Group B: Semantic-Collaborative Disagreement

These papers support the idea that ambiguity is strongest where semantic similarity and collaborative structure disagree.

### PRISM

- Local file: [PRISM.pdf](/home/leejt/OneRec/papers/PRISM.pdf)
- arXiv: https://arxiv.org/abs/2601.16556
- Core signal:
  - collaborative signals are noisy
  - tokenization benefits from **adaptive collaborative denoising** and **hierarchical semantic anchoring**
- Borrowable idea:
  - disagreement between semantic structure and raw collaborative structure is not something to ignore; it is precisely where denoising becomes necessary
- Relevance to us:
  - very strong support for using **semantic-collaborative inconsistency** as an ambiguity indicator

### DiscRec

- Local file: [DiscRec - Disentangled Semantic-Collaborative Modeling for Generative Recommendation.pdf](/home/leejt/OneRec/papers/DiscRec%20-%20Disentangled%20Semantic-Collaborative%20Modeling%20for%20Generative%20Recommendation.pdf)
- arXiv: https://arxiv.org/abs/2506.15576
- Core signal:
  - semantic and collaborative signals have different distributions and should be disentangled before fusion
  - gating should happen after disentanglement, not before
- Borrowable idea:
  - disagreement can be estimated through **cross-branch inconsistency**
  - semantic branch and collaborative branch should each retain their own structure
- Relevance to us:
  - supports defining an ambiguity proxy from **mismatch between semantic neighbors and collaborative neighbors**

### DisCo

- Local file: [DisCo - Towards Harmonious Disentanglement and Collaboration between Tabular and Semantic Space for Recommendation.pdf](/home/leejt/OneRec/papers/DisCo%20-%20Towards%20Harmonious%20Disentanglement%20and%20Collaboration%20between%20Tabular%20and%20Semantic%20Space%20for%20Recommendation.pdf)
- arXiv: https://arxiv.org/abs/2406.00011
- Core signal:
  - semantic and collaborative spaces each contain both shared and unique information
  - good models should preserve both consistency and specificity
- Borrowable idea:
  - ambiguity is especially likely where a point sits in the **shared-but-conflicted region** between two spaces
- Relevance to us:
  - supports a proxy that explicitly measures whether semantic and collaborative structure agree on local grouping

### CARec

- Local file: [CARec - Collaborative Semantic Alignment in Recommendation Systems.pdf](/home/leejt/OneRec/papers/CARec%20-%20Collaborative%20Semantic%20Alignment%20in%20Recommendation%20Systems.pdf)
- arXiv: https://arxiv.org/abs/2310.09400
- Core signal:
  - semantic representations need collaborative alignment
  - but alignment must preserve semantic meaning
- Borrowable idea:
  - disagreement can be quantified as **alignment cost** or **alignment residual**
- Relevance to us:
  - useful for designing a semantic-collaborative discrepancy score before SID is finalized

### DAS

- Local file: [DAS.pdf](/home/leejt/OneRec/papers/DAS.pdf)
- Core signal:
  - one-stage alignment avoids some information loss seen in two-stage alignment pipelines
  - multi-view contrastive alignment is helpful
- Borrowable idea:
  - disagreement is not only a problem; it can become a **training signal**
- Relevance to us:
  - suggests that ambiguity proxies can be tied directly to alignment losses, not only heuristic statistics

### What We Can Borrow for Our Setting

- `semantic_collab_disagreement(i)` candidate definitions:
  - Jaccard distance between semantic `k`-NN and collaborative `k`-NN
  - overlap-weighted divergence between semantic and graph neighborhood distributions
  - alignment residual between semantic embedding and collaborative-smoothed embedding
- Strong design principle:
  - **high disagreement** means “semantic SID is likely missing collaborative structure exactly where it matters”

---

## Group C: Graph Competition Strength and Multi-Scale Structure

These papers support the idea that ambiguity is not just density; it is often about **competition among plausible graph neighbors**.

### GSPRec

- Local file: [GSPRec - Temporal-Aware Graph Spectral Filtering for Recommendation.pdf](/home/leejt/OneRec/papers/GSPRec%20-%20Temporal-Aware%20Graph%20Spectral%20Filtering%20for%20Recommendation.pdf)
- arXiv: https://arxiv.org/abs/2505.11552
- Core signal:
  - low-frequency signals capture global trends
  - band-pass signals capture user-level or mid-scale patterns
  - sequential augmentation improves graph construction
- Borrowable idea:
  - ambiguity can depend on which frequency band is strong
  - items with strong mid-band competition may be exactly our difficult local disambiguation cases
- Relevance to us:
  - strongly supports our `G_mid` line and suggests a **spectral competition score**

### FaGSP

- Local file: [Frequency-aware Graph Signal Processing for Collaborative Filtering.pdf](/home/leejt/OneRec/papers/Frequency-aware%20Graph%20Signal%20Processing%20for%20Collaborative%20Filtering.pdf)
- arXiv: https://arxiv.org/abs/2402.08426
- Core signal:
  - user/item preference needs both common and unique characteristics
  - low-pass and high-pass components contribute differently
- Borrowable idea:
  - ambiguity may be high when common and unique graph signals are both strong
- Relevance to us:
  - supports building a graph-based competition proxy from **multi-band energy**

### How Do Graph Signals Affect Recommendation

- Local file: [How Do Graph Signals Affect Recommendation Unveiling the Mystery of Low and High-Frequency Graph Signals.pdf](/home/leejt/OneRec/papers/How%20Do%20Graph%20Signals%20Affect%20Recommendation%20Unveiling%20the%20Mystery%20of%20Low%20and%20High-Frequency%20Graph%20Signals.pdf)
- arXiv: https://arxiv.org/abs/2512.15744
- Core signal:
  - both low- and high-frequency graph signals can matter
  - the real issue is controlled smoothing
- Borrowable idea:
  - ambiguity is not always “more local is better”; it may be about the right balance of graph smoothness
- Relevance to us:
  - supports moving from raw graph strength to **competition after scaling/filtering**

### Collaboration and Transition

- Local file: [Collaboration and Transition Distilling Item Transitions into Multi-Query Self-Attention for Sequential Recommendation.pdf](/home/leejt/OneRec/papers/Collaboration%20and%20Transition%20Distilling%20Item%20Transitions%20into%20Multi-Query%20Self-Attention%20for%20Sequential%20Recommendation.pdf)
- arXiv: https://arxiv.org/abs/2311.01056
- Core signal:
  - collaborative context and short-range transitions are different signals
  - both are needed
- Borrowable idea:
  - ambiguity may be high when global collaborative support and local transition support disagree
- Relevance to us:
  - supports defining a competition proxy from **transition-vs-collaboration conflict**

### Seq2Graph / SGRec

- Local file: [Discovering Collaborative Signals for Next POI Recommendation with Iterative Seq2Graph Augmentation.pdf](/home/leejt/OneRec/papers/Discovering%20Collaborative%20Signals%20for%20Next%20POI%20Recommendation%20with%20Iterative%20Seq2Graph%20Augmentation.pdf)
- arXiv: https://arxiv.org/abs/2106.15814
- Core signal:
  - sparse transitions can be densified through cross-sequence graph augmentation
- Borrowable idea:
  - ambiguity can be hidden when the graph is too sparse
  - strong local competition may only appear after graph augmentation
- Relevance to us:
  - suggests our graph competition score should not rely only on raw local edges

### CAGCN

- Local file: [CAGCN - Collaboration-Aware Graph Convolutional Network for Recommender Systems.pdf](/home/leejt/OneRec/papers/CAGCN%20-%20Collaboration-Aware%20Graph%20Convolutional%20Network%20for%20Recommender%20Systems.pdf)
- arXiv: https://arxiv.org/abs/2207.06221
- Core signal:
  - introduces a topological measure, `Common Interacted Ratio (CIR)`, to estimate how collaboratively useful a neighbor is
- Borrowable idea:
  - ambiguity is not just node degree; it can be about the **competition profile among neighbors**
- Relevance to us:
  - very promising for defining graph competition strength in a more principled way than raw degree or raw fanout

### A Topology-aware Analysis of Graph Collaborative Filtering

- Local file: [A Topology-aware Analysis of Graph Collaborative Filtering.pdf](/home/leejt/OneRec/papers/A%20Topology-aware%20Analysis%20of%20Graph%20Collaborative%20Filtering.pdf)
- arXiv: https://arxiv.org/abs/2308.10778
- Core signal:
  - dataset topology strongly affects graph recommendation behavior
- Borrowable idea:
  - ambiguity may depend on graph topology statistics, not only per-item similarity
- Relevance to us:
  - supports using local topology features such as clustering or neighborhood redundancy as ambiguity priors

### What We Can Borrow for Our Setting

- `graph_competition_strength(i)` candidates:
  - entropy of normalized graph neighbor weights
  - number of strong neighbors above a threshold
  - `CIR`-style common-neighbor overlap
  - tension between collaborative graph and transition graph
  - spectral mid-band energy around the item
- Strong design principle:
  - **high graph competition** means “many graph-supported alternatives are plausible, so leaf assignment should be more careful”

---

## Group D: Quantization-Time Uncertainty

These papers support the idea that some ambiguity can be estimated directly from the quantization process itself.

### PIT

- Local file: [PIT.pdf](/home/leejt/OneRec/papers/PIT.pdf)
- arXiv: https://arxiv.org/abs/2602.08530
- Core signal:
  - collaborative signals are volatile
  - tokenizer and recommender should co-evolve instead of being statically separated
- Borrowable idea:
  - instability during token assignment is itself meaningful
- Relevance to us:
  - supports using quantization-time confidence as a signal for whether graph supervision should intervene

### ReSID (again)

- Local file: [ReSID.pdf](/home/leejt/OneRec/papers/ReSID.pdf)
- arXiv: https://arxiv.org/abs/2602.02338
- Core signal:
  - orthogonal quantization should reduce semantic ambiguity and prefix-conditional uncertainty
- Borrowable idea:
  - code assignment quality is not only about nearest code distance, but about how predictable the prefix becomes
- Relevance to us:
  - suggests we should monitor level-wise assignment confidence, not just final reconstruction

### HiD-VAE (again)

- Local file: [HiD-VAE.pdf](/home/leejt/OneRec/papers/HiD-VAE.pdf)
- arXiv: https://arxiv.org/abs/2508.04618
- Core signal:
  - uniqueness loss explicitly fights latent overlap and collision
- Borrowable idea:
  - quantization ambiguity can be proxied by local overlap to neighboring code regions
- Relevance to us:
  - motivates codebook-margin or overlap-aware weighting

### LCRec / TokenRec / VQ-Rec / UTGRec / ETEGRec

- Local files:
  - [LCRec.pdf](/home/leejt/OneRec/papers/LCRec.pdf)
  - [TokenRec.pdf](/home/leejt/OneRec/papers/TokenRec.pdf)
  - [VQ-Rec.pdf](/home/leejt/OneRec/papers/VQ-Rec.pdf)
  - [UTGRec.pdf](/home/leejt/OneRec/papers/UTGRec.pdf)
  - [ETEGRec.pdf](/home/leejt/OneRec/papers/ETEGRec.pdf)
- Core signal:
  - quantization quality should be judged relative to downstream recommendation behavior
  - tokenization and recommendation should not be treated as completely separate
- Borrowable idea:
  - ambiguity proxies should not be only geometric; they should be tied to downstream usefulness
- Relevance to us:
  - supports mixing structural priors with online quantization confidence rather than using pure geometry alone

### USD

- Local file: [Uncertainty-Aware Semantic Decoding for LLM-Based Sequential Recommendation.pdf](/home/leejt/OneRec/papers/Uncertainty-Aware%20Semantic%20Decoding%20for%20LLM-Based%20Sequential%20Recommendation.pdf)
- arXiv: https://arxiv.org/abs/2508.07210
- Core signal:
  - decoding uncertainty can be measured and used to adjust recommendation behavior
- Borrowable idea:
  - uncertainty over semantically similar candidates can be explicitly modeled
- Relevance to us:
  - although it is an inference-time method, it supports the broader framing that ambiguity can be quantified through **cluster-level uncertainty**

### What We Can Borrow for Our Setting

- `quantization_uncertainty(i, level)` candidates:
  - nearest-code margin: `d_2 - d_1`
  - normalized margin: `(d_2 - d_1) / (|d_1| + eps)`
  - soft assignment entropy if Sinkhorn or softmax distances are exposed
  - residual norm after each quantization level
- Strong design principle:
  - **high quantization uncertainty** means “the current level is not making a confident discrete decision”

---

## A Concrete Synthesis for Our Project

Given the current evidence, the cleanest next-step ambiguity design is:

### Offline structural ambiguity prior

For each item `i`, estimate:

- `semantic_density(i)`
- `semantic_collab_disagreement(i)`
- `graph_competition_strength(i)`

This prior is train-only and does **not** require the final SID.

### Online quantizer ambiguity

During tokenizer training, estimate for each level:

- code assignment margin
- assignment entropy
- residual difficulty

These are fully available before final SID generation.

### Final use

Instead of a post-hoc patch:

- high ambiguity -> stronger graph regularization
- low ambiguity -> stronger semantic retention / anti-overcorrection anchor

This is consistent with the current empirical story:

- `hierarchy` helps most on crowded local cases
- but over-corrects some already-stable baseline cases

So the literature supports a next-step method that is not “more graph everywhere”, but:

> **ambiguity-aware graph supervision with semantic retention**

---

## Most Actionable Borrowable Ideas

If we only choose a few modules to carry forward:

1. **semantic-collaborative disagreement score**
   - Borrow from `PRISM`, `DiscRec`, `DisCo`, `CARec`
   - This is probably the strongest single idea for our setting.

2. **graph competition score**
   - Borrow from `GSPRec`, `FaGSP`, `CAGCN`, `Collaboration and Transition`
   - This is likely the best way to formalize “crowded local neighborhood”.

3. **quantizer margin / entropy**
   - Borrow from `ReSID`, `PIT`, `HiD-VAE`, `USD`
   - This is the cleanest online signal because it comes directly from the code assignment process.

4. **semantic retention anchor**
   - Borrow from `PRISM`, `CARec`, `DisCo`
   - This directly addresses the current `broken` samples where hierarchy destroys already-correct local retention.

---

## Current Recommendation

Do **not** jump straight to a fully free learned gate.

The literature and our own results suggest a better next-step order:

1. define ambiguity proxies
2. use them to **weight** fixed level-wise graph regularization
3. add a semantic retention / anti-overcorrection anchor
4. only then consider a more flexible gate

This is cleaner, easier to interpret, and much better aligned with the evidence we already have.
