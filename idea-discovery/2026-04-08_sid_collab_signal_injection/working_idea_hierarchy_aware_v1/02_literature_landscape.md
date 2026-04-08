# Literature Landscape

## Core question for this round

Existing papers already show that collaborative information can be injected into front-end tokenization. The gap is narrower:

> once collaborative information enters a hierarchical SID tokenizer, should every SID level consume the same collaborative view, or do different levels need different collaborative resolutions and different denoising treatments?

## Literature clusters

## 1. Front-end collaborative or end-to-end tokenizer redesign

These papers establish that front-end collaborative injection is a serious and competitive direction.

| Paper | Main takeaway | Relevance to us |
|---|---|---|
| `ETEGRec` | tokenizer and generative recommender should be aligned end-to-end | supports front-end integration as a valid target |
| `PIT` | tokenization can be dynamic/personalized; collaborative volatility is a real challenge | supports our stability concern |
| `LCRec` | collaborative semantics should be integrated, not left to pure language semantics | supports our motivation |
| `Align3GR` | multi-level alignment matters, but at token/behavior/preference level | adjacent but not the same as SID-level allocation |
| `UTGRec` | collaborative signals can be integrated during tokenization together with content | supports front-end collaborative learning |

### Reading

The field is clearly not moving toward “semantic-only tokenization”. The question is no longer whether collaboration should enter the tokenizer.

## 2. Purification, stability, and collapse prevention

These papers explain why naive global front-end fusion can fail.

| Paper | Main takeaway | Relevance to us |
|---|---|---|
| `PRISM` | collaborative denoising and hierarchical semantic anchoring are needed to avoid impure tokenization and codebook collapse | directly supports our denoising premise |
| `ReSID` | tokenization should be recommendation-native and reduce prefix-conditional uncertainty | directly supports our hierarchy/predictability framing |
| `DIGER` | differentiable semantic IDs need careful codebook stabilization | supports stability concerns |
| `QuaSID` | not all collisions are equal; collaborative-aware tokenization needs more refined control | useful for collision nuance |

### Reading

These papers support a strong claim:

- front-end collaboration is useful
- but raw or uniform injection is dangerous

This is exactly the tension observed in our repo.

## 3. Semantic-collaborative disentanglement and gating

These papers are not exactly about SID-level allocation, but they point toward a more structured treatment of collaborative signals.

| Paper | Main takeaway | Relevance to us |
|---|---|---|
| `DiscRec` | semantic and collaborative signals should be disentangled and gated differently | very close in spirit, but not explicitly SID-level |
| `DAS` | semantic IDs should be aligned with collaborative objectives in one stage | supports alignment-centric formulation |

### Reading

This cluster suggests that “one fused representation for all purposes” is too crude.

## 4. Evidence from broader recommendation literature: signal scales matter

These works are not all SID papers, but they matter for our idea.

| Paper / Direction | Main takeaway | Relevance to us |
|---|---|---|
| `GSPRec` | different graph frequencies capture different recommendation structure | supports multi-resolution view design |
| `DRCSD` | multi-order collaborative signals should be decomposed/denoised instead of mixed blindly | supports view-wise purification |
| `How Do Graph Signals Affect Recommendation?` | recommendation quality depends on the type of graph signal extracted | supports non-uniform signal utility |
| `Collaboration and Transition` | long-range collaboration and short-range transition play different roles | directly supports our coarse/mid/local intuition |

### Reading

The broader literature repeatedly says that:

- not all collaborative signals are the same
- different scales carry different utility and different noise

This strongly supports our move from “single collaborative feature” to “view bank”.

## What seems missing in the literature

After reading the relevant clusters, the most interesting remaining gap is:

### Missing design question

Current papers largely study:

- whether front-end collaboration helps
- how to denoise collaborative signals
- how to couple tokenization and recommendation

But they rarely make the following claim explicit:

> **the correct collaborative signal should depend on the SID level itself**

In particular, I did not find a clear mainstream formulation that says:

- upper SID levels should absorb one kind of purified collaborative structure
- middle levels should absorb another
- deeper levels should absorb a different one
- and this allocation should be learned and validated as a hierarchy-level principle

## Literature takeaway for our next step

The most differentiated path is not:

- “another collaborative tokenizer”

It is:

- a **multi-resolution collaborative view bank**
- with **level-aware allocation**
- and **view-specific purification**

That is the basis for the idea ranking in the next section.

