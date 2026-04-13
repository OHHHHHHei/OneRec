# Modules Mapped to the Three Core Questions

## Purpose

This note maps the most relevant papers we have searched to the three core questions in:

- `CURRENT_TASK_ALIGNMENT.md`

The purpose is not to summarize whole papers.  
The purpose is to extract:

- what module or design idea each paper offers
- which of our three core questions it helps answer
- whether it is suitable for the first version of `MGR-SID`

## Our three core questions

### Q1. What graph should carry collaborative information?

We need graph structures that can serve as collaborative-information carriers for SID construction.

### Q2. How do we make the method hierarchy-aware?

We need a level-aware mechanism so that different SID levels use collaborative graph structure differently.

### Q3. How do we fuse graph-structured collaboration with MiniOneRec's pure semantic SID?

We need a fusion mechanism that enhances semantic SID without collapsing its hierarchy.

## Recommended reading priority

If we only read a small subset carefully, the strongest order is:

1. `PRISM`
2. `ReSID`
3. `GSPRec`
4. `Collaboration and Transition`
5. `DiscRec`
6. `PIT`

## Paper-to-module mapping

## 1. `PRISM`

- **arXiv**: `2601.16556`
- **Role for us**: semantic-safe collaborative injection

### Borrowable modules

- adaptive collaborative denoising
- hierarchical semantic anchoring
- anti-collapse tokenizer stabilization

### Maps to our questions

- `Q1`: weak  
  It does not tell us exactly what graph to use.
- `Q2`: medium  
  It supports the idea that hierarchy must be protected.
- `Q3`: very strong  
  This is one of the best references for how collaborative information can enter tokenization without destroying semantic structure.

### What we should learn

- collaborative signals are noisy enough to collapse tokenization
- semantic hierarchy must remain the backbone
- denoising should be treated as a first-class module, not a minor preprocessing step

### Best use in `MGR-SID`

- semantic anti-collapse anchor
- justification for view-specific denoising
- support for “graph as enhancement, not replacement”

### First-version recommendation

- **Adopt the principle directly**
- **Do not copy the full framework**

## 2. `ReSID`

- **arXiv**: `2602.02338`
- **Role for us**: hierarchy-aware motivation from predictability

### Borrowable modules

- prefix-conditional uncertainty perspective
- tokenizer quality tied to generative predictability
- recommendation-native view of SID construction

### Maps to our questions

- `Q1`: weak  
  It is not a graph paper.
- `Q2`: very strong  
  It gives a principled reason why different SID levels matter.
- `Q3`: medium  
  It supports objective design more than fusion mechanics.

### What we should learn

- SID levels matter because they affect predictability differently
- hierarchy-awareness must be tied to generation difficulty, not only structure aesthetics

### Best use in `MGR-SID`

- motivation for level-wise graph supervision
- justification that deeper levels may need different collaborative help

### First-version recommendation

- **Adopt as theoretical motivation**
- **Not a direct module to implement**

## 3. `PIT`

- **arXiv**: `2602.08530`
- **Role for us**: volatility warning for collaborative tokenization

### Borrowable modules

- collaborative volatility awareness
- tokenizer instability analysis
- collaborative signal alignment caution

### Maps to our questions

- `Q1`: weak  
  It does not define our graph bank.
- `Q2`: weak to medium  
  More about dynamics than SID levels.
- `Q3`: strong  
  Very useful for understanding why naive collaborative fusion fails.

### What we should learn

- collaborative information is not a clean static feature
- front-end fusion must control instability

### Best use in `MGR-SID`

- justify view-specific purification
- motivate why we need semantic anchoring and no-collapse constraints

### First-version recommendation

- **Adopt as caution and design constraint**
- **Do not follow PIT into dynamic personalized tokenization in v1**

## 4. `DiscRec`

- **arXiv**: `2506.15576`
- **Role for us**: disentangled semantic/collaborative fusion

### Borrowable modules

- semantic/collaborative disentanglement
- dual-branch modeling
- gated fusion rather than early entanglement

### Maps to our questions

- `Q1`: weak  
  It does not specify graph construction.
- `Q2`: medium  
  Gating can inspire level-wise allocation.
- `Q3`: very strong  
  One of the most useful references for how not to naively merge semantics and collaboration.

### What we should learn

- semantic and collaborative signals should not be collapsed into one space too early
- flexible fusion beats blunt unification

### Best use in `MGR-SID`

- semantic branch vs graph branch separation
- level-wise allocation as a cleaner version of gated fusion

### First-version recommendation

- **Adopt the separation principle**
- **Do not turn the whole method into an embedding-layer dual-branch baseline**

## 5. `ETEGRec`

- **arXiv**: `2409.05546`
- **Role for us**: tokenization should serve recommendation

### Borrowable modules

- tokenization and recommendation should be aligned
- tokenizer should not be treated as an isolated preprocessor

### Maps to our questions

- `Q1`: weak
- `Q2`: weak
- `Q3`: medium to strong

### What we should learn

- graph information should not only shape preprocessing artifacts
- if graph enters SID construction, it should influence tokenization learning objectives

### Best use in `MGR-SID`

- support for graph-regularized quantization
- support for objective-level integration instead of feature-only integration

### First-version recommendation

- **Adopt as integration principle**

## 6. `CoST`

- **arXiv**: `2404.14774`
- **Role for us**: relation-preserving tokenization

### Borrowable modules

- neighborhood / item relation preservation during tokenization
- quantization should not ignore recommender-specific structure

### Maps to our questions

- `Q1`: weak
- `Q2`: medium
- `Q3`: strong

### What we should learn

- tokenization can explicitly preserve useful item relations
- this is very close to our planned graph regularization idea

### Best use in `MGR-SID`

- conceptual template for `graph-regularized quantization`

### First-version recommendation

- **Adopt as one of the closest supporting precedents**

## 7. `GSPRec`

- **arXiv**: `2505.11552`
- **Role for us**: graph-scale decomposition

### Borrowable modules

- low-pass global trend view
- band-pass user-specific or medium-scale view
- sequentially informed graph construction

### Maps to our questions

- `Q1`: very strong  
  This is one of the best references for building `G_coarse`, `G_mid`, and part of `G_local`.
- `Q2`: medium  
  It helps justify why different graph scales play different roles.
- `Q3`: weak to medium  
  It is more about graph signals than SID fusion.

### What we should learn

- `G_mid` should probably be a true mid-frequency or band-pass graph view
- graph roles should be defined by structure or frequency, not by arbitrary windows

### Best use in `MGR-SID`

- direct inspiration for `G_mid`
- support for multi-resolution graph bank

### First-version recommendation

- **Adopt directly for `G_mid` candidate design**

## 8. `How Do Graph Signals Affect Recommendation`

- **arXiv**: `2512.15744`
- **Role for us**: non-uniform graph signal justification

### Borrowable modules

- low-frequency and high-frequency graph signals can both matter
- recommendation value is not restricted to one graph regime

### Maps to our questions

- `Q1`: strong
- `Q2`: medium
- `Q3`: weak

### What we should learn

- do not assume the best graph signal is always the smoothest one
- this supports our belief that deep SID levels may need different graph signal characteristics

### Best use in `MGR-SID`

- justification for having multiple graph views
- support for non-uniform level-wise graph use

### First-version recommendation

- **Use as support for design choice**
- **No need to import its full method**

## 9. `FaGSP`

- **arXiv**: `2402.08426`
- **Role for us**: graph filtering and neighborhood hierarchy

### Borrowable modules

- cascaded filters
- parallel filters
- neighborhood hierarchy through graph filters

### Maps to our questions

- `Q1`: very strong
- `Q2`: medium
- `Q3`: weak

### What we should learn

- mid-scale structure can be produced by graph operators, not just by hop count
- graph neighborhood hierarchy can be an explicit design object

### Best use in `MGR-SID`

- candidate construction for `G_mid`
- candidate operator for `G_coarse` vs `G_mid`

### First-version recommendation

- **Directly useful for `G_mid` prototype search**

## 10. `Collaboration and Transition`

- **arXiv**: `2311.01056`
- **Role for us**: coexistence of collaboration and transition

### Borrowable modules

- transition-aware embedding distillation
- explicit separation of collaborative vs transition signals

### Maps to our questions

- `Q1`: very strong
- `Q2`: weak to medium
- `Q3`: medium

### What we should learn

- local transition signal is not only sequential noise
- transition structure can calibrate broader collaborative patterns

### Best use in `MGR-SID`

- direct support for `G_local`
- support for keeping `G_local` separate from `G_coarse`

### First-version recommendation

- **Adopt as `G_local` motivation and design reference**

## 11. `TGSRec`

- **arXiv**: `2108.06625`
- **Role for us**: temporal collaborative graph construction

### Borrowable modules

- temporal graph weighting
- coexistence of sequential and collaborative signals

### Maps to our questions

- `Q1`: strong
- `Q2`: weak
- `Q3`: weak to medium

### What we should learn

- graph edges can carry temporal information
- `G_local` should probably not be a pure static count graph

### Best use in `MGR-SID`

- recency weighting for `G_local`
- time-aware edge confidence

### First-version recommendation

- **Adopt as a local-graph construction cue**

## 12. `Seq2Graph`

- **arXiv**: `2106.15814`
- **Role for us**: densifying sparse local graphs

### Borrowable modules

- sequence-to-graph augmentation
- graph-based propagation of sparse collaborative signals

### Maps to our questions

- `Q1`: strong
- `Q2`: weak
- `Q3`: weak

### What we should learn

- if `G_local` is too sparse, graph augmentation can help
- local graph quality is not only about denoising; it is also about recovering missing useful edges

### Best use in `MGR-SID`

- optional augmentation strategy when `G_local` coverage is too low

### First-version recommendation

- **Keep as backup**
- **Do not add unless plain local graph is too sparse to be useful**

## Compact mapping table

| Paper | Q1 Graph Carrier | Q2 Hierarchy-Aware | Q3 Semantic Fusion | Most Borrowable Module | First-Version Priority |
|------|---|---|---|---|---|
| `PRISM` | Low | Medium | High | collaborative denoising + semantic anchor | High |
| `ReSID` | Low | High | Medium | prefix-conditional uncertainty motivation | High |
| `PIT` | Low | Low-Med | High | volatility caution | Medium |
| `DiscRec` | Low | Medium | High | semantic/collaborative disentanglement | High |
| `ETEGRec` | Low | Low | Med-High | objective-level integration | Medium |
| `CoST` | Low | Medium | High | relation-preserving tokenization | High |
| `GSPRec` | High | Medium | Low-Med | band-pass / low-pass graph decomposition | High |
| `Graph Signals` | High | Medium | Low | multi-regime graph signal support | Medium |
| `FaGSP` | High | Medium | Low | graph filtering hierarchy | High |
| `Collaboration and Transition` | High | Low-Med | Medium | transition-aware local signal | High |
| `TGSRec` | High | Low | Low-Med | temporal graph weighting | Medium |
| `Seq2Graph` | High | Low | Low | sparse local graph augmentation | Backup |

## What we can directly assemble into `MGR-SID`

If we only borrow the most useful pieces and assemble a minimal first version, the natural combination is:

### For `Q1`

- from `GSPRec` and `FaGSP`:
  - define `G_mid` as a real mid-scale graph operator, not a hand-picked 2-hop proxy
- from `Collaboration and Transition` and `TGSRec`:
  - define `G_local` as a transition-aware temporal graph

### For `Q2`

- from `ReSID`:
  - justify why different SID levels need different treatment
- from `PRISM`:
  - preserve hierarchy while introducing collaborative structure
- from our own method:
  - implement level-wise allocation and level-wise graph regularization

### For `Q3`

- from `DiscRec`:
  - keep semantic and collaborative signals disentangled
- from `PRISM` and `PIT`:
  - denoise collaborative information before injection
- from `CoST` and `ETEGRec`:
  - make relation-preserving or recommendation-aware constraints part of tokenization learning

## Recommended first-version module stack

For a restrained and paper-friendly first implementation, the best stack is:

1. `Semantic backbone`
   - inherit MiniOneRec semantic SID construction as the base hierarchy

2. `Multiplex graph bank`
   - `G_coarse`: debiased collaborative graph
   - `G_mid`: band-pass / diffusion-residual / community-aware graph
   - `G_local`: temporal transition graph

3. `View-specific purification`
   - inspired by `PRISM` and `PIT`

4. `Level-wise graph supervision`
   - motivated by `ReSID`
   - implemented as allocation + graph regularization

5. `Semantic anti-collapse anchor`
   - inspired by `PRISM`

## What not to borrow into v1

To keep the first version focused, we should avoid borrowing:

- the full personalized dynamic tokenization route from `PIT`
- full graph-encoder competitions
- very heavy graph augmentation before we know whether plain `G_local` is sufficient
- too many graph views beyond coarse / mid / local

## Bottom line

The literature does not yet hand us a complete ready-made method.  
But it does give us a very usable module library:

- `PRISM` gives the denoising and anchoring philosophy
- `ReSID` gives the hierarchy-aware motivation
- `DiscRec` gives the semantic/collaborative separation principle
- `GSPRec` and `FaGSP` give the mid-scale graph design cues
- `Collaboration and Transition` plus `TGSRec` give the local transition graph design cues
- `CoST` and `ETEGRec` support relation-aware tokenization objectives

This is enough to justify and scaffold a solid first version of `MGR-SID`.
