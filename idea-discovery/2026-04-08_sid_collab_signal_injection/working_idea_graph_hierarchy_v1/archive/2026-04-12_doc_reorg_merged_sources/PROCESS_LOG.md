# Process Log

**Date**: 2026-04-09  
**Trigger**: continue `idea-discovery` under the direction of graph structure + denoising + hierarchy-aware collaborative information

## Goal

Use a fresh discovery round to answer a sharper question than the previous `MRC-SID` iteration:

> if collaborative information should enter SID tokenization hierarchically, should it enter as level-wise graph structure rather than as a set of fused feature views?

## Inputs used

- current direction note: `../RESEARCH_DIRECTION.md`
- previous working idea: `../archive/2026-04-08_working_idea_hierarchy_aware_v1_superseded/`
- repo evidence:
  - `results/v05_r1_industrial/summary.json`
  - `results/collaborative_diagnostics/industrial_best_summary.json`
  - `results/collaborative_diagnostics/office_best_summary.json`
  - `data/Amazon/index/*.index.json`
- local paper library:
  - `papers/PRISM.pdf`
  - `papers/ReSID.pdf`
  - `papers/PIT.pdf`
  - `papers/ETEGRec.pdf`
  - `papers/TokenRec.pdf`
  - `papers/UTGRec.pdf`
- additional online literature searches around graph signal processing, graph denoising, and multi-scale recommendation

## Main pivot in this round

The earlier `MRC-SID` direction was:

- multiple collaborative views
- level-wise learned allocation

This round asks whether that is still too weak as a paper story.

The key shift is:

- from `feature fusion`
- to `graph-regularized hierarchical quantization`

## Working conclusion

The strongest balanced direction in this round is not:

- post-hoc refinement
- simple graph feature injection
- full graph-defined tokenization from scratch

It is:

**`MGR-SID`: Multiplex Graph-Regularized Hierarchical Semantic IDs**

Core thesis:

> collaborative information should supervise SID learning through level-specific graph structure preservation, not only through globally fused embeddings.

## Important evidence collected

### Existing repo evidence

- naive front-end collaborative fusion can collapse the SID structure badly
- collision is not the main bottleneck
- errors cluster in local same-prefix leaf ambiguity

### New lightweight graph pilot

Using train-only graph proxies:

- `coarse_graph` is strongest globally
- `local_trans` is sparse but can be strongest on the deepest ambiguity bucket
- naive `mid_graph` as plain 2-hop diffusion is not good enough

This suggests that graph structure matters, but `mid-scale` collaborative modeling must be designed carefully rather than approximated by a hand-picked window or diffusion depth.
