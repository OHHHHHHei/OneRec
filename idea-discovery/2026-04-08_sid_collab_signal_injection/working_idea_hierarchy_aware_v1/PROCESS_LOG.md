# Process Log

**Date**: 2026-04-08  
**Direction**: Hierarchy-Aware Collaborative Signal Fusion for SID-based Generative Recommendation

## Scope

This round follows the `idea-discovery` workflow, but is grounded in the already-fixed direction document:

- `../RESEARCH_DIRECTION.md`

The goal is not to revisit whether the direction is meaningful, but to refine it into:

- concrete method candidates
- a sharper top idea
- a claim-driven experiment plan
- multiple rounds of critical review

## Inputs used

### Repo evidence

- `results/v05_r1_industrial/summary.json`
- `results/collaborative_diagnostics/industrial_best_summary.json`
- `results/collaborative_diagnostics/office_best_summary.json`
- current baseline / v0.5 evidence already established in this repo

### Local papers

- `papers/PRISM.pdf`
- `papers/ReSID.pdf`
- `papers/PIT.pdf`
- `papers/ETEGRec.pdf`
- `papers/UTGRec.pdf`
- `papers/DAS.pdf`
- `papers/Unified Multi-Level Alignment for LLM-based Generative.pdf`
- `papers/LCRec.pdf`
- `papers/HiD-VAE.pdf`
- `papers/TokenRec.pdf`
- `papers/VQ-Rec.pdf`

### Online search focus

- front-end collaborative tokenization
- denoising / collapse prevention
- multi-granularity collaborative signals
- disentangled semantic vs collaborative modeling
- graph-signal / short-vs-long range recommendation evidence

## Additional lightweight pilot done in this round

A train-only collaborative probe was built from the existing train/test/result files to compare different collaborative views:

- `coarse10`: full history collaborative compatibility
- `mid3`: recent-3 collaborative compatibility
- `local2`: recent-2 collaborative compatibility
- `fine1`: last-item transition compatibility

This was used as a low-cost pilot to test whether different collaborative granularities help different error buckets.

