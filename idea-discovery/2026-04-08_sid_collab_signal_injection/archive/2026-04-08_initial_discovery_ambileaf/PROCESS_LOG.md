# Process Log

**Date**: 2026-04-08  
**Goal**: Run an isolated idea-discovery pass for collaborative signal injection in SID-based generative recommendation.

## Inputs used in this round

- Current conversation context only:
  - collision exists but is not the dominant bottleneck
  - current SID construction is text-driven and lacks direct collaborative injection
  - many failures are `prefix correct, leaf wrong`
  - prefix semantics are not stable enough for strong leaf predictability
- Current repo evidence only:
  - `results/v05_r1_industrial/summary.json`
  - `results/collaborative_diagnostics/industrial_best_summary.json`
  - `results/collaborative_diagnostics/office_best_summary.json`
  - `data/Amazon/index/Industrial_and_Scientific.index.json`
  - `data/Amazon/index/Industrial_and_Scientific.v05_e1.index.json`
  - `data/Amazon/index/Industrial_and_Scientific.v05_c1_shuffled.index.json`
  - `logs/sid_train_industrial_tdcf_v05_e1_20260406_222051.log`
  - `logs/sid_train_industrial_tdcf_v05_c1_20260406_222148.log`
- Local papers in `papers/`
- Fresh web novelty search on recent arXiv papers

## Deliberate scope control

- This run was written into a brand-new folder to keep it separate from older discovery outputs.
- The goal here is not to defend a preselected answer such as `selective`, but to compare front-end, back-end, and hybrid directions on equal footing.
- No new GPU pilot was launched in this discovery pass. Ranking is based on current repo evidence plus literature/novelty analysis. Any idea that still needs empirical validation is marked as such.

## Core evidence snapshot

- Baseline Industrial SID collision rate is low: `0.00434`
- Naive front-end global fusion collapses SID structure:
  - `text + cf`: `0.74037`
  - `text + shuffled-cf`: `0.62832`
- Collaborative signal is especially useful in same-prefix local errors:
  - Industrial `same_l2` best-case target-better rate: `0.2284`
  - Office `same_l2` best-case target-better rate: `0.4048`
- ACLR-lite gains show both truths at once:
  - global total gain is largest
  - local/ambiguity modes have higher gain per activated sample

