# Experiment Plan

**Problem**: Stage-2 showed that tokenizer-side structural gains do not
automatically turn into stronger downstream ranking.  
**Method Thesis**: The missing axis is not only structure, but also SID-space
stability, prefix-conditioned semantic consistency, and downstream
learnability.  
**Date**: 2026-04-13

## Terminology Calibration

In this document, `transfer` or `interface` does **not** mean preserving the
ability of a previous SFT checkpoint.

Each SFT run is retrained from the base model.

So the real question is:

> does a new SID space become easier or harder for a freshly trained downstream model to learn?

All uses of `transfer cost`, `interface problem`, or `rearrangement cost` in
this plan should be read in that sense.

Also, these diagnostics are **not** the final selection rule for tokenizer
variants.

They exist to explain why a tokenizer branch may win or lose downstream.
Final selection should still be made by full `SFT -> evaluate`, not by
diagnostic stability scores alone.

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|-----------------|-----------------------------|---------------|
| C1 | The current stage-2 failure is largely a SID target-space learnability problem rather than a tokenizer-structure failure | A tokenizer branch with stronger structure but worse downstream can be shown to have high SID rearrangement or lower prefix stability | B1, B4 |
| C2 | Code-level semantic consistency and downstream learnability are missing diagnostic axes | We can measure token semantic drift / polysemy and show they vary meaningfully across tokenizer variants | B2, B3 |

## Paper Storyline

- Main paper does not need all of these diagnostics immediately.
- But the next research move should be informed by them.
- These blocks are primarily for internal decision-making and mechanism clarity.

## Experiment Blocks

### Block 1: Prefix Stability / SID Rearrangement

- Claim tested:
  - tokenizer variants that change too much of the SID space incur a downstream learnability cost
- Why this block exists:
  - current evidence strongly suggests `R202a` helped hard cases but hurt many already-stable examples because the SID space changed too broadly
- Dataset / split / task:
  - Industrial tokenizer outputs and corresponding generated indices
- Compared systems:
  - `current v2`
  - `R202a`
  - `R202b-r075`
  - `R205`
- Metrics:
  - changed `l1` rate
  - changed `l2` rate
  - changed full-SID rate
  - l1-prefix neighbor overlap / Jaccard
  - l2-prefix neighbor overlap / Jaccard
- Setup details:
  - pure analysis on generated `index.json`
- Success criterion:
  - we can quantify whether `R202a` introduces large prefix rearrangement relative to `current v2`
- Failure interpretation:
  - if rearrangement is small, then the downstream learnability problem must come more from code semantics or downstream training dynamics
- Table / figure target:
  - one table + one histogram / overlap summary
- Priority:
  - MUST-RUN

### Block 2: Code Polysemy / Semantic Consistency

- Claim tested:
  - code tokens differ in semantic stability, and excessive prefix-conditioned polysemy may hurt downstream learnability
- Why this block exists:
  - our current diagnostics do not measure whether shared code tokens are semantically coherent
- Dataset / split / task:
  - item semantic embeddings + generated SID indices on Industrial
- Compared systems:
  - `current v2`
  - `R202a`
  - optionally strongest original semantic SID
- Metrics:
  - per-token semantic spread for `a`, `b`, `c`
  - prefix-conditioned drift for `b` and `c`
  - token reuse count vs semantic variance
- Setup details:
  - for each token, collect covered items and compute embedding variance
  - for `b` and `c`, group by parent prefix and compare semantic centers
- Success criterion:
  - identify whether structural refinements increase or decrease token semantic overload
- Failure interpretation:
  - if token consistency barely changes, then semantic overload is unlikely the main cause of downstream loss
- Table / figure target:
  - per-level token-consistency table, a few illustrative tokens
- Priority:
  - MUST-RUN

### Block 3: SID Learnability Probes

- Claim tested:
  - some SID spaces are structurally clean but harder for the downstream model to learn
- Why this block exists:
  - we need a lighter-weight way to estimate downstream compatibility before full SFT/RL
- Dataset / split / task:
  - Industrial
  - train/valid probe built from history and target SID labels
- Compared systems:
  - `current v2`
  - `R202a`
  - optionally strongest original SID
- Metrics:
  - level-wise predictability (`a`, `b`, `c`)
  - prefix-to-leaf predictability
  - next-level conditional accuracy
- Setup details:
  - lightweight probe, not full LLM
  - can start with simple classifier / shallow decoder over history-side features
- Success criterion:
  - detect whether `R202a` improves hard-case local predictability while degrading global/prefix learnability
- Failure interpretation:
  - if probes do not separate variants, then transfer loss may be more decode- or objective-related
- Table / figure target:
  - per-level probe performance table
- Priority:
  - NICE-TO-HAVE but high value

### Block 4: Structure-to-Downstream Transfer Attribution

- Claim tested:
  - tokenizer-side structural improvements help only a subset of examples, while a different subset is harmed because old routing structure is broken
- Why this block exists:
  - this is the working explanation for `R208`
- Dataset / split / task:
  - Industrial test set
- Compared systems:
  - `current v2_on_p05 SFT`
  - `R208`
- Metrics:
  - per-item SID change indicators
  - per-item downstream rank delta
  - relation between SID-prefix changes and `improved / worsened`
- Setup details:
  - join:
    - old SID
    - new SID
    - prefix stability
    - top-k migration
- Success criterion:
  - establish whether large prefix change correlates with worsened downstream examples
- Failure interpretation:
  - if worsened examples are not associated with large SID changes, then the issue may be more about token semantics than rearrangement
- Table / figure target:
  - improved vs worsened attribution table
- Priority:
  - MUST-RUN

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M1 | quantify SID rearrangement | `R301` | if prefix stability is clearly low, prioritize conservative tokenizer updates | low | interpretation ambiguity |
| M2 | measure token semantic consistency | `R302` | if code polysemy is high, prioritize interface-friendly tokenizer updates | low | token stats may be noisy |
| M3 | connect structure to downstream loss | `R303` | if worsened examples align with prefix changes, avoid full SID rearrangement in next tokenizer round | low | requires careful joins |
| M4 | optional learnability probe | `R304` | if probes separate variants, use them as pre-screen before new SFT runs | medium | probe design may itself be noisy |

## Compute and Data Budget

- Total estimated GPU-hours:
  - near-zero for `R301/R302/R303`
  - small for `R304`
- Data preparation needs:
  - reuse existing Industrial outputs, SFT test results, and index files
- Biggest bottleneck:
  - analysis implementation quality, not compute

## Risks and Mitigations

- Risk:
  - diagnostics remain descriptive and do not isolate causal factors
  - Mitigation:
    - always compare improved vs worsened sets

- Risk:
  - token semantic variance is difficult to interpret alone
  - Mitigation:
    - report it jointly with token reuse count and prefix-conditioned drift

- Risk:
  - learnability probe overfits or does not correlate with full SFT
  - Mitigation:
    - treat probe as supporting evidence, not final gate

## Final Checklist

- [x] Prefix stability quantified
- [x] Code polysemy measured
- [x] Structure-to-downstream transfer attribution completed
- [x] Optional learnability probe executed
- [x] Next tokenizer step informed by interface diagnostics, not structure alone
