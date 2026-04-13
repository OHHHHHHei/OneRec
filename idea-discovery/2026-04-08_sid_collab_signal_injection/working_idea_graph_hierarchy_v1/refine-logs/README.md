# Refine Logs

This folder keeps the active execution plan documents for the current method line.

## Current Canonical Files

- `EXPERIMENT_PLAN_TOKENIZER_V2.md`
  - current tokenizer-first plan
  - the active execution path before larger ablations

- `EXPERIMENT_TRACKER_TOKENIZER_V2.md`
  - run-by-run tracker for the tokenizer-first line

- `EXPERIMENT_PLAN_STAGE2_RETENTION.md`
  - next-stage plan after the first full `v2 -> SFT -> RL` confirmation
  - focuses on closing the remaining `top5/top10` retention gap

- `EXPERIMENT_TRACKER_STAGE2_RETENTION.md`
  - execution tracker for the retention-targeted refinement stage
  - this is the current active execution path
  - current readout:
    - `R202a` is the best tokenizer-side branch so far
    - `R205` is a completed negative result in its current KL form
    - `R208` completed, but did not beat current `v2_on_p05` downstream

## Reading Policy

Read these only after:

1. `../00_ACTIVE_CONTEXT.md`
2. `../18_mgr_sid_v2_ambiguity_aware_method.md`

These tracker files are execution-oriented.
They are not the best entry point for understanding the method itself.
