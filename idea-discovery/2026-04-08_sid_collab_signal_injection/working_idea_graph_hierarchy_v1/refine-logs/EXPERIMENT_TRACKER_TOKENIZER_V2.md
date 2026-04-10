# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| `R001` | `M0` | proxy freeze | offline combined ambiguity prior | Industrial analysis | hard/easy separation, hard-item enrichment | MUST | REUSE | usable for first `v2` run |
| `R002` | `M0` | proxy freeze | offline + online ambiguity | Industrial analysis | same as `R001` | MUST | REUSE | currently not recommended |
| `R003` | `M1` | tokenizer anchor | MiniOneRec baseline | Industrial | final SID collision, `H(level3|level1,l2)`, target `l2` fanout | MUST | REUSE | canonical baseline |
| `R004` | `M1` | tokenizer anchor | `v1 hierarchy_reg` | Industrial | same as `R003` | MUST | REUSE | current best graph-aware tokenizer reference |
| `R005` | `M1` | tokenizer main run | `v2 full` with offline combined ambiguity prior only | Industrial | same as `R003` | MUST | TODO | first real `v2` decision run |
| `R006` | `M1` | final SID generation | `v2 full` `sid-generate` | Industrial | final collision and generated index stats | MUST | TODO | paired with `R005` |
| `R007` | `M2` | tokenizer diagnosis | baseline vs `v1` vs `v2` local ambiguity comparison | Industrial | target `l2` leaf count, multi-leaf rates, weighted entropy | MUST | TODO | run immediately after `R006` |
| `R008` | `M3` | deferred ablation | no semantic-structure retention | Industrial tokenizer | tokenizer-side metrics only | DEFERRED | TODO | only after `R005-R007` are positive |
| `R009` | `M3` | deferred ablation | uniform weighting instead of ambiguity-aware weighting | Industrial tokenizer | tokenizer-side metrics only | DEFERRED | TODO | only after `R005-R007` are positive |
| `R010` | `M3` | deferred ablation | alternative ambiguity prior variants | Industrial tokenizer | tokenizer-side metrics only | DEFERRED | TODO | only after `R005-R007` are positive |

## Immediate Launch Order

1. `R005`: launch the first tokenizer `v2` run with the offline combined ambiguity prior only.
2. `R006`: generate final SID from the `R005` best checkpoint.
3. `R007`: run local ambiguity analysis against the MiniOneRec baseline and current `v1`.

## Stop / Go Rules

- If `R005-R006` do not beat the MiniOneRec baseline tokenizer-side, stop and revise the weighting design before any downstream experiments.
- If `R005-R006` improve collision but `R007` shows no local ambiguity benefit, stop and revisit the proxy-to-weight mapping.
- Do not launch `R008-R010` until `R005-R007` establish a clear tokenizer-side win.
