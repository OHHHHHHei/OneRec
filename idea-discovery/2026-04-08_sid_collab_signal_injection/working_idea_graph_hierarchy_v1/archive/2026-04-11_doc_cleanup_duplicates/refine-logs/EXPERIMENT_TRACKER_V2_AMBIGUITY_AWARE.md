# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| `R001` | `M0` | minimal proxy sanity | `P1` offline combined ambiguity prior | Industrial train-only | separation of hard local ambiguity vs easy stable items | MUST | DONE | borderline-positive: `AUC=0.7187`, top-10% hard rate `0.6141`; use for first `v2` run |
| `R002` | `M0` | minimal proxy sanity | `P2` offline + online ambiguity | Industrial tokenizer | same as `R001` | MUST | DONE | negative in current form: online uncertainty weakens separation and increases worsened@3 concentration |
| `R007` | `M1` | tokenizer anchor | `T0` MiniOneRec baseline | Industrial | final SID collision / weighted `H(l3|l1,l2)` / target `l2` fanout | MUST | REUSE | already available canonical baseline |
| `R008` | `M1` | tokenizer anchor | `T1` current `v1 hierarchy_reg` | Industrial | same as `R007` | MUST | REUSE | already available v1 reference |
| `R009` | `M1` | tokenizer main result | `T2` `v2 full` ambiguity-aware + semantic-structure retention | Industrial | same as `R007` | MUST | TODO | first real decision run |
| `R011` | `M2` | downstream anchor | `S0` MiniOneRec baseline | Industrial | `NDCG@3/10`, `HR@3/10`, top-k structural metrics | MUST | REUSE | already available canonical baseline |
| `R012` | `M2` | downstream main result | `S1` `v2 full` | Industrial | same as `R011` + broken/fixed counts | MUST | TODO | only if `R009` is positive |
| `R013` | `M3` | ablation | no online uncertainty | Industrial tokenizer | final SID metrics + hard-case buckets | DEFERRED | TODO | only after baseline is beaten |
| `R014` | `M3` | ablation | no offline ambiguity prior | Industrial tokenizer | same as `R013` | DEFERRED | TODO | only after baseline is beaten |
| `R015` | `M3` | ablation | no semantic-structure retention | Industrial tokenizer | same as `R013` + over-correction markers | DEFERRED | TODO | only after baseline is beaten |
| `R016` | `M4` | tokenizer generalization | Office `T0/T2` | Office | tokenizer-side metrics only | DEFERRED | TODO | only if Industrial is clearly positive |
| `R017` | `M4` | downstream generalization | Office `S0/S1` | Office | `NDCG@3/10`, `HR@3/10` | DEFERRED | TODO | optional appendix support |

## Immediate Launch Order

1. `R001-R002`: do the lightest possible proxy sanity so we are not training with a blind ambiguity prior.
2. `R009`: run tokenizer-only `v2` and compare it directly to the MiniOneRec baseline and current `v1`.
3. `R012`: only after tokenizer-side evidence is positive, launch the first downstream `SFT -> evaluate` run against the MiniOneRec baseline.

## Stop / Go Rules

- If `R001-R002` show no usable proxy signal, pause `v2` and redesign the ambiguity prior before training.
- If `R009` does not beat the MiniOneRec baseline tokenizer-side, do not launch `R012`.
- If `R012` still cannot beat the MiniOneRec baseline downstream, pause ablations and revisit the core transfer design first.
