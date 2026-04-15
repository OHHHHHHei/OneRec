# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| `R201` | `M0` | freeze diagnosis | reuse `v2_on_p05` SFT/RL analyses | Industrial | top-k, same-prefix, collided-target hit | MUST | REUSE | current gap is `top5/top10` retention |
| `R202a` | `M1` | tokenizer micro-fix | `v2 + stop-grad hierarchy isolation` train | Industrial tokenizer | final SID collision, local ambiguity metrics | MUST | COMPLETED | healthy branch, best collision `0.10065`, forward to `R203/R204` |
| `R202b` | `M1` | tokenizer micro-fix | `v2 + stop-grad + level-1 compensation` train | Industrial tokenizer | final SID collision, local ambiguity metrics | MUST | REGRESSED | compensation branch failed badly, best collision `0.95849`; keep as archived branch for smaller-coefficient retry later |
| `R202b-r075` | `M1` | tokenizer micro-fix | `v2 + stop-grad + smaller level-1 compensation` train | Industrial tokenizer | final SID collision, local ambiguity metrics | CONDITIONAL | COMPLETED | stable retry: best train collision `0.11150`, final generated collision `12/3686`; however structure is clearly worse than `R202a`, so keep as archived tradeoff branch |
| `R203` | `M1` | tokenizer generation | `R202a` -> `sid-generate` | Industrial tokenizer | final generated SID stats | MUST | COMPLETED | generated collision `0.0035268584`, max conflict `2` |
| `R204` | `M1` | tokenizer diagnosis | current `v2` vs `R202a` | Industrial tokenizer | `l2` fanout, multi-leaf same-`l2`, weighted entropy | MUST | COMPLETED | mixed-but-promising: mean target `l2` leaf count `4.3422 -> 3.6148`, target entropy `1.1001 -> 1.0308`, deep crowded `l2>=4` improves, but multi-leaf `same_l2` slightly worsens |
| `R205` | `M2` | tokenizer micro-fix | block-2 best + semantic retention KL train | Industrial tokenizer | same as `R202a` | MUST | COMPLETED | finished cleanly, but underperformed `R202a`: best train collision `0.11367` at epoch `1149`, final collision `0.11557`; keep as completed negative result unless later retuned |
| `R205b` | `M2` | fallback tokenizer micro-fix | current `v2` + semantic retention KL train | Industrial tokenizer | same as `R202a` | MUST | READY | config prepared, hold unless stop-grad branch proves unstable |
| `R206` | `M2` | tokenizer generation | best of `R205/R205b` -> `sid-generate` | Industrial tokenizer | final generated SID stats | MUST | COMPLETED | `R205` generated collision improves to `12/3686`, but this does not survive structure diagnosis |
| `R207` | `M2` | tokenizer diagnosis | current `v2` vs best of `R205/R205b` | Industrial tokenizer | same as `R204` | MUST | COMPLETED | `R205` regresses strongly after generate: mean target `l2` leaf count `4.3422 -> 4.9572`, entropy `1.1001 -> 1.2623`; do not push semantic-retention branch downstream in current form |
| `R208` | `M3` | downstream screen | block-2 best -> `SFT/evaluate` on `title_on + desc_p05` | Industrial | `HR@10`, `NDCG@10`, top-k diagnostics | MUST | COMPLETED | `R202a` does not beat current `v2_on_p05`: `NDCG@10 = 0.09974`, `HR@10 = 0.14251`; structural gain did not survive downstream strongly enough |
| `R209` | `M3` | downstream screen | block-3 best -> `SFT/evaluate` on `title_on + desc_p05` | Industrial | `HR@10`, `NDCG@10`, top-k diagnostics | MUST | HOLD | do not launch in current form because `R205` is a negative tokenizer result after `R206/R207` |
| `R209s` | `M3` | seed confirmation | best of `R208/R209` with second seed, only if gain is small | Industrial | `HR@10`, `NDCG@10` | CONDITIONAL | HOLD | no stage-2 downstream winner exists yet |
| `R210` | `M4` | RL confirmation | best of `R208/R209` (or `R209s`) -> `RL/evaluate` | Industrial | `HR@10`, `NDCG@10`, collided-target hit | MUST | HOLD | gate not met: no stage-2 SFT candidate beats current `v2_on_p05` |

## Immediate Launch Order

1. `R202a -> R203/R204`: carry the winning stop-grad branch forward.
2. `R205/R205b -> R206/R207`: implement and test semantic-retention KL, with a fallback branch that does not depend on stop-grad succeeding.
3. `R208/R209`: downstream SFT screen on the fixed `title_history2sid_on + desc_align_p05` recipe.

## Stop / Go Rules

- `R202a` is the only surviving block-2 branch. Do not push `R202b` downstream unless it is restarted later with a smaller compensation coefficient and shows recovery.
- If `R205/R205b` do not preserve the current ambiguity-cleanup behavior, do not continue the semantic-retention branch downstream.
- If neither `R208` nor `R209` beats current `v2_on_p05 SFT` on both `HR@10` and `NDCG@10`, stop and do not launch new RL.
- If the best SFT gain is only small, run `R209s` before RL.
- Only launch `R210` for a candidate that clearly improves the current `v2_on_p05 SFT`.
