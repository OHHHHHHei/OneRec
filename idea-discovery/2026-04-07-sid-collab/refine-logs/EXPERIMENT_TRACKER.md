# EXPERIMENT TRACKER

Date: 2026-04-07

## Status legend

- `todo`
- `running`
- `done`
- `stopped`

## Tracking table

| ID | Experiment | Dataset | Status | Key question | Notes |
|---|---|---|---|---|---|
| E0-1 | Freeze baseline and current local rerank numbers | Industrial | done | What is the current reference point? | baseline `0.073241`, global `0.083830`, same_l2 `0.081624` |
| E0-2 | Freeze baseline and current local rerank numbers | Office | done | What is the current reference point? | baseline `0.085697`, global `0.088368`, same_l2 `0.087957`, ambiguity_l2 regresses |
| E1-1 | Design calibrated ambiguity score | Industrial | todo | Can we beat current `ambiguity_l2` clearly? | Start from leaf count + beam flatness + local density |
| E1-2 | Validate calibrated ambiguity score | Industrial | todo | Can we match or beat `same_l2` without near-global activation? | Must stay selective |
| E2-1 | Candidate-set ablation | Industrial | todo | Which local subset definition works best? | Compare same_l1 / same_l2 / dynamic subset |
| E3-1 | Collaborative score ablation | Industrial | todo | What is the minimal effective score? | Best vs mean vs normalized |
| E1-3 | Transfer calibrated score | Office | todo | Does the trigger generalize? | Run only if Industrial is positive |
| E4-1 | Optional training consistency objective | Industrial | todo | Can we make the story less post-hoc? | Run only if earlier blocks pass |

## Current decision

Current evidence supports continuing with `E1-1` first. No long tokenizer training should be launched yet.
