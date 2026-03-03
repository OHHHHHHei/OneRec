# Audit Report v3

## Scope

This note summarizes the main engineering issues identified during the refactor and the current
cleanup posture of the repository.

## Confirmed issues that were addressed in the refactored mainline

- Dataset base processing now filters `None` samples.
- Unsafe `eval` usage in the mainline path was replaced with safe parsing.
- The SFT freeze-LLM path was corrected so the original vocabulary boundary is tracked.
- RL semantic reward handling no longer raises on invalid generated items.
- The mainline no longer depends on hard-coded one-off paths for its primary entrypoints.

## Remaining operational constraints

- Full training is not validated in this local environment.
- Remote-server smoke runs are still required for SFT, RL, and evaluation.
- Legacy wrappers remain for compatibility and should eventually be phased out after migration.

## Repository hygiene

- Mainline code lives in `src/minionerec`.
- Experimental and historical code should stay under `archive/`.
- Large datasets and generated outputs should stay out of version control.
