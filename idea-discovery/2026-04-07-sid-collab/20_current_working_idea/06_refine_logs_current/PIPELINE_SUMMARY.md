# Pipeline Summary

**Problem**: MiniOneRec's remaining errors are concentrated in local same-prefix SID ambiguity rather than global collision or total tokenizer failure.
**Final Method Thesis**: Add a train-only collaborative residual only for ambiguous `(a,b)` prefixes so third-token SID prediction becomes behavior-aware exactly where the current model is still brittle.
**Final Verdict**: READY
**Date**: 2026-04-06

## Final Deliverables

- Proposal: `FINAL_PROPOSAL.md`
- Review summary: `REVIEW_SUMMARY.md`
- Experiment plan: `EXPERIMENT_PLAN.md`
- Experiment tracker: `EXPERIMENT_TRACKER.md`

## Contribution Snapshot

- **Dominant contribution**: ambiguity-aware collaborative residualization of the leaf SID token
- **Optional supporting contribution**: diagnostic-driven ambiguity profiler for selective activation
- **Explicitly rejected complexity**: full collaborative tokenizer rebuild, graph-heavy encoder, RL redesign

## Must-Prove Claims

- ACLR improves recommendation by reducing local same-prefix leaf mistakes.
- Selective local activation is better than generic global collaborative bias.

## First Runs To Launch

1. Reproduce the `cf_top_l2` rerank pilot and archive the exact outputs.
2. Integrate the static ambiguity-aware leaf bias into evaluation.
3. Train the Industrial SFT ACLR variant with the leaf-level auxiliary loss.

## Main Risks

- **Risk**: gains are too small after training-time integration
  - **Mitigation**: keep heuristic-only and training-only ablations separate
- **Risk**: the method looks like reranking only
  - **Mitigation**: show the local training loss matters
- **Risk**: Office gains stay small
  - **Mitigation**: anchor the paper on Industrial and treat Office as transfer confirmation

## Next Action

- Proceed to `/run-experiment`
