# Pipeline Summary

This idea-discovery pass started from three motivations around MiniOneRec SIDs:

1. missing collaborative information
2. unstable prefix semantics
3. SID collision

After checking the repo and the latest arXiv literature, the main conclusion is:

- all three motivations are real
- but the strongest repo bottleneck is local same-prefix ambiguity, not raw collision
- and the latest literature is already crowded on global collaborative tokenizer redesign

The recommended direction is:

- `ACLR-Plus`: Ambiguity-Calibrated Local Collaborative Repair

One lightweight local pilot already supports the direction:

- collaborative reranking improves both Industrial and Office
- local-prefix-constrained rerank also helps
- the current ambiguity trigger is too crude, which motivates a stronger calibrated trigger

Next action:

- implement and test a better ambiguity score before doing any expensive retraining
