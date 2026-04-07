# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | Reproduce offline pilot | Baseline vs `cf_top_l2` rerank on Industrial | Industrial test | top-1 hit, same-prefix subgroup hit | MUST | TODO | Should match current notebook/script numbers |
| R002 | M0 | Reproduce offline pilot | Baseline vs `cf_top_l2` rerank on Office | Office test | top-1 hit, same-prefix subgroup hit | MUST | TODO | Confirm smaller but non-negative gain |
| R003 | M1 | Sanity integration | Static ambiguity profiler only | Industrial valid/test | subgroup metrics | MUST | TODO | Verify no leakage |
| R004 | M1 | Heuristic integration | Static local leaf bias in eval path | Industrial test | HR/NDCG, same-prefix errors | MUST | TODO | Should approximate offline rerank |
| R005 | M2 | Main method run | ACLR training-only on best SFT recipe | Industrial | HR/NDCG, same-prefix errors | MUST | TODO | First real training gate |
| R006 | M3 | Transfer check | ACLR training-only on best SFT recipe | Office | HR/NDCG, same-prefix errors | MUST | TODO | Check dataset generality |
| R007 | M3 | Strongest pipeline check | ACLR full on best RL path | Industrial | HR/NDCG, same-prefix errors | MUST | TODO | Only after R005 is positive |
| R008 | M4 | Novelty isolation | ACLR full vs gate-off global bias | Industrial | HR/NDCG, ambiguity-bucket gains | MUST | TODO | Defends local selective story |
| R009 | M4 | Mechanism isolation | ACLR inference-only vs training-only vs full | Industrial | HR/NDCG, subgroup rescue | MUST | TODO | Distinguish heuristic from learned gain |
| R010 | M4 | Simplicity check | Simple residual vs heavier residual | Industrial | metrics + compute | NICE | TODO | Only if core method is already positive |
