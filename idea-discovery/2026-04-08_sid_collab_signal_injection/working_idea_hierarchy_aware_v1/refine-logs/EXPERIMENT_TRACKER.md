# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| `R001` | `M0` | Revalidate coarse/mid/local premise | train-only probe: `coarse10` / `mid3` / `local2` / `fine1` | Industrial | target-better, same-`l2`, coverage | MUST | TODO | No retraining |
| `R002` | `M0` | Revalidate coarse/mid/local premise | train-only probe: `coarse10` / `mid3` / `local2` / `fine1` | Office | target-better, same-`l2`, coverage | MUST | TODO | No retraining |
| `R003` | `M1` | Tokenizer health check | 3-view builder + purification cache | Industrial | cache stats, view sparsity | MUST | TODO | Ensure view construction is stable |
| `R004` | `M1` | Tokenizer health check | `MRC-SID` forward / SID generation smoke test | Industrial | collision, code usage, runtime | MUST | TODO | Stop if collapse appears |
| `R005` | `M2` | Baseline anchor | `S0` semantic-only baseline reproduction | Industrial | HR@1, NDCG@10, same-`l2` | MUST | TODO | Reproduce current anchor |
| `R006` | `M2` | Parameter-matched baseline | `S2` uniform all-view fusion | Industrial | HR@1, NDCG@10, collision | MUST | TODO | Same view bank as `MRC-SID` |
| `R007` | `M2` | Heuristic control | `S3` fixed `coarse/mid/local` assignment | Industrial | HR@1, NDCG@10, collision | MUST | TODO | Manual per-level mapping |
| `R008` | `M2` | Main method | `S4` `MRC-SID` | Industrial | HR@1, NDCG@10, same-`l2`, gates | MUST | TODO | 1 seed first |
| `R009` | `M3` | Novelty isolation | `S5` swapped allocation control | Industrial | HR@1, NDCG@10, gates | MUST | TODO | Stress-test hierarchy claim |
| `R010` | `M3` | Purification necessity | `S6` no-purification `MRC-SID` | Industrial | HR@1, NDCG@10, collision | MUST | TODO | Same architecture, no purification |
| `R011` | `M3` | Interpretation | gate visualization / per-level weight summary | Industrial | gate weights, entropy | MUST | TODO | Needed for paper figures |
| `R012` | `M4` | Generalization | `S2` uniform all-view fusion | Office | HR@1, NDCG@10, same-`l2` | MUST | TODO | Run only after Industrial go |
| `R013` | `M4` | Generalization | `S3` fixed assignment | Office | HR@1, NDCG@10, same-`l2` | MUST | TODO | Run only after Industrial go |
| `R014` | `M4` | Generalization | `S4` `MRC-SID` | Office | HR@1, NDCG@10, same-`l2`, gates | MUST | TODO | Run only after Industrial go |
| `R015` | `M4` | Robustness | `S4` second/third seeds | Industrial | mean/std HR@1, NDCG@10 | NICE | TODO | Launch only if `R008` is positive |
| `R016` | `M4` | Robustness | `S4` second/third seeds | Office | mean/std HR@1, NDCG@10 | NICE | TODO | Launch only if `R014` is positive |
| `R017` | `M4` | Simplicity stress test | `S7` overbuilt ambiguity-aware extension | Industrial | HR@1, NDCG@10 | NICE | TODO | Only if core method already works |

