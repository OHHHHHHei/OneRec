# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| `R001` | `M0` | choose `G_mid` | `G_mid = diffusion residual` probe | Industrial | same-`l2`, same-`l1`, coverage | MUST | TODO | compare raw vs purified |
| `R002` | `M0` | choose `G_mid` | `G_mid = band-pass` probe | Industrial | same-`l2`, same-`l1`, coverage | MUST | TODO | primary candidate from `GSPRec` |
| `R003` | `M0` | choose `G_mid` | `G_mid = community-aware` probe | Industrial | same-`l2`, same-`l1`, coverage | MUST | TODO | primary candidate from graph clustering view |
| `R004` | `M0` | tie-break if needed | best 2 `G_mid` candidates | Office | same-`l2`, coverage | NICE | TODO | only if Industrial inconclusive |
| `R005` | `M1` | purification sanity | raw vs purified graph bank diagnostics | Industrial | degree skew, popularity corr, coverage | MUST | TODO | cache outputs for reuse |
| `R006` | `M1` | pipeline sanity | graph cache + bucket scripts validation | Industrial | reproducibility, bucket counts | MUST | TODO | CPU-heavy |
| `R007` | `M2` | baseline anchor | `S0` semantic-only MiniOneRec | Industrial | HR@1, NDCG@10, collision | MUST | TODO | reuse current best settings |
| `R008` | `M2` | baseline anchor | `S1` graph feature fusion | Industrial | HR@1, NDCG@10, collision | MUST | TODO | parameter-match as closely as possible |
| `R009` | `M2` | baseline anchor | `S2` uniform graph regularization | Industrial | HR@1, NDCG@10, collision | MUST | TODO | same graph bank as `S3` |
| `R010` | `M2` | main method | `S3` `MGR-SID` | Industrial | HR@1, NDCG@10, same-`l2` | MUST | TODO | 1 seed decision run |
| `R011` | `M3` | novelty isolation | `S4` fixed level assignment | Industrial | HR@1, NDCG@10, same-`l2` | MUST | TODO | coarse/mid/local hard mapping |
| `R012` | `M3` | novelty isolation | `S5` swapped assignment | Industrial | HR@1, NDCG@10, same-`l2` | MUST | TODO | intentionally bad mapping |
| `R013` | `M3` | stability ablation | `S7` no-denoising `MGR-SID` | Industrial | HR@1, NDCG@10, collision | MUST | TODO | tests purification necessity |
| `R014` | `M3` | stability ablation | `S8` weak-anchor `MGR-SID` | Industrial | HR@1, NDCG@10, collision | MUST | TODO | tests semantic anchor |
| `R015` | `M3` | seed confirmation | `S2` best baseline, seed 2/3 | Industrial | HR@1, NDCG@10 | MUST | TODO | run only if `R010` is positive |
| `R016` | `M3` | seed confirmation | `S3` main method, seed 2/3 | Industrial | HR@1, NDCG@10 | MUST | TODO | run only if `R010` is positive |
| `R017` | `M3` | optional simplicity test | `S9` overbuilt variant | Industrial | HR@1, NDCG@10, stability | NICE | TODO | local augmentation or confidence router |
| `R018` | `M4` | generalization | `S0` semantic-only | Office | HR@1, NDCG@10 | MUST | TODO | one seed first |
| `R019` | `M4` | generalization | `S2` uniform graph regularization | Office | HR@1, NDCG@10 | MUST | TODO | one seed first |
| `R020` | `M4` | generalization | `S3` `MGR-SID` | Office | HR@1, NDCG@10, same-`l2` | MUST | TODO | one seed first |
| `R021` | `M4` | mechanism analysis | bucketed gain + allocation analysis | Industrial + Office | same-`l1`, same-`l2`, gain per activated sample | MUST | TODO | produces paper figures |
