# Experiment Tracker

| Run ID | Milestone | Purpose（目的） | System / Variant（系统 / 变体） | Split | Metrics（指标） | Priority | Status | Notes（备注） |
|---|---|---|---|---|---|---|---|---|
| R500 | M0 | 属性抽取小试 | `Attr-Raw-TextPhrase` 全量构图 | Industrial | 覆盖率、属性质量、图密度 | MUST | COMPLETED | 覆盖率 `0.9997`，图能成，但唯一属性数 `9602`，噪声偏重 |
| R501 | M0 | 过滤与融合 | `Attr-Fused-TextPhrase` 全量构图 | Industrial | 覆盖率、连通性、冷启动接通率 | MUST | COMPLETED | 当前最佳 `M0` 候选；唯一属性数压到 `3254`，图仍高连通 |
| R502 | M0 | 控制组构图 | `Attr-Heuristic-Title` 全量构图 | Industrial | 同上 | MUST | COMPLETED | 控制组明显更弱；图更碎，最大连通分量只有 `0.6356` |
| R510 | M1 | 直接替换 `G_mid` | `T1: G_mid <- G_attr_fused` | Industrial | tokenizer 健康性、collision、局部歧义 | MUST | COMPLETED | generate 后 `collision = 11 / 3686`，但相对 `v2` 的结构结果是 mixed（混合）：mean target `l2` leaf count 降到 `3.6848`，同时 multi-leaf `same_l2` 比例升到 `0.5277` |
| R511 | M1 | 混合 `G_mid` | `T2: G_mid <- mix(fagsp_mid_base, G_attr_fused)` | Industrial | 同上 | MUST | COMPLETED | 第一版 `0.5 / 0.5` 混合没有带来更稳结果；generate 后 `collision = 18 / 3686`，明显差于 `R510` 的 `11 / 3686` |
| R512 | M1 | 边重加权 | `T3: attr-gated coarse/local` | Industrial | 同上 | NICE | TODO | 检查属性图是否更适合去噪而非替换 |
| R530 | M2 | 下游最终裁判 | best TAGCF-inspired branch -> `SFT -> evaluate` | Industrial | `NDCG/HR@1/3/5/10` | MUST | TODO | 只推最像样的一个分支 |
| R540 | M3 | `LLM` 必要性 | best placement + `Attr-Heuristic` | Industrial | full `SFT -> evaluate` | NICE | TODO | 检查是否简单文本标签也够 |
| R541 | M3 | raw vs fused | best placement + `Attr-LLM-Raw` | Industrial | full `SFT -> evaluate` | NICE | TODO | 检查过滤与融合是否关键 |
| R542 | M3 | fused 主版本 | best placement + `Attr-LLM-Fused` | Industrial | full `SFT -> evaluate` | NICE | TODO | 对应预期最好版本 |
| R550 | M4 | RL 确认 | best TAGCF-inspired `SFT` winner -> `RL -> evaluate` | Industrial | `NDCG/HR@5/10/20` | NICE | TODO | 仅在 `R530` 明显正向后开启 |
