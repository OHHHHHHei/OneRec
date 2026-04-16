# Experiment Tracker（实验跟踪表）

Status（状态）: `snapshot（快照）`
Snapshot date（快照日期）: `2026-04-16`

This tracker is frozen after `S000`.

- `TODO` rows below are historical proposals（历史提案）, not active next runs（活跃下一步实验）.
- Do not revive them based only on prior diagnostics（前验诊断）.

| Run ID | Milestone（里程碑） | Purpose（目的） | System / Variant（系统 / 变体） | Split（划分） | Metrics（指标） | Priority（优先级） | Status（状态） | Notes（备注） |
|---|---|---|---|---|---|---|---|---|
| D600 | M0 | pair diagnostics（物品对诊断） | semantic-near + graph-weak / graph-non-neighbor pair statistics | Industrial | retired | MUST | RETIRED | historical pair-construction snapshot（历史物品对构造快照）, no longer a decision gate（决策门） |
| D601 | M0 | pair diagnostics（物品对诊断） | semantic-near + user-segment inconsistent pair statistics | Industrial | retired | NICE | DROPPED | dropped with the diagnostics-first（诊断优先） workflow |
| R610a | M1 | tokenizer screen（分词器筛选） | base `v2` + selective separation on `L3` only | Industrial | downstream not run | MUST | FROZEN | completed `sid-generate`（SID 生成） only; not promotable without downstream verdict（下游裁决） |
| R610b | M1 | tokenizer screen（分词器筛选） | base `v2` + selective separation on `L2 + L3` | Industrial | retired gate | MUST | DROPPED | old promotion logic depended on retired diagnostics |
| R611a | M2 | ablation（消融） | naive non-edge repulsion | Industrial | retired gate | MUST | DROPPED | do not launch from diagnostics-only evidence |
| R611b | M2 | ablation（消融） | semantic-near candidate pairs with uniform separation weight | Industrial | retired gate | MUST | DROPPED | same reason |
| R611c | M2 | ablation（消融） | reliability-aware selective separation | Industrial | retired gate | MUST | DROPPED | same reason |
| R612a | M3 | ablation（消融） | pair source = semantic-near + graph-weak | Industrial | retired gate | MUST | DROPPED | pair-source ranking by prior diagnostics is retired |
| R612b | M3 | ablation（消融） | pair source = semantic-near + user-segment inconsistent | Industrial | retired gate | MUST | DROPPED | same reason |
| R612c | M3 | ablation（消融） | pair source = semantic-near + same-prefix competitor | Industrial | retired gate | NICE | DROPPED | same reason |
| R620 | M4 | downstream verdict（下游裁决） | best selective-separation tokenizer -> `title_history2sid_on + desc_align_p05` | Industrial | `NDCG@10`, `HR@10`, beam pattern | MUST | BLOCKED | only revive after a new variant is justified without retired diagnostics |
