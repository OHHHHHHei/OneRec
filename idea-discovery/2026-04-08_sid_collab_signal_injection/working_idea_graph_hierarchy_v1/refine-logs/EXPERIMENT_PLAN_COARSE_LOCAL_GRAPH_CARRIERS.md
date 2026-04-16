# Experiment Plan（实验计划）

**Problem（问题）**: Current `graph carrier`（图载体） exploration has repeatedly modified `G_mid`（中尺度图）, but `G_coarse`（粗粒度图） and `G_local`（局部图） remain relatively crude and under-tested.
**Method Thesis（方法主张）**: The next meaningful tokenizer（分词器） gains are more likely to come from improving edge quality / user-diversity signal in `G_coarse` or coverage / transition reach in `G_local`, rather than from another pure `G_mid` replacement.
**Date（日期）**: `2026-04-15`

## Claim Map（主张映射）

| Claim（主张） | Why It Matters（重要性） | Minimum Convincing Evidence（最小可信证据） | Linked Blocks（关联模块） |
|---|---|---|---|
| C1. Repeated negative `G_mid` replacement results do not imply tokenizer-side graph carriers are exhausted; under-explored `G_coarse / G_local` may still have headroom. | This is the most actionable fork after `R510 / R511 / R520`. | At least one coarse/local candidate shows healthier graph-side diagnostics and tokenizer-side evidence than current `v2`-aligned carrier changes, then survives promotion to downstream `SFT -> evaluate`（监督微调到评测）. | B1, B2, B3, B4 |
| C2. The next gain should come from either better edge quality / diversity in `G_coarse` or better supervision coverage in `G_local`, not from another open-ended design-space expansion. | Prevents another round of low-information branching. | A compact 2-mainline + 1-control plan is enough to decide whether to keep pushing graph carriers or pivot elsewhere. | B0, B1, B2, B3 |

## Paper Storyline（论文叙事）

- Main paper must prove（主文必须证明）:
  - graph-structured collaborative supervision（图结构协同监督） still has nontrivial headroom beyond current `v2`, but the leverage point may sit outside `G_mid`
- Appendix can support（附录可支持）:
  - graph topology diagnostics（图拓扑诊断）
  - candidate graph sanity tables（候选图合理性表）
- Experiments intentionally cut（明确不做）:
  - another broad `G_mid` brainstorm（中图大范围发散）
  - pure attribute / semantic `G_mid` replacement（纯属性 / 语义中图替换）
  - full 3-layer hierarchy rewrite（全三层层级重写）

## Experiment Blocks（实验模块）

### Block B0: Graph Diagnostics Gate（图诊断门）

- Claim tested（检验主张）:
  - candidate graphs actually change the intended structural property, instead of only renaming the same signal.
- Why this block exists（存在原因）:
  - after `R202a / R401b / R401d / R510 / R511 / R520`, we should not promote a candidate just because it sounds new.
- Dataset / split / task（数据 / 划分 / 任务）:
  - `Industrial_and_Scientific`
  - offline graph construction and topology summary
- Compared systems（对比系统）:
  - current `coarse_purified`
  - current `local_purified`
  - candidate `G_coarse` with `CIR`（边可靠性） reweighting
  - candidate `G_coarse` with `user-segment`（用户分群） conditioning
  - candidate `G_local` with multi-hop diffusion（多跳扩散）
- Metrics（指标）:
  - edge count / avg degree / connected components（连通分量）
  - row sparsity / orphan-node ratio（孤立节点比例）
  - overlap with current graph（与当前图的边重叠）
  - for local graph: item coverage ratio（受监督覆盖比例）, 2-hop / 3-hop expansion ratio
  - for coarse graph: segment diversity score（分群多样性得分） or CIR distribution（CIR 分布）
- Setup details（设置）:
  - no tokenizer training
  - export a compact comparison table
- Success criterion（成功标准）:
  - candidate graph changes the intended property in a clear way without topology collapse（拓扑塌缩）
- Failure interpretation（失败解释）:
  - if a candidate barely changes the graph or breaks connectivity badly, cut it before training
- Table / figure target（目标表 / 图）:
  - appendix or internal decision table
- Priority（优先级）: `MUST-RUN（必跑）`

### Block B1: `G_local` Coverage Expansion via Multi-Hop Diffused Transition（用多跳扩散扩展 `G_local` 覆盖）

- Claim tested（检验主张）:
  - the current `local_purified` may be too sparse, so many items receive weak or zero `L3` collaborative supervision（L3 协同监督）.
- Why this block exists（存在原因）:
  - this directly targets the current `HR@10` / `mid-beam retention`（中束保持） gap better than yet another `G_mid` replacement.
- Dataset / split / task（数据 / 划分 / 任务）:
  - `Industrial_and_Scientific`
  - tokenizer train + generate
- Compared systems（对比系统）:
  - base `v2` tokenizer graph bank
  - `R530a`: `L3 <- A + α A^2`
  - `R530b`: `L3 <- A + α A^2 + α^2 A^3`
- Metrics（指标）:
  - generated collision（生成后冲突率）
  - same-`l2` ambiguity（同 `l2` 歧义）
  - local coverage ratio（局部图覆盖比例）
  - target-weighted local branching / entropy（目标加权局部分叉 / 熵）
- Setup details（设置）:
  - keep `L1 <- coarse_purified`
  - keep `L2 <- fagsp_mid_base`
  - only modify `L3`
  - try a small grid such as `α ∈ {0.35, 0.50}`
- Success criterion（成功标准）:
  - no tokenizer-side regression worse than current `v2` on final generated collision
  - plus clear improvement in local coverage or ambiguity metrics
- Failure interpretation（失败解释）:
  - if multi-hop only densifies the graph but gives no structural or generate benefit, then local sparsity is probably not the main bottleneck
- Table / figure target（目标表 / 图）:
  - candidate-carrier table, local coverage figure
- Priority（优先级）: `MUST-RUN（必跑）`

### Block B2: `G_coarse` Diversity Injection via User-Segment Co-Occurrence（用用户分群共现注入 `G_coarse` 多样性）

- Claim tested（检验主张）:
  - current coarse graph ignores user-type diversity（用户类型多样性）, so its edges may over-reflect a narrow subset of users.
- Why this block exists（存在原因）:
  - this is the strongest `information-expanding carrier`（信息扩张型图载体） candidate that still stays behavior-aligned.
- Dataset / split / task（数据 / 划分 / 任务）:
  - `Industrial_and_Scientific`
  - tokenizer train + generate
- Compared systems（对比系统）:
  - base `v2` tokenizer graph bank
  - `R540a`: `G_coarse` weighted by cross-segment support（跨分群支持）
  - `R540b`: stronger segment diversity weighting（更强分群多样性加权）
- Metrics（指标）:
  - generated collision（生成后冲突率）
  - prefix distribution balance（前缀分布平衡）
  - same-`l2` ambiguity（同 `l2` 歧义）
  - graph-side segment diversity score（图侧分群多样性得分）
- Setup details（设置）:
  - cluster users with a lightweight interaction-profile feature
  - keep `L2` and `L3` fixed to the current base
  - recommended initial `K`（分群数） in `{4, 8}`
- Success criterion（成功标准）:
  - no regression relative to current `v2` in generated collision
  - plus a plausible improvement in coarse routing-oriented diagnostics（粗层路由诊断）
- Failure interpretation（失败解释）:
  - if user segmentation adds instability without better graph quality, then this diversity view is too noisy or too weak on Industrial
- Table / figure target（目标表 / 图）:
  - coarse-graph diagnostics table
- Priority（优先级）: `MUST-RUN（必跑）`

### Block B3: `G_coarse` Low-Risk Control via CIR Reweighting（用 CIR 重加权做 `G_coarse` 低风险对照）

- Claim tested（检验主张）:
  - before we claim coarse-level new information is needed, we should test whether coarse edge quality is already the simpler bottleneck.
- Why this block exists（存在原因）:
  - this is the cleanest low-risk control for the coarse branch.
- Dataset / split / task（数据 / 划分 / 任务）:
  - `Industrial_and_Scientific`
  - tokenizer train + generate
- Compared systems（对比系统）:
  - base `v2`
  - `R541`: `G_coarse` with `CIR` reweighting
- Metrics（指标）:
  - generated collision（生成后冲突率）
  - prefix balance（前缀平衡）
  - same-`l2` ambiguity（同 `l2` 歧义）
  - CIR score distribution（CIR 分布）
- Setup details（设置）:
  - keep `L2 / L3` fixed
  - no new signal source, only reweight current coarse edges
- Success criterion（成功标准）:
  - if this simple control already helps, we should prefer it before more speculative coarse carriers
- Failure interpretation（失败解释）:
  - if even the low-risk coarse cleanup gives nothing, then either coarse is not the bottleneck or current injection interface is the real limit
- Table / figure target（目标表 / 图）:
  - control row in coarse-branch table
- Priority（优先级）: `MUST-RUN（必跑）`

### Block B4: Promotion to Downstream `SFT -> evaluate`（推进到下游 `SFT -> evaluate`）

- Claim tested（检验主张）:
  - at least one new coarse/local carrier can survive beyond tokenizer-only evidence.
- Why this block exists（存在原因）:
  - tokenizer-side structure alone is not enough; we need a downstream promotion gate.
- Dataset / split / task（数据 / 划分 / 任务）:
  - `Industrial_and_Scientific`
  - `title_history2sid_on + desc_align_p05`
- Compared systems（对比系统）:
  - current `v2_on_p05`
  - only the single best promoted candidate from B1 / B2 / B3
- Metrics（指标）:
  - `NDCG@1/3/5/10`
  - `HR@1/3/5/10`
  - focused reading on `HR@10`（十位命中率） and mid-beam pattern（中束模式）
- Setup details（设置）:
  - do not promote more than one candidate at first
  - use the current strongest `v2`-aligned recipe（配方）
- Success criterion（成功标准）:
  - at least match `v2_on_p05` SFT on `NDCG@10 / HR@10`, or show a clearly better gap pattern worth RL promotion
- Failure interpretation（失败解释）:
  - if the best coarse/local carrier still cannot beat `v2_on_p05` in downstream SFT, graph-carrier-only exploration should be reconsidered
- Table / figure target（目标表 / 图）:
  - main branch decision table
- Priority（优先级）: `MUST-RUN（必跑）`

## Run Order and Milestones（执行顺序与里程碑）

| Milestone（里程碑） | Goal（目标） | Runs（运行） | Decision Gate（决策门） | Cost（成本） | Risk（风险） |
|---|---|---|---|---|---|
| M0 | Graph diagnostics gate（图诊断门） | `D530` local multi-hop diagnostics, `D540` segment-coarse diagnostics, `D541` CIR-coarse diagnostics | Cut any carrier that collapses topology or barely changes the graph | CPU-only / low | Over-reading graph stats without training evidence |
| M1 | Low-risk coarse control（低风险粗图对照） | `R541` CIR-coarse tokenizer + generate | If totally flat or worse, coarse-only cleanup is weak | 1 tokenizer run | May still be too small a change |
| M2 | Main local branch（主局部分支） | `R530a`, optional `R530b` | Promote only if local coverage improves and generate does not regress | 1-2 tokenizer runs | Multi-hop may over-smooth local distinctions |
| M3 | Main coarse diversity branch（主粗图多样性分支） | `R540a`, optional `R540b` | Promote only if coarse diagnostics improve and generate is competitive | 1-2 tokenizer runs | User clustering may be noisy on a small dataset |
| M4 | Single promotion to downstream（单一候选下游推进） | `R550` best candidate -> `SFT -> evaluate` | If not competitive with `v2_on_p05`, stop graph-carrier expansion | 1 SFT/eval chain | Tokenizer gains may not survive downstream |
| M5 | Optional combo only if justified（仅在有理据时做组合） | combine best `G_coarse` and best `G_local` | Run only if at least one single-branch candidate is clearly positive | 1 tokenizer run + maybe downstream | Easy to overbuild before single-branch evidence is clear |

## Compute and Data Budget（算力与数据预算）

- Total estimated GPU-hours（总 GPU 小时）:
  - tokenizer runs: `4-6`
  - one downstream `SFT -> evaluate`: `1`
  - optional combo: `1`
- Data preparation needs（数据准备需求）:
  - user clustering features for segment-coarse
  - multi-hop transition matrix export
  - CIR score precompute
- Human evaluation needs（人工评估需求）:
  - none at this stage
- Biggest bottleneck（最大瓶颈）:
  - avoiding another round of tokenizer-side false positives（分词器侧假阳性）

## Risks and Mitigations（风险与缓解）

- Risk（风险）: We again optimize tokenizer-side proxies（分词器侧代理指标） that do not survive downstream.
- Mitigation（缓解）: Promote only one best candidate, and only after graph diagnostics + generate evidence both pass.

- Risk（风险）: User segmentation is unstable on a small dataset.
- Mitigation（缓解）: keep `K` small and compare against the low-risk `CIR` control.

- Risk（风险）: Multi-hop diffusion over-smooths local distinctions.
- Mitigation（缓解）: start with shallow diffusion and keep a small `α` grid.

- Risk（风险）: We accidentally reopen open-ended graph design exploration.
- Mitigation（缓解）: no more than two main lines plus one control before the next downstream verdict.

## Final Checklist（最终检查表）

- [ ] Main branch decision is compact and claim-linked
- [ ] `G_coarse` and `G_local` are both tested directly
- [ ] One low-risk control exists
- [ ] Only one candidate is promoted to downstream first
- [ ] Open-ended `G_mid` expansion is paused
