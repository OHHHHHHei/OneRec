# Experiment Plan（实验计划）

Status（状态）: `snapshot（快照）`
Snapshot date（快照日期）: `2026-04-16`

Retirement update（退役更新）:

- After `S000`, the diagnostics-first（诊断优先） workflow in this file is no longer active.
- Blocks that rely on prior diagnostics（前验诊断） as decision gates should now be read as historical design snapshots（历史设计快照）, not current execution rules（当前执行规则）.
- If selective separation（选择性分离） continues, future variants must be judged by downstream `SFT -> evaluate`（监督微调到评测）, not by these diagnostics blocks alone.

**Problem（问题）**: Current `MGR-SID` graph supervision（图监督） is still dominated by `attraction-only graph smoothness`（仅吸引式图平滑）. It can model which items should move closer, but it cannot explicitly separate `semantic-close but collaboratively inconsistent`（语义接近但协同不一致） items, which is now one of the most plausible sources of local `SID` ambiguity（局部 `SID` 歧义）.
**Method Thesis（方法主张）**: The next stage should keep the current `v2` attraction + semantic-retention backbone（吸引 + 语义保持骨干）, but add a **reliability-aware selective separation**（可靠性感知的选择性分离） mechanism that only pushes apart high-risk confusing pairs rather than all graph non-neighbors（图上无边对）.
**Date（日期）**: `2026-04-16`

## Claim Map（主张映射）

| Claim（主张） | Why It Matters（重要性） | Minimum Convincing Evidence（最小可信证据） | Linked Blocks（关联模块） |
|---|---|---|---|
| C1. `SID` tokenizer（`SID` 分词器） quality is currently limited not only by missing attraction（吸引） but also by missing selective separation（选择性分离） for `semantic-close but collaboratively inconsistent` items. | This turns the current motivation（动机） into a testable next-stage method thesis（方法主张）. | A selective-separation variant improves tokenizer-side local ambiguity indicators without the collapse seen in naive broader graph edits, and at least one promoted variant survives downstream `SFT -> evaluate`（监督微调到评测）. | B0, B1, B4 |
| C2. The gain, if any, must come from **reliable negative pair construction**（可靠负对构造）, not from generic global repulsion（全局排斥）. | This is the main anti-failure rule; otherwise the method will likely over-separate semantically valid neighbors. | Reliability-aware separation beats or matches naive non-edge repulsion（无边排斥） while being more stable. | B2, B3 |

## Paper Storyline（论文叙事）

- Main paper must prove（主文必须证明）:
  - graph-aware `SID` learning（图感知 `SID` 学习） needs both attraction（吸引） and selective separation（选择性分离）
  - the missing piece is not “push everything apart” but “reliably push apart the right confusing pairs”
- Appendix can support（附录可支持）:
  - pair diagnostics（物品对诊断）
  - reliability histograms（可靠性分布图）
  - failure cases of naive repulsion（朴素排斥的失败案例）
- Experiments intentionally cut（明确不做）:
  - another open-ended graph carrier（图载体） search before the selective-separation thesis is tested
  - full differentiable tokenizer（可微分词器） redesign
  - all-layer repulsion from the first run

## Borrowed Modules（借鉴模块）

- **NCL / NESCL**:
  - use semantic neighborhood（语义邻域） and structural neighborhood（结构邻域） jointly to define candidate confusing pairs
- **HDCCF**:
  - treat hard negatives（困难负样本） and true negatives（真实负样本） differently
- **Affinity Uncertainty-based Hard Negative Mining**:
  - assign confidence / uncertainty（置信度 / 不确定性） to candidate negative pairs instead of uniformly pushing all of them
- **DirectAU / ProtoAU**:
  - keep alignment（对齐） and uniformity（均匀性） in balance so the representation space does not collapse or over-scatter
- **DIGER / HiD-VAE**:
  - kept as future interface inspiration（后续接口灵感）, not phase-1 implementation targets

## Experiment Blocks（实验模块）

### Block B0: Pair Diagnostics Gate（物品对诊断门）

- Claim tested（检验主张）:
  - we can define a compact and meaningful set of candidate separation pairs instead of treating all graph non-neighbors as negatives
- Why this block exists（存在原因）:
  - the biggest risk of the new direction is false negatives（假负样本）
- Dataset / split / task（数据 / 划分 / 任务）:
  - `Industrial_and_Scientific`
  - offline diagnostics only
- Compared systems（对比系统）:
  - current semantic neighborhood（语义邻域）
  - current graph neighborhood（图邻域）
  - candidate negative pair rules
- Candidate pair rules（候选对规则）:
  - `P1`: semantic-near + graph-non-neighbor（语义近 + 图上无邻接）
  - `P2`: semantic-near + weak collaborative affinity（语义近 + 协同亲和度低）
  - `P3`: semantic-near + user-segment inconsistent（语义近 + 用户分群不一致）
  - `P4`: semantic-near + same-prefix competitor（语义近 + 同前缀竞争对） if a reliable prefix source is available
- Metrics（指标）:
  - candidate pair count（候选对数量）
  - semantic similarity distribution（语义相似度分布）
  - collaborative affinity distribution（协同亲和度分布）
  - overlap with current graph neighbors（与当前图邻居的重叠）
  - reliability score distribution（可靠性分数分布）
- Setup details（设置）:
  - no tokenizer training
  - export top confusing-pair examples（最易混淆物品对示例）
- Success criterion（成功标准）:
  - at least one pair rule isolates a compact set of semantically close but structurally unsupported pairs
- Failure interpretation（失败解释）:
  - if all candidate rules are noisy or nearly random, the selective-separation hypothesis needs a better pair source before training
- Table / figure target（目标表 / 图）:
  - pair diagnostics table + histogram
- Priority（优先级）: `MUST-RUN（必跑）`

### Block B1: Minimal Selective Separation in Tokenizer Training（最小选择性分离训练）

- Claim tested（检验主张）:
  - a small, well-targeted separation term can improve local `SID` discrimination（局部 `SID` 判别） without destabilizing the tokenizer（分词器）
- Why this block exists（存在原因）:
  - this is the first real test of the new method thesis（方法主张）
- Dataset / split / task（数据 / 划分 / 任务）:
  - `Industrial_and_Scientific`
  - tokenizer train + generate
- Compared systems（对比系统）:
  - current strongest tokenizer-side `v2` backbone
  - `R610a`: `L3`-only selective separation（仅 `L3` 选择性分离）
  - `R610b`: `L2 + L3` selective separation（`L2 + L3` 选择性分离）
- Metrics（指标）:
  - generated collision（生成后冲突率）
  - same-`l2` ambiguity（同 `l2` 歧义）
  - conditional leaf entropy（条件叶子熵）
  - target-weighted branching（目标加权分叉）
- Setup details（设置）:
  - keep current base graph bank（基础图组） fixed for the first pass
  - keep ambiguity-aware attraction（歧义感知吸引） and semantic retention（语义保持）
  - only add one new separation term
- Success criterion（成功标准）:
  - no severe collision explosion（冲突爆炸）
  - plus improvement in local ambiguity diagnostics relative to base `v2`
- Failure interpretation（失败解释）:
  - if even `L3`-only separation collapses, then the pair definition or separation strength is too noisy
- Table / figure target（目标表 / 图）:
  - tokenizer-side main method table
- Priority（优先级）: `MUST-RUN（必跑）`

### Block B2: Reliability Ablation（可靠性消融）

- Claim tested（检验主张）:
  - reliable pair weighting（可靠物品对加权） is necessary; naive repulsion（朴素排斥） is not enough
- Why this block exists（存在原因）:
  - this is the cleanest anti-claim check（反主张检验）
- Dataset / split / task（数据 / 划分 / 任务）:
  - `Industrial_and_Scientific`
  - tokenizer train + generate
- Compared systems（对比系统）:
  - `R611a`: naive non-edge repulsion（朴素无边排斥）
  - `R611b`: semantic-near + graph-weak pairs with uniform weight（语义近 + 图弱连接统一加权）
  - `R611c`: reliability-aware weighted separation（可靠性感知加权分离）
- Metrics（指标）:
  - generated collision（生成后冲突率）
  - local ambiguity metrics（局部歧义指标）
  - training stability（训练稳定性）
- Setup details（设置）:
  - same base backbone
  - same pair source
  - only vary reliability handling（可靠性处理）
- Success criterion（成功标准）:
  - reliability-aware version is more stable and at least as good as naive separation
- Failure interpretation（失败解释）:
  - if all three behave similarly, reliability scoring is not informative enough yet
- Table / figure target（目标表 / 图）:
  - ablation table on negative-pair reliability
- Priority（优先级）: `MUST-RUN（必跑）`

### Block B3: Pair Source Ablation（物品对来源消融）

- Claim tested（检验主张）:
  - not all confusing-pair definitions are equally useful
- Why this block exists（存在原因）:
  - we need to know whether the gain comes from semantic-graph disagreement（语义-图不一致）, user-segment inconsistency（用户分群不一致）, or prefix-level competition（前缀竞争）
- Dataset / split / task（数据 / 划分 / 任务）:
  - `Industrial_and_Scientific`
  - tokenizer train + generate
- Compared systems（对比系统）:
  - `R612a`: semantic-near + graph-weak
  - `R612b`: semantic-near + user-segment inconsistent
  - `R612c`: semantic-near + same-prefix competitor
- Metrics（指标）:
  - generated collision（生成后冲突率）
  - same-`l2` ambiguity（同 `l2` 歧义）
  - prefix competition diagnostics（前缀竞争诊断）
- Setup details（设置）:
  - use the best reliability handling from B2
  - keep all other knobs fixed
- Success criterion（成功标准）:
  - one pair source clearly dominates or two complementary sources emerge
- Failure interpretation（失败解释）:
  - if all pair sources are flat, the bottleneck may sit in the loss interface rather than the pair construction
- Table / figure target（目标表 / 图）:
  - pair-source comparison table
- Priority（优先级）: `MUST-RUN（必跑）`

### Block B4: Downstream Promotion to `SFT -> evaluate`（推进到下游 `SFT -> evaluate`）

- Claim tested（检验主张）:
  - selective separation helps not only tokenizer-side structure but also downstream recommendation
- Why this block exists（存在原因）:
  - tokenizer-side evidence alone is not enough for this project
- Dataset / split / task（数据 / 划分 / 任务）:
  - `Industrial_and_Scientific`
  - `title_history2sid_on + desc_align_p05`
- Compared systems（对比系统）:
  - current `v2_on_p05`
  - only the single best promoted selective-separation variant
- Metrics（指标）:
  - `NDCG@1/3/5/10`
  - `HR@1/3/5/10`
  - focused reading on `HR@10`（十位命中率） and `mid-beam retention`（中束保持）
- Setup details（设置）:
  - do not promote more than one variant at first
  - keep the same strongest `v2`-aligned recipe（与 `v2` 对齐的最强配方）
- Success criterion（成功标准）:
  - at least match `v2_on_p05` SFT on the main metrics, or show a clearly better gap pattern worth RL promotion
- Failure interpretation（失败解释）:
  - if even the best selective-separation tokenizer cannot survive downstream, the current loss interface is probably still too indirect
- Table / figure target（目标表 / 图）:
  - main decision table for the next paper-stage fork
- Priority（优先级）: `MUST-RUN（必跑）`

## Run Order and Milestones（执行顺序与里程碑）

| Milestone（里程碑） | Goal（目标） | Runs（运行） | Decision Gate（决策门） | Cost（成本） | Risk（风险） |
|---|---|---|---|---|---|
| M0 | Pair diagnostics gate（物品对诊断门） | `D600`, optional `D601` | Stop if pair rules are too noisy or too broad | CPU-only / low | Overfitting to a handcrafted pair rule |
| M1 | Minimal method test（最小方法测试） | `R610a`, `R610b` | Continue only if one variant improves local diagnostics without collapse | 1-2 tokenizer runs | Separation term may be too strong |
| M2 | Reliability ablation（可靠性消融） | `R611a`, `R611b`, `R611c` | Continue only if reliability-aware design beats naive repulsion | 2-3 tokenizer runs | Reliability scores may be weak on a small dataset |
| M3 | Pair source ablation（物品对来源消融） | `R612a`, `R612b`, optional `R612c` | Promote only one best pair source | 2 tokenizer runs | User-segment signals may still be too noisy |
| M4 | Downstream verdict（下游裁决） | `R620` best selective-separation variant -> `SFT -> evaluate` | Stop if not competitive with current `v2_on_p05` | 1 SFT/eval chain | Tokenizer-side gains may not survive downstream |

## Compute and Data Budget（算力与数据预算）

- Total estimated GPU-hours（总 GPU 小时）:
  - tokenizer runs: `5-8`
  - one downstream `SFT -> evaluate`: `1`
- Data preparation needs（数据准备需求）:
  - semantic-neighborhood export（语义邻域导出）
  - graph affinity export（图亲和度导出）
  - optional user-segment label export（可选用户分群标签导出）
- Human evaluation needs（人工评估需求）:
  - none at this stage
- Biggest bottleneck（最大瓶颈）:
  - false negative control（假负样本控制）

## Risks and Mitigations（风险与缓解）

- Risk（风险）: We define too many false negatives（假负样本） and destroy valid semantic neighborhoods（有效语义邻域）.
- Mitigation（缓解）: run B0 before any training and require reliability-aware weighting（可靠性感知加权） before promotion.

- Risk（风险）: The new direction reduces to generic contrastive learning（通用对比学习） and loses the `SID` story.
- Mitigation（缓解）: keep the intervention local, pair-driven, and explicitly attached to `L2 / L3` representations（`L2 / L3` 表示） inside the current `RQ-VAE` pipeline.

- Risk（风险）: We again optimize tokenizer-side proxies（分词器侧代理指标） that do not survive downstream.
- Mitigation（缓解）: promote only one best candidate and judge it by the current strongest `v2` recipe（配方） immediately.

## Final Checklist（最终检查表）

- [ ] Main method claim is compact and testable
- [ ] Selective separation is defined by explicit pair rules
- [ ] Reliability is tested against naive repulsion
- [ ] Only one best tokenizer is promoted downstream first
- [ ] Open-ended graph-carrier branching is paused for this stage
