# Current State（当前状态）

Status（状态）: `canonical（权威）`

Last updated（更新日期）: `2026-04-16`

## 一句话当前主线

当前 strongest validated line（最强已验证主线）仍然是 `v2_on_p05 -> RL`；`S000` 之后，所有当前前验 diagnostics（前验诊断）都已经从 active decision workflow（活跃决策工作流）里退役，所以现在**不存在**一个可信的“先看诊断再推进 tokenizer（分词器）”路径。

## 当前问题

我们的目标不是让新的 SID space（SID 空间）尽量贴近旧 baseline（基线），而是构造一个更好的 SID codebook space（SID 码本空间），让 fresh downstream SFT（全新下游 SFT）更容易学出更强的推荐行为。

当前最可信的工作假设是：

- 仅靠更干净的 tokenizer-side structure（分词器侧结构）还不够。
- 仅靠更强的 deeper conditional learnability（更深层条件可学习性）也还不够。
- 当前主要瓶颈已经更具体地收敛为：现有图监督仍然主要是 attraction-only graph smoothness（仅吸引式图平滑）；它能建模“谁应该靠近”，但还不能显式处理 `semantic-close but collaboratively inconsistent`（语义接近但协同不一致）的物品分离。
- `R510 / R511 / R520` 说明“继续换 `G_mid`（中尺度图）来源”还没有给出正信号；`R530a / R542a` 则说明单纯继续改 `G_local / G_coarse`（局部图 / 粗粒度图）也还不足以自动解决这个缺口。
- selective separation（选择性分离）仍然是一个合理的方法假设，因为当前图监督主要还是 attraction-only graph smoothness（仅吸引式图平滑），缺少对 `semantic-close but collaboratively inconsistent`（语义接近但协同不一致）物品的显式分离。
- 但这些分支之前依赖的 offline diagnostics gate（离线诊断门）现在已经统一退役，后续不能再用它们来决定“谁值得推进”。

## 当前方法骨架

- semantic tokenizer backbone（语义分词器骨干）仍然是 MiniOneRec 的 `RQ-VAE -> sid-generate`。
- graph-structured collaborative supervision（图结构协同监督）仍然作为 structural supervision（结构监督）注入 tokenizer（分词器）训练，而不是后验 patch（后验修补）。
- 当前最强已验证方法骨架仍然是 ambiguity-aware graph supervision（歧义感知图监督） + semantic-structure retention（语义结构保持）。
- 当前方法上的一个关键开放缺口是：缺少对 `semantic-close but collaboratively inconsistent`（语义接近但协同不一致）物品的显式选择性分离。
- 因此方法探索上仍然可以考虑：
  - 保留当前 `v2` attraction（吸引） + retention（保持）骨架
  - 额外引入 `reliability-aware selective separation`（可靠性感知的选择性分离）
- 但执行规则已经改变：
  - 不再允许任何 retired prior diagnostic（已退役前验诊断）充当 tokenizer promotion gate（分词器推进门槛）
- 当前稳定的层级分工是：`L1 <- coarse_purified`，`L2 <- fagsp_mid_base`，`L3 <- local_purified`。
- 当前正在启动的新分支是一个更简化的 selective separation（选择性分离）版本：
  - 只在 `L2`（第 2 层）保留 `pull / push`（拉近 / 推远）
  - 不再把 coarse/local pull（粗层 / 局部层拉近）和 semantic retention（语义保持）继续叠进同一个实验

## Baseline 口径

- 主 baseline（主基线）：original MiniOneRec strongest SFT（原版最强 SFT）和 strongest RL（原版最强 RL）。
- strongest original SFT：`title_history2sid_off + desc_align_p05`，Industrial 上 `NDCG@10 = 0.10372`，`HR@10 = 0.15089`。
- strongest original RL：同一 recipe（配方）的 RL 链，Industrial 上 `NDCG@10 = 0.10726`，`HR@10 = 0.15133`。
- recipe-aligned original baseline（配方对齐原版基线）：原版 MiniOneRec 在和 `v2` 相同 task recipe（任务配方）下的对照。
- internal control（内部对照）：`mgr_upstream_baseline` / `mgr_upstream_hierarchy`，只用于机制诊断，不是主 baseline。

## 当前 strongest validated line

- `v2_on_p05 -> RL`
- 对应 recipe（配方）：`title_history2sid_on + desc_align_p05`
- 当前结果：`NDCG@10 = 0.10432`，`HR@10 = 0.14185`

这条线已经超过 strongest original SFT（原版最强 SFT）的 `NDCG@10`，但还没有超过 strongest original RL（原版最强 RL），所以现在仍然不能宣称 end-to-end overall best（端到端总体最优）。

## 已经被证明的结论

- graph-structured collaborative information（图结构协同信息）对 SID 构建是有用的，这个方向已经被 `v1` 和 `v2` 支撑。
- `v2` 不只是 tokenizer-side artifact（分词器侧假象）；它在 recipe isolation（配方隔离）之后确认了自己的最佳下游 recipe（配方）是 `title_history2sid_on + desc_align_p05`。
- `title_history2sid_off` 是 `v2` 与 strongest original recipe（原版最强配方）之间的主要 mismatch（失配）来源，`desc_align_p05` 本身不是主要问题。
- `v2_on_p05` 可以稳定推进到 full downstream RL（完整下游 RL），说明 graph-aware SID（图感知 SID）不是只停留在 tokenizer（分词器）侧。
- stage-2 `R202a` 和 stage-3 `R401b/R401d` 都说明：更强的结构指标不自动等于更好的 downstream SID space（下游 SID 空间）。

## 当前没有被证明的结论

- `v2_on_p05 -> RL` 还没有超过 strongest original RL（原版最强 RL）。
- 现在还不能宣称 end-to-end overall best（端到端总体最优）。
- `TAGCF` 分支和更完整 `FaGSP` 分支还没有产出新的 strongest validated tokenizer（最强已验证分词器）。
- 目前还没有证据表明“再换一种 `G_mid`（中尺度图）”比优先重构 `G_coarse`（粗粒度图）更有希望。

## 当前进行中的实验

- `R610a`：base `v2` + `L3`-only `reliability-aware selective separation`（仅 `L3` 的可靠性感知选择性分离）已完成：
  - 当前只保留一个很弱的事实：
    - 这是一个已经落地并完成 `sid-generate`（SID 生成）的 tokenizer（分词器）变体
  - 但它**不是**当前 active candidate（活跃候选）：
    - supporting diagnostics（支撑诊断）已经退役
    - downstream transfer（下游迁移）还没有验证
- `R630a / R630b / R630c`：mid-only pull/push（仅中层拉近/推远）分支已正式发起：
  - `R630a = pull-only`（仅拉近）
  - `R630b = push-only`（仅推远）
  - `R630c = pull + push`（拉近 + 推远）
  - 共同特点：
    - 只干预 `L2`（第 2 层）
    - `pull`（拉近）只用 `fagsp_mid_base`
    - `push`（推远）只用 `semantic-near + mid-graph-weak`（语义接近 + 中图弱连接）物品对
    - 不再使用任何 retired prior diagnostic（已退役前验诊断）做推进门槛
  - 当前作用：
    - 这是 selective separation（选择性分离）方向的首个 clean attribution（干净归因）实验组
    - 目标是先回答“真正有用的是 pull、push，还是两者都要”
- `S000`：diagnostic audit（诊断审计）已完成：
  - 审计文件：
    - `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_diagnostic_audit_industrial/SUMMARY.md`
  - 审计范围：
    - 只使用 Industrial 上 recipe-aligned（配方对齐）的 historical downstream comparisons（历史下游对比）
    - 统一检查当前常用前验 diagnostics（前验诊断）对 `NDCG@10 / HR@10` 的 pairwise consistency（成对一致率）
  - 关键结论：
    - 当前**没有**任何一个前验 diagnostic（前验诊断）达到 decision-usable（可用于决策）的强度
    - generated collision（生成后冲突率）最差：
      - vs `NDCG@10`: `0.1111`
      - vs `HR@10`: `0.0556`
    - local ambiguity（局部歧义）类指标同样不可靠：
      - test mean `l2` leaf count（测试平均 `l2` 叶子数）vs `NDCG@10`: `0.2609`
      - test mean `l3` entropy（测试平均 `l3` 熵）vs `NDCG@10`: `0.3043`
    - prefix collaborative consistency（前缀协同一致性）略好，但仍不足以做推进门槛：
      - consistent crowded fraction（协同一致拥挤占比）vs `NDCG@10`: `0.4783`
      - inconsistent crowded fraction（协同不一致拥挤占比）vs `NDCG@10`: `0.3478`
  - interpretation（解读）:
    - 这些脚本现已统一从 active decision workflow（活跃决策工作流）里退役
    - 只保留为 archived provenance（归档来源记录），不再作为当前判断依据
- 最近完成的 graph-carrier（图载体）分支结果：
  - `R510`：属性图纯替换 `G_mid`，完整下游 `SFT -> evaluate`（监督微调到评测）为负
  - `R511`：属性图混合 `G_mid`，generated collision（生成后冲突率）退步到 `18 / 3686`
  - `R520`：`FaGSP cascade`（FaGSP 级联）`G_mid`，generated collision（生成后冲突率）为 `14 / 3686`，仍弱于当前 `v2`
  - `R530a`：`L3 <- local_multihop (A + αA^2, α=0.35)`（局部多跳 `A + αA^2`）已完成，generated collision（生成后冲突率）恶化到 `107 / 3686 = 0.02903`，属于明确负结果
  - `R542a`：`L1 <- coarse_mgdcf`, `L2 <- fagsp_mid_mgdcf`, `L3 <- local_purified` 已完成，generated collision（生成后冲突率）为 `42 / 3686 = 0.01139`
  - `R542b / R542c`：`MGDCF` coarse-only isolation（仅粗图隔离实验）目前没有活跃进程，仍属于未收尾证据，不再作为当前 active line（活跃主线）

这些结果共同说明：

> 仅靠继续替换 graph carrier（图载体）还不足以形成新的主线突破；selective separation（选择性分离）仍然值得做，但现在必须摆脱“先看前验诊断、再决定要不要推进”的旧工作流。当前唯一可靠的主线裁决仍然是 downstream evaluate（下游评测）。

## 已退役诊断

- 以下内容已统一退役，不再用于推进新 tokenizer（分词器）或排序候选：
  - generated collision（生成后冲突率）
  - local ambiguity（局部歧义）
  - prefix collaborative consistency（前缀协同一致性）
  - stage-2 / stage-3 interface diagnostics（阶段 2 / 阶段 3 接口诊断）
  - coarse/local graph diagnostics（粗图 / 局部图诊断）
  - selective-separation pair diagnostics（选择性分离物品对诊断）
- 归档位置：
  - `/home/leejt/OneRec/research-progress-log/archive/2026-04-16_retired_prior_diagnostics/README.md`
  - `/home/leejt/OneRec/scripts/archive/retired_prior_diagnostics/README.md`

## 下一步最合理的动作

1. 先完成 `R630a / R630b / R630c` 这组三路并行 tokenizer（分词器）训练，确认简化后的 `pull-only / push-only / pull+push`（仅拉近 / 仅推远 / 拉近加推远）哪条最值得继续。
2. 三条线训练完成后，不再用任何前验 diagnostic（前验诊断）筛选；如果要裁决，直接把最合理的一条推进到最小 `SFT -> evaluate`（监督微调到评测），优先仍使用 `title_history2sid_on + desc_align_p05`。
3. 未来如果真的想重新引入新诊断，必须先过 retrospective audit（回顾性审计）；过不了审计，就不能进入当前主线。

## 使用方式

- 这是唯一应该持续维护的 current-state document（当前状态文档）。
- 想看完整实验账本，请查 `/home/leejt/OneRec/experiment_results.csv`。
- 想看阶段实验记录，请查 `/home/leejt/OneRec/research-progress-log/experiment_launches/README.md`。
- 想看代码对齐的方法公式，请查 `/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/19_mgr_sid_current_method_code_aligned_formulas.md`。
