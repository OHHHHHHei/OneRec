# Current State（当前状态）

Status（状态）: `canonical（权威）`

Last updated（更新日期）: `2026-04-15`

## 一句话当前主线

当前 strongest validated line（最强已验证主线）仍然是 `v2_on_p05 -> RL`；当前 active exploration（当前活跃探索）已经从 stage-3 codebook-space search（stage-3 码本空间搜索）切到 graph-carrier upgrade（图载体升级），重点验证 `R511` 和 `R520` 能否给出更好的 `G_mid`（中尺度图）。

## 当前问题

我们的目标不是让新的 SID space（SID 空间）尽量贴近旧 baseline（基线），而是构造一个更好的 SID codebook space（SID 码本空间），让 fresh downstream SFT（全新下游 SFT）更容易学出更强的推荐行为。

当前最可信的工作假设是：

- 仅靠更干净的 tokenizer-side structure（分词器侧结构）还不够。
- 仅靠更强的 deeper conditional learnability（更深层条件可学习性）也还不够。
- 当前主要瓶颈更像是 graph carrier quality（图载体质量），尤其是 `G_mid`，以及它最终能否转化成更好的 downstream SID usability（下游 SID 可用性）。

## 当前方法骨架

- semantic tokenizer backbone（语义分词器骨干）仍然是 MiniOneRec 的 `RQ-VAE -> sid-generate`。
- graph-structured collaborative supervision（图结构协同监督）仍然作为 structural supervision（结构监督）注入 tokenizer（分词器）训练，而不是后验 patch（后验修补）。
- 当前最强已验证方法骨架仍然是 ambiguity-aware graph supervision（歧义感知图监督） + semantic-structure retention（语义结构保持）。
- 当前稳定的层级分工是：`L1 <- coarse_purified`，`L2 <- fagsp_mid_base`，`L3 <- local_purified`。

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

## 当前进行中的实验

- `R511`：`G_mid <- 0.5 * fagsp_mid_base + 0.5 * G_attr_fused`
  当前状态：`IN_PROGRESS`
- `R520`：`G_mid <- fagsp_mid_cascade`
  当前状态：`PENDING`

它们的共同目标都不是再修一个 retention-only tweak（仅保持项微调），而是直接回答：

> 更好的 graph carrier（图载体），尤其是更好的 `G_mid`，是不是比继续修 tokenizer loss（分词器损失）更可能带来下一步实质收益。

## 下一步最合理的动作

1. 先完成 `R511`，判断 attribute topology（属性拓扑）作为 additive mid signal（增量中图信号）是否比 pure replacement（纯替换）更稳。
2. 并行完成 `R520`，判断更完整的 FaGSP item-side cascade（FaGSP 物品侧级联）是否优于当前 `fagsp_mid_base`。
3. 只有当新的 graph carrier（图载体）在 tokenizer-side evidence（分词器侧证据）上足够有说服力时，才把它推进到新的 downstream（下游） `SFT -> evaluate`；在此之前，不重新打开 retention-only branch（仅保持分支）的大范围发散。

## 使用方式

- 这是唯一应该持续维护的 current-state document（当前状态文档）。
- 想看完整实验账本，请查 `/home/leejt/OneRec/experiment_results.csv`。
- 想看阶段实验记录，请查 `/home/leejt/OneRec/research-progress-log/experiment_launches/README.md`。
- 想看代码对齐的方法公式，请查 `/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/19_mgr_sid_current_method_code_aligned_formulas.md`。
