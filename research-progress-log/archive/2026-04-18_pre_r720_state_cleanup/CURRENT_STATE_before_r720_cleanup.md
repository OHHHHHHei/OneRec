# Current State（当前状态）

Status（状态）: `canonical（权威）`

Last updated（更新日期）: `2026-04-18`

## 一句话当前主线

当前 strongest validated line（最强已验证主线）仍然是 `v2_on_p05 -> RL`，但当前 active method candidate（活跃方法候选）已经收敛为 `R720a: ambiguity-aware stop-gradient L2 ranking contrastive SID`（歧义感知停止梯度中层排序对比 SID）。后续不要再横向发散新主线，主要围绕 `R720a` 的 loss weight（损失权重）、graph source（图来源）、positive/negative pair construction（正负样本构造）做小范围微调。

`R720a` 已经接入代码并通过 smoke run（冒烟运行）：它在 `L1/L3`（第一层/第三层）保留轻量 graph pull（图拉近），在 `L2`（第二层）用 ranking contrastive loss（排序对比损失）表达“语义相近候选中，协同正样本应该比协同弱负样本更近”。完整代码入口见 `config/experiments/sid_train_industrial_mgr_sid_r720a_l2_ranking_contrastive.yaml`、`scripts/experiment_mgr_sid_r720a_l2_ranking_contrastive_train_generate.sh` 和 `src/onerec/experiments/mgr_sid/train_v2.py`。

## 当前问题

我们的目标不是让新的 SID space（SID 空间）尽量贴近旧 baseline（基线），而是构造一个更好的 SID codebook space（SID 码本空间），让 fresh downstream SFT（全新下游 SFT）更容易学出更强的推荐行为。

当前最可信的工作假设是：

- 仅靠更干净的 tokenizer-side structure（分词器侧结构）还不够。
- 仅靠更强的 deeper conditional learnability（更深层条件可学习性）也还不够。
- 当前主要瓶颈已经更具体地收敛为：现有图监督仍然主要是 attraction-only graph smoothness（仅吸引式图平滑）；它能建模“谁应该靠近”，但还不能显式处理 `semantic-close but collaboratively inconsistent`（语义接近但协同不一致）的物品分离。
- `R510 / R511 / R520` 说明“继续换 `G_mid`（中尺度图）来源”还没有给出正信号；`R530a / R542a` 则说明单纯继续改 `G_local / G_coarse`（局部图 / 粗粒度图）也还不足以自动解决这个缺口。
- selective separation（选择性分离）仍然是一个合理的方法假设，因为当前图监督主要还是 attraction-only graph smoothness（仅吸引式图平滑），缺少对 `semantic-close but collaboratively inconsistent`（语义接近但协同不一致）物品的显式分离。
- `R630c` 的 posterior output diagnosis（后验输出诊断）进一步说明：当前更具体的失败模式不是单纯 same-prefix confusion（同前缀混淆）或 collision（碰撞），而是 beam retention（候选束保留）/ neighborhood retention（邻域保留）不足。
- 更细的 loss-item graph/semantic analysis（损失物品图/语义分析）还说明：
  - 一批最差热点来自 dense semantic variant families（稠密语义变体家族），尤其是多颜色 `3D` 打印耗材
  - 这些物品在语义空间里非常接近，但它们的语义近邻在当前 `coarse / mid / local`（粗图 / 中图 / 局部图）里经常几乎全是零亲和
  - 因此当前最差失败不完全像“被 `push`（推远）推坏了”，而更像 graph carrier blind spot（图载体盲区）：语义上很近的一整个变体邻域，根本没有被图有效看见
- 进一步的 high-order collaborative audit（高阶协同审计）又说明：
  - 这些 blind spot（盲区）里的很多物品对，虽然 direct edge（直接边）接近于零
  - 但它们共享明显相似的 predecessor context（前驱上下文）
  - 因此当前最具体的新假设是：direct `item-item` graph（直接物品图）漏掉了 cross-sequence predecessor sharing（跨序列前驱共享）形成的 high-order collaboration（高阶协同）
  - 当前 carrier（载体）分支是 `Seq2Graph-lite`（轻量 `Seq2Graph`） high-order rescue graph（高阶补盲图）：先增强 `G_coarse`（粗图），再重建 `G_mid`（中图），而不是继续加新 loss（损失）
  - `D640` offline graph audit（离线图审计）已经完成：
    - 三个 `Seq2Graph-lite`（轻量 `Seq2Graph`） coarse variants（粗图变体）都不是 near-baseline tweak（近基线微调），而是真正改动了邻域结构
    - `coarse_seq2g_rel`（可靠性感知粗图）把 hotspot semantic-pair visibility（热点语义对可见率）从 `0.1667` 提高到 `0.3667`
    - `coarse_seq2g_rel` 和 `coarse_seq2g_rel_masked`（可靠性感知加掩码粗图）都把 predecessor-sharing direct-zero visibility（前驱共享且直接零连接可见率）从 `0.0` 提高到 `0.8`
    - 其中 `rel`（可靠性版）更偏 coverage（覆盖），`rel_masked`（可靠性加掩码版）更偏 rescue purity（补盲纯度）
  - `R640` tokenizer screen（分词器筛选）已经给出第一轮正式裁决：
    - `R640b = rel`（可靠性版）从首次可见评估开始就接近完全塌缩：first eval collision（首次评估冲突率）`0.9997`，best train collision（训练最佳冲突率）`0.9284`，generated collision（生成后冲突率）`0.4121`
    - `R640c = rel_masked`（可靠性加掩码版）则成功恢复到可推进状态：best train collision（训练最佳冲突率）`0.1243`，generated collision（生成后冲突率）`12 / 3686 = 0.0032556`
    - `R640b` 的核心问题不是“高阶协同没用”，而是 reliability-only rescue（仅可靠性感知补盲）保留了太多已经 direct-strong（直接强连接）的边，导致 dense same-brand families（稠密同品牌家族）被进一步过度平滑
  - `R645 = R640c -> SFT -> evaluate`（监督微调到评测）已经完成，结果为负：`NDCG@10 = 0.09306`，`HR@10 = 0.13126`
  - 这说明 `Seq2Graph-lite rel_masked`（轻量 `Seq2Graph` 可靠性感知加掩码版）作为 carrier-only smoothness（仅图载体加平滑监督）不能直接推广为主线
  - 但这不是对 high-order carrier + explicit push-pull（高阶载体 + 显式推远拉近）方向的否定；它更像是在提醒：只改图并继续 attraction-only smoothness（仅吸引式平滑）不够
- `R650a = Seq2Graph-lite rel_masked + mid-only pull-push`（轻量 Seq2Graph 加掩码版 + 仅中层拉近推远）已经完成 `SFT -> evaluate`（监督微调到评测），结果为负：
  - 把 `R640c` 的 `fagsp_mid_seq2g_rel_masked`（Seq2Graph 中层图）放进 `R630c` 的 mid-only `pull + push`（仅中层拉近加推远）框架
  - `pull`（拉近）使用 `fagsp_mid_seq2g_rel_masked`
  - `push`（推远）物品对也用 semantic-near + `fagsp_mid_seq2g_rel_masked` weak（语义近 + Seq2Graph 中图弱连接）重新生成
  - tokenizer/generate（分词器训练与生成）不是 catastrophic failure（灾难性失败）：generated collision（生成后冲突）为 `11 / 3686 = 0.0029842648`，`max_conflict = 2`
  - 但 downstream verdict（下游裁决）为负：`NDCG@10 = 0.09518`，`HR@10 = 0.13236`
  - 这高于 `R640c` carrier-only smoothness（仅图载体加平滑监督）的 `0.09306 / 0.13126`，也高于 `R630c` 的 `0.09261 / 0.12972`，但仍明显低于 current `v2_on_p05`（当前 `v2_on_p05`）SFT 的 `0.10271 / 0.14626`
  - 因此当前这版 high-order carrier + explicit push-pull（高阶载体 + 显式拉近推远）不能推进到 `RL`（强化学习）
- `R650a` 后验诊断把新问题进一步收窄到 L1/L2 organization quality（第一层/第二层组织质量）：
  - active L1（活跃第一层码）数量本身不是因果变量；`v2` 和 `R650a` 都在约 200 个 active L1 附近，但 downstream（下游）表现差异很大
  - 更合理的判断是：`L1` 应该承担 semantic-dominant coarse routing（语义主导的粗路由）角色，低歧义、同质、同品牌/同类型物品应该倾向于在 `L1` 更凝聚
  - `L2/L3`（第二层/第三层）再承载 collaborative refinement（协同细分）和局部分辨，而不是让 `L2` 目标无保护地改写 `L1` 的语义入口角色
- `R660a` 已完成 tokenizer/generate（分词器训练与生成），用来验证 `R650a` 的负结果是否主要来自移除了 `L1/L3/semantic`（第一层/第三层/语义）约束：
  - 保持 `R650a` 的 `coarse_seq2g_rel_masked / fagsp_mid_seq2g_rel_masked`（Seq2Graph 加掩码粗图 / 中图）和 mid-only `push-pull`（仅中层推远拉近）
  - 恢复 `v2` 风格的 `coarse_weight = 0.05`、`local_weight = 0.05`、`semantic_coarse_weight = 0.05`、`semantic_mid_weight = 0.025`
  - tokenizer/generate（分词器训练与生成）不是 catastrophic failure（灾难性失败）：generated collision（生成后冲突）为 `12 / 3686 = 0.0032555616`，`max_conflict = 2`
  - active L1（活跃第一层码）从 `R650a` 的 `199` 降到 `181`，但 generated collision（生成后冲突）没有优于 `R650a`
  - 当前状态：tokenizer-side（分词器侧）不是强正信号；如果继续做 `title_history2sid_on + desc_align_p05` 的 `SFT -> evaluate`（监督微调到评测），应视作 diagnostic downstream check（诊断性下游检查）
- `R670a` 已完成 tokenizer/generate（分词器训练与生成），结果为明确负结论：
  - 设计目标是回到更干净的层级分工：
    - 不再堆 `v2` full constraints（全套约束）
    - `L1`（第一层）只用 high-confidence semantic pull（高置信语义拉近）保护粗粒度语义入口
    - `L2`（第二层）用 base `fagsp_mid_base`（基础中层图）做 collaborative pull（协同拉近），并用 semantic-near + mid-weak pairs（语义近但中图弱连接物品对）做 selective push（选择性推远）
    - 打开 `hierarchy_stopgrad_previous_levels=true`，避免 `L2` 辅助 loss（损失）无保护地反传改写 `L1`
  - 但最终 tokenizer（分词器）空间明显塌缩：
    - generated collision（生成后冲突）=`162 / 3686 = 0.04395`
    - active L1（活跃第一层码）=`19`
    - unique L2 pairs（唯一第二层前缀数）=`375`
  - 这说明 `L1` 高置信语义凝聚 + stop-gradient prefix（前缀停梯度）在当前形态下过强地压扁了前缀空间，不建议推进到 `SFT`（监督微调）
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
- 当前正在验证的新分支是 `Seq2Graph-lite`（轻量 `Seq2Graph`） high-order carrier（高阶载体）版本：
  - `R640b`：`L1 <- coarse_seq2g_rel`，`L2 <- fagsp_mid_seq2g_rel`，`L3 <- local_purified`
  - `R640c`：`L1 <- coarse_seq2g_rel_masked`，`L2 <- fagsp_mid_seq2g_rel_masked`，`L3 <- local_purified`
  - `R640c` 下游正式裁决已完成且为负，因此不能推进到 `RL`（强化学习）
  - 如果继续沿这个方向推进，`R640c` 只能作为 future push-pull（后续推远拉近）的候选 carrier substrate（图载体基底），不能作为 standalone method（独立方法）
- `R650a` 是 high-order carrier + explicit push-pull（高阶载体 + 显式推远拉近）的第一版最小实现，已完成 `SFT -> evaluate`（监督微调到评测）且结论为负：
  - `coarse_weight = 0.0`
  - `mid_weight = 0.15`
  - `local_weight = 0.0`
  - `selective_separation_weight = 0.01`
  - 只干预 `L2`（第 2 层），避免重新堆复杂损失
  - generated collision（生成后冲突）为 `11 / 3686 = 0.0029842648`
  - downstream `SFT -> evaluate`（下游监督微调到评测）为 `NDCG@10 = 0.09518`，`HR@10 = 0.13236`
  - 不推进 `RL`（强化学习）
- `R660a` 是 constraint restoration（约束恢复）版本：
  - 保持 `R650a` 的 high-order carrier + mid-only push-pull（高阶载体 + 仅中层推远拉近）
  - 恢复 `v2` 的 `L1/L3/semantic`（第一层/第三层/语义）约束
  - 目的不是把 SID（语义 ID）拉回 baseline（基线），而是测试有明确层级分工保护时，`push-pull`（推远拉近）是否仍可能带来正向信号
  - 当前 tokenizer/generate（分词器训练与生成）已完成：generated collision（生成后冲突）为 `12 / 3686 = 0.0032555616`，`max_conflict = 2`
  - active L1（活跃第一层码）为 `181`，unique L2 pairs（唯一第二层前缀数）为 `2598`
  - 仍需 `SFT -> evaluate`（监督微调到评测）才能形成 downstream verdict（下游裁决）
- `R670a` 是 clean hierarchy（干净层级分工）版本：
  - 删除 `coarse graph / local graph / semantic_mid`（粗图 / 局部图 / 第二层语义保持）等额外项
  - 只保留 `RQ`（残差量化）、`L1` high-confidence semantic graph smoothness（第一层高置信语义图平滑）、`L2` base collaborative graph smoothness（第二层基础协同图平滑）和 `L2` selective separation（第二层选择性分离）
  - 使用 stop-gradient prefix（前缀停梯度）让 `L2` 目标主要塑造 `q2`，不直接拉散 `q1`
  - tokenizer/generate（分词器训练与生成）已完成，但给出明确负结果：
    - train best collision（训练最佳冲突率）=`0.4851`
    - generated collision（生成后冲突）=`162 / 3686 = 0.04395`
    - max conflict（最大冲突簇）=`35`
    - active L1（活跃第一层码）=`19`
    - unique L2 pairs（唯一第二层前缀数）=`375`
  - 结论：这是一次 tokenizer collapse（分词器塌缩），不进入后续 `SFT -> evaluate`（监督微调到评测）

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

- `R680a`：`L1 smooth + L2 contrastive + L3 smooth`（第一层平滑 + 第二层对比式 + 第三层平滑）已完成 tokenizer/generate（分词器训练与生成）：
  - 输出目录：
    - `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-18_mgr_sid_r680_l1_smooth_l2_contrastive_multihop_industrial`
  - 当前设计：
    - `L1 <- coarse_purified`
    - `L2 <- local_multihop`
    - `L3 <- local_purified`
    - `hierarchy_stopgrad_previous_levels = true`
    - `coarse_weight = 0.05`
    - `mid_weight = 0.0`
    - `local_weight = 0.05`
    - `l2_contrastive_pull_weight = 0.15`
    - `selective_separation_weight = 0.01`
  - pair source（物品对来源）：
    - `mid_view_name = local_multihop`
    - `semantic_pair_count = 82596`
    - `weak_pair_count = 1738`
    - `weak_pair_item_coverage_rate = 0.4881`
    - `weak_threshold = 0.00280704`
  - 当前结果：
    - `tmux`（终端复用器） session `mgr_r680a_l1_smooth_l2_contrastive_multihop` 已结束
    - train best collision（训练最佳冲突率）: `0.0984807379`
    - generated collision（生成后冲突）: `11 / 3686 = 0.0029842648`
    - max conflict（最大冲突簇）: `2`
    - active L1（活跃第一层码）: `226`
    - unique L2 pairs（唯一第二层前缀数）: `2833`
  - 当前作用：
    - 这是第一条真正把 `L2` supervision interface（第二层监督接口）从 graph smoothness（图平滑）切到 pairwise pull + selective push（成对拉近 + 选择性推远）的 clean test（干净测试）
    - 它已经完成 `2` 卡 effective-batch-aligned（有效批大小对齐）的 `SFT -> evaluate`（监督微调到评测）：
      - `batch_size = 1024`
      - `micro_batch_size = 2`
      - `nproc_per_node = 2`
      - `gradient_accumulation_steps = 256`
      - effective batch（有效批大小）仍保持 `1024`
    - downstream result（下游结果）为负，但优于近期多个负分支：
      - `NDCG@1/3/5/10 = 0.06883 / 0.08497 / 0.09038 / 0.09864`
      - `HR@1/3/5/10 = 0.06883 / 0.09707 / 0.11008 / 0.13567`
      - 相比 `R650a`：
        - `NDCG@10`: `0.09518 -> 0.09864`
        - `HR@10`: `0.13236 -> 0.13567`
      - 但相比 current `v2_on_p05`（当前 `v2_on_p05`）仍然落后：
        - `NDCG@10`: `0.10271 -> 0.09864`
        - `HR@10`: `0.14626 -> 0.13567`
    - 当前结论：
      - `R680a` 不能推进到 `RL`（强化学习）
      - 但它说明 `L2 contrastive interface`（第二层对比式接口）这条线并非无效，只是当前版本还不够强
- `R690`：`CoST-inspired contrastive quantization`（受 CoST 启发的对比式量化）双分支已正式启动：
  - 输出目录：
    - `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-18_mgr_sid_r690_cost_inspired_contrastive_quantization_industrial`
  - 共享设计：
    - 使用 `fagsp_mid_base`（基础中层图）作为 `L2`（第二层）正样本来源
    - 使用 semantic-near + mid-weak（语义近但中图弱连接）物品对作为 `L2 InfoNCE`（第二层对比学习损失）负样本来源
    - 共享 pair source（物品对来源）统计：
      - `semantic_pair_count = 82596`
      - `weak_pair_count = 1211`
      - `weak_pair_item_coverage_rate = 0.2797`
      - `weak_threshold = 0.00161124`
  - `R690a`：
    - `RQ + L2 graph-guided InfoNCE`（残差量化 + 第二层图引导对比损失）
    - 不加 `L1/L3`（第一层/第三层）保护
    - `hierarchy_stopgrad_previous_levels = false`
    - `mid_view_name = fagsp_mid_base`
    - 结果：
      - best train collision（训练最佳冲突率）: `0.0887140532`
      - generated collision（生成后冲突）: `11 / 3686 = 0.0029842648`
      - active L1（活跃第一层码）: `118`
      - unique L2 pairs（唯一第二层前缀数）: `2527`
  - `R690b`：
    - `RQ + L1 semantic pull + L2 graph-guided InfoNCE + L3 local pull`（残差量化 + 第一层语义拉近 + 第二层图引导对比损失 + 第三层局部拉近）
    - 打开 stop-gradient prefix（前缀停梯度）保护前层
    - `mid_view_name = fagsp_mid_base`
    - 结果：
      - best train collision（训练最佳冲突率）: `0.1120455779`
      - generated collision（生成后冲突）: `14 / 3686 = 0.0037981552`
      - active L1（活跃第一层码）: `33`
      - unique L2 pairs（唯一第二层前缀数）: `1989`
  - 当前作用：
    - 这是第一条正式把 `CoST`（基于对比量化的语义分词）思路和我们当前 graph-structured collaborative signal（图结构协同信号）合并的支线
    - 目前已经可以回答一个关键问题：
      - 这两个实验的 `mid graph`（中图）都不是 `local_multihop`（局部多跳图），而是 `fagsp_mid_base`（基础中层图）
    - 当前判读：
      - `R690a` 是 non-catastrophic tokenizer（非灾难性分词器）候选，更值得先推进 `SFT -> evaluate`（监督微调到评测）
      - `R690b` 没有灾难性失败，但有明显的 prefix over-compression（前缀过度压缩）风险
- `R693a`：hierarchical collaboration-only multihop（层级纯协同多跳版）已完成 tokenizer/generate（分词器训练与生成）：
  - 输出目录：
    - `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-18_mgr_sid_r693_hier_collab_only_multihop_industrial`
  - 当前设计：
    - 不再使用显式 semantic graph supervision（语义图监督）
    - `L1 <- coarse_purified`（净化粗图）的高置信正边图
    - `L2 <- local_multihop`（局部多跳图）作为 `InfoNCE`（对比学习损失）正样本来源
    - `L2` 负样本来自 `coarse candidate + multihop weak`（粗图候选 + 多跳弱连接）
    - `L3 <- local_purified`（净化局部图）
    - `hierarchy_stopgrad_previous_levels = true`
  - 当前图源统计：
    - `l1_graph_undirected_edge_count = 2191`
    - `l1_graph_item_coverage_rate = 0.7982`
    - `weak_pair_count = 4519`
    - `weak_pair_item_coverage_rate = 0.8304`
    - `weak_threshold_mean = 0.04627`
  - 运行状态：
    - 第一轮长跑曾因 `L2 InfoNCE`（第二层对比学习损失）空行数值稳定性问题失败
    - 该 `InfoNCE`（对比学习损失）缺陷已修复，修复后重跑完整完成
    - tokenizer registry（分词器总账）已记录有效重跑结果
  - 当前作用：
    - 这是目前最接近“语义由 `RQ-VAE`（残差量化变分自编码器）`主干保留，显式辅助项只注入协同信息`”的新主线候选
    - 它直接测试：`R690b` 的主问题到底更像是 `L1` 监督源不对，还是 `L2` 载体不对
  - 当前结果：
    - best train collision（训练最佳冲突率）: `0.1009224091`
    - generated collision（生成后冲突）: `12 / 3686 = 0.0032555616`
    - max conflict（最大冲突簇）: `2`
    - active L1（活跃第一层码）: `90`
    - unique L2 pairs（唯一第二层前缀数）: `2274`
  - 当前结论：
    - `R693a` 是 non-catastrophic tokenizer candidate（非灾难性分词器候选），但下游 `SFT -> evaluate`（监督微调到评测）已经给出负裁决
    - 相比 `R690b`，它明显缓解了 prefix over-compression（前缀过度压缩）：active L1（活跃第一层码）从 `33` 回升到 `90`
    - 相比 `R690a`，它略微更收紧：active L1（活跃第一层码）从 `118` 降到 `90`，但 generated collision（生成后冲突）从 `11` 变成 `12`
    - 下游结果：`NDCG@1/3/5/10 = 0.06309 / 0.07972 / 0.08678 / 0.09731`，`HR@1/3/5/10 = 0.06309 / 0.09155 / 0.10876 / 0.14163`
    - 相比 `R680a`，`HR@10`（命中率@10）更高，但 `NDCG@1/3/5/10`（归一化折损累计增益）全线更低；相比 current `v2_on_p05`（当前 v2_on_p05）仍明显落后
    - 结论：不推进 `RL`（强化学习）
- `D640`：`Seq2Graph-lite`（轻量 `Seq2Graph`） offline graph audit（离线图审计）已完成：
  - 输出目录：
    - `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_d640_seq2graph_lite_graph_audit_industrial`
  - 当前结果：
    - `coarse_seq2g_ctx_only`
      - `connected_item_rate = 0.9891`
      - `rescue_edge_ratio = 0.5407`
    - `coarse_seq2g_rel`
      - `connected_item_rate = 0.9891`
      - `rescue_edge_ratio = 0.5325`
      - hotspot visibility（热点可见性）最好：`0.1667 -> 0.3667`
    - `coarse_seq2g_rel_masked`
      - `connected_item_rate = 0.9891`
      - `rescue_edge_ratio = 0.6049`
      - direct-zero rescue purity（直接零连接补盲纯度）最好
  - 当前结论：
    - 这一步已经足够说明 `Seq2Graph-lite`（轻量 `Seq2Graph`）不是空转分支
    - `R640a / R640b / R640c` 都值得进入 tokenizer-side（分词器侧）最小正式筛选
    - 其中 `R640b`（可靠性版）和 `R640c`（可靠性加掩码版）最值得优先看
- `R640b / R640c`：`Seq2Graph-lite`（轻量 `Seq2Graph`） tokenizer screen（分词器筛选）已完成：
  - 输出目录：
    - `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_r640_seq2graph_lite_industrial`
  - `R640b`
    - `L1 <- coarse_seq2g_rel`
    - `L2 <- fagsp_mid_seq2g_rel`
    - `L3 <- local_purified`
    - first eval collision（首次评估冲突率）: `0.9997287032`
    - best train collision（训练最佳冲突率）: `0.9283776451`
    - generated collision（生成后冲突率）: `0.4120998372`
    - 具体表现：
      - `1519 / 3686` 个 item（物品）发生冲突
      - 最大冲突簇 `max_conflict = 310`
    - 失败诊断：
      - `rel`（可靠性版）相对 `rel_masked`（可靠性加掩码版）多保留了 `17,960` 条边，占全部 `rel` rescue edges（补盲边）的 `15.97%`
      - 这批被 `mask`（掩码）删掉的边全部都是 direct-strong（直接强连接）边：`direct_support >= 0.5`
      - 其平均 direct support（直接支持度）达到 `9.13`，而 `rel_masked` 保留下来的边平均仅 `0.00063`
      - 它们高度集中在 dense same-brand families（稠密同品牌家族），例如 `HATCHBOX`、`Small Parts`、`uxcell`、`3D Solutech`
    - 当前结论：
      - 这是一次明确的 catastrophic failure（灾难性失败）
      - 当前不能再把 reliability-only rescue（仅可靠性感知补盲）当作可直接推进的主分支
  - `R640c`
    - `L1 <- coarse_seq2g_rel_masked`
    - `L2 <- fagsp_mid_seq2g_rel_masked`
    - `L3 <- local_purified`
    - first eval collision（首次评估冲突率）: `0.9997287032`
    - best train collision（训练最佳冲突率）: `0.1242539338`
    - generated collision（生成后冲突率）: `0.0032555616`
    - 具体表现：
      - 仅 `12 / 3686` 个 item（物品）发生冲突
      - 最大冲突簇 `max_conflict = 3`
    - 当前结论：
      - 这是 `R640` 三路里唯一通过 catastrophic failure filter（灾难性失败过滤）的候选
      - 已按 `R645` 推进到 `title_history2sid_on + desc_align_p05` 下游裁决
- `R645`：`R640c -> SFT -> evaluate`（监督微调到评测）已完成：
  - 输出目录：
    - `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-17_mgr_sid_r640c_sft_eval_industrial`
  - recipe（配方）：
    - `title_history2sid_on + desc_align_p05`
  - 当前状态：
    - `FINISHED_NEGATIVE`
  - 结果：
    - `NDCG@10 = 0.09305728`
    - `HR@10 = 0.13125965`
    - `NDCG@1 = 0.06265167`
    - `HR@50 = 0.22678138`
  - 当前作用：
    - 这是 `Seq2Graph-lite`（轻量 `Seq2Graph`） high-order carrier（高阶载体）方向的第一次正式 downstream verdict（下游裁决）
    - 结果低于 current `v2_on_p05`（当前 `v2_on_p05`）SFT 的 `NDCG@10 = 0.10271`，也低于 strongest original SFT（原版最强 SFT）的 `NDCG@10 = 0.10372`
    - 因此 `R640c` carrier-only smoothness（仅图载体加平滑监督）不应推进到 `RL`（强化学习）
- `R650a`：`Seq2Graph-lite`（轻量 `Seq2Graph`） high-order carrier + mid-only push-pull（高阶载体 + 仅中层推远拉近）已完成 `SFT -> evaluate`（监督微调到评测）：
  - 输出目录：
    - `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-17_mgr_sid_r650_seq2graph_push_pull_industrial`
  - recipe（配方）：
    - `title_history2sid_on + desc_align_p05`
  - 当前状态：
    - `COMPLETED_NEGATIVE`
  - pair source（物品对来源）：
    - `mid_view_name = fagsp_mid_seq2g_rel_masked`
    - `semantic_pair_count = 82596`
    - `weak_pair_count = 1190`
    - `weak_pair_item_coverage_rate = 0.2990`
    - `weak_threshold = 0.00169437`
  - 当前作用：
    - 直接检验“`R640c` 图放进 `push-pull`（推远拉近）里”是否能避免 carrier-only smoothness（仅图载体加平滑监督）的负结果
    - 给 high-order carrier + explicit push-pull（高阶载体 + 显式推远拉近）第一次完整 downstream verdict（下游裁决）
  - 当前结果：
    - train best collision（训练最佳冲突率）: `0.1142159523`
    - best loss（最佳损失）: `0.2820739150`
    - generated collision（生成后冲突）: `11 / 3686 = 0.0029842648`
    - max conflict（最大冲突簇）: `2`
  - 下游结果：
    - `NDCG@1/3/5/10 = 0.06530 / 0.08132 / 0.08778 / 0.09518`
    - `HR@1/3/5/10 = 0.06530 / 0.09354 / 0.10920 / 0.13236`
    - constraint invalid total（约束失配总数）: `0`
  - 当前限制：
    - 虽然它略高于 `R640c` 和 `R630c`，但远低于 current `v2_on_p05`（当前 `v2_on_p05`）SFT 的 `NDCG@10 = 0.10271`，也低于 strongest original SFT（原版最强 SFT）的 `0.10372`
    - 因此 `R650a` 不应推进到 `RL`（强化学习）
- `R660a`：constraint restoration（约束恢复）tokenizer/generate（分词器训练与生成）已完成：
  - 输出目录：
    - `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-17_mgr_sid_r660_constraint_restoration_industrial`
  - `tmux`（终端复用器） session：
    - `mgr_r660a_constraint_restoration`，已结束
  - runtime GPU（运行显卡）：
    - `CUDA_VISIBLE_DEVICES=7`
  - 设计：
    - `R660a = R650a + v2-style full constraints`（R650a 加回 v2 风格全套约束）
    - `coarse_weight = 0.05`
    - `mid_weight = 0.15`
    - `local_weight = 0.05`
    - `semantic_coarse_weight = 0.05`
    - `semantic_mid_weight = 0.025`
    - `selective_separation_weight = 0.01`
  - 当前作用：
    - 验证 `R650a` 负结果是否主要来自 L1/L3/semantic（第一层/第三层/语义）约束缺失
    - tokenizer/generate（分词器训练与生成）已经通过非灾难性检查：generated collision（生成后冲突）为 `12 / 3686 = 0.0032555616`，`max_conflict = 2`
    - active L1（活跃第一层码）为 `181`，低于 `R650a` 的 `199`，但 tokenizer-side（分词器侧）没有优于 `R650a`
    - 下一步如果继续该分支，应接 `title_history2sid_on + desc_align_p05` 下游裁决，但定位为诊断性检查而非强候选推进
- `R670a`：clean L1 semantic + L2 push-pull（干净第一层语义 + 第二层推远拉近）已完成：
  - 输出目录：
    - `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-18_mgr_sid_r670_clean_l1_semantic_l2_push_pull_industrial`
  - 设计：
    - `semantic_coarse_weight = 0.08`
    - `mid_weight = 0.15`
    - `selective_separation_weight = 0.01`
    - `coarse_weight = 0.0`
    - `local_weight = 0.0`
    - `semantic_mid_weight = 0.0`
    - `hierarchy_stopgrad_previous_levels = true`
  - 当前结果：
    - best train collision（训练最佳冲突率）=`0.4850786761`
    - generated collision（生成后冲突）=`162 / 3686 = 0.0439500814`
    - max conflict（最大冲突簇）=`35`
    - active L1（活跃第一层码）=`19`
    - unique L2 pairs（唯一第二层前缀数）=`375`
  - 当前结论：
    - 这不是边缘负结果，而是前缀空间明显塌缩
    - 因此不进入后续 `SFT -> evaluate`（监督微调到评测）
- `R610a`：base `v2` + `L3`-only `reliability-aware selective separation`（仅 `L3` 的可靠性感知选择性分离）已完成：
  - 当前只保留一个很弱的事实：
    - 这是一个已经落地并完成 `sid-generate`（SID 生成）的 tokenizer（分词器）变体
  - 但它**不是**当前 active candidate（活跃候选）：
    - supporting diagnostics（支撑诊断）已经退役
    - downstream transfer（下游迁移）还没有验证
- `R630a / R630b / R630c`：mid-only pull/push（仅中层拉近/推远）分支已完成训练与 `sid-generate`（SID 生成）：
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
  - 当前结果：
    - `R630a`: `16 / 3686 = 0.0043407488`
    - `R630b`: `15 / 3686 = 0.0040694520`
    - `R630c`: `11 / 3686 = 0.0029842648`
  - 当前可得结论：
    - `pull-only`（仅拉近）不够
    - `push-only`（仅推远）也不够
    - 只有 `pull + push`（拉近 + 推远）在这组三路里给出了明显更强的 tokenizer-side（分词器侧）结果
  - 但需要明确保留的限制：
    - `R630c` 的 `11 / 3686` 只是 tokenizer-side（分词器侧）最强
    - 它目前只**匹配**了已经被下游否掉的 `R510`
    - 所以现在还不能把 `R630c` 直接当成新 strongest validated line（最强已验证主线）
- `R630c -> SFT -> evaluate`（监督微调到评测）已正式发起：
  - recipe（配方）：
    - `title_history2sid_on + desc_align_p05`
  - 目的：
    - 给当前 selective separation（选择性分离）方向唯一值得继续的 tokenizer candidate（分词器候选）一次最小正式下游裁决
  - 当前状态：
    - `COMPLETED`
  - 当前结果：
    - `NDCG@10 = 0.09261`
    - `HR@10 = 0.12972`
  - 对比 current `v2_on_p05`（当前 `v2_on_p05`）：
    - `NDCG@10`: `0.10271 -> 0.09261`
    - `HR@10`: `0.14626 -> 0.12972`
  - 当前结论：
    - 这是一次**明确负结论**
    - `R630c` 虽然 tokenizer-side（分词器侧）优于 current `v2` 和 `R610a`
    - 但这种改进没有转化成更强的 downstream `SFT`（下游 `SFT`）表现
    - posterior output diagnosis（后验输出诊断）显示：
      - 很多失败不是“完全找不到 target（目标）”
      - 而是 target（目标）从 `top10` 掉到 `11-50` 或 `>50`
      - 因此当前更像 beam-retention failure（候选束保留失败），而不是简单的 local ambiguity（局部歧义）没有解决
    - 因此当前这版 `mid-only pull + push`（仅中层拉近加推远）不能进入 strongest line（最强主线）
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
- 后续允许保留的诊断方式：
  - posterior output diagnosis（后验输出诊断）
  - 但它的职责只是解释模型输出错在什么地方，不能反过来充当前验 tokenizer promotion gate（分词器前验推进门槛）
- 归档位置：
  - `/home/leejt/OneRec/research-progress-log/archive/2026-04-16_retired_prior_diagnostics/README.md`
  - `/home/leejt/OneRec/scripts/archive/retired_prior_diagnostics/README.md`

## 下一步最合理的动作

1. 不要把 `R630c`、`R640c` 或 `R650a` 推进到 `RL`（强化学习）；它们的 `SFT -> evaluate`（监督微调到评测）都已经给出负裁决。
2. 不要把 `R670a` 推进到 `SFT`（监督微调）；它的 tokenizer（分词器）空间已经明显塌缩，继续下游只会浪费算力。
3. `R680a` 已经给出完整下游裁决：它是“较强负结果”，不推进 `RL`（强化学习），但可以作为后续 `L2`（第二层）对比式接口线的参考锚点。
4. 当前主动作已经从 `R690a / R690b` 收束到 `R693a`，并且 `R693a` 修复后已经完成 tokenizer/generate（分词器训练与生成）和 `SFT -> evaluate`（监督微调到评测）：
   - `R690b` 说明“分层对比式骨架”本身是有方法感的，但当前 `L1`（第一层）监督仍然偏语义、`L2`（第二层）载体仍然是 `fagsp_mid_base`（基础中层图）。
   - `R693a` 把这两个问题一次性改掉：
     - `L1` 改成高置信 `coarse_purified`（净化粗图）正边
     - `L2` 改成 `local_multihop`（局部多跳图）正样本
   - 修复后 generated collision（生成后冲突）=`12 / 3686`
   - active L1（活跃第一层码）=`90`
   - unique L2 pairs（唯一第二层前缀数）=`2274`
   - 下游 `NDCG@10 = 0.09730760`，`HR@10 = 0.14162806`
   - 这是一个有效但负向的下游裁决，不推进 `RL`（强化学习）。
5. 当前最合理的近期动作是回到 `R690/R693` 这条分层 `InfoNCE`（对比学习损失）主线做归因：`R693a` 改善了 `L1`（第一层）形态但没有改善排序质量，说明需要检查 `L2`（第二层）对比目标和负样本构造是否把可学习排序信号削弱了。
6. 如果后续要继续沿 `R680a` 线改，不该再把目标表述成“让 SID 更稳定”，而应更明确地围绕：
   - 如何让 `L2` 的对比分辨增益延续到 `@10`
   - 如何避免 `L1` 仍然偏碎
   - 如何让浅层 routing（浅层路由）改善转化成更深层 candidate quality（候选质量）改善

## 使用方式

- 这是唯一应该持续维护的 current-state document（当前状态文档）。
- 想看完整实验账本，请查 `/home/leejt/OneRec/research-progress-log/experiment_registry/README.md`。
- 想看下游指标排序，请查 `/home/leejt/OneRec/research-progress-log/experiment_registry/downstream_scoreboard.csv`。
- 想看阶段实验记录，请查 `/home/leejt/OneRec/research-progress-log/experiment_launches/README.md`。
- 想看代码对齐的方法公式，请查 `/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/19_mgr_sid_current_method_code_aligned_formulas.md`。
