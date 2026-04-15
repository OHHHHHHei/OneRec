# Experiment Plan

**问题**：当前 `graph bank`（图库）主要来自行为投影得到的 `item-item graph`（物品-物品图），这条路已经证明“有用”，但也越来越像当前 `MGR-SID` 的瓶颈之一。我们怀疑，真正卡住我们的，不一定只是 `SID structure`（SID 结构）或 `loss`（损失）形式，而可能是“承载协同信息的图本身还不够好”。  
**方法主张**：借鉴 `TAGCF` 的“turn semantics into topology（把语义变成拓扑）”思想，构造 `item-attribute-item`（物品-属性-物品）属性拓扑视图，把它作为新的 `graph carrier`（图载体）接入当前 `SID tokenizer`（SID 分词器）训练，优先验证“更好的图”能否带来更好的 `SID codebook space`（SID 码本空间）。  
**日期**：2026-04-14  
**定位**：探索性支链（exploratory branch，探索支链），不是当前 stage-3 主执行线的替代品。

## 一、问题锚点

这条支链要回答的不是：

- 当前 `R401*` 这类保守 refinement（细化）是否还值得继续；

而是更上游的问题：

> 如果当前 `graph bank`（图库）本身太粗，那么继续在现有图上修 `SID`，会不会已经接近“错的局部最优”？

我们已经有几个现实信号支持这个怀疑：

1. `v2` 已经证明图结构协同信息是有用的，但收益没有继续稳定放大。
2. 很多 tokenizer 分支能把结构指标做漂亮，却不能稳定带来更好的下游 `evaluate`（评测）。
3. 当前 `G_mid`（中尺度图）虽然有效，但更像“从已有行为图做谱变换出来的视图”，而不是一张独立、扎实、信息源更丰富的中尺度图。
4. `TAGCF` 提醒我们：语义信息不一定只能进 embedding（嵌入），也可以先进拓扑，再让推荐系统去消费它。

这条支链的核心假设是：

> 对当前 `MGR-SID` 来说，“更好的图载体”可能比“更复杂的 retention（保持）项”更值得优先探索。

## 二、Claim Map（主张地图）

| Claim | 为什么重要 | 最低说服证据 | 对应实验块 |
|---|---|---|---|
| C1: 语义转拓扑得到的属性图，有机会成为比当前纯行为投影图更好的 `G_mid`（中尺度图）载体。 | 如果成立，说明当前瓶颈可能真在图设计，而不是只在 `SID` 结构细调。 | 至少一个 `TAGCF-inspired`（受 TAGCF 启发）分支，在 full `SFT -> evaluate` 下优于当前 `v2_on_p05`。 | B1, B2, B4 |
| C2: 真正有价值的不是“加更多文本特征”，而是“把语义变成拓扑，再接进当前 tokenizer（分词器）主干”。 | 这决定我们是在做 graph upgrade（图升级），还是只是又加一层语义噪声。 | 属性拓扑分支优于简单关键词/原始文本投影控制组。 | B1, B3, B4 |

### 需要排除的 Anti-claims（反主张）

- A1：任何多加一点语义信息都会涨，和“拓扑化”无关。
- A2：一定要照搬 `TAGCF` 的整套异构 `GNN`（图神经网络）主干才有用。
- A3：属性图对三层 `SID` 都一样重要，不存在最适合的注入位置。
- A4：当前图设计已经不是瓶颈，收益真正来自下游 `SFT/RL` recipe（配方）而不是 tokenizer 侧图载体。

## 三、为什么这条支链值得做

### 3.1 它和当前主线不冲突

当前主线仍然是：

- 找到更好的 `SID codebook space`（SID 码本空间）
- 最终用 full downstream `SFT -> evaluate` 选择

这条 `TAGCF` 支链只是换一个切入点：

- 主线在问“怎样得到更好的码本空间？”
- 这条支链在问“是不是图载体本身就该升级？”

### 3.2 它比直接堆更复杂 loss（损失）更干净

如果我们连图都还比较粗，就继续堆：

- `retention loss`（保持损失）
- `predictability regularizer`（可预测性正则）
- 更多层级约束

很容易出现一种情况：

> 我们在一个不够好的图载体上，不断精修训练目标，但始终没有碰到真正限制质量的上游因素。

### 3.3 它可以低侵入接入现有代码

这条支链最吸引人的地方在于：

- 不需要改 `MiniOneRec` 下游主干
- 不需要先上重型异构 `GNN`
- 可以先通过新增 graph view（图视图）接入当前 `transplanted_graph_bank.py`

也就是说，它非常适合做“先小范围验证，再决定是否值得升级成完整方法”。

## 四、方法范围：借什么，不借什么

### 4.1 借什么

从 `TAGCF` 借这三件事：

1. **语义转拓扑（semantics to topology，语义转拓扑）**
   - 不把语义只当 embedding（嵌入）
   - 把语义先转成属性节点，再转成图

2. **属性节点（attribute nodes，属性节点）作为中介结构**
   - 先构 `item -> attribute -> item`
   - 再投影得到新的 `item-item` 图

3. **过滤与融合（filtering and fusion，过滤与融合）**
   - 原始属性不能直接用
   - 需要做低频裁剪、同义合并、噪声属性清洗

### 4.2 暂时不借什么

第一波不借这些：

- 不直接照搬 `TAGCF` 的异构 `GNN`（图神经网络）主干
- 不做每个 `user-item pair`（用户-物品对）的 `LLM`（大语言模型）意图推理
- 不在第一轮就把整个 tokenizer（分词器）训练目标重写成属性驱动

理由很简单：

- 太重
- 太贵
- 和当前 `MGR-SID` 代码接口不对
- 很难判断到底是“新图有用”还是“模型变复杂了”

## 五、拟探索的方法族

## 5.1 核心对象：属性拓扑图

目标不是直接训练一个新推荐器，而是先离线构造一个新 graph family（图族）：

- `G_attr_raw`
  - 原始属性投影图
- `G_attr_fused`
  - 经过过滤与融合后的属性投影图
- `G_attr_mix`
  - 与当前 `fagsp_mid_base` 混合后的中尺度图

构法直觉：

1. 为每个 item（物品）从 `title / desc / metadata`（标题/描述/元数据）抽 3-8 个属性。
2. 构造 `item-attribute incidence matrix`（物品-属性关联矩阵）`A`。
3. 用 `A A^T` 得到 `item-item attribute projection`（物品-物品属性投影图）。
4. 做归一化、裁剪、融合，形成候选 graph view（图视图）。

## 5.2 三个第一波候选注入方式

### Variant T1：`G_mid` 直接替换

- 现有：
  - `L1 <- G_coarse`
  - `L2 <- fagsp_mid_base`
  - `L3 <- G_local`
- 新版：
  - `L1 <- G_coarse`
  - `L2 <- G_attr_fused`
  - `L3 <- G_local`

**要回答的问题**：
属性拓扑能不能直接承担当前 `G_mid` 的角色？

### Variant T2：`G_mid` 混合

- `L2 <- mix(fagsp_mid_base, G_attr_fused)`

例如：

- `G_mid_mix = lambda * fagsp_mid_base + (1 - lambda) * G_attr_fused`

**要回答的问题**：
属性图是不是更适合作为中尺度补充，而不是完全替代当前行为中频图？

### Variant T3：属性图做边重权（edge reweighting，边重加权）

- 保留当前三层图角色不变
- 用属性图相似度去重加权：
  - `G_coarse`
  - `G_local`
  - 或二者之一

**要回答的问题**：
属性拓扑的最佳用法，会不会不是“替代 `G_mid`”，而是“给现有协同行为图去噪/加权”？

## 5.3 一个重要控制组

我们需要一个明确控制组，防止把“语义拓扑化”误判成“任何文本信号都行”。

建议控制组：

- `Attr-Heuristic`（启发式属性）
  - 不用 `LLM`
  - 只用 `title/desc` 的关键词或名词短语

和它对比的是：

- `Attr-LLM-Raw`（LLM 原始属性）
- `Attr-LLM-Fused`（LLM 过滤融合属性）

这样才能回答：

> 真正有用的是 `TAGCF` 那种“语义转拓扑”思想，还是只是简单文本标签也够了？

## 六、Experiment Blocks（实验块）

### Block 1：属性拓扑构图可行性验证

- **Claim tested（检验主张）**：
  - 我们可以构造出一张质量可接受、覆盖面足够、不过度稀碎的属性拓扑图。
- **Why this block exists（为什么做）**：
  - 如果图本身都构不稳，后面训练没有意义。
- **Dataset / task（数据与任务）**：
  - Industrial
  - 只做离线属性抽取与图构建
- **Compared systems（比较系统）**：
  - `A0`: `Attr-Heuristic`
  - `A1`: `Attr-LLM-Raw`
  - `A2`: `Attr-LLM-Fused`
- **Metrics（指标）**：
  - 属性覆盖率：有属性的 item 占比
  - 平均每个 item 的属性数
  - 唯一属性节点数
  - 投影后图密度
  - 最大连通分量占比
  - 冷启动 item 接通率
  - 与当前 `G_mid` 的 top-k 邻居重合度
  - 人工抽样质量检查：随机 100 个 item，看属性是否可读、可解释、不过度噪声
- **Success criterion（成功标准）**：
  - `Attr-LLM-Fused` 形成的图不是塌的：
    - 覆盖率高
    - 图不是过度稠密或过度碎裂
    - 属性样本质量显著优于 `Attr-Heuristic`
- **Failure interpretation（失败解释）**：
  - 如果连这里都不稳定，说明第一轮不该急着上 tokenizer（分词器）训练，应该先把属性抽取和融合流程打磨好。
- **Target artifact（目标产物）**：
  - `item_attributes.jsonl`
  - `attribute_vocab.json`
  - `item_attribute_matrix.npz`
  - `G_attr_raw.npz`
  - `G_attr_fused.npz`
- **Priority（优先级）**：MUST-RUN（必须）

### Block 2：属性拓扑进入 tokenizer（分词器）的注入位置筛选

- **Claim tested（检验主张）**：
  - 属性拓扑进入 `SID tokenizer` 的最佳位置，很可能是 `G_mid`，而不是一上来全图替换。
- **Why this block exists（为什么做）**：
  - `TAGCF` 的启发要变成我们的方法，关键不只是“有没有属性图”，而是“它放在哪一层最值”。
- **Dataset / task（数据与任务）**：
  - Industrial
  - `sid-train -> sid-generate`
- **Compared systems（比较系统）**：
  - `B0`: 当前 best `v2`
  - `B1`: `T1` = `G_mid <- G_attr_fused`
  - `B2`: `T2` = `G_mid <- mix(fagsp_mid_base, G_attr_fused)`
  - `B3`: `T3` = `attr-gated coarse/local`
- **Metrics（指标）**：
  - 训练健康性：
    - reconstruction（重建）是否异常
    - codebook usage（码本使用）是否崩坏
  - 生成后健康性：
    - final collision（最终冲突）
    - local ambiguity（局部歧义）相关指标
    - `R302` 代码多义性
  - 图侧解释：
    - 新图与原图的边覆盖差异
    - 新图是否让 `L2` 附近的 item 组织更有解释性
- **Success criterion（成功标准）**：
  - 至少有一个 `TAGCF-inspired` 候选没有出现明显训练/生成异常，并形成一个值得推下游的候选码本空间。
- **Failure interpretation（失败解释）**：
  - 如果三种放法都不行，问题可能不在注入位置，而在属性抽取噪声或图构法本身。
- **Table / figure target（表图目标）**：
  - tokenizer-side（分词器侧）对比表
  - graph-view（图视图）可视化附图
- **Priority（优先级）**：MUST-RUN（必须）

### Block 3：`LLM`（大语言模型）必要性与“拓扑化”必要性检查

- **Claim tested（检验主张）**：
  - 真正有价值的是“高质量属性拓扑”，而不是“随便多加一点文本标签”。
- **Why this block exists（为什么做）**：
  - 否则 reviewer（审稿人）很容易说：你只是给 item 加了点文本 side information（旁路信息）。
- **Dataset / task（数据与任务）**：
  - Industrial
  - 优先在 Block 2 里最好的注入位置上做
- **Compared systems（比较系统）**：
  - `C0`: `Attr-Heuristic`
  - `C1`: `Attr-LLM-Raw`
  - `C2`: `Attr-LLM-Fused`
- **Metrics（指标）**：
  - tokenizer-side（分词器侧）健康指标
  - full downstream `SFT -> evaluate`
- **Success criterion（成功标准）**：
  - `Attr-LLM-Fused` 至少在一个关键层面明显优于 `Attr-Heuristic`
  - 如果下游差距也存在，说明 `LLM + filtering/fusion`（LLM + 过滤融合）确实有必要
- **Failure interpretation（失败解释）**：
  - 如果 `Attr-Heuristic` 和 `Attr-LLM-Fused` 差不多，说明“语义转拓扑”可能对，但 `LLM` 不是必要部件
- **Priority（优先级）**：NICE-TO-HAVE（有价值但可后置）

### Block 4：下游最终裁判

- **Claim tested（检验主张）**：
  - 属性拓扑图如果真的更好，最终应该体现在 full `SFT -> evaluate` 上，而不是只在 tokenizer-side（分词器侧）好看。
- **Why this block exists（为什么做）**：
  - 当前项目已经明确：最终裁判不是 prefix stability（前缀稳定性），也不是 linear probeability（线性可分性），而是 full downstream（全量下游）指标。
- **Dataset / task（数据与任务）**：
  - Industrial
  - `SFT -> evaluate`
  - recipe（配方）固定：
    - `title_history2sid_on + desc_align_p05`
- **Compared systems（比较系统）**：
  - `D0`: 当前 `v2_on_p05`
  - `D1`: best TAGCF-inspired tokenizer branch
  - `D2`: strongest original MiniOneRec baseline（最强原始基线）
- **Metrics（指标）**：
  - `NDCG@1/3/5/10`
  - `HR@1/3/5/10`
  - fanout-stratified（按候选扇出分层）分析
- **Success criterion（成功标准）**：
  - best `TAGCF-inspired` 分支在关键指标上超过当前 `v2_on_p05`
  - 至少优先看 `NDCG@10 / HR@10`
  - 不接受通过严重牺牲 `@1/@3` 来换取中段指标
- **Failure interpretation（失败解释）**：
  - 如果 tokenizer-side（分词器侧）看着不错，但下游不涨，说明“属性拓扑图”还不是更好的 `graph carrier`（图载体），或者注入位置仍不对。
- **Priority（优先级）**：MUST-RUN（必须）

### Block 5：RL（强化学习）确认

- **Claim tested（检验主张）**：
  - 如果属性拓扑分支在 `SFT`（监督微调）阶段有效，收益是否能传到 `RL`（强化学习）阶段。
- **Why this block exists（为什么做）**：
  - 只有在 `SFT` 真的有正信号时才值得烧算力。
- **Dataset / task（数据与任务）**：
  - Industrial
  - `RL -> evaluate`
- **Compared systems（比较系统）**：
  - `E0`: 当前最强 `v2_on_p05 -> RL`
  - `E1`: best TAGCF-inspired `SFT` winner -> RL
- **Metrics（指标）**：
  - `NDCG@5/10/20`
  - `HR@5/10/20`
- **Success criterion（成功标准）**：
  - 至少在一个核心 `top-k`（前 k）层面带来净提升
- **Priority（优先级）**：NICE-TO-HAVE（有价值但后置）

## 七、Run Order and Milestones（执行顺序与里程碑）

| Milestone | Goal（目标） | Runs（运行） | Decision Gate（决策门） | Cost（成本） | Risk（风险） |
|---|---|---|---|---|---|
| M0 | 验证属性抽取和图构建是否可行 | `R500-R502` | `Attr-LLM-Fused` 图是否健康、可解释 | 低到中 | 属性太脏、图太碎 |
| M1 | 先筛选注入位置 | `R510-R512` | 至少一个分支 tokenizer-side 不崩，并值得推下游 | 中 | 图接进训练后副作用大 |
| M2 | 用 full `SFT -> evaluate` 决定这条支链值不值得继续 | `R530` | 是否超过当前 `v2_on_p05` | 高 | tokenizer-side 好看但下游不涨 |
| M3 | 检查 `LLM` 与 filtering/fusion（过滤融合）的必要性 | `R540-R542` | `LLM-Fused` 是否真优于控制组 | 中到高 | 最终发现简单关键词也够 |
| M4 | 如果 SFT 正向，再做 RL 确认 | `R550` | 收益能否穿透到 RL | 很高 | 只在 SFT 有效 |

## 八、建议的首批运行清单

### First three runs to launch（第一批建议先跑）

1. `R500`
   - `Attr-LLM-Raw` 小规模试抽
   - 目标：先看属性格式、覆盖率、噪声形态

2. `R501`
   - `Attr-LLM-Fused`
   - 目标：把 `filtering and fusion`（过滤与融合）真正做出来

3. `R510`
   - `T1: G_mid <- G_attr_fused`
   - 目标：最直接检验属性拓扑能不能承担 `G_mid`

### 如果 `R510` 不理想

优先补：

4. `R511`
   - `T2: G_mid <- mix(fagsp_mid_base, G_attr_fused)`

### 如果 `R510` 或 `R511` 有像样候选

再推：

5. `R530`
   - best TAGCF-inspired branch -> full `SFT -> evaluate`

## 九、Suggested Implementation Path（建议实现路径）

### 9.1 新增离线属性构建模块

建议新增：

- `scripts/experiment_mgr_sid_tagcf_build_attributes.py`
- `src/onerec/experiments/mgr_sid/semantic_topology.py`

职责：

- 读 item 文本
- 产出属性列表
- 做过滤与融合
- 生成属性投影图

### 9.2 接入现有 graph bank（图库）

最自然的接入口是：

- `src/onerec/experiments/mgr_sid/transplanted_graph_bank.py`

新增 view（视图）即可：

- `tagcf_attr_mid_raw`
- `tagcf_attr_mid_fused`
- `tagcf_attr_mid_mix`
- `tagcf_attr_gate_local`

这样不需要先改训练主干。

### 9.3 产物落盘建议

- 轻量资产：
  - `results/tagcf_branch/...`
  - `research-progress-log/...`
- 大型 tokenizer 权重：
  - `/data/leejt/OneRec/output_weights/...`

## 十、风险与规避

- **风险 1**：属性抽取噪声大，图变得更差  
  - **规避**：必须先做 `filtering and fusion`（过滤与融合），不要直接用 raw 属性。

- **风险 2**：属性图和当前协同行为图表达的是不同东西，强行替代 `G_mid` 反而伤性能  
  - **规避**：保留 `mix`（混合）分支，不把“完全替代”当成唯一方案。

- **风险 3**：`LLM` 成本太高，第一轮跑不动  
  - **规避**：先在 Industrial 上做单数据集验证；必要时保留 `Attr-Heuristic` 作为廉价控制组。

- **风险 4**：结果只是 tokenizer-side（分词器侧）好看，下游不涨  
  - **规避**：严格执行 full `SFT -> evaluate` 作为主裁判，不在中间诊断上过度乐观。

## 十一、最终判断

我对这条支链的判断是：

- **值得做**
- **而且值得尽快做**
- 但第一轮必须克制

最好的第一步不是：

- 全量照搬 `TAGCF`
- 上异构 `GNN`
- 大改下游模型

而是：

> 先做一个 `item-attribute-item -> item-item projection`（物品-属性-物品到物品-物品投影）的轻量属性拓扑分支，看它能不能成为比当前 `G_mid` 更好的 graph carrier（图载体）。

如果这一点都不成立，那说明 `TAGCF` 给我们的更多只是思路启发。  
如果这一点成立，再考虑更接近论文原貌的更强版本。

