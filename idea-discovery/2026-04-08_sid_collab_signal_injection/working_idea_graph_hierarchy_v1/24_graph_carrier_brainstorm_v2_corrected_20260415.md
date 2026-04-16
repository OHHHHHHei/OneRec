# Graph Carrier Brainstorm V2（纠偏重做版）

Status（状态）: `discussion-only（仅讨论）`

Created（创建日期）: `2026-04-15`

Supersedes（替代）: `23_graph_carrier_brainstorm_20260415.md`

---

## 极短纠偏：上一轮哪些判断是错的

### 判断错误 1：把 Cross-View Consensus Graph 排在第一位

上一轮我把"语义 + 行为交集图"排在 Top 1，核心逻辑是"语义参与边筛选 → 更高 precision → 更好的下游效果"。

**现在必须纠正**：`R510` 把 `G_mid` 换成了 `G_attr_fused`（一种"语义驱动拓扑"的属性图），完成了完整 `SFT → evaluate`，结果是明确负向：

- `NDCG@10`：`0.09758` vs `v2_on_p05` 的 `0.10271`（**-5.0%**）
- `HR@10`：`0.13148` vs `v2_on_p05` 的 `0.14626`（**-10.1%**）

`R510` 正是"让语义信息参与图结构"这个思路的直接实验。它失败了。

这说明**"语义参与图构建"这个大方向，在当前 pipeline 中，至少不是一个默认高优先级方向**。Cross-View Consensus Graph 和 Semantic-Behavioral Intersection Graph 都属于同一个大类，必须下调。

### 判断错误 2：把 Community-Initialized Hierarchical Graph 排在 Top 3

上一轮我认为"当前三层图来自三种不同构建方法是最大结构性缺陷"，因此社区检测层级图可以从根本上解决这个问题。

**现在需要修正**：这个判断本身不一定错，但它的优先级被高估了。原因是：

- `R510`、`R511`、`R520` 三轮实验表明，**单纯换 `G_mid` 的来源**（不管是属性图、混合图、还是 cascade 滤波）目前都没有产出正信号
- community 层级图要替换全部三层，风险面更大
- 当前没有任何证据说明"三层图来自不同方法"本身就是问题——这只是一个逻辑上的不优雅，不一定是性能瓶颈

### 判断错误 3：低估了"R510/R511/R520 全部没有正信号"这个事实的含义

上一轮我的核心判断是"问题在信号不够精确"。但如果这个判断是对的，那 R510（属性图）应该带来至少部分正信号——因为属性图从一个全新的信号源构建，"精度"应该和原始 co-occurrence 不同。

事实是 R510 不仅没正信号，还明显负向。这迫使我重新考虑：

> 问题可能不只是"信号精度"，而是**当前 graph smoothness loss 机制本身对新图信号的转化效率不够**——或者说，**当前方法框架下，只靠更换图载体能获得的增量可能有限，需要同时引入真正新类型的协同信息**。

### 哪些候选的优先级必须下调

| 候选 | 上一轮排位 | 现在必须下调的原因 |
|------|-----------|-------------------|
| Cross-View Consensus Graph | Top 1 | `R510` 已证明"语义参与图构建"在当前框架下不默认有效 |
| Semantic-Behavioral Intersection Graph | 候选 8 | 和 Cross-View Consensus 同类，受同一实验证据约束 |
| Community-Initialized Hierarchical Graph | Top 3 | R510/R511/R520 表明纯换图载体暂时无正信号，全面替换三层的风险更大 |
| Attribute-Bridge Item Graph | 候选 5 | `R510` 本身就是属性图方向的直接实验，且已负向 |

### 哪些候选仍然成立，但需要换解释方式

| 候选 | 为什么仍然成立 | 需要换的解释方式 |
|------|---------------|-----------------|
| CIR-Reweighted Coarse Graph | 它不是换图源，而是对现有图做边质量评估；R510 失败不影响这个方向 | 不能再说"提升精度就能赢"，应该说"当前 G_coarse 的边质量未经验证，CIR 是唯一可以做 edge-level quality scoring 的方式" |
| Multi-Hop Diffused Transition Graph | 它扩展了 local_purified 的覆盖范围，这和 R510/R511 的 mid-graph 替换是不同的问题 | 不能再说"多跳补密度就够了"，应该强调"1-hop transition 的 coverage 极低，这可能导致 L3 graph supervision 对大量 item 根本不起作用" |
| Cascade-Filtered G_mid | R520 已经跑了，但 generate collision = 14/3686 不如 v2 的 13/3686 | 这条不算"仍然成立"——R520 已经给出了初步负信号，应该等 R520 的结构诊断再判断 |

---

## 一、A 类：Same-Source Graph Refinement（同源图精修）

这一类的共同特征是：**不引入 interaction data 之外的新信号源**，只是用更好的方式重新处理现有边。

### A1. CIR-Reweighted Coarse Graph（CIR 重加权粗粒度图）

**它到底承载了什么新的 collaborative information**：

不是新信息，而是**边级别的可靠性评估**。CIR (Common Interacted Ratio) 衡量的是：一条 item-item 边是否被多条 user-mediated 偶数长度路径（even-length path）交叉验证。高 CIR 边 = 多个 user 共同支撑的关系；低 CIR 边 = 偶然共现。

**和当前 `coarse / mid / local` 相比，新增信息到底在哪里**：

当前 `coarse_purified` 的边质量控制只有两步：
- `support_prune(min=2.0)`：只看共现次数是否 ≥ 2
- `debias_by_popularity(α=0.5)`：只调整 popularity 偏差

这两步都不检验"这条边在图中是否被多条路径支撑"。CIR 检验的正是这个——一条边的两端节点是否通过大量中间节点相互验证。

**新增的信息类型**：同源精修（更干净的图，不是新信息的图）。

**更可能改善哪种 evaluate gap**：

- **NDCG@10** 和 **HR@10**：如果当前 G_coarse 中确实有大量低质量边在 graph smoothness loss 中产生 false pull（虚假拉力），那 CIR reweighting 应该让 L1 codebook 的 routing 更准确，从而改善整体排序
- **routing accuracy**（路由准确性）：最可能改善的是 L1 的 routing，因为 CIR 直接改善了 G_coarse 的边质量

**适合放在哪里**：`G_coarse`，替换当前 `coarse_purified`。同时因为 `fagsp_mid_base` 是从 `coarse_purified` 衍生的，更好的 coarse → 更好的 mid 输入。

### A2. Multi-Hop Diffused Transition Graph（多跳扩散转移图）

**它到底承载了什么新的 collaborative information**：

不是新信息源，但**大幅扩展了 transition signal 的覆盖范围**。当前 `local_purified` 只有 1-hop history→target 边，对称化后：`A_diff = A + α·A² + α²·A³`（α < 1）。

**和当前 `coarse / mid / local` 相比，新增信息到底在哪里**：

当前 `local_purified` 极度稀疏——它只包含"直接从 history item 转移到 target item"的关系。大量 item pair 虽然频繁出现在相似的 browsing chain 中，但因为中间隔了一两步就没有边。

multi-hop diffusion 捕捉的是："A→B→C 这种 2-step chain 中 A 和 C 的间接转移关系"。这在 1-hop 图中完全不存在。

**新增的信息类型**：同源精修（扩展 coverage，不是新信号源）。

**更可能改善哪种 evaluate gap**：

- **HR@10** 和 **beam diversity**（束搜索多样性）：L3 graph supervision 目前对很多 item 可能根本不起作用（因为它们在 local_purified 中没有边），multi-hop 可以让更多 item 受到 L3 supervision，改善 tail item 的 leaf-level separability（叶级可分离性）
- **NDCG@10**：间接改善——如果 L3 的 fine-grained 区分度提升，downstream beam search 在最后一步的选择更准确

**适合放在哪里**：`G_local`，替换当前 `local_purified`。或者对称化后的 multi-hop transition 也可以作为 `G_mid` 的 alternative view。

### A3. User-Cluster-Projected Item Graph（用户聚类投影物品图）

**它到底承载了什么新的 collaborative information**：

当前 `coarse_purified` 把所有用户的 session co-occurrence 混在一起。User-cluster projection 的逻辑是：先对 user 做 K-means 聚类（基于 interaction profile），每个 cluster 贡献一份独立的 item-item co-occurrence subgraph，最终图的每条边带有"被多少个不同 user cluster 支撑"的 count。

**和当前 `coarse / mid / local` 相比，新增信息到底在哪里**：

新增的是**边的 user-diversity 维度**。当前 co-occurrence 只关心"i 和 j 共现了多少次"，不关心"这些共现来自多少种不同类型的用户"。一条被 5 个不同 user cluster 支撑的边，比一条只被 1 个 cluster 的高频用户撑起来的边，可靠性更高。

**新增的信息类型**：同源精修（同一批 interaction data，但引入了 user-side diversity 视角）。

**更可能改善哪种 evaluate gap**：

- **HR@10**：user-diversity weighted edges 应该更好地反映"广泛受欢迎的 item 关系"而非"被少数 power user 主导的关系"，这可能让 codebook 中的 cluster 更加 representative
- **NDCG@10**：如果 user-diversity weighting 能去掉被少数 power user 制造的 spurious co-occurrence，L1 routing 会更准确

**适合放在哪里**：`G_coarse`，作为 `coarse_purified` 的替代或增强。

### A4. Community-Guided Layer Decomposition（社区引导的层级分解）

**它到底承载了什么新的 collaborative information**：

不是新信息，而是**用同一份 co-occurrence 图的多分辨率社区结构来定义三层图的层级分工**。当前三层图来自三种不同的构建方法（co-occurrence / spectral band-pass / transition），community decomposition 则是从同一图的不同粒度出发。

**和当前 `coarse / mid / local` 相比，新增信息到底在哪里**：

严格来说没有新增信息，它只是重新组织了已有信息。但组织方式的改变可能有实质影响：G_coarse = 社区间图（社区级别的 item 关系），G_mid = 社区内图（社区内部的 fine-grained 关系），G_local = 最小社区内的 transition。

**新增的信息类型**：同源精修（重新组织，不是新信号）。

**更可能改善哪种 evaluate gap**：

- **routing accuracy**：如果粗社区数 ≈ 256 且和 L1 codebook size 对齐，那 L1 supervision 会变得更 sharp——"同社区 = 同 L1 code"比"在 coarse graph 上 embedding 平滑"更明确
- **NDCG@1**：更清晰的 L1 routing → L2/L3 的条件预测更有效 → top-1 准确率可能提升

**适合放在哪里**：理论上替换全部三层。但 **R510/R511/R520 全部没有正信号的现实** 使得全面替换的风险很大。更稳妥的做法是只用社区结构来增强 `G_coarse`（让 L1 supervision 更 sharp），保留当前 `G_mid` 和 `G_local` 不动。

---

## 二、B 类：Information-Expanding Graph Carrier（信息扩张型图载体）

这一类的共同特征是：**引入 interaction data 之外的新关系信号**，不只是重新处理同一批边。

### B1. User-Segment Co-Occurrence：按用户类型分离的协同图

**它到底承载了什么新的 collaborative information**：

当前所有图都把 user 视为匿名的 session 贡献者。这种图引入了**user-type-conditioned co-occurrence**（按用户类型条件化的共现）——不同类型的用户对 item-item 关系的贡献被分开维护。

**和当前 `coarse / mid / local` 相比，新增信息到底在哪里**：

假设有一个 Industrial & Scientific 物品池，"实验室设备采购员"和"DIY 爱好者"是两类不同的用户。它们各自产生的 item-item co-occurrence 模式完全不同：
- 采购员："试管 → 试管架 → 量杯" 是一条 chain
- DIY 爱好者："螺丝刀 → 胶水 → 砂纸" 是另一条 chain

当前 `coarse_purified` 把两者混在一起。如果我们用 user-segment-conditioned graph，就能区分"哪些 item 关系是跨 segment 普遍存在的"（更可靠）和"哪些只在特定 segment 内存在"（更个性化）。

**新增的信息类型**：**真正的新信息**——user-type diversity 不在当前任何图中。

**更可能改善哪种 evaluate gap**：

- **HR@10**：cross-segment universality 作为边权重的加分项，能让 codebook 中的 cluster 更反映"广泛适用的协同关系"，而非被单一用户类型主导
- **NDCG@10**：如果 user-segment 能帮助区分"真正的全局共现"和"segment-specific noise"，L1 routing 会更准确

**适合放在哪里**：`G_coarse`（用 cross-segment universality 作为 coarse graph 的边权增强）或 `G_mid`（不同 segment 的 subgraph 可以提供不同 resolution 的视角）。

### B2. Item Metadata Co-Attribute Graph（物品元数据共属性图）

**它到底承载了什么新的 collaborative information**：

从 item 的 category / brand / price-range / keyword 等 metadata 中提取属性，构建 item-item 共属性图：两个 item 共享越多 attribute 则边权越高。

**和当前 `coarse / mid / local` 相比，新增信息到底在哪里**：

当前所有图都源自 interaction data。共属性图的信息源完全独立——它不依赖任何用户行为，只依赖 item 自身的属性。对 cold/sparse item（交互数据很少的物品），这种图提供的是 interaction graph 中完全不存在的连接。

**新增的信息类型**：**真正的新信息**——item metadata relationship 不在当前任何图中。

**⚠️ 关键风险提醒**：`R510` 用的 `G_attr_fused` 也是属性图方向，已经负向。但 `R510` 是用属性图**完全替换** `G_mid`，且属性图的构建方式是 TAGCF-inspired 的 semantic-to-topology。这里的 B2 和 R510 的区别在于：
1. B2 建议用 item metadata（category/brand/price 等结构化属性），不是 LLM-extracted concepts
2. B2 建议作为**增量信号补充**而非完全替换
3. B2 最合理的放置位置是 `G_coarse`（提供 interaction-independent 的 broad structure），不是 `G_mid`

但仍然必须承认：**这个方向和 R510 有思路上的重叠，风险不小**。

**更可能改善哪种 evaluate gap**：

- **HR@10**：metadata co-attribute 可以为 sparse item 提供额外的 structural anchor，让它们不被完全忽略
- **beam diversity**：如果 codebook 中有些 cluster 纯粹是因为 interaction graph 连通性不足而形成的"孤岛"，metadata graph 可以提供跨孤岛的连接

**适合放在哪里**：`G_coarse` 作为补充（mixed，不是替换）。明确不建议放 `G_mid`——R510 已经证明属性图替换 mid 图不行。

### B3. Temporal-Partitioned Co-Occurrence：时间分片共现图

**它到底承载了什么新的 collaborative information**：

当前 `coarse_purified` 把所有时间窗口的 session co-occurrence 合并在一起。Temporal-partitioned 的做法是：把训练数据按时间段切分（如 early / mid / late），每个时间段独立构建 co-occurrence graph，然后用不同时间段图之间的**边一致性**（temporal stability，时间稳定性）来加权最终图。

**和当前 `coarse / mid / local` 相比，新增信息到底在哪里**：

新增的是**时间维度的边稳定性**。当前 co-occurrence 不区分"一条边是在整个训练期间都稳定存在"还是"只在某个短暂时间窗口出现过"。如果一条边在 3 个时间段中的 2 个以上都出现，那它大概率是真正的 collaborative signal 而非 temporal noise（时间噪声）。

**新增的信息类型**：**部分新信息**——时间分片间的 edge consistency 目前不在任何图中，但原始数据源仍然是 interaction data。

**更可能改善哪种 evaluate gap**：

- **NDCG@10**：temporally stable edges 更可能代表持久的 collaborative 关系，用它们作为 supervision 可以让 codebook 更 robust
- **routing accuracy**：L1 routing 如果被 temporally unstable edges 误导，会把不相关的 item 分到同一个 code prefix

**适合放在哪里**：`G_coarse`（temporal stability 是 global-level 的边质量信号）。

### B4. Explicit Negative-Pair Graph：显式负样本对图

**它到底承载了什么新的 collaborative information**：

当前所有图都只编码"哪些 item 应该在 embedding 空间中接近"（正关系）。但我们也有信号表明"哪些 item 应该被分开"——例如：
- 同一用户在短时间内浏览了 item A 和 B，但最终只买了 A 而跳过了 B → A 和 B 应该被区分
- 两个 item 在 semantic embedding 上很近，但在 interaction pattern 上完全不同 → 它们需要不同的 SID code

**和当前 `coarse / mid / local` 相比，新增信息到底在哪里**：

当前 graph smoothness loss 只有**拉力**（把 graph neighbor 的 embedding 拉近），没有**推力**（把应该分开的 item 推远）。这意味着：即使我们能精确识别出"这两个 item 不应该共享 code prefix"，当前框架也无法表达这种约束。

**⚠️ 重要限制**：这个方向涉及 loss 设计的改变（需要引入 contrastive 或 repulsion 项），超出了纯"graph carrier"的范畴。但 graph carrier 的角度仍然可以定义"negative pair graph"——即一张图的边表示"这对 item 应该被分开"。

**新增的信息类型**：**真正的新信息**——"哪些 item 不应该在一起"目前完全不在 supervision 中。

**更可能改善哪种 evaluate gap**：

- **NDCG@1** 和 **NDCG@10**：如果 codebook 中有些 code prefix 把语义不同但行为偶然共现的 item 混在了一起，negative-pair supervision 可以直接纠正这种 mixing
- **routing accuracy**：直接约束"哪些 item 不应该走同一条 routing path"

**适合放在哪里**：`mixed graph`——negative pair 的约束应该同时作用于 L1（不应该同 L1 code）和 L2/L3（不应该同 leaf）。但实现上需要扩展 loss，不是纯 graph carrier change。

### B5. Cross-Dataset Transfer Graph：跨数据集迁移图

**它到底承载了什么新的 collaborative information**：

如果 Office Products 数据集中存在 item-item 关系（相同或类似 item 出现在两个数据集中），可以把 Office 的 co-occurrence graph 映射到 Industrial 的 item 空间，作为额外的 collaborative signal。

**和当前 `coarse / mid / local` 相比，新增信息到底在哪里**：

当前所有图都只来自 Industrial & Scientific 数据集自身的 interaction data。如果两个数据集有 item 或 category 的交叉，跨数据集的 co-occurrence 提供了一种完全独立的 collaborative view。

**新增的信息类型**：**真正的新信息**（如果 item 交叉存在的话）。

**⚠️ 关键风险**：Industrial & Scientific 和 Office Products 的 item 交叉可能很少，使得这张图极度稀疏甚至为空。需要先验证 item overlap。

**更可能改善哪种 evaluate gap**：如果 overlap 足够，理论上可以改善所有指标，但风险太高。

**适合放在哪里**：如果可行的话，`G_coarse` 作为补充。

---

## 三、明确区分："更干净的图" vs "带来新信息的图"

| 候选 | 类型 | 更干净还是新信息 | 关键判断 |
|------|------|-----------------|---------|
| A1. CIR-Reweighted | A 类 | **更干净** | 同一批边的可靠性评估 |
| A2. Multi-Hop Diffusion | A 类 | **更干净 + 覆盖扩展** | 同一信号源的 coverage 放大 |
| A3. User-Cluster-Projection | A 类 → B 类边缘 | **边缘新信息** | 同一 interaction data，但 user-diversity 维度是新的 |
| A4. Community-Guided Decomposition | A 类 | **重新组织** | 没有新信号，只是换了层级分工方式 |
| B1. User-Segment Co-Occurrence | B 类 | **新信息** | user-type conditioning 是全新维度 |
| B2. Metadata Co-Attribute | B 类 | **新信息** | 完全独立于 interaction 的信号源 |
| B3. Temporal-Partitioned | B 类边缘 | **新信息（时间维度）** | 同一 interaction data，但 temporal stability 是新的 |
| B4. Negative-Pair | B 类 | **新信息（负信号）** | 当前完全没有"推力"信号 |
| B5. Cross-Dataset Transfer | B 类 | **新信息** | 完全独立的数据源 |

---

## 四、重新排序 Top 5

排序标准（按重要性排）：
1. 最可能帮助 final evaluate（最终评测）超过 baseline（基线）
2. 最可能带来真实新增信息
3. 最不依赖"当前图只是再精修一下就会赢"这个假设

### Top 1：User-Segment Co-Occurrence Graph（用户群体分离共现图，B1）

**为什么排第一**：

1. **唯一一种"真正新信息 + 低实现风险"的组合**。user-type conditioning 引入了一个 interaction data 中**已经存在但从未被利用**的维度——不同类型的用户对 item 关系的贡献模式不同。这不需要外部数据、不需要新模型、不需要改 loss。

2. **直接回应 R510 的教训**。R510 失败的一个可能原因是：属性图引入的"新信号"和 downstream task（下游任务）的关联太弱——item 共享 attribute 不代表它们在 collaborative 意义上应该在一起。User-segment co-occurrence 的信号强度天然和 downstream task 对齐，因为它直接来自用户行为。

3. **cross-segment universality 是一个高质量的边权重信号**。如果一条 item-item 边被 3 种不同类型的用户群体都支撑，那它比只被 1 种用户群体撑起来的边可靠得多。这种 edge-level quality signal 和 CIR 思想互补，但维度完全不同。

4. **实现路径清晰**：(a) 对训练集 user 做 K-means（基于 interaction profile），K=5~10；(b) 每个 cluster 独立构建 session co-occurrence subgraph；(c) 最终图的边权 = cross-cluster count / K × 原始共现权重。

**更可能改善的 evaluate gap**：`HR@10`（让 codebook cluster 更反映广泛共识而非少数用户偏好）和 `NDCG@10`（更准确的 L1 routing）。

**适合放在哪里**：`G_coarse`。

### Top 2：CIR-Reweighted Coarse Graph（CIR 重加权粗粒度图，A1）

**为什么排第二**：

1. **R510/R511/R520 全部失败的一个共同背景是：它们都改了 G_mid，但 G_coarse 从未被质疑过**。而 `fagsp_mid_base` 是从 `coarse_purified` 衍生的——如果 `coarse_purified` 本身的边质量有问题，那再怎么改 mid 也是在脏数据上做文章。CIR 是对 coarse graph 的边级质量审计（edge-level quality audit），可能解决更上游的问题。

2. **CIR 的信息维度和 popularity debiasing 完全正交**。当前 debiasing 只处理"热门 item 的边权过高"，但不处理"这条边是否被图中多条路径交叉验证"。一条 low-popularity-pair 的边如果只来自单一 session 的偶然共现，support_prune=2.0 可能放过它，但 CIR 会给它低分。

3. **和现有 pipeline 的兼容性最好**——只改边权，不改拓扑、不改层级、不改 loss。

**更可能改善的 evaluate gap**：`routing accuracy`（L1 routing）和 `NDCG@10`。

**适合放在哪里**：`G_coarse`。

### Top 3：Multi-Hop Diffused Transition Graph（多跳扩散转移图，A2）

**为什么排第三**：

1. **L3 的 graph supervision 可能对大量 item 根本不起作用**。`local_purified` 是 1-hop history→target 有向图，经过 support_prune(min=1.0) 后可能极度稀疏。如果大量 item 在 `local_purified` 中没有入边或出边，那 L3 的 graph smoothness loss 对它们 = 0——即 L3 codebook 对这些 item 完全没有 collaborative supervision，只靠 reconstruction loss 在引导。

2. **multi-hop diffusion 的实现成本接近零**：`A_diff = A + α·A² + α²·A³`，只是矩阵乘法。对 3686×3686 的矩阵可以瞬间完成。

3. **R510/R511/R520 的教训不影响这个方向**——它们改的都是 G_mid，L3 的 local graph 从未被动过。这可能是一个被忽视的低垂果实。

**更可能改善的 evaluate gap**：`HR@10`（更多 item 受到 L3 supervision → leaf-level separability 提升）和 `beam diversity`。

**适合放在哪里**：`G_local`，替换当前 `local_purified`。

### Top 4：Temporal-Partitioned Co-Occurrence（时间分片共现图，B3）

**为什么排第四**：

1. **temporal stability 是一个几乎免费的边质量信号**。把训练数据按时间等分为 3 段，每段独立构建 co-occurrence graph，然后统计每条边在多少个时间段中出现。这和 CIR 的逻辑类似（都是"交叉验证"），但验证维度不同——CIR 看图结构路径，temporal partition 看时间一致性。

2. **两者可以组合**：`w_final = w_raw × CIR_score × temporal_stability_score`。一条边如果在图结构上被交叉验证（高 CIR）、且在时间上持续存在（高 temporal stability），那它的可靠性非常高。

3. **直接和 R510 的教训互补**：R510 引入了一种"新的信号源"（属性图）但失败了。temporal partitioning 不是新信号源，但它提供了一种**时间维度的去噪方式**——去掉那些只在特定时间窗口出现的 temporal noise。

**更可能改善的 evaluate gap**：`NDCG@10`（temporally stable edges → 更 robust 的 codebook）。

**适合放在哪里**：`G_coarse`。

### Top 5：Community-Guided Coarse Supervision（社区引导的粗粒度监督，A4 的降级版）

**为什么排第五（不是更高）**：

1. 上一轮我把 community-initialized hierarchical graph 排 Top 3，建议全面替换三层。现在必须降级，因为 R510/R511/R520 表明"换图"的风险很大。

2. **但 community structure 对 L1 supervision 的 sharpening 价值仍然存在**。当前 L1 的 graph smoothness loss 说的是"在 coarse graph 上 embedding 应该平滑"——这很模糊。如果我们用 community detection 找到 coarse graph 的社区结构，然后给 graph smoothness loss 增加一个"同社区 item 的 L1 representation 应该更接近"的约束，这比全面替换三层安全得多。

3. **不建议替换全部三层**。只用 community 信息来增强 G_coarse → L1 的 supervision。

**更可能改善的 evaluate gap**：`routing accuracy`（L1 routing 更 sharp）→ `NDCG@1`。

**适合放在哪里**：增强 `G_coarse`，不替换。

---

## 五、What I Was Still Too Anchored On Last Time

### 锚定 1：我被"semantic 参与图构建一定有价值"这个假设锚住了

上一轮我把 Cross-View Consensus Graph（语义 + 行为交集）排在 Top 1，把 Semantic-Behavioral Intersection Graph 也列入候选。这背后的隐含假设是："Qwen semantic embedding 包含有用的 item 关系信息，只要让它参与图构建就能提升边质量。"

**R510 打破了这个假设。** R510 的 `G_attr_fused` 就是一种"语义驱动拓扑"的图——它用 LLM-extracted attribute concepts 来构建 item-item 连接。结果是明确负向。

这迫使我承认：**在当前 pipeline 下，"语义参与图构建"不是一个安全的默认赌注。** 语义相似性和 collaborative usefulness（协同有用性）之间可能存在 systematic gap（系统性差距），尤其在 Industrial & Scientific 这种垂直领域。

### 锚定 2：我把"同源精修"误当成了"新图载体"

上一轮 Top 5 中有 3 个（Cross-View Consensus、CIR-Reweight、Cascade Filter）本质上都是"用更好的方式处理同一批 co-occurrence 边"。我在措辞上把它们和"新信息"混在了一起。

**现在必须明确**：Cross-View Consensus 虽然用了 semantic graph 做 gating，但它的核心操作仍然是对 co-occurrence graph 做 precision filtering——这是 A 类（同源精修），不是 B 类（信息扩张）。把它当成"新信息"是误导性的。

### 锚定 3：我低估了"完全换 G_mid 的风险"

上一轮我在 Top 5 中有 2 个方向（Cascade Filter、Community Hierarchy）涉及替换 G_mid 或全部三层。现在 R510（属性图替换 mid）、R511（混合 mid）、R520（cascade 替换 mid）三轮实验全部没有正信号。

**我现在最想纠正的排序偏差**：

> 上一轮我默认"改 G_mid 就是最高杠杆点"，因为 CURRENT_STATE.md 说"G_mid 是瓶颈"。但三轮实验表明：**G_mid 也许确实是瓶颈，但解决方案不一定是"换一种 G_mid"——也许应该先检查 G_coarse 和 G_local 的问题，因为它们从未被认真审视过。**

---

## 六、最终只押 2 条方向

### 押注 A（来自 A 类）：CIR-Reweighted Coarse Graph

**为什么选这条 A 类方向**：

R510/R511/R520 反复改 G_mid 但全无正信号。一个被忽视的可能性是：**问题不在 G_mid 本身，而在 G_coarse**。原因：
- `fagsp_mid_base` 是从 `coarse_purified` 经谱分解衍生的——如果 coarse 的边质量有问题，mid 再怎么改也是在脏数据上做文章
- `coarse_purified` 的边质量控制（support_prune=2.0 + popularity_debias α=0.5）从未被检验其充分性——没人知道有多少低质量边通过了这两步
- CIR 是唯一一种可以做 edge-level quality audit（边级质量审计）的方式，它检验的是"这条边在图中是否被多条独立路径支撑"，这是 support count 和 popularity debias 完全覆盖不到的维度

**为什么不把两个都押在 A 类**：

纯靠 A 类（同源精修）有一个根本限制：**它不能引入 interaction data 中不存在的信号**。如果当前图的核心问题不是"同一批边的权重不对"，而是"整个 interaction data 的 coverage 不足以提供足够的 collaborative supervision"，那再怎么精修也不够。

所以必须至少一条押在 B 类。

### 押注 B（来自 B 类）：User-Segment Co-Occurrence Graph

**为什么选这条 B 类方向**：

在所有 B 类候选中：
- B2（Metadata Co-Attribute）已经被 R510 间接打了负号——属性驱动的图构建在当前框架下风险高
- B4（Negative-Pair Graph）需要改 loss，超出纯 graph carrier 范畴
- B5（Cross-Dataset Transfer）依赖 item overlap，可能不可行
- B3（Temporal Partition）有价值但更像 A 类的变体

只剩 B1（User-Segment Co-Occurrence）满足：
1. 引入了真正新的信息维度（user-type diversity）
2. 信号天然和 downstream task 对齐（来自用户行为，不是 metadata 或语义）
3. 实现简单且风险可控
4. 不需要改 loss 或 encoder

**为什么不把两个都押在 B 类**：

纯靠 B 类（信息扩张）有一个实际风险：**R510 已经证明"引入新信号源"在当前框架下不默认有效**。如果我们把两个都押在 B 类，而 B 类的核心假设（"当前图缺乏新信号"）也是错的，那就两条路都走不通。

A 类（CIR Reweighting）的价值在于：它不依赖"需要新信号"这个假设。它假设的是"当前信号中有噪声，去噪就能帮忙"。这和 B 类的假设（"需要新信号"）是正交的——分别押一条，可以覆盖两种不同的 failure mode（失败模式）。

### 两条方向的关系

这两条方向是**可以组合的**：
- 先用 CIR reweighting 清理 G_coarse 的边质量
- 再用 user-segment diversity scoring 增强 G_coarse 的边权重
- 最终 `w_final = w_raw × CIR_score × segment_diversity_score`

这种组合同时解决了"边质量"和"信号维度"两个问题，且都只作用于 G_coarse——不动 G_mid 和 G_local，风险最小。

---

## 附录：事实校准表

```
最新实验事实（2026-04-15）：

v2_on_p05 → SFT:  NDCG@10=0.10271, HR@10=0.14626
v2_on_p05 → RL:   NDCG@10=0.10432, HR@10=0.14185
original strongest SFT: NDCG@10=0.10372, HR@10=0.15089
original strongest RL:  NDCG@10=0.10726, HR@10=0.15133

需要超过的线：NDCG@10 > 0.10726, HR@10 > 0.15133

R510 (attr_mid → SFT→eval): NDCG@10=0.09758, HR@10=0.13148 → 负向
R511 (mix_mid):   collision 18/3686 → 明显退步于 R510 的 11/3686
R520 (cascade_mid): collision 14/3686 → 不如 v2 的 13/3686
R202a (stage2):   NDCG@10=0.09974, HR@10=0.14251 → 不如 v2_on_p05
R401b (stage3):   NDCG@10=0.09905, HR@10=0.13854 → 负向
R401d (stage3):   NDCG@10=0.09354, HR@10=0.13148 → 明显负向

核心事实：
- 所有尝试替换/修改 G_mid 的实验（R510/R511/R520）目前都没有正信号
- 所有尝试做更强 tokenizer-side structure 的实验（R202a/R401b/R401d）也没超过 v2_on_p05
- G_coarse 和 G_local 从未被单独作为实验变量
```
