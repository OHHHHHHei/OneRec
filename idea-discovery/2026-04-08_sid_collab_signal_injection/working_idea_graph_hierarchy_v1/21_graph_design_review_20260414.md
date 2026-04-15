# 图设计复盘与文献补检（2026-04-14）

Status（状态）: `discussion-only（仅讨论）`

Discussion date: `2026-04-14`

这份文档是 graph design review（图设计复盘）笔记，不是 current-state document（当前状态文档）。

它的作用是回答：

- 当前图设计哪里还粗糙
- 哪些论文模块值得借
- 图载体应该往哪里升级

如果你要看最新主线和实验结论，请先读：

1. [CURRENT_STATE.md](/home/leejt/OneRec/research-progress-log/CURRENT_STATE.md)
2. [experiment_results.csv](/home/leejt/OneRec/experiment_results.csv)

## 写在前面

这份笔记回答下面几个问题：

1. 我们当前的图设计是不是还比较粗糙？
2. 当前的中频带通图，是否严格按照 `FaGSP` 实现？
3. 目前只用 `item-item` 图够不够？
4. `user-item` 图有没有可能提供额外帮助？
5. 当前 `G_coarse / G_mid / G_local` 三层图设计是否合理？
6. 我们有没有检索过 arXiv 上“图承载协同信息帮助推荐”的工作，还有什么值得借鉴？

我会把每个问题拆开回答，并尽量区分三件事：

- **当前代码里真正发生了什么**
- **当前方法叙事在说什么**
- **从文献里还能借什么，但目前还没吸收**

---

## 一、先给结论

### 结论 1：当前图设计是“有明确方向，但仍然偏粗糙”的

它不是拍脑袋的随意设计，因为：

- `G_coarse -> L1`
- `G_mid -> L2`
- `G_local -> L3`

这条映射在概念上是顺的，而且已经有正结果支撑。

但它也确实还比较粗糙，粗糙主要体现在三点：

1. **图源不够丰富**  
   当前主干还是从用户历史投影出来的 `item-item` 关系，缺少对 `user-item` 二部图结构的直接利用。

2. **中尺度图目前更像一个“谱变换视图”，不是一个完整图建模模块**  
   它有 `FaGSP` 风格，但不是 `FaGSP` 的严格实现。

3. **三层图目前不是三张完全独立的信息源**  
   `G_mid` 目前是从 `coarse` 图再做谱重建得到的，所以更准确地说是：
   - 一张 broad collaborative 图
   - 这张 broad 图的中频变换视图
   - 一张 local transition 图

所以我会说：

> 当前图设计已经足够支撑“第一代有效方法”，但还没有达到“图设计本身已经成熟、可以不再追问”的程度。

---

## 二、问题 1：我们当前的图设计是不是还比较粗糙？

### 我的回答

**是，仍然比较粗糙。**

但这里的“粗糙”不是说它没用，而是说它还停留在一个**有效的 first-generation graph bank** 阶段，而不是一个已经非常扎实的 graph modeling framework。

### 具体为什么说它粗糙

#### 1. 当前图主要还是从行为序列里做了比较直接的投影

当前底层 graph bank 里，最基础的两张图是：

- `build_coarse_graph`：从历史序列和目标 item 构一个去重后的共现图
- `build_local_graph`：从最近历史到目标 item 构一个有方向的局部转移图

代码位置：

- [graph_bank.py](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/graph_bank.py:154)
- [graph_bank.py](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/graph_bank.py:180)

这说明当前图设计的起点很清楚：

- `G_coarse` 是 broad co-occurrence
- `G_local` 是 short-range transition

这个起点并不坏，但也意味着：

- 你现在并没有直接建模 `user-item` 二部图的高阶拓扑
- 也没有直接建模 `user-user` 或 `item-item` 相关性增强后的联合邻接
- 更没有学习式图构建

所以它是一个**干净、轻量、可控的起点**，但不是信息最充分的图建模。

#### 2. `G_mid` 的确是最强，但也是当前最“半成品”的部分

`train_v2.py` 里真正送进训练的三张图是：

- `coarse`: `views["coarse_purified"]`
- `mid`: `views["fagsp_mid_base"]`
- `local`: `views["local_purified"]`

代码位置：

- [train_v2.py](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/train_v2.py:253)

问题在于：

`fagsp_mid_base` 不是从一套独立的中尺度行为图构出来的，  
而是对 `coarse_purified` 做谱域重建之后得到的一个中频视图。

也就是说，当前三层图并不是：

- 三张独立构建的图

而更像：

- 一张粗粒度协同图
- 这张图的中频视图
- 一张局部转移图

这在早期方法里完全合理，但如果以后要把“graph bank design”写得非常硬，就还不够。

#### 3. 当前图设计仍然偏“静态”

现在的图：

- 不带用户条件
- 不带 session 条件
- 不带时间可靠性估计
- 不带边级别置信度学习

它们都是**全局 item 图**，然后统一作用于 tokenizer。

这对于“静态 item tokenizer”是匹配的，但也意味着它会天然忽略：

- 同一个 item 在不同 user context 下的协同角色差异
- 不同时间段边是否可靠
- 高频/低频边是否应该被动态重权

所以当前图设计更准确地说是：

> 对静态 tokenizer 很友好的全局协同载体设计  
> 但不是最充分的协同图建模

---

## 三、问题 2：当前的中频带通图，是否严格按照 `FaGSP` 实现？

### 我的回答

**不是。**

更准确的说法应该是：

> 当前实现是一个 `FaGSP-inspired` 的谱中频视图移植，而不是 `FaGSP` 方法本体的严格复现。

### 为什么不是严格实现

当前代码里的 `fagsp_mid_base` 是这样构的：

1. 先拿一个 `base_graph`
2. 做对称归一化
3. 算前 `rank` 个特征值/特征向量
4. 只截取 `[band_low, band_high]` 这一段频带
5. 用这一段重构一个非负稀疏图

代码位置：

- [_spectral_reconstruct](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/paper_transplants.py:109)
- [build_fagsp_mid_view](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/paper_transplants.py:145)

而仓库里当时自己写的 transplant note 也说得很诚实：

> `GSPRec / FaGSP`-inspired spectral middle-resolution graph views

见：
[14_paper_transplant_probe_run_2026-04-09.md](/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/archive/2026-04-12_doc_reorg_merged_sources/14_paper_transplant_probe_run_2026-04-09.md:10)

### 那么真正的 `FaGSP` 在说什么

`FaGSP` 原文强调的是：

- 在协同过滤里，用户和物品的偏好同时有 common 和 unique 两种成分
- 需要同时利用高通和低通滤波
- 还要并行利用高阶邻域层次

arXiv 原文：
[FaGSP](/home/leejt/OneRec/papers/Frequency-aware%20Graph%20Signal%20Processing%20for%20Collaborative%20Filtering.pdf)
[arXiv:2402.08426](https://arxiv.org/abs/2402.08426)

摘要里说得很明确：

- 串联高通+低通来捕获 unique/common characteristics
- 并行低通来利用高阶邻域层次  
见 [arXiv] (https://arxiv.org/abs/2402.08426)

### 当前实现和 `FaGSP` 的差别

#### 你们当前实现有的

- 频域分解这个思想
- 中频/带通视图这个思想
- 把图当成多尺度信号载体这个思想

#### 你们当前实现没有的

- 不是直接在 `user-item` 二部图上做 `FaGSP`
- 没有 `FaGSP` 那种完整的串联/并行滤波模块
- 没有学习式频率组合
- 没有把高通、低通、中频都显式纳入模型
- 没有复现它完整的推荐框架，只是取了“谱带视图”这个部件

### 所以最准确的判断

当前 `fagsp_mid_base` 不是“严格复现 FaGSP”，而是：

> 用 `FaGSP` 的频率分解直觉，给当前 item 图构造了一个中频 collaborative view

这个做法是合理的，但在写论文或做方法自评时，应该明确写成：

- `FaGSP-inspired`
- `spectral mid-view transplant`

而不是“我们实现了 FaGSP”

---

## 四、问题 3：目前只用 `item-item` 图够不够？

### 我的回答

**对于当前这条 tokenizer 主线，它“够用到能出正结果”，但“不够到可以放心认为图设计已经到头了”。**

我会把这个问题拆成两层。

### 第一层：为什么 `item-item` 图在当前问题里是合理起点

你们当前优化的是一个**静态 item tokenizer**。

它训练时面对的是：

- item embedding
- item SID
- item-level graph supervision

所以从接口匹配上说，`item-item` 图非常自然：

- 它直接约束 item representation
- 直接作用在 tokenizer 上
- 不需要引入 user encoder
- 不会把方法改成用户条件化 tokenizer

这也是为什么现在这条线能比较干净：

- 图只负责承载协同信息
- tokenizer 只负责学习 item codebook space

如果一开始就把 `user-item` 二部图整套搬进训练，方法复杂度会一下子大很多。

### 第二层：为什么 `item-item` 图又明显不是全部

`item-item` 投影会丢掉三类东西：

#### 1. 用户侧异质性

两个 item 在投影后的 `item-item` 图上可能连得很强，  
但这条边可能来自**完全不同类型用户的混合**。

也就是说：

- `item-item` 图保留了“共现”
- 但丢掉了“是谁把它们连起来”

而这件事对中尺度结构尤其重要。

#### 2. 二部图高阶拓扑

很多 graph CF（图协同过滤）工作其实并不先把它压成 `item-item`，  
而是直接在 `user-item` 二部图上做传播、滤波、分解。

例如：

- [FaGSP](https://arxiv.org/abs/2402.08426)
- [JGCF](https://arxiv.org/abs/2306.03624)
- [A Topology-aware Analysis of Graph CF](https://arxiv.org/abs/2308.10778)

这些工作都把**二部图本身**视为主要协同载体。

#### 3. 边的可靠性差异

投影成 `item-item` 以后，很容易把：

- 高频热门边
- 偶然共现边
- 时序不可靠边

都混在一起。

而像 [GraphDA](https://arxiv.org/abs/2304.03344) 和 [DeBaTeR](https://arxiv.org/abs/2411.09181) 这类工作都在强调：

> 图的邻接本身需要去噪、重权、增强  
> 不能默认原始交互图就是干净的协同信息

### 所以结论不是“item-item 不够，必须上 user-item”

而是：

> 对当前静态 tokenizer 路线，`item-item` 是一个合理而干净的第一代接口；  
> 但如果你们要把 graph design 再往前做，不能只停留在“从用户历史投影一个 item-item 图”这一层。

---

## 五、问题 4：`user-item` 图有没有可能提供额外帮助？

### 我的回答

**有，而且我觉得是有帮助空间的。**

但注意：

> `user-item` 图对你们最有价值的用法，不一定是“直接把 user-item 图送进 loss”，  
> 而更可能是“先从 user-item 图提炼出更好的 item-side graph operator（图算子）或 edge confidence（边可靠性）”。  

这是关键区别。

### 为什么我不建议一上来“直接用 user-item 图监督 tokenizer”

因为当前 tokenizer 是 item-only 的。

也就是说它学的是：

- item codebook
- item representation

不是用户条件化的编码器。

所以如果你把 `user-item` 图直接塞进来，会遇到接口问题：

1. user 节点怎么进入 tokenizer 训练？
2. user 侧信号是固定先验，还是动态 batch 相关？
3. 会不会把方法变成 quasi-graph-CF，而不是 SID tokenizer？

这会让方法叙事变得很重。

### 更现实、更适合你们当前主线的三种 `user-item` 用法

#### 用法 A：从 `user-item` 二部图导出更好的 `G_mid`

这是我觉得最值得考虑的一条。

你们现在的 `G_mid` 来自：

- `coarse_purified` 的谱重建

但更自然的办法其实是：

- 先在 `user-item` 二部图上做频率分解 / 高阶传播 / 时间增强
- 再投影成 item-side 的 mid-scale operator

这样得到的 `G_mid` 可能会比“coarse 图的频带切片”更像一个真正的中尺度协同图。

#### 用法 B：用 `user-item` 图给边做去噪和重权

例如借鉴：

- [GraphDA](https://arxiv.org/abs/2304.03344)
- [DeBaTeR](https://arxiv.org/abs/2411.09181)

你可以不直接用二部图训练 tokenizer，  
而是先用它做：

- edge reliability
- temporal denoising
- sparse-edge augmentation

然后再产出更干净的 `item-item` 图。

这条和你们当前“graph 只是信息载体”的思路最兼容。

#### 用法 C：用 `user-item` 图构 ambiguity prior（歧义先验）

现在你们的 `offline_combined` prior 已经很重要了。  
但它仍然主要是：

- semantic density
- semantic-collab disagreement
- graph competition

以后完全可以再加：

- bipartite inconsistency
- time-aware edge reliability
- user-cluster disagreement

也就是：

> 不直接改 graph supervision 本身，  
> 而是先让“哪一些 item 应该被更强 graph supervision 修正”这件事更准。

### 所以我的判断

`user-item` 图不是“要不要上”的问题，  
而是“用在哪一层最划算”的问题。

对你们当前项目，我会按价值排序：

1. **先拿来改 `G_mid` 的构造**
2. **再拿来做 edge denoising / edge weighting**
3. **最后才考虑是否直接进训练主图**

---

## 六、问题 5：当前 `G_coarse / G_mid / G_local` 三层图设计是否合理？

### 我的回答

**整体上是合理的，但合理性强弱不一样。**

我会分别评价。

### 1. `G_coarse`：合理，而且最容易解释

当前 `G_coarse` 是 broad co-occurrence after purification（净化后的粗粒度共现图）。

它适合放在 `L1`，因为：

- `L1` 应该承担粗分类、粗路由
- broad collaborative consistency 和这个目标是对齐的

所以 `G_coarse -> L1` 基本没有什么大的概念问题。

### 2. `G_local`：合理，但当前实现可能偏稀疏、偏弱

当前 `G_local` 本质上是：

- 最近历史 item 指向 target item 的加权转移

代码在：
[graph_bank.py](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/graph_bank.py:180)

它放在 `L3` 也很顺，因为：

- `L3` 本来就是最细粒度
- transition/local correction 和 leaf disambiguation 天然相关

问题不在于它的角色，而在于它当前实现可能：

- 太 sparse（过稀）
- 太直接
- 太依赖短窗口
- 没做足够的时序去噪

所以 `G_local` 的**角色是对的**，但**实现还可以更强**。

### 3. `G_mid`：最关键，也最值得继续怀疑

你们自己很多文档其实已经承认了：

> `G_mid` 是全方法最关键、也最容易变得 hand-wavy（说不清）的部分

见：

- [01_PROBE_AND_EARLY_EVIDENCE.md](/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/01_PROBE_AND_EARLY_EVIDENCE.md)
- [02_RELATED_WORK_AND_MODULE_MAP.md](/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/02_RELATED_WORK_AND_MODULE_MAP.md)
- [CURRENT_TASK_ALIGNMENT.md](/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/CURRENT_TASK_ALIGNMENT.md)

我同意这个判断。

现在的 `G_mid` 有两个优点：

- 实验上确实强
- 直觉上确实比 `G_coarse` 和 `G_local` 更接近“真正卡住 tokenizer 的局部协同结构”

但它也有两个明显问题：

#### 问题 A：它不是一个真正独立的数据源

它来自 `coarse_purified` 的谱重建。  
所以它不是“另一张独立图”，而是“coarse 图的一个频带视图”。

这并不错误，但会让“三图三层”的叙事稍微弱一点。

#### 问题 B：它的物理含义还不够可解释

你可以说它是：

- middle-resolution
- band-pass
- user-level pattern carrier

这些都对。

但如果 reviewer 追问：

> 这个图到底对应了什么行为统计对象？

你现在的回答还不够硬。

所以我会给当前三图设计一个总评价：

### 总评价

- `G_coarse -> L1`：**合理且稳定**
- `G_local -> L3`：**合理，但实现还偏弱**
- `G_mid -> L2`：**方向对，但仍然是最值得继续打磨的图**

也就是说，三层角色分配本身没有大问题，  
真正没完全成熟的是 `G_mid` 的来源和表达。

---

## 七、问题 6：我们有没有检索过 arXiv 上“图承载协同信息帮助推荐”的论文？还能借什么？

### 我的回答

**有，而且不止一次。**

仓库里已经有几轮相关扫描：

- [02_RELATED_WORK_AND_MODULE_MAP.md](/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/02_RELATED_WORK_AND_MODULE_MAP.md)
- [17_ambiguity_proxy_literature_scan.md](/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/17_ambiguity_proxy_literature_scan.md)
- 历史扫描：
  [11_arxiv_related_work_by_question.md](/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/archive/2026-04-12_doc_reorg_merged_sources/11_arxiv_related_work_by_question.md)

我这次又补查了一轮更聚焦的问题：

- 图频率分解 / band-pass / spectral filtering
- `user-item` 二部图能否提供更强的协同载体
- 协同图去噪 / 重权 / 增强
- transition graph 与 collaboration graph 的分工

### 我认为最值得继续借的几类东西

#### 1. 从 `FaGSP / GSPRec / JGCF` 借“二部图频率分解”，而不是只借中频切片直觉

相关文献：

- [FaGSP](https://arxiv.org/abs/2402.08426)
- [GSPRec](https://arxiv.org/abs/2505.11552)
- [JGCF](https://arxiv.org/abs/2306.03624)

它们共同提示的是：

- 推荐中的图信号最好先在 `user-item` 图上理解
- 低频、中频、高频承担不同作用
- 中频不应该只是“从 item-item 图上切一刀”，而应该来自更完整的频域建模

**对你们最可借的点：**

- 让 `G_mid` 来源于二部图谱信号，而不是只来源于 `coarse item-item`
- 不只保留一个中频图，可以显式保留“粗 + 中”的互补信息

#### 2. 从 `GraphDA / DeBaTeR` 借“先净化图，再拿图做监督”

相关文献：

- [GraphDA](https://arxiv.org/abs/2304.03344)
- [DeBaTeR](https://arxiv.org/abs/2411.09181)

这两篇最有启发的一点是：

> 用户交互图本身并不天然干净  
> 应该先做边增强、边去噪、边重权，再谈图监督

这对你们很重要，因为你们现在虽然做了：

- popularity debias
- semantic anchor purification

但对边可靠性的建模还比较初级。

**对你们最可借的点：**

- 用时间信息评估边可靠性
- 用 user-item 稀疏性 / 偏置来重权边
- 再输出更好的 `item-item` 图

#### 3. 从 `CAGCN` 借“图里不是所有邻居都一样有用”

相关文献：

- [CAGCN](https://arxiv.org/abs/2207.06221)

这篇的核心启发不是一定要用 GCN，  
而是它提出了一个推荐导向的拓扑度量：

- 邻居不只是“有没有边”
- 还要看这个邻居和其他邻居形成的协同结构是否真的有用

**对你们最可借的点：**

- `ambiguity prior` 里加入更 topology-aware 的竞争分数
- `G_coarse` / `G_mid` / `G_local` 的边不只看强度，还看拓扑价值

#### 4. 从 `Collaboration and Transition` 借“协同”和“转移”要分开建模，还可以看它们是否冲突

相关文献：

- [Collaboration and Transition](https://arxiv.org/abs/2311.01056)

这篇很贴你们，因为它明确把：

- collaborative signal
- transition signal

分开看。

你们现在也在这么做：

- `G_coarse/G_mid` 更偏协同
- `G_local` 更偏转移

但还能再往前走一步：

> 不只是把两种图分开，  
> 还可以把“它们是否冲突”当成一个 ambiguity signal。

#### 5. 从 topology-aware analysis 借“先量化图形态，再讨论为什么方法有效”

相关文献：

- [A Topology-aware Analysis of Graph Collaborative Filtering](https://arxiv.org/abs/2308.10778)

这类工作提醒你们：

> 方法效果和图拓扑之间是有系统关系的，  
> 不能只看最后指标，不看图本身长什么样。

这对你们尤其有用，因为你们已经很擅长做 tokenizer diagnostics。  
下一步完全可以把一部分诊断前移到图层：

- 稀疏度
- 邻居竞争度
- band energy
- coarse-local disagreement

---

## 八、把你的问题合起来之后，我的总判断是什么？

### 1. 当前图设计不是错，而是“有效但未饱和”

它已经足够支撑正结果，也足够形成一篇有逻辑的方法论文。  
所以**现在不需要因为图设计不完美，就推翻当前主线**。

### 2. 但如果以后继续打磨 graph bank，最值得动的不是 `G_coarse`，而是 `G_mid`

因为：

- `G_coarse` 角色已经比较稳定
- `G_local` 角色也清晰，只是实现偏弱
- `G_mid` 同时承担了最大收益和最大不确定性

### 3. `user-item` 图是很值得考虑的，但更适合“间接进入”

不是直接把 user 节点拖进 tokenizer loss，  
而是先做：

- 二部图谱分解
- 边去噪
- 时间重权
- item-side operator 提取

### 4. 如果以后要开“图侧改进”实验，我会这样排优先级

#### Priority 1：只改 `G_mid` 的来源，不改整套训练

做一个新的 `mid` 候选：

- 从 `user-item` 二部图出发
- 做谱分解 / 频带提取
- 再投影成 item-side 中尺度图

这是最可能既增强图设计，又不把方法搞重的方向。

#### Priority 2：增强 `G_local`

不是换角色，而是增强实现：

- temporal denoising
- sparse augmentation
- transition reliability

#### Priority 3：把 topology-aware 分数并入 ambiguity prior

这条成本低，而且和你们现有 `offline_combined` 体系很兼容。

### 5. 但在当前时点，这些都不应该抢 `R401b/R401d -> SFT` 的优先级

因为你们现在最需要先知道的是：

> 当前最强的两个新码本空间，最终到底能不能在下游赢 `v2_on_p05`

这个答案出来以后，才值得决定：

- 继续沿 tokenizer 主线加图
- 还是把图设计问题收成论文讨论点

---

## 九、我给你的直接回答

如果把所有问题压成最短回答：

1. **我们当前图设计是有效的，但确实还比较粗糙。**
2. **当前 `fagsp_mid_base` 不是严格复现 `FaGSP`，而是 `FaGSP-inspired` 的谱中频视图。**
3. **只用 `item-item` 图作为第一代 tokenizer supervision 是合理的，但不应该被当成最终完备答案。**
4. **`user-item` 图很可能有帮助，但更适合作为更好 `item-side` 图算子的来源，而不是直接硬塞进当前 tokenizer loss。**
5. **当前三层图设计里，`G_coarse` 和 `G_local` 的角色比较稳，`G_mid` 最关键也最值得继续打磨。**
6. **你们确实已经做过相关 arXiv 检索，而且从现有文献看，最值得继续借的是：二部图谱分解、图去噪/重权、拓扑竞争度量、协同与转移的冲突建模。**

---

## 参考来源

### 当前代码与项目文档

- [paper_transplants.py](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/paper_transplants.py)
- [transplanted_graph_bank.py](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/transplanted_graph_bank.py)
- [graph_bank.py](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/graph_bank.py)
- [train_v2.py](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/train_v2.py)
- [02_RELATED_WORK_AND_MODULE_MAP.md](/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/02_RELATED_WORK_AND_MODULE_MAP.md)
- [17_ambiguity_proxy_literature_scan.md](/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/17_ambiguity_proxy_literature_scan.md)
- [14_paper_transplant_probe_run_2026-04-09.md](/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/archive/2026-04-12_doc_reorg_merged_sources/14_paper_transplant_probe_run_2026-04-09.md)

### 补充查看的 arXiv 论文

- FaGSP: https://arxiv.org/abs/2402.08426
- GSPRec: https://arxiv.org/abs/2505.11552
- JGCF: https://arxiv.org/abs/2306.03624
- GraphDA: https://arxiv.org/abs/2304.03344
- DeBaTeR: https://arxiv.org/abs/2411.09181
- CAGCN: https://arxiv.org/abs/2207.06221
- A Topology-aware Analysis of Graph Collaborative Filtering: https://arxiv.org/abs/2308.10778
- Collaboration and Transition: https://arxiv.org/abs/2311.01056
