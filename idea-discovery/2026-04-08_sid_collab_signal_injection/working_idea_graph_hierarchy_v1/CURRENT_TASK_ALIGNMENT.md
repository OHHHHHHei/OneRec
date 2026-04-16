# 当前任务对齐：MGR-SID 的四个核心问题

Status（状态）: `reference（参考）`

Last updated（更新日期）: `2026-04-16`

## 用途

这份文档只做一件事：

- 把我们当前在 `MGR-SID` 方向上真正要解决的核心问题固定下来

后续无论是继续 brainstorm、做实现设计、写实验计划，还是拆分具体任务，都应优先和这份文档对齐。

它是一个长期有效的问题定义文档，不是 current-state document（当前状态文档）。

如果你要看最新实验结论、当前最强主线、进行中实验，请先读：

1. [CURRENT_STATE.md](/home/leejt/OneRec/research-progress-log/CURRENT_STATE.md)
2. [experiment_results.csv](/home/leejt/OneRec/experiment_results.csv)

## 当前方向的一句话定义

我们当前的研究方向是：

> 用图结构作为协同信息的载体，并以层级感知的方式，将这种 graph-structured collaborative information（图结构协同信息）融合进 MiniOneRec 的纯语义 SID 构建。

这里还要再补一个当前已经非常明确的点：

- 我们不只需要建模“哪些物品应该在协同上靠近”
- 还需要显式处理“哪些物品虽然语义接近，但在协同上应该被分开”

这里要额外明确：

- `baseline`（基线）只是参考锚点，不是优化目标
- 我们不是要让新的 SID 尽量靠近旧的 SID
- 我们要找到的是一个更好的 SID 码本空间，它可能接近 baseline，也可能明显偏离 baseline
- 最终判断标准是全量下游 `evaluate`（评测），不是 tokenizer（分词器）空间是否足够稳定

这里的重点不是：

- 比谁的 graph encoder（图编码器）更强
- 比较强图模型和弱图模型本身的优劣

这里的重点是：

- 如何让图承载 collaborative information（协同信息）
- 如何让这种协同信息真正增强 SID 的层级结构与局部判别性

## 核心问题 1：用什么图来承载协同信息

### 问题定义

我们首先需要决定：

- 什么样的图最适合作为 collaborative information（协同信息）的结构化载体

这一步的核心不是盲目选择某个现成的 GNN，而是先回答：

- 我们想让图承载哪一种 collaborative relation（协同关系）
- 哪一种图结构最适合服务于 SID 构建

### 当前讨论共识

当前最合理的图对象主要包括：

- item-item 协同图
- item-item transition 图
- 可能的 multiplex graph

图的复杂度不是当前第一优先级。  
像 `LightGCN` 这样的图方法，可以作为一种图信号提取手段，但不是当前问题的核心本体。

### 当前未决点

- `G_coarse` 应该如何定义得既稳定又不过分受 popularity 支配
- `G_mid` 应该如何定义，才能真正承载“比 global 更局部、比 transition 更稳定”的中尺度协同结构
- `G_local` 应该如何表达局部转移信息，同时避免过稀疏和过噪声

## 核心问题 2：怎么做到层级感知

### 问题定义

即便图已经构建出来，我们仍然需要回答：

- 不同 SID level 应该如何使用不同的图结构协同信息

这不是一句简单的：

- Level 1 用 coarse
- Level 2 用 mid
- Level 3 用 local

就能解决的问题。

真正的问题是：

- 为什么不同 level 对协同信息的需求不同
- 这种不同应该通过什么机制体现在 SID learning 里

### 当前讨论共识

当前我们更倾向于：

- 让层级感知体现在 `level-wise allocation`
- 并进一步体现在 `level-wise graph regularization`

也就是说，图不是统一地全局注入，而是：

- 不同 level 接收不同的图结构约束
- 这些约束的强度和来源应当按 level 区分

### 当前未决点

- 层级感知究竟主要靠 graph mixture allocation 实现，还是主要靠 graph regularization 实现
- 哪一种层级感知机制最像“图参与 SID learning”，而不是退化成普通 feature fusion
- 如何证明 learned allocation 的非均匀性是真正有意义的，而不是训练噪声

## 核心问题 3：怎么和 MiniOneRec 的纯语义 SID 融合

### 问题定义

我们不是从零开始造一个全新的 tokenizer，而是在 MiniOneRec 现有纯语义 SID 的基础上引入图结构协同信息。

因此必须回答：

- graph-structured collaborative information 应该以什么形式进入现有 semantic SID pipeline

### 当前讨论共识

当前最重要的原则是：

- 语义 SID 仍然是 backbone
- 图结构协同信息的作用是增强，而不是替代
- 融合不能退化为简单 embedding concat

更准确地说，我们当前倾向于：

- semantic structure 提供基础 hierarchy
- graph structure 修补 semantic SID 在局部判别和层级组织上的盲点
- 融合发生在 quantization learning / structural constraint 层，而不是只发生在 feature 拼接层

### 当前未决点

- graph information 进入 SID 时，最小可行接口是什么
- 如何保证融合后不重演之前 naive collaborative fusion 的 collapse
- semantic anchor、graph regularization、code usage health 之间应如何平衡

## 核心问题 4：怎么显式分离语义接近但协同不一致的物品

### 问题定义

当前方法已经能通过 graph smoothness（图平滑）让图上相邻的物品在量化空间里更靠近，但这还不够。

我们还需要回答：

- 哪些物品虽然在语义上很近，却应该在协同上被显式区分
- 这种“协同上应分离”的关系，应该如何进入 SID learning（SID 学习）

这个问题之所以重要，是因为当前生成式推荐里一个非常关键的失败模式正是：

- 物品名称、描述、语义 embedding（语义嵌入）都很接近
- 但用户群体、购买场景、使用人群明显不同
- 如果没有显式的 separation（分离）或 selective repulsion（选择性排斥），这些物品很容易在 SID codebook space（SID 码本空间）里仍然挤在一起

### 当前讨论共识

当前我们已经可以比较明确地说：

- 只建模“谁应该靠近”是不够的
- 不能把所有 non-neighbor（非邻居）都当成负样本直接推开
- 更合理的目标是：只对 `semantic-close but collaboratively inconsistent`（语义接近但协同不一致）的物品做选择性分离

也就是说，我们需要的不是一个粗暴的全局 repulsion（排斥）机制，而是一个：

- 由协同信息驱动
- 对局部易混淆样本敏感
- 不破坏已有稳定语义结构

的判别式约束

### 当前未决点

- 这种分离约束应该建立在什么 pair（物品对）定义上
- 如何避免把“图上没边但只是未观测到”的物品错误当成负对
- 这种分离信号应当作用在哪个 SID level（SID 层级）最有效
- 它应当作为 graph loss（图损失）的补充项，还是应当进入更直接的 code assignment（码分配）判别机制

## 当前最重要的非目标

为了避免任务跑偏，当前阶段我们暂时不把下面这些当成主目标：

- 证明某个 graph encoder 比另一个更强
- 一上来追求 full personalized dynamic tokenization
- 先写复杂公式，再倒推方法
- 先扩很多 graph view，而不先把三层逻辑讲清楚

## 当前建议的思考顺序

为了减少设计空间失控，后续讨论建议优先按下面顺序推进：

1. 先明确问题 3 的融合原则  
   也就是：图信息进入 SID 的方式必须是“增强 SID 的结构学习”，而不是普通特征拼接。

2. 再明确问题 4 的分离目标  
   也就是：哪些语义接近但协同不一致的物品，应该被显式拉开。

3. 再明确问题 2 的层级机制  
   也就是：层级感知到底通过什么机制落地。

4. 最后再细化问题 1 的具体图构建  
   也就是：在已经明确融合目标和层级机制后，再选择最合适的图载体与图算子。

## 这份文档对应的当前一句话任务

如果把当前任务再压缩成一句话，就是：

> 设计一种方法，使图结构承载的协同信息能够以层级感知、结构约束式的方式，有效地融合进 MiniOneRec 的纯语义 SID 构建中，并形成更好的 SID 码本空间用于下游推荐学习。

这句话还可以再压缩成一个更直接的版本：

> 设计一种方法，让协同信息既能把真正相关的物品在 SID 空间里拉近，又能把语义接近但协同上应区分的物品显式拉开。
