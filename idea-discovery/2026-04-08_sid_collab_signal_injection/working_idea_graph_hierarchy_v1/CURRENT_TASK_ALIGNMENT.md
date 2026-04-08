# 当前任务对齐：MGR-SID 的三个核心问题

## 用途

这份文档只做一件事：

- 把我们当前在 `MGR-SID` 方向上真正要解决的核心问题固定下来

后续无论是继续 brainstorm、做实现设计、写实验计划，还是拆分具体任务，都应优先和这份文档对齐。

## 当前方向的一句话定义

我们当前的研究方向是：

> 用图结构作为协同信息的载体，并以层级感知的方式，将这种 graph-structured collaborative information 融合进 MiniOneRec 的纯语义 SID 构建。

这里的重点不是：

- 比谁的 graph encoder 更强
- 比较强图模型和弱图模型本身的优劣

这里的重点是：

- 如何让图承载 collaborative information
- 如何让这种协同信息真正增强 SID 的层级结构与局部判别性

## 核心问题 1：用什么图来承载协同信息

### 问题定义

我们首先需要决定：

- 什么样的图最适合作为 collaborative information 的结构化载体

这一步的核心不是盲目选择某个现成的 GNN，而是先回答：

- 我们想让图承载哪一种 collaborative relation
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

2. 再明确问题 2 的层级机制  
   也就是：层级感知到底通过什么机制落地。

3. 最后再细化问题 1 的具体图构建  
   也就是：在已经明确融合目标和层级机制后，再选择最合适的图载体与图算子。

## 这份文档对应的当前一句话任务

如果把当前任务再压缩成一句话，就是：

> 设计一种方法，使图结构承载的协同信息能够以层级感知、结构约束式的方式，稳定地融合进 MiniOneRec 的纯语义 SID 构建中。
