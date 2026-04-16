# Graph Carrier Brainstorm：到底应该构建什么样的图来承载协同信息

Status（状态）: `discussion-only（仅讨论）`

Created（创建日期）: `2026-04-15`

## 前言：本文的核心问题

> 我们当前 SID tokenizer（SID 分词器）使用三层图（coarse_purified / fagsp_mid_base / local_purified）来承载 collaborative information（协同信息），并以 graph smoothness loss（图平滑损失）注入 RQ-VAE 训练。当前最强已验证线 `v2_on_p05 -> RL` 在 NDCG@10 上已超过 original strongest SFT（原版最强 SFT），但尚未超过 original strongest RL（原版最强 RL）。
>
> 本文的任务：独立调研并 brainstorm（头脑风暴）——到底应该构建什么样的图，来更好地承载 collaborative information（协同信息），从而最有可能把 final evaluate（最终评测）指标刷过 baseline（基线）。

---

## 一、论文借鉴总结

以下 10 个图构造思路来自定向文献调研，每个只回答三个问题。

### 1. TAGCF：Semantics → Topology（语义 → 拓扑）

**原论文**：*Turning Semantics into Topology: LLM-Driven Attribute Augmentation for CF*（arXiv 2502.21099）

- **原本解决什么问题**：交互稀疏性（interaction sparsity）下，LLM 语义信息无法有效进入 CF 图结构。现有方法要么直接用 LLM 预测（幻觉风险大），要么做 embedding fusion（维度失配、目标不对齐）。
- **核心 graph idea（图思想）**：不用语义 embedding（嵌入），而是把 LLM 推理出的"attribute concept（属性概念）"变成**图中的结构节点**——构造 User-Attribute-Item 三部图（tripartite graph）。语义信息通过新增 `u → a → i` 路径来改善图连通性（connectivity），而不是通过特征拼接。属性节点用可学习 ID embedding（可学习 ID 嵌入），不用文本向量。
- **对我们 SID tokenizer 的潜在帮助**：我们当前三层图全都是从 interaction data（交互数据）衍生的（co-occurrence / spectral filter / transition）。TAGCF 的思路提示：可以从 item metadata（物品元数据）中抽取 attribute node（属性节点）作为中间枢纽，让原本在交互图中无法直接连接的 item 通过共享属性节点获得新的 collaborative path（协同路径）。这种新路径带来的信息是原始交互图中不存在的，尤其对 long-tail item（长尾物品）可能有本质帮助。

### 2. CAGCN：Edge Reliability via CIR（基于 CIR 的边可靠性）

**原论文**：*Collaboration-Aware Graph Convolutional Network for Recommender Systems*（WWW 2023, arXiv 2207.06221）

- **原本解决什么问题**：GCN-based CF 对所有 neighbor edge（邻居边）做均匀 message passing（消息传递），但并非所有边都承载同等质量的协同信号。低质量边会"污染"用户/物品表示。
- **核心 graph idea**：提出 Common Interacted Ratio (CIR)——衡量一条边的两端节点是否和同一批其他节点有丰富的偶数长度路径（even-length path）。高 CIR 边说明这对 item 的协同关系被大量共同用户交叉验证过；低 CIR 边则是孤立的、无交叉验证的偶然共现。CIR 用于**非对称边重加权**（asymmetric edge reweighting），而非删边。
- **对我们 SID tokenizer 的潜在帮助**：我们当前 coarse_purified 只做了 popularity debiasing（流行度去偏）和 support pruning（支持度裁剪），但没有评估"一条边的协同路径是否在 graph 中被交叉验证过"。CIR-style（CIR 风格）的边重加权可以直接应用在 coarse/mid/local 任何一层图上，让 graph smoothness loss（图平滑损失）更多地被高质量边驱动。这是一种"图质量提升"而非"换图"的思路。

### 3. FaGSP：Cascade Spectral Filter（级联谱滤波）

**原论文**：*Frequency-Aware Graph Signal Processing for Collaborative Filtering*（arXiv 2402.08426）

- **原本解决什么问题**：GSP-based CF 只做 low-pass smoothing（低通平滑），丢失了 distinctive/personalized interaction（独特/个性化交互）信号。
- **核心 graph idea**：**先高通、后低通的级联滤波**（cascade filter）。先用 ideal high-pass filter（理想高通滤波器）在频谱上分离出"高频/独特"交互，将它们的权重放大 α₁ 倍，再在增强后的信号上做 low-pass reconstruction（低通重建）。另设一个 parallel module（并行模块）在 item-side 和 user-side 分别用高阶多项式低通滤波捕捉多跳邻域。
- **对我们 SID tokenizer 的潜在帮助**：我们当前 fagsp_mid_base 只做了 band-pass filter（带通滤波）——取第 25%~65% 的 eigenvector（特征向量），但没有做"先识别独特交互、再放大、再平滑"这种级联。FaGSP 的级联逻辑暗示：如果我们在构造 G_mid 时先**识别出"被当前 low-pass 滤掉的个性化信号"**并有选择地保留，可能比简单 band-pass 更有效。此外，item-side parallel filter 的高阶多项式低通 `f(λ) = 1 - λ^k` 可以直接替代当前 eigenvector truncation（特征向量截断）方式。

### 4. GSPRec：Transition-Aware Spectral Graph（转移感知谱图）

**原论文**：*Temporal-Aware Graph Spectral Filtering for Recommendation*（arXiv 2505.11552）

- **原本解决什么问题**：标准 GSP-CF 丢弃了交互序列中的顺序信息（sequential dynamics），且只用 low-pass 丢失了 mid-frequency 个性化信号。
- **核心 graph idea**：在 item-item block 中直接编码 sequential transition（序列转移）——如果 item i 在任何用户历史中直接出现在 item j 前面，则加一条有向边 i→j，再对称化后做多跳扩散 `S^(d) = Σ α^(k-1)(S')^k`。然后把这个 transition-enriched item-item block 塞进 bipartite graph 的 item-item 区域，在整个增广图上做 Gaussian band-pass filter（高斯带通滤波）。
- **对我们 SID tokenizer 的潜在帮助**：我们的 local_purified 已经捕捉 history→target transition（历史→目标转移），但只用于 L3。GSPRec 暗示：transition 信号经过对称化 + 多跳扩散后可以成为一种**结构上更丰富的 item-item graph**（不再只是 1-hop transition），可以被用于 G_mid 甚至整个图空间，而不仅仅被限制在 local 层。

### 5. FREEDOM：Freeze Semantic + Denoise Interaction（冻结语义 + 去噪交互）

**原论文**：*A Tale of Two Graphs*（ACM MM 2023, arXiv 2211.06924）

- **原本解决什么问题**：多模态推荐中，semantic item-item graph（语义物品-物品图）和 interaction graph（交互图）被混在一起联合学习，增加了参数和不稳定性。
- **核心 graph idea**：**角色分离**——semantic graph 在训练前构建一次然后 freeze（冻结），interaction graph 则每 epoch（轮次）做 degree-aware stochastic edge pruning（度感知随机边裁剪）。两种图各走各的 LightGCN propagation（LightGCN 传播），最终 item embedding = CF embedding + semantic residual（CF 嵌入 + 语义残差）。
- **对我们 SID tokenizer 的潜在帮助**：我们当前的 semantic_knn_graph 也已经是 freeze 的（cosine KNN from Qwen embeddings），但它只作为 retention loss（保持损失）用于 L1/L2，权重很小（0.05/0.025）。FREEDOM 的启示是：如果 semantic graph 的角色应该是 **stable anchor（稳定锚点）** 而不是次要的 regularization（正则化），那可能应该**增大**它在低层（L1）的权重，同时对 interaction-derived graph（交互衍生图）在高层做更激进的 denoising。

### 6. GraphAug / GIB：Information Bottleneck Denoising（信息瓶颈去噪）

**原论文**：*Graph Augmentation for Recommendation*（arXiv 2403.16656）

- **原本解决什么问题**：图对比学习（graph contrastive learning）中随机 augmentation（增强）会破坏真实结构。
- **核心 graph idea**：用 Graph Information Bottleneck (GIB) 训练一个 edge predictor（边预测器），学出"保留哪些边对 downstream prediction（下游预测）最有信息量、丢弃哪些边是噪声"。目标是 minimize $I(Z'; A)$（压缩对原始邻接矩阵的依赖）while maximize $I(Z'; Y)$（保留对推荐目标的预测力）。
- **对我们 SID tokenizer 的潜在帮助**：GIB 原则性地解决了"哪些边是有用的、哪些是噪声"这个问题。如果我们把 GIB 的逻辑搬到 graph construction（图构建）阶段——即在构造 coarse/mid/local 图之前，先用一个轻量 bottleneck model（瓶颈模型）过滤掉"对最终推荐预测无贡献"的边——理论上可以得到一套更干净的图作为 tokenizer supervision（分词器监督）。不过这需要额外的预训练步骤。

### 7. Behavior-Conditioned Diffusion Denoising（行为条件扩散去噪）

**原论文**：*IGDMRec: Behavior Conditioned Item Graph Diffusion for Multimodal Recommendation*（arXiv 2512.19983）

- **原本解决什么问题**：从多模态特征构建的 semantic item-item graph 包含大量 false positive edge（假阳性边）——语义相似但用户行为完全不同的物品被错误连接。
- **核心 graph idea**：把 semantic item graph 视为"noisy version（噪声版本）"的理想图，用一个 diffusion model（扩散模型）学习去噪，去噪过程以 user behavior pattern（用户行为模式）为条件——只保留那些既语义相似、又在用户行为上 compatible（兼容）的边。
- **对我们 SID tokenizer 的潜在帮助**：直接对应我们的问题——当前 coarse_purified 和 fagsp_mid_base 都源于 co-occurrence（共现），但和 item 的语义特征之间可能存在 mismatch（失配）。如果我们构建一个"语义 + 行为交叉验证"的图——只保留在语义图和行为图中都确认的边——可以得到更高 precision（精度）的 item-item graph。不过 diffusion model 本身的训练成本可能不划算（3686 个 item 可能不需要这么重的方法）。

### 8. Community-Detection Graph Cleaning（社区检测图清洗）

**原论文**：*ALDA4Rec: Adaptive Long-term Embedding with Denoising and Augmentation*（arXiv 2504.13614）

- **原本解决什么问题**：item-item graph 中不尊重 community partition（社区划分）的边是噪声。
- **核心 graph idea**：先对 item-item graph 做 community detection（社区检测），然后 prune（裁剪）所有跨社区但 weight 低于阈值的边。社区内的边被保留和增强，社区间的低质量连接被切断。
- **对我们 SID tokenizer 的潜在帮助**：**社区结构天然对应 SID 层级**。如果我们在 coarse graph 上做 community detection，那么同一 community 的 item 理论上应该共享 L1 prefix（前缀）。这种"先发现社区、再用社区结构指导 tokenizer 码本"的逻辑非常直觉化，且实现成本低。关键问题是 community 的粒度控制——太粗则 L1 区分度不够，太细则退化成 item-level。

### 9. Graph RQ-VAE / MoToRec：图结构直接参与量化

**原论文**：
- *MMGRec: Multimodal Generative Recommendation with Transformer Model*（arXiv 2404.16555）
- *MoToRec: Sparse-Regularized Multimodal Tokenization for Cold-Start Recommendation*（arXiv 2602.11062）

- **原本解决什么问题**：标准 RQ-VAE 的 codebook 学习不知道 item 之间的 collaborative structure（协同结构）。
- **核心 graph idea**：在 RQ-VAE 的 encoder 端引入 graph encoder（图编码器），让 item embedding 在进入 quantization（量化）之前已经融合了图邻域信息。MoToRec 进一步加了 sparse regularization（稀疏正则化）来保证 codebook 的可组合性。
- **对我们 SID tokenizer 的潜在帮助**：这是"改 encoder（编码器），不改图"的思路。我们当前 encoder 是纯语义的（Qwen embedding → MLP），图信息只通过 auxiliary loss（辅助损失）注入。如果把 graph neighborhood aggregation（图邻域聚合）直接接入 encoder output（编码器输出）再送入 quantizer（量化器），图信息对 codebook assignment（码本分配）的影响会更直接。**但这已经涉及 encoder 架构改动，而非 graph carrier 本身的改进。**

### 10. Multi-Hop Diffusion + Cross-Modal Consensus（多跳扩散 + 跨模态共识）

**综合自多篇论文**（2502.08071, 2406.12501, 2505.11552）

- **原本解决什么问题**：单模态/单跳图只捕捉了部分协同信号。
- **核心 graph idea**：(a) 对 item-item transition graph 做 multi-hop diffusion `Σ α^k A^k` 捕捉更长距离的 sequential pattern；(b) 只保留在多个 view（视图）中都确认的边（cross-modal consensus）——例如同时在 co-occurrence view 和 semantic view 中都是邻居的边才保留。
- **对我们 SID tokenizer 的潜在帮助**：我们的 local_purified 只有 1-hop transition，丢失了"item A → B → C" 这种 2-hop 甚至 3-hop 链式结构。对 local_purified 做 multi-hop diffusion 可以把 sparse transition graph 变成 denser（更密集的）、更稳定的近邻图。cross-modal consensus 则是一种简单但有效的 denoising 方式——只保留同时被 interaction data 和 semantic embedding 支持的边。

---

## 二、Graph Carrier 候选设计空间

以下每一类候选图都明确定义了节点、边、承载的协同信息、相对当前的新增信息、以及建议放置的层级。

### 候选 1：CIR-Reweighted Coarse Graph（CIR 重加权粗粒度图）

| 维度 | 定义 |
|------|------|
| **节点** | 全部 3686 个 item |
| **边** | 和当前 coarse_purified 相同的 co-occurrence edge（共现边） |
| **边权** | 在 popularity debiasing（流行度去偏）之后，再乘以 CIR 分数：`w_ij = w_ij_current × φ(i,j)`。CIR 衡量 i 和 j 是否被大量共同用户"交叉验证"过。|
| **承载的协同信息** | "经过路径级交叉验证的全局协同关系"——不仅要共现，还要在 graph 的更大结构中被多条路径支撑 |
| **相对当前 coarse_purified 新增了什么** | 当前只用 co-occurrence count + popularity debiasing，不考虑"这条边在图中是否被交叉验证"。CIR 引入了 edge-level quality signal（边级质量信号） |
| **建议放置** | **G_coarse**（替换当前 coarse_purified） |

### 候选 2：Cascade-Filtered G_mid（级联滤波中尺度图）

| 维度 | 定义 |
|------|------|
| **节点** | 全部 3686 个 item |
| **边** | 从 coarse_purified 衍生，但不再是简单 band-pass |
| **构建方式** | FaGSP-style cascade：(1) 对 coarse_purified 做 ideal high-pass，识别出"独特/非主流"的 item-item 关系；(2) 将这些关系的权重放大 α₁ 倍后加回原矩阵；(3) 在增强后的矩阵上做 top-p₂ low-pass reconstruction。最终结果既保留主流社区结构、又不丢失个性化信号 |
| **承载的协同信息** | "主流社区结构 + 被保护的个性化信号"——一种 frequency-aware collaboration signal（频率感知协同信号） |
| **相对当前 fagsp_mid_base 新增了什么** | 当前 fagsp_mid_base 只做 band-pass（丢弃 top-25% 和 bottom-35%），这实际上丢弃了部分独特但有价值的高频信号。cascade filter 先识别再保护这些信号 |
| **建议放置** | **G_mid**（替换当前 fagsp_mid_base） |

### 候选 3：Multi-Hop Diffused Transition Graph（多跳扩散转移图）

| 维度 | 定义 |
|------|------|
| **节点** | 全部 3686 个 item |
| **边** | 基于 local_purified，但做 2~3 hop diffusion：`A_diff = A + α A² + α² A³` |
| **边权** | 多跳扩散后的累积权重（recency decay × hop decay），然后 row-normalize |
| **承载的协同信息** | "多步 sequential transition pattern（多步序列转移模式）"——不仅 A→B 的直接转移，还有 A→B→C 的间接链式关系 |
| **相对当前 local_purified 新增了什么** | 当前 local_purified 只有 1-hop history→target 边，且是有向的。multi-hop diffusion 捕捉了更长距离的 sequential context（序列上下文），且对称化后变成稳定的 item-item graph |
| **建议放置** | **G_local**（替换当前 local_purified）或 **G_mid 的 alternative view（备选视图）** |

### 候选 4：Cross-View Consensus Graph（跨视图共识图）

| 维度 | 定义 |
|------|------|
| **节点** | 全部 3686 个 item |
| **边** | 只保留同时在 co-occurrence view（共现视图）**和** semantic view（语义视图）中都是 top-K 近邻的 item pair |
| **边权** | `w_ij = w_cooccurrence(i,j) × sim_semantic(i,j)`，即行为和语义的乘积 |
| **承载的协同信息** | "行为上共现 + 语义上相似"的双重确认关系——precision 最高的 collaboration signal |
| **相对当前任何一层图新增了什么** | 当前三层图都只来自 interaction data（交互数据），semantic_knn_graph 只作为 retention loss。这种 consensus graph 把 semantic information 从"辅助正则"提升为"图构建的 gating 条件（门控条件）" |
| **建议放置** | **G_coarse** 或 **G_mid**——视为 high-precision backbone graph（高精度骨干图） |

### 候选 5：Attribute-Bridge Item Graph（属性桥接物品图）

| 维度 | 定义 |
|------|------|
| **节点** | 3686 个 item + K 个 attribute node（属性节点，从 item metadata 中提取） |
| **边** | item-attribute 边（item 具有某属性则连边）+ item-item 边可通过 2-hop path `i → a → j` 间接构成 |
| **构建方式** | 从 item 的 category、brand、price range、description keywords 等 metadata 提取 attribute（属性）；对 attribute 做 frequency filtering（频率过滤，去掉过普遍和过罕见的）；构建 bipartite item-attribute graph，然后 project（投影）成 item-item graph：`A_attr = B × B^T`（B 是 item-attribute 关联矩阵）|
| **承载的协同信息** | "基于属性共享的隐式协同关系"——即使两个 item 从未在同一用户历史中出现过，只要它们共享足够多的 attribute，就建立协同连接 |
| **相对当前新增了什么** | 当前所有图都源自 interaction data。这种图引入了 interaction-independent（不依赖交互）的 item 关系，对 cold/sparse item 有本质帮助 |
| **建议放置** | **G_coarse**（作为 global structure 补充）或 **mixed with G_mid** |

### 候选 6：Degree-Aware Epoch-Pruned Interaction Graph（度感知逐轮裁剪交互图）

| 维度 | 定义 |
|------|------|
| **节点** | 全部 3686 个 item |
| **边** | 每个 training epoch（训练轮次），从 coarse_purified 中 stochastic subsample（随机子采样）一个子图，sampling probability 和 node degree 反相关（高度节点的边更可能被 drop） |
| **承载的协同信息** | "去除 popularity-driven spurious edge（流行度驱动的虚假边）后的核心协同结构" |
| **相对当前新增了什么** | 当前 coarse_purified 虽然做了 popularity debiasing，但 debiasing 只调整权重、不删边。stochastic pruning 进一步做**结构级去噪** |
| **建议放置** | **G_coarse** 或全部三层的 augmentation（数据增强）策略 |

### 候选 7：Community-Initialized Hierarchical Graph（社区初始化层级图）

| 维度 | 定义 |
|------|------|
| **节点** | 全部 3686 个 item |
| **边** | 和当前相同的 co-occurrence 边 |
| **构建方式** | 先对 coarse_purified 做 multi-resolution community detection（多分辨率社区检测）——例如 Louvain 在不同 resolution parameter（分辨率参数）下得到 3 层社区结构：~256 个粗社区 / ~256×256 个中社区 / item 级。然后根据社区归属关系重新定义三层图：G_coarse = 粗社区的 inter-community graph（社区间图）；G_mid = 同一粗社区内的 intra-community graph（社区内图）；G_local = 同一中社区内的 fine-grained graph（细粒度图） |
| **承载的协同信息** | "层级化的社区结构"——每层图的边只在特定粒度的社区内/间起作用 |
| **相对当前新增了什么** | 当前三层图的"层级分工"完全来自不同的 graph construction method（图构建方法）（co-occurrence / spectral / transition），而非来自同一图的不同粒度。community-based 的层级划分和 SID 的 3-level codebook 在语义上更 aligned（对齐） |
| **建议放置** | **替换全部三层**，或者至少替换 G_coarse + G_mid |

### 候选 8：Semantic-Behavioral Intersection Graph（语义-行为交集图）

| 维度 | 定义 |
|------|------|
| **节点** | 全部 3686 个 item |
| **边** | 取 semantic_knn_graph 的 top-K 近邻集合与 coarse_purified 的 top-K 近邻集合的**交集** |
| **边权** | 取两个图中边权的 geometric mean（几何平均）：`w = sqrt(w_semantic × w_cooccurrence)` |
| **承载的协同信息** | "语义和行为双重确认的核心关系"——去掉了仅靠语义相似或仅靠共现频率的 spurious edge（虚假边） |
| **相对当前新增了什么** | 当前 semantic graph 和 interaction graph 完全分开使用（不同 loss 项），没有做**集合运算级别的交叉过滤** |
| **建议放置** | **G_mid**——作为一种 high-precision mid-scale graph |

### 候选 9：User-Cluster-Projected Item Graph（用户聚类投影物品图）

| 维度 | 定义 |
|------|------|
| **节点** | 全部 3686 个 item |
| **边** | 先对 user 做 clustering（聚类），每个 user cluster 定义了一个 item subset（物品子集）= 该 cluster 中所有 user 交互过的 item。然后对每个 cluster 内的 item 做 pairwise co-occurrence → item-item edge。最终图 = 所有 cluster-projected subgraph 的 union（并集），但每条边带有"来自多少个不同 cluster"的 count 作为权重 |
| **承载的协同信息** | "user-segment-level collaborative signal（用户群体级协同信号）"——不同类型的用户群各自贡献了哪些 item-item 关系 |
| **相对当前新增了什么** | 当前 coarse_purified 混合了所有用户的共现，不区分用户群体。这种投影方式让每条 item-item edge 都带有 user-type diversity（用户类型多样性）信息——如果一条边只来自一个很小的 user cluster，它的可靠性更低 |
| **建议放置** | **G_coarse** 或 **G_mid** |

### 候选 10：Polynomial Item-Side Low-Pass Graph（多项式物品侧低通图）

| 维度 | 定义 |
|------|------|
| **节点** | 全部 3686 个 item |
| **边** | 从 item co-occurrence matrix `O_I = R^T R` 衍生，应用多项式低通滤波 `F_I = I - (I - O_I)^k`，k=2~4 |
| **构建方式** | 先从 user-item interaction matrix 得到 item-item co-occurrence `O_I`，然后用 FaGSP-style 多项式滤波器 `f(λ) = 1 - λ^k` 做 nonlinear concave low-pass |
| **承载的协同信息** | "多阶 item-side collaborative neighborhood（多阶物品侧协同邻域）"——不是简单共现，而是通过 user 中介的高阶共现路径 |
| **相对当前新增了什么** | 当前没有直接使用 item-item co-occurrence matrix `O_I = R^T R`（当前 coarse 是 session-level co-occurrence 而非 matrix product）。这种方式通过 user-mediated path 捕捉了更全局的 item 关系 |
| **建议放置** | **G_coarse** 或 **G_mid**——多项式阶数 k 控制平滑程度（k 大 = 更 global） |

---

## 三、高价值候选 Top 5

排序标准：最可能提升 final evaluate（最终评测）> 最可能带来真实新信息 > 最不只是换壳重排同一批边。

### Top 1：Cross-View Consensus Graph（跨视图共识图，候选 4）

**为什么排第一**：
- 我们当前三层图全部来自 interaction data，semantic graph 只做辅助 retention loss。但"item 是否在语义上也相似"这个信息从未被用来**过滤** interaction graph 的边。
- Cross-view consensus 是一种简单但高 precision 的去噪手段——被 co-occurrence 和 semantic embedding 同时确认的边，大概率是真正的 collaborative signal。
- 实现成本极低：只需要对 coarse_purified 和 semantic_knn_graph 的 top-K 近邻取交集，不需要新的模型或预训练。
- **最可能带来的效果**：更干净的 G_coarse 或 G_mid 意味着 graph smoothness loss 的方向更准确，减少把语义不相关的 item 拉到一起的 false pull。

**风险**：交集可能太稀疏（如果 co-occurrence 和 semantic 的 overlap 很小），需要调 K 值。

### Top 2：CIR-Reweighted Coarse Graph（CIR 重加权粗粒度图，候选 1）

**为什么排第二**：
- CIR 是对"边质量"的 principled measure（原则性度量），比单纯 support pruning + popularity debiasing 更有信息量。
- 它不改变图的拓扑结构（不删边、不加边），只做 reweight（重加权），所以和现有 pipeline 的兼容性最好。
- 当前 coarse_purified 经过 popularity debiasing 后，仍然可能包含大量"只出现在一两个 session 中、没有被多条路径交叉验证的"弱边。CIR 可以系统性地 down-weight 这些边。
- **最可能带来的效果**：更高质量的 G_coarse 直接改善 L1 的 graph supervision，同时因为 fagsp_mid_base 是从 coarse_purified 衍生的，更好的 coarse → 更好的 mid。

**风险**：3686 个 item 的 CIR 计算复杂度 OK（离线预计算），但需要验证在小规模图上 CIR 的区分度是否足够。

### Top 3：Community-Initialized Hierarchical Graph（社区初始化层级图，候选 7）

**为什么排第三**：
- 这是当前方案中 **逻辑上最大的结构性缺陷** 的直接解决方案：我们的三层图来自三种不同的图构建方法（co-occurrence / spectral band-pass / transition），但 SID 的三层 codebook 应该反映同一图的不同粒度。
- Community detection → 多分辨率层级结构天然映射到 256/256/256 的 codebook 结构：粗社区数 ≈ L1 code 数，中社区数 ≈ L1×L2 组合数，等。
- Louvain 等算法在 3686 个节点上跑不到 1 秒，实现零成本。
- **最可能带来的效果**：layer-wise graph 和 codebook 层级在语义上 aligned，L1 code 真正对应一个 co-occurrence community，而不是 spectral filter 的 artifact。

**风险**：community detection 的 resolution parameter 需要仔细调——如果粗社区数远不等于 256，则和 codebook 的 alignment 会打折扣。

### Top 4：Cascade-Filtered G_mid（级联滤波中尺度图，候选 2）

**为什么排第四**：
- 当前 fagsp_mid_base 的 band-pass 丢弃了 top-25% eigenvector（最平滑的全局结构）和 bottom-35%（最高频的噪声）。但"高频"不全是噪声——其中包含"个性化/非主流"的有价值信号。
- FaGSP cascade 的核心价值是：先 **识别** 这些高频中的有价值部分（用 high-pass filter 分离出来），然后**有选择地保留**，再做 low-pass reconstruction。
- 这是对当前 mid graph 的 **precision 提升**——不是换一种全新的图，而是用更好的滤波方式从同一份数据中提取更多有效信号。
- **最可能带来的效果**：更丰富的 G_mid 信号（保留了部分个性化高频）+ 更干净的结构（low-pass 重建去噪），直接改善 L2 的 graph supervision。

**风险**：需要调 α₁（高频信号放大系数）和 p₂（low-pass 截断维度）两个超参数。

### Top 5：Multi-Hop Diffused Transition Graph（多跳扩散转移图，候选 3）

**为什么排第五**：
- 当前 local_purified 只有 1-hop history→target 边，是所有三层图中最稀疏的。multi-hop diffusion 是**零成本**的密度增强方式（矩阵乘法 A² + αA³）。
- 对称化后的 multi-hop transition graph 可以捕捉"两个 item 经常出现在同一个 2-step 或 3-step browsing chain（浏览链）中"的模式，这是 1-hop graph 无法表达的。
- **最可能带来的效果**：让 L3 的 graph supervision 不再受限于过稀疏的 1-hop transition，给 fine-grained codebook 更多有效结构信号。

**风险**：多跳扩散会引入更多间接关系，可能包含噪声。需要配合 hop-decay factor α < 1 来控制。

---

## 四、伪创新排除

### 伪创新 1：只换 graph encoder 不换图

- **典型做法**：把 LightGCN 换成 GAT / GraphSAGE / GIN 等"更强"的 encoder 来处理同一批图
- **为什么是伪创新**：我们的核心问题不在 "graph encoder 不够强"——graph smoothness loss 本身根本不用 GNN encoder，它直接对 adjacency matrix 和 representation 做 MSE。换 encoder 不改变图承载的信息
- **本质**：把同一批边用不同的方式聚合，但信息源没变

### 伪创新 2：给 co-occurrence 图加更多归一化/正则化变体

- **典型做法**：把 popularity debiasing 从 `α=0.5` 调到 `α=0.7`，或者换一种归一化方式（symmetric vs. random-walk normalized）
- **为什么是伪创新**：归一化只改变现有边的相对权重，不引入新的信息。当前 coarse_purified 的问题不是"归一化不对"，而是"有些边本身就不该在"
- **本质**：同一批数据的同一种信号，只是换了缩放方式

### 伪创新 3：把 transition graph 换成 attention-weighted transition graph

- **典型做法**：给 history→target 的 transition 加一个 learned attention weight（学习的注意力权重），而不是固定的 1/recency_rank
- **为什么是伪创新**：transition 的权重不管是 recency decay 还是 learned attention，信息源都是"history 中第 k 个 item → target item"。attention weight 只是在同一批边上做微调
- **本质**：同一批 transition 边的权重微调

### 伪创新 4：把 band-pass 的区间从 [25%, 65%] 调到 [20%, 70%]

- **典型做法**：调 fagsp_mid_base 的 `eigen_ratio_low` 和 `eigen_ratio_high`
- **为什么是伪创新**：这只是调超参数，不改变 mid graph 的信号来源和构造逻辑。真正的问题是 band-pass 的 cascade 逻辑（或者说 band-pass 本身是否是最好的滤波方式），而不是区间 boundary
- **本质**：超参数搜索

### 伪创新 5：构建 user-item bipartite graph 然后 project 成 item-item graph

- **典型做法**：直接用 `A_item = R^T × R`（R 是 user-item interaction matrix）得到 item-item co-occurrence
- **为什么不算真创新**：虽然这和当前 session-level co-occurrence 的构造方式不完全一样，但信息来源高度重叠——都是"哪些 item 被同一批 user 交互过"。差异只是"session window 内的共现 vs. user 级别的共现"
- **部分例外**：如果 user-mediated co-occurrence 和 session-level co-occurrence 的 overlap 真的很小（因为 history_k=10 导致很多 user-level co-occurrence 被 session window 截断），那这就不算伪创新，而是一种 information source expansion。**需要实际度量两种图的边集 overlap**

---

## 五、独立判断：如果只能押 2 条图载体方向

### 押注 1：Cross-View Consensus Graph（跨视图共识图）作为新 G_coarse 或 G_mid

**核心论据**：

1. **当前最大的未利用信号**：我们有 Qwen embedding 提供的 semantic similarity，但它从未被用来"筛选" interaction graph 的边。当前 semantic_knn_graph 只作为 retention loss（权重 0.05 / 0.025），作用极弱。Cross-view consensus 直接把语义信息提升为边选择的 gate（门控），利用效率质变。

2. **直接解决一个具体的 downstream failure mode（下游失败模式）**：stage-2 和 stage-3 的教训表明"tokenizer-side 结构指标好 ≠ downstream 好"。一个合理的解释是：当前 graph smoothness loss 把某些语义不相关但行为上偶然共现的 item 拉到了一起，导致 codebook 中出现"语义不一致的 cluster（语义不一致的聚类）"——downstream model 在这些 cluster 上学不动。Consensus graph 直接消除这类 false pull。

3. **实现成本接近零**：只需一行 set intersection + 一行 geometric mean。不需要新模型、不需要预训练、不需要新超参数（K 值可以直接复用当前 topk=32）。

4. **失败的 downside（下行风险）有限**：即使 consensus graph 太稀疏导致 graph supervision 太弱，也只需要调大 K 值。不会像"换 encoder"或"加 diffusion model"那样引入新的不可控因素。

**为什么最可能帮我们刷过 baseline**：

当前 `v2_on_p05 → RL` 和 original strongest RL 的差距在 HR@10 上是 -0.0095。这个差距的一个可能来源是：tokenizer 产生了某些 code cluster 中的 item 语义不一致，导致 downstream SFT/RL 在这些 cluster 的 beam search 中排序错误。Consensus graph 通过消除"语义不确认的 co-occurrence 边"来减少这类 cluster 的形成概率，从而可能在 HR@10 上回收一部分损失。

### 押注 2：Community-Initialized Hierarchical Graph（社区初始化层级图）全面替换三层图的层级划分逻辑

**核心论据**：

1. **解决当前最大的结构性问题**：当前三层图的"层级分工"是人为规定的（co-occurrence → L1, spectral band-pass → L2, transition → L3），和 SID codebook 的 3-level hierarchy 之间没有 principled alignment（原则性对齐）。为什么 spectral band-pass 应该对应 L2 而不是 L1？没有答案。Community detection 提供了一种 **从数据出发的层级结构**——粗社区、中社区、细粒度近邻——和 codebook 的 coarse→mid→fine 直接映射。

2. **从根本上改变"三层图承载了三种不同信号"的假设**：当前方案假设 L1 需要 co-occurrence、L2 需要 spectral mid-band、L3 需要 transition——但这三种信号之间的关系从未被验证。如果它们其实是同一份 co-occurrence 数据在不同粒度上的 view（视图），那用 community detection 做 multi-resolution decomposition 才是 coherent（连贯的）的做法。

3. **天然适配 codebook size**：Louvain 的 resolution parameter（分辨率参数）可以调到让粗社区数 ≈ 256，这恰好等于 L1 的 codebook size。如果我们能让 L1 code 和粗社区一一对应，那 graph smoothness loss 对 L1 的 supervision 就变成了"同社区的 item 应该得到相同的 L1 code"——这比当前模糊的"在 coarse graph 上 embedding 应该平滑"要 sharp（清晰）得多。

4. **实现简单**：`python-louvain` / `leidenalg` 在 3686 节点图上 < 1s。multi-resolution 用不同 resolution 参数跑 3 次即可。

**为什么最可能帮我们刷过 baseline**：

当前 v2 的 L1 prefix 在 stage-2/3 探索中反复暴露出"prefix 不够稳定"或"prefix 和 downstream learnability 不对齐"的问题。一个根本原因可能是：当前 G_coarse 产生的 graph smoothness loss 对 L1 codebook 的 supervision 太"糊"——它只说"co-occurrence 近邻的 embedding 应该接近"，但没说"哪些 item 应该分到同一个 L1 code"。Community structure 提供了一种更 discrete、更 sharp 的 supervision：同一粗社区 = 同一 L1 code prefix。这种 sharper supervision 可能让 L1 routing（路由）更准确，进而让 L2/L3 的 conditional prediction（条件预测）更有效，最终改善 downstream evaluate。

---

## 附录：思考链总结

```
当前状态：
  v2_on_p05 → RL: NDCG@10=0.10432 (> original SFT, < original RL)
  HR@10=0.14185 (< original SFT, < original RL)
  gap to beat: NDCG@10 需要 > 0.10726, HR@10 需要 > 0.15133

核心约束：
  只改图载体，不改 encoder / loss 架构 / downstream pipeline
  最终判断标准是 full downstream evaluate，不是 tokenizer-side metrics

关键洞察 from 文献调研：
  1. 边质量 > 边数量（CAGCN）
  2. 层级应来自同一图的不同粒度，而非不同图（community detection 文献）
  3. 语义信息应参与边选择，不只是辅助 loss（FREEDOM, TAGCF）
  4. 频率感知应做 cascade，不只做 band-pass（FaGSP）
  5. sequential transition 可以 multi-hop 扩散成更稳定的结构（GSPRec）

我的两个押注的共同逻辑：
  问题不在"图的信号太弱"，而在"图的信号不够精确"。
  - Cross-view consensus 通过 precision filtering（精度过滤）提升边质量
  - Community hierarchy 通过 structural alignment（结构对齐）提升层级监督的 sharpness
  两者都不是"加更多信号"，而是"更精确地使用已有信号"。
```
