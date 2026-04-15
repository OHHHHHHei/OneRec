# Stage-2 后的方法方向再审视：如何让层级感知图正则的结构收益真正传导到下游

**日期**: 2026-04-13  
**定位**: 在 Stage-2 首轮实验（stop-grad / semantic retention KL / interface diagnostics）全部完成后，重新审视方法论方向。

---

## 1. 当前困境的精确描述

### 1.1 我们已经证明的

1. **图结构协同信息对 SID 构建有用**（probe 阶段，阶段 1-2）
2. **层级感知的图正则比均匀正则更好**（v1 hierarchy_reg vs uniform_reg）
3. **歧义感知的加权进一步改善了 tokenizer 结构**（v2 vs v1）
4. **v2 的收益能存活到 end-to-end RL**（v2_on_p05 → RL）
5. **v2 RL 在 top1、top3、top50 和 collided-target 解析上优于原始 MiniOneRec RL**

### 1.2 我们还没有解决的

v2_on_p05 RL 在 **@5/@10/@20** 的中段 beam retention 上仍然落后于原始 MiniOneRec RL。

Stage-2 试图通过 tokenizer-side 微调来补这个 gap，但发现：

> **tokenizer 结构改善（R202a stop-grad）没有传导到下游。**

### 1.3 根因分析（来自 R301-R304 interface diagnostics）

- R301：stop-grad 导致 **99.65% item 改变了 l1 前缀**，全局 SID 重排
- R302：token 语义多义性没有变差 → 排除了"code 变得更混乱"的假说
- R303：**改善的样本 = l1 Jaccard 高（l1 路由保留好）+ l2 Jaccard 低（l2 被积极重写）**；恶化的样本则相反
- R304：R202a 让 a 的预测变容易了，但 **b|a 和 c|a,b 的预测变难了**

**核心矛盾**：stop-grad 让每层独立优化各自的图正则目标，结构上确实更干净了；但独立优化的副作用是**层间的条件可预测性下降了**——level 1 不再为 level 2/3 提供好的条件上下文。

---

## 2. 为什么"冻结 L1/L2 只调 L3"不是好的研究方向

用户的判断是对的。冻结前两层的问题是：

1. **与论文的核心 claim 冲突**。我们的 story 是"不同 SID level 应该感知不同粒度的协同结构"。如果冻结 L1/L2，就意味着放弃了在前两层注入协同信息的机会，整个层级感知的叙事退化成了"只在 leaf level 做 collaborative refinement"——这跟 ACLR 的早期想法没有本质区别。

2. **回避了真正的问题**。真正的问题不是"改了 l1 就一定坏"，而是"改了 l1 之后下游模型学不会新的 l1 路由"。冻结是一种回避，不是解决。

3. **上限有限**。如果只能改 L3，那能改善的只是叶子级消歧。但当前证据说最大的协同信号来自 G_mid（中尺度），而 G_mid 作用于 level 2。冻结 L2 就放弃了最有价值的信号源。

---

## 3. 重新定义问题

把问题重新表述为：

> **如何在保持层级感知图正则的同时，让 SID 空间的变化对下游模型友好？**

这个问题有三个子问题：

- (A) **如何让 level 1 的变化更温和**，而不是彻底冻结它？
- (B) **如何让新的 SID 空间在条件可预测性上不退化**？
- (C) **如何在 tokenizer 训练时就考虑下游可学习性**？

---

## 4. 五个候选方向

### 方向 A：渐进式层级训练（Curriculum Hierarchical Training）

**灵感来源**：HiD-VAE 的层级监督 + RQ-VAE staged training 的思想 + SoundStorm/VAR 的 level-by-level 生成

**核心思路**：不是同时训练三层 + 三张图，而是**分阶段逐层引入图正则**：

- **Phase 1**（warmup）：正常训练 RQ-VAE，不加任何图正则。让三层 codebook 先到达稳定的语义量化均衡。这个阶段产生的 SID 空间就是原始语义 tokenizer 的空间。
- **Phase 2**（coarse injection）：冻结 L2/L3 codebook，**只对 L1 施加 G_coarse 正则**。L1 codebook 温和地调整，适应粗粒度协同结构。因为 L2/L3 冻结，L1 的调整不会导致全量 SID 重排——只是 L1 token 的分配变了，L2/L3 保持不变。
- **Phase 3**（mid injection）：解冻 L2（L1/L3 保持冻结），**对 L2 施加 G_mid 正则**。L2 在 L1 已经稳定的条件下做中尺度优化。
- **Phase 4**（local injection）：解冻 L3（L1/L2 保持冻结），**对 L3 施加 G_local 正则**。

**为什么这保持了层级感知的 claim**：三层都被图正则影响了，只是不是同时。每一层都感知了不同粒度的协同结构，只不过是按从粗到细的顺序逐步注入的。

**为什么这解决了 SID 重排问题**：每一阶段只有一层在变，其他两层锁定。所以每次下游需要适应的变化量只有 1/3。而且因为是从粗到细的顺序，L1 先稳定后 L2 再调整，L2 稳定后 L3 再调整——这保证了**层间的条件可预测性**。

**与 R304 发现的关系**：R304 显示 R202a 的 b|a 和 c|a,b 可预测性下降了。渐进式训练直接解决这个问题——因为 L2 是在 L1 已经稳定的前提下训练的，所以 b|a 的可预测性天然得到保证。

**实现复杂度**：中等。需要修改训练循环来支持分阶段冻结/解冻，但不需要新的 loss 函数或新的模型组件。

### 方向 B：SID 可预测性正则化（Predictability-Aware Regularization）

**灵感来源**：ReSID 的 prefix-conditional entropy 最小化 + R304 learnability probe 的发现

**核心思路**：在 tokenizer 训练目标中，**显式加入一个 SID 可预测性正则项**。当前的训练目标是：

$$\mathcal{L} = \mathcal{L}_{rec} + \mathcal{L}_{rq} + \sum_l \lambda_l \mathcal{L}_{graph}^{(l)}$$

在此基础上，增加一个**条件熵正则**：

$$\mathcal{L}_{pred} = H(z^{(2)} | z^{(1)}) + H(z^{(3)} | z^{(1)}, z^{(2)})$$

直觉是：图正则鼓励 item 的量化表示在图邻域上平滑，但它不关心最终的离散 token 序列是否容易被下游模型预测。可预测性正则显式要求：**在 codebook 分配之后，deeper level 的 token 应该能从 earlier level 的 token 被合理预测**。

**为什么这保持了层级感知的 claim**：所有三层仍然接受不同粒度的图正则。可预测性正则不是替代图正则，而是一个**补充约束**，确保层级结构对下游友好。

**ReSID 的具体做法**：ReSID 使用 Globally Aligned Orthogonal Quantization (GAOQ)，联合最小化重建熵和 prefix-conditional 熵。我们可以借鉴其 prefix-conditional entropy 的计算方式，但不需要采用它的全部框架。

**具体实现**：
- 在每个 batch 内，统计 (z¹, z²) 的联合分布和 z¹ 的边际分布
- 计算条件熵 H(z² | z¹)
- 加入到总 loss 中，鼓励 z² 的选择在给定 z¹ 条件下是"可预测的"（低熵）

**R304 作为 evaluation metric**：R304 的 learnability probe 可以直接作为这个方向的 offline evaluation metric——如果可预测性正则有效，R304 的 b|a 和 c|a,b accuracy 应该不降反升。

**实现复杂度**：中等偏低。条件熵可以在 batch 内近似计算（batch 内统计 token co-occurrence），不需要全局统计。

### 方向 C：Codebook 锚定正则（Codebook Anchor Regularization）

**灵感来源**：DCCL 的 EWC-style codebook 保护 + Backward-Compatible Training 文献 + Codebook Transfer 的思想

**核心思路**：不是冻结 codebook，而是给 codebook 加一个**弹性锚定约束**：允许 codebook 向量移动，但高使用频率的 codebook entry 被惩罚移动得太远。

$$\mathcal{L}_{anchor} = \sum_{l=1}^{3} \sum_{k=1}^{K} w_k^{(l)} \| c_k^{(l)} - c_k^{(l),\text{init}} \|^2$$

其中 $c_k^{(l),\text{init}}$ 是训练开始时的 codebook 向量（来自 v2 的 pretrained checkpoint），$w_k^{(l)}$ 是该 codebook entry 的重要性权重（可以用使用频率、或 Fisher information、或下游 loss 的梯度幅度来估计）。

**为什么这比冻结更好**：
- 冻结是硬约束（$w_k = \infty$）；锚定是软约束（$w_k$ 有限）
- 高频使用的 codebook entry（对应大量 item 的 SID token）被强约束保持稳定
- 低频使用的 codebook entry（对应少量 item）被允许移动以适应图正则
- 这自然产生了"稳定多数 + 改善少数"的效果——恰好是 R303 告诉我们需要的行为

**与 R303 的关系**：R303 显示恶化的样本往往是 easy/stable 的——它们对应的 codebook entry 使用频率高。锚定正则会自动保护这些高频 entry，避免 easy case 的路由被破坏。

**与层级感知 claim 的关系**：每一层的 codebook 都在被图正则调整，只是调整的幅度受锚定约束控制。这完全符合"层级感知协同融合"的叙事——我们没有放弃在任何层级注入协同信息，只是让注入的方式更温和、更保结构。

**实现复杂度**：低。只需要在训练开始时保存 codebook snapshot，然后在 loss 中加一个 L2 正则项。

### 方向 D：唯一性损失 + 对比级别感知正则（Uniqueness + Contrastive Level-Aware Regularization）

**灵感来源**：HiD-VAE 的 Uniqueness Loss + 对比学习思想

**核心思路**：当前的 graph smoothness loss 只有"拉近"力（鼓励图邻居表示接近），没有"推开"力。HiD-VAE 的 uniqueness loss 提供了一个显式的"推开"机制——对于被分配到相同 SID 的不同 item，惩罚它们的连续表示过于相似。

但我们可以做得更好：**把 uniqueness 做成 level-aware 的**。

- **Level 1**：对被分配到同一个 a-token 的 item，不施加 uniqueness（因为 level 1 就是要把它们分在一起）
- **Level 2**：对被分配到同一个 (a,b) 前缀的 item，如果它们**不是图邻居**，则用对比 loss 推开它们的 level-2 表示
- **Level 3**：对被分配到同一个 (a,b,c) 完整 SID 的 item（即 collision），用最强的推开力

这比当前的 smoothness-only loss 更精细：
- Graph smoothness 只拉近邻居
- Level-aware contrastive 同时拉近邻居 + 推开非邻居
- 而且推开力的强度随层级递增（level 1 最弱，level 3 最强）

**为什么这可能解决 SID 重排问题**：当前 smoothness-only loss 的一个问题是它可以通过"把所有 item 拉到一起"来最小化——这给了 codebook 太多自由度去重排。加入对比项后，每个 codebook entry 不仅需要覆盖正确的图邻居，还需要排斥非邻居。这大大约束了 codebook 的解空间，使得 codebook 的变化更受控。

**实现复杂度**：中等。需要在 batch 内构造正负样本对，但不需要外部数据或新的模型组件。

### 方向 E：双阶段 tokenizer（语义 warmstart → 图正则 finetune），配合 codebook 锚定

**核心思路**：把方向 A（渐进训练）和方向 C（codebook 锚定）组合成一个更简洁的两阶段方案：

- **Stage 1**：用原始 MiniOneRec 的方式训练 RQ-VAE（纯语义，无图正则），得到一个稳定的语义 tokenizer。
- **Stage 2**：从 Stage 1 的 checkpoint 出发，加入层级感知图正则 + codebook 锚定正则继续训练。

Stage 2 的 loss 是：

$$\mathcal{L} = \mathcal{L}_{rec} + \mathcal{L}_{rq} + \sum_l \lambda_l \mathcal{L}_{graph}^{(l)} + \alpha \mathcal{L}_{anchor}$$

其中 $\mathcal{L}_{anchor}$ 锚定到 Stage 1 的 codebook。

**为什么这是一个好的平衡**：
- Stage 1 保证了 SID 空间从一个稳定的语义起点出发
- Stage 2 的图正则注入协同信息，但锚定正则防止 codebook 偏离太远
- 最终的 SID 空间既包含了协同结构（来自图正则），又保持了对下游的友好性（来自锚定约束）

**与当前 v2 的关系**：当前 v2 实际上已经是一种隐式的"warmstart → finetune"——它从 scratch 训练但同时加了 reconstruction loss + graph regularization。方向 E 的改进是**把这两个阶段显式分开**，并在第二阶段加入锚定约束。

**这不就是重新训练了吗？**：不完全是。关键区别在于 $\alpha \mathcal{L}_{anchor}$ 的存在。没有锚定时，Stage 2 可以任意重排 codebook（就像 R202a 那样）。有锚定时，codebook 只能在原始语义解的附近做小幅调整——这恰好是我们想要的："语义为主，协同为辅，结构性注入而非破坏性替换"。

---

## 5. 优先级排序与推荐

| 优先级 | 方向 | 成本 | 风险 | 与 claim 的兼容性 | 与当前代码的兼容性 |
|---|---|---|---|---|---|
| **P0** | **E：双阶段 warmstart + codebook 锚定** | 低 | 低 | 完全兼容 | 只需加一个 L2 正则项 |
| **P0** | **B：SID 可预测性正则** | 中低 | 中 | 完全兼容 | 需要在 batch 内计算条件熵 |
| **P1** | **C：codebook 锚定正则（独立）** | 低 | 低 | 完全兼容 | 同上 |
| **P1** | **A：渐进式层级训练** | 中 | 中 | 完全兼容 | 需要修改训练循环 |
| **P2** | **D：对比级别感知正则** | 中 | 中高 | 完全兼容 | 需要构造正负样本对 |

### 推荐的组合方案

我最推荐的是 **E + B 的组合**：

1. **双阶段训练**：Stage 1 纯语义 → Stage 2 加层级感知图正则
2. **Codebook 锚定**：Stage 2 对 codebook 加 L2 anchor 到 Stage 1 的 checkpoint
3. **可预测性正则**：Stage 2 额外加 prefix-conditional entropy 项，确保层间条件可预测性不退化

这个组合的叙事非常干净：

> "我们的层级感知图正则 tokenizer 分两阶段训练。第一阶段学习稳定的语义量化，第二阶段在语义基础上注入层级感知的协同结构。通过 codebook 锚定和可预测性正则，我们确保协同信息的注入是渐进且保结构的，避免了全量 SID 重排对下游模型学习的破坏。"

---

## 6. 具体实验设计

### Block A：Codebook 锚定实验（最小验证）

**目的**：验证 codebook 锚定能否在保持 R202a 的结构收益的同时减少 SID 重排。

**实现**：
1. 从当前 v2 的 best checkpoint 出发
2. 保持 stop-grad hierarchy isolation
3. 加入 codebook anchor loss：$\alpha \sum_{l,k} f_k^{(l)} \| c_k^{(l)} - c_k^{(l),\text{init}} \|^2$
   - $f_k^{(l)}$ = 该 codebook entry 在 v2 checkpoint 中的使用频率（归一化）
   - $\alpha$ 从小到大试几个值（0.01, 0.05, 0.1）
4. 训练完成后运行 R301 检查 prefix stability

**成功标准**：
- R301 的 l1 pair retention 从 41.4% 提升到 > 70%
- R204 的结构指标不明显回归（mean l2 leaf count 仍 < 4.0）
- 如果 R301 + R204 都通过，推到 SFT screen

### Block B：可预测性正则实验

**目的**：验证 prefix-conditional entropy 正则能否改善 R304 的 b|a 可学习性。

**实现**：
1. 在 Block A 的最佳 α 基础上，额外加入：
   $$\mathcal{L}_{pred} = \beta \cdot \hat{H}(z^{(2)} | z^{(1)})$$
   其中 $\hat{H}$ 是 batch 内近似的条件熵
2. β 从小到大试（0.01, 0.05）

**成功标准**：
- R304 的 b|a accuracy 不低于 v2 的 0.2392
- R301 的 prefix stability 仍然高

### Block C：下游 SFT Screen

**目的**：验证 Block A/B 的最佳候选是否在下游超过 v2_on_p05。

**设置**：固定 recipe（title_on + desc_p05），单 seed first。

**成功标准**：HR@10 和 NDCG@10 至少持平 v2_on_p05 SFT。

---

## 7. 这些方向为什么与层级感知的 claim 完全兼容

关键理解是：**层级感知的核心 claim 不是"每层的 codebook 必须自由变化"，而是"不同层应该感知不同粒度的协同结构"**。

- 方向 A-E 都保持了三层图正则的层级分配（L1←G_coarse, L2←G_mid, L3←G_local）
- 它们添加的约束（codebook 锚定、可预测性正则、渐进训练）不是在取消层级感知，而是在**约束层级感知的实现方式**——让它更温和、更保结构、对下游更友好
- 论文可以把这些约束叙述为"层级感知协同注入的训练策略"——它们是 method 的一部分，不是对 method 的削弱

**类比**：就像 knowledge distillation 不是在"放弃 student model 的学习能力"，而是在"约束 student 的学习方式使其更高效"。Codebook 锚定和可预测性正则是对层级感知图正则的"学习策略约束"，不是对其的否定。

---

## 8. 一句话总结

> Stage-2 的核心发现是"层级感知图正则改善了结构但破坏了 SID 空间的下游友好性"。解决这个问题不需要放弃层级感知（冻结前两层），而是需要**让层级感知的注入方式更温和**——通过 codebook 锚定、可预测性正则、或渐进式训练来约束 SID 空间的变化幅度，在保持层级感知 claim 的同时实现结构收益到下游的稳定传导。
