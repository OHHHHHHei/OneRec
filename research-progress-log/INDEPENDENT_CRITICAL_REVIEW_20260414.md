# 独立深度审视：MGR-SID 的真实瓶颈与下一步方向

Status（状态）: `snapshot（快照）`

Snapshot date: `2026-04-14`

这份文档是一份去锚定 critical review snapshot（批判性复盘快照），不是 current-state summary（当前状态摘要）。

它最适合用来：

- 回看当时对方法瓶颈的独立批判
- 提炼后续 paper discussion（论文讨论）或 future work（未来工作）素材

如果你只想同步现在的项目状态，请先读：

1. [CURRENT_STATE.md](/home/leejt/OneRec/research-progress-log/CURRENT_STATE.md)
2. [experiment_results.csv](/home/leejt/OneRec/experiment_results.csv)

**日期**: 2026-04-14  
**定位**: 作为独立研究合作者的去锚定批判分析。不默认当前主线正确，不复述已有结论。

---

## 核心判断摘要

在展开之前，先把我最重要的判断放在这里：

1. **我们对 FaGSP 的借鉴是表层的。** 当前的 `_spectral_reconstruct` 只是对称归一化矩阵的特征值切片，而 FaGSP 的核心是高通→支撑选择→低通级联 + 并行高阶邻域滤波。我们用了"频率"这个词，但没有用到 FaGSP 的关键机制。这可能是 G_mid 信号弱于预期的原因之一。

2. **当前最大的单一瓶颈很可能是超参数从未认真调过。** 五组核心图权重（coarse=0.05, mid=0.15, local=0.05, sem_coarse=0.05, sem_mid=0.025）从 v2 一路沿用到 R401d，跨越了 5 个 tokenizer 变体。在说"方向不行"之前，我们其实不知道"这套配置是不是根本就不在图正则的最优工作区间里"。

3. **"结构好但下游不涨"的现象，有一个比"结构指标不重要"更精确的解释：我们的图正则 loss 只有拉力没有推力。** `MSE(H, A@H)` 鼓励邻居靠近，但不惩罚非邻居挤在一起。RQ-VAE 的离散量化空间中，"减少某区域的拥挤"需要推力，而不仅仅是拉力。这可能是结构指标改善但下游不稳定改善的根因——结构指标衡量的是"邻居是否靠近"，但下游需要的是"non-neighbors 是否被正确分开"。

4. **ReSID 的 prefix-conditional entropy 目标才是我们真正缺失的核心组件。** 我们的 learnability probe (R304) 是 ReSID 思想的诊断版本，但我们从未把它变成训练目标。R401b/R401d 的 c|ab learnability 意外提升（0.4712/0.4829 vs v2 的 0.4365）暗示方向是对的，但这个提升是 warm-start + retention 的副产品，不是被主动优化的。

5. **v2 → original 的 recipe preference inversion（v2 需要 title_on, original 需要 title_off）是一个被严重低估的信号。** 它说明 v2 的码本空间真的在结构上不同——LLM 需要额外的上下文信号才能消费它。这不是缺陷，而是一种尚未被充分利用的特性。

---

## A. 方向层面的根本审问

### A1. 我们是不是过度关注结构指标了？

**是的，而且我能精确指出偏移发生在哪里。**

让我追溯时间线：

- v2 阶段：evaluate 是最终裁判，结构指标只是解释工具 → **正确**
- R202a 阶段：R208 downstream 失败 → 做了 R301-R304 interface diagnostics → 开始把 pair retention 和 learnability 当成 token 质量的代理指标 → **还算合理**
- Stage-3 plan 阶段：把 l1 pair retention ≥ 70% 和 learnability halfway recovery 设为 hard gate → **开始偏离**
- R401b/R401d 阶段：发现 pair retention 并未如预期提高，但结构指标大幅改善 → 围绕"为什么 retention 不 work"展开了大量分析 → **明确偏离了**

偏离的根源不是"分析了结构指标"，而是**把结构指标从"解释工具"升级成了"筛选门坎"**。

但我想追问一个更深的问题：**为什么我们会自然地走到这一步？**

我的推测是：因为跑一个 tokenizer 只要几小时，但推到 SFT evaluate 要一天。诊断指标提供了一种"不用跑 SFT 就能判断 tokenizer 好坏"的捷径。这个捷径在 R202a 上看起来有解释力（结构好但 SID 重排大 → 下游差），所以我们自然地推广了它。

**但这个推广是错误的。** R202a 的失败和"SID 重排"之间的因果关系并没有被严格验证。另一个同样合理的假说是：R202a 失败是因为 stop-grad 破坏了层间梯度流，导致 codebook 空间的条件可预测性变差，**而这和 SID 重排是共因（stop-grad）的两个后果，不是因果链**。

### A2. "结构更好"为什么没有稳定转化为"下游更强"？

我认为现有的解释都不够精确。让我提出一个新假设：

**核心假设：当前的结构指标衡量的是"局部一致性"，但下游 LLM 需要的是"全局判别性"。**

展开说：

- `mean l2 leaf count` 下降 = 同一 l2 prefix 下的 item 更少 = 局部不拥挤了
- `H(l3|l1,l2)` 下降 = 给定前缀后叶子的不确定性更低 = 前缀→叶子的映射更清晰了

这些都是**局部**指标——它们衡量的是"一个 subtree 内部的清洁度"。

但 LLM 做 beam search 时面临的是一个**全局**决策：

1. 先预测 a-token → 需要在 256 个 a-token 中选出正确的 → 这需要 a-token 之间有足够的**判别力**
2. 给定 a，预测 b-token → 需要在 256 个 b-token 中选出正确的 → 这需要 b-token 在给定 a 的条件下有足够的判别力
3. beam 展开后，需要在多条路径中保留正确的一条

在这个过程中，"某个 subtree 内部不拥挤"并不直接帮助"在多个 subtree 之间正确选择"。

**我们的图正则只做了前者（拉近邻居 → 减少 subtree 内拥挤），但几乎没有做后者（推远非邻居 → 增强 subtree 间判别）。**

这就是为什么结构指标可以持续改善但下游不稳定：我们在优化的东西和下游需要的东西之间存在**目标不对齐 (objective misalignment)**。

### A3. 当前真正的主瓶颈

我认为当前瓶颈是**三个因素的叠加**，但不是等权的：

| 排名 | 瓶颈 | 证据强度 | 可干预性 |
|---:|---|---|---|
| 1 | **超参数从未认真调过** | 强（配置文件直接可查） | 高（成本低，可快速扫） |
| 2 | **图正则 loss 缺少推力 / 判别性目标** | 中（有理论动机，缺实验验证） | 中（需要改代码） |
| 3 | **图设计偏粗糙（中频图不是真正的 FaGSP）** | 中（代码对比可查） | 中高（graph bank 改动不侵入训练循环） |

让我展开说为什么第 1 个是最大的：

当前 v2 的 NDCG@10 = 0.10271 vs strongest original SFT 0.10372，差距只有 0.001。v2 的 top-1 已经超过 original（0.07059 vs 0.06706）。这意味着 **v2 的方法本身已经接近工作了**，差的可能真的只是一些关键配比。

但我们从 v2 到 R401d，添加了 stop-grad、retention、codebook anchor 等新机制，**同时图正则的核心权重一直没动**。这就像你调味道的时候，盐一直固定在同一个量，然后不停地加新的调料——也许问题就在盐的量上。

### A4. 继续沿"结构改进"这条线的方向性风险

**风险是真实的，但需要精确定义：**

- ❌ 错误的 framing："结构改进方向不对"
- ✅ 正确的 framing："继续只盲做结构改进而不改变注入方式和超参数，大概率是低回报的"

原因：R401b/R401d 的 mean l2 leaves 已经从 4.34 降到 2.70/2.57，H(l3|l1,l2) 从 1.10 降到 0.74/0.72。这些已经是非常激进的结构改善了。如果这样的结构改善在 SFT 后仍然不能beat v2_on_p05，再把结构指标压到 2.0 也大概率没用。

**但如果 R401b/R401d SFT 正向呢？** 那说明当前这套"warm-start + retention + 原有图正则"恰好找到了一个好的 codebook space，此时应该做的不是继续雕结构，而是回去理解**为什么这个空间好**，然后看能不能从 scratch 复现或进一步强化。

---

## B. 图设计层面的深挖

### B1. 当前中频图是否严格实现了 FaGSP？

**不是。差距很大。**

让我做一个精确的对比：

| 维度 | FaGSP 论文 | 当前 `_spectral_reconstruct` |
|---|---|---|
| 基础矩阵 | 归一化拉普拉斯或交互矩阵的 SVD | 对称归一化的邻接矩阵的 `eigsh` |
| 频率选择 | 级联：高通(尾部奇异向量) → 支撑选择 → 低通(头部奇异向量) | 单次：直接取 `[lo:hi]` 范围的特征向量做投影 |
| 非线性处理 | 高通输出经过 item-wise 分位数阈值化 → 二值化 → 增强 | 无。纯线性投影 |
| 并行分支 | 有。item/user 高阶邻域滤波 `F_I = I - (I - O_I)^k` | 无 |
| 输出组合 | `α₂·P₁ + P₂ + P₃`（级联+两个并行分支） | 单一重建矩阵 |

**实质差距**：当前实现是 FaGSP 最粗糙的近似——相当于"取邻接矩阵的中间特征值段做投影"。这是一个有效的band-pass proxy，但**丢失了 FaGSP 的几个关键设计决定**：

1. 高通→支撑选择→低通的级联（非线性步骤）
2. 并行的高阶邻域分支
3. 自适应的频率带宽确定

这意味着：**如果我们想在论文中 claim "使用了 FaGSP 风格的中频图"，这个 claim 在技术上是站得住的（因为确实用了频谱 band selection）。但如果我们想 claim "这就是 FaGSP"，那不成立。**

更重要的是：FaGSP 的高通→支撑选择→低通级联是有信息论动机的——高通先找到"独特的/判别性的"交互，然后用低通平滑它们。这恰好是我们缺少的"判别性"组件。

### B2. 只用 item-item 图够不够？user-item 二部图有没有可能带来关键增益？

**当前阶段不建议引入 user-item 二部图。**

理由：

1. **数据规模限制**：Industrial 只有 3686 items。user-item 图的密度和信号质量在这个规模上不确定。
2. **方法复杂度**：引入 user-item 图需要改变图构建管线和存储方式，侵入性较高。
3. **已有的 item-item 图本身就是从 user-item 交互投影来的**：`build_coarse_graph` 从用户历史构建 item-item 共现，`build_local_graph` 从序列转移构建 item-item 转移。这些已经是 user-item 信息的间接利用。
4. **FaGSP 的并行模块确实用了 user-side 滤波 (`O_U = R·R^T`)**——但那是在交互矩阵上直接操作的，而我们的设置中 RQ-VAE 只处理 item embeddings，没有自然的 user-item 矩阵入口。

**如果要利用 user-side 信息，更自然的方式是**：在 item-item 图构建阶段更好地利用 user 共现模式（比如 CAGCN 的 Common Interacted Ratio），而不是在 tokenizer 训练中直接引入 user embeddings。

### B3. 三层图的角色分配合理吗？

**角色分配在概念上是合理的，但实现上有一个容易被忽视的问题：G_mid 不是独立的信息源。**

当前的三张图：

- `G_coarse = purify(coarse_raw)` → 来自共现
- `G_mid = spectral_reconstruct(G_coarse, [0.25:0.65])` → **来自 G_coarse 的频谱变换**
- `G_local = purify(local_raw)` → 来自转移

**G_mid 是 G_coarse 的线性变换**。它不引入任何新的边或新的信息源。它只是对 G_coarse 的信号做了频段选择。

这意味着：G_coarse 和 G_mid 之间存在**信息冗余**。L1 和 L2 的图正则实际上都在优化同一张底层图的不同视角。如果 G_coarse 本身就有问题（比如过度受 popularity 支配），那 G_mid 也会继承这些问题。

**更好的方案**：G_mid 应该融合独立的信息源。比如 `gsprec_mid_prism`（混合 coarse 和 local 后做频谱重建）在概念上更好，因为它至少引入了转移信息。但当前代码中 `gsprec_mid_prism` 没有被选为训练用的 mid view——`train_v2.py` 固定选的是 `fagsp_mid_base`。

### B4. 哪一层图最弱？

**G_coarse 最弱，因为它同时是两个问题的来源：**

1. 它本身的构建方式偏粗糙（位置衰减加权的共现图 + 最低权重阈值过滤 + popularity 去偏）
2. 它是 G_mid 的输入，所以 G_mid 的质量被它的天花板限制

具体来说，`build_coarse_graph` 的逻辑是：

```python
for i, src in enumerate(unique_seq):
    for j in range(i + 1, len(unique_seq)):
        weight = 1.0 / (j - i)
```

这是一个**极其简单的距离衰减共现权重**。它没有：
- 考虑共现频率（两个 item 共现 100 次和 1 次的权重一样）
- 考虑共现的上下文质量（在长序列中间共现 vs 在短序列中紧邻共现）
- 考虑 CAGCN 式的拓扑可靠性

### B5. 如果重构 graph bank，优先改什么？

我的优先级：

1. **首先试用 `gsprec_mid_prism` 替代 `fagsp_mid_base`**。零代码修改（只需改 config），但引入了独立信息源（转移）到 G_mid 中。这是最低成本的探索。

2. **改进 G_coarse 的边权设计**。两个候选：
   - 加入共现频率因子：`weight = (1.0 / (j - i)) * log(1 + pair_count)`
   - 加入 CAGCN 式的共同交互比率（CIR）：优先保留那些和目标用户有更多共同邻居的边

3. **考虑更完整的 FaGSP band-pass**：在 `_spectral_reconstruct` 中加入高通→支撑选择→低通级联，而非仅做特征值切片。

---

## C. 方法设计层面的深挖

### C1. Graph smoothness 的表达力上限

当前的 graph smoothness loss：

```python
L = MSE(H, A @ H) = Σᵢ wᵢ ||hᵢ - Σⱼ aᵢⱼhⱼ||²
```

**表达力分析**：

这个 loss 等价于 Dirichlet energy 的加权版本。它的梯度方向是让每个 item 的表示靠近其图邻居的加权平均。

**能做到的事**：
- 让图邻居在表示空间中靠近
- 从而让图邻居更可能被分配到相同/相近的 SID prefix

**做不到的事**：
- 让非邻居远离（没有对比/推远力）
- 控制 codebook 的全局布局（不作用于 codebook 向量本身）
- 控制离散分配后的条件可预测性（作用在连续空间，不直接约束离散 token）
- 区分"有益的邻居靠近"和"无益的过度平滑"

**上限估计**：在只有拉力没有推力的情况下，graph smoothness 能做到的最好效果是"让邻居足够近以被分到同一 subtree"。但它无法保证"非邻居被分到不同 subtree"。后者需要显式的判别性目标。

### C2. 哪些外部论文值得直接复用模块？

按"可复用价值/实现成本"排序：

**最值得直接复用的：**

1. **ReSID 的 prefix-conditional entropy 目标**
   - 不是搬整个 GAOQ，而是把 `H(c_l | C_{<l})` 作为训练目标加到现有 loss 中
   - 具体做法：在每个 batch 内统计 `(c_1, c_2)` 和 `(c_1, c_2, c_3)` 的联合分布，计算条件熵，加到 loss 中
   - 这直接解决了"结构好但不可预测"的问题
   - 实现成本：中低。需要在 forward 中拿到离散 token indices（`model.get_indices` 已有），然后做 batch-level 统计
   - **证据支撑**：R304 probe 已经证明 learnability 和下游有关联，把它从诊断变成训练目标是自然延伸

2. **CAGCN 的 Common Interacted Ratio (CIR) 边权**
   - 不是搬 CAGCN 的 GNN，而是用 CIR 公式重新计算 G_coarse 的边权
   - CIR(i,j) = |N(i) ∩ N(j)| / |N(i) ∪ N(j)|，其中 N(i) 是 item i 的用户集合
   - 这给图边加上了"拓扑可靠性"，让图正则只拉那些真正有协同价值的邻居
   - 实现成本：低。离线预计算，只改 `build_coarse_graph`
   - **证据支撑**：当前 G_coarse 的简单距离衰减权重从未被验证过是最优的

**值得认真考虑但实现成本更高的：**

3. **FaGSP 的级联滤波模块**（高通→支撑选择→低通）
   - 改进 G_mid 的构建质量
   - 需要在 `paper_transplants.py` 中实现完整的 FaGSP 级联，替换当前的 `_spectral_reconstruct`
   - 实现成本：中。需要额外的 SVD + 阈值化步骤

**现在不应该做的：**

4. ReSID 的 FAMAE（需要结构化字段特征，我们只有文本嵌入）
5. ReSID 的完整 GAOQ（需要替换整个量化架构，改动太大）
6. FaGSP 的并行滤波模块（需要 user-item 矩阵，我们没有直接入口）

### C3. 哪些模块可以低侵入接入？

**真正低侵入的（只改 graph bank 或 loss 函数，不改 RQ-VAE 架构）：**

1. CIR 边权重新计算 → 改 `graph_bank.py` 的 `build_coarse_graph`
2. prefix-conditional entropy loss → 在 `train_v2.py` 的 training loop 中新增一个 loss 项
3. 图正则加推远力 → 在 `_weighted_graph_smoothness_loss` 旁边加一个 `_weighted_graph_contrastive_loss`
4. 换 mid view 为 `gsprec_mid_prism` → 改 `_build_graph_tensors` 的选择

**中度侵入的（需要改训练循环但不改模型）：**

5. 更完整的 FaGSP 级联滤波 → 改 `paper_transplants.py`
6. warm-start + predictability regularization 的联合训练 → 改 training loop

### C4. 哪些虽然看起来高级但不值得现在做？

1. **替换 RQ-VAE 为 GAOQ**：代价太大，且和我们的增量式研究方式不兼容
2. **引入 user-item 二部图**：数据规模不够，且改动链太长
3. **HiD-VAE 的 uniqueness loss**：概念上有趣但与当前瓶颈不直接对应
4. **多数据集扩展**：在 Industrial 上都还没解决的问题，扩展到 Office 不会提供新信息

---

## D. 实验策略层面的重排

### D1. 超参数真的没调过吗？

**从配置文件来看，以下核心数值从 v2 到 R401d 完全固定：**

```yaml
coarse_weight: 0.05      # 从未变过
mid_weight: 0.15          # 从未变过
local_weight: 0.05        # 从未变过
semantic_coarse_weight: 0.05  # 从未变过
semantic_mid_weight: 0.025    # 从未变过
graph_scale_min: 0.5      # 从未变过
graph_scale_max: 1.5      # 从未变过
band_low: 0.25            # 从未变过
band_high: 0.65           # 从未变过
spectral_rank: 48         # 从未变过
anchor_topk: 32           # 从未变过
semantic_mix: 0.35        # 从未变过
```

唯一真正改过的是：
- R202a：加了 `hierarchy_stopgrad_previous_levels: true`
- R202b_retry075：`coarse_weight: 0.075`（然后因为 collapse 被否了）
- R401b：加了 `prefix_retention_l1_weight: 0.05, prefix_retention_l2_weight: 0.05`
- R401d：加了 `codebook_anchor_l1_weight: 0.05, codebook_anchor_l2_weight: 0.05`

**这意味着：我们花了大量精力设计新的 loss 组件（stop-grad, retention, anchor），但从未回头审视基础 loss 的配比是否合适。**

一个合理的怀疑：如果 `mid_weight` 从 0.15 改成 0.30 或 0.05，效果可能比加 retention loss 更大。我们不知道，因为没试过。

### D2. 如果只允许设计少量高价值实验

我的排序和理由：

#### 第零优先级（必须先完成）
**推 R401b 和 R401d 到 SFT evaluate。**

不是因为我认为它们一定会 work，而是因为我们已经投入了资源生成了这两个码本空间，不看结果就转方向是浪费。而且它们的 learnability probe 信号是正向的。

#### 第一优先级：基础图/loss 配比 sweep

在**当前 v2 配置上**（不是 R401，因为 v2 的下游结果已知，可以用来 calibrate），做一组 2x3 的小 sweep：

| 实验 | 改什么 | 动机 |
|---|---|---|
| v2-mid30 | `mid_weight: 0.30` | G_mid 是信号最强的视角，但权重只有 0.15——不到 recon loss 的零头 |
| v2-mid05 | `mid_weight: 0.05` | 反向验证：如果 mid_weight 减小反而更好，说明 G_mid 的信号质量有问题 |
| v2-local15 | `local_weight: 0.15` | G_local 目前权重太低，可能被压制了 |
| v2-no-sem | `semantic_coarse_weight: 0, semantic_mid_weight: 0` | 语义 retention 可能在拖后腿——特别是在 v2 已经有 ambiguity-aware 的情况下 |
| v2-gsprec | 把 mid view 从 `fagsp_mid_base` 换成 `gsprec_mid_prism` | 测试转移信息融入 G_mid 是否有帮助 |
| v2-band-wide | `band_low: 0.15, band_high: 0.75` | 更宽的频带，测试频段敏感度 |

每个只跑 tokenizer → generate → 推一个到 SFT（取结构最好的 2-3 个）。

**为什么是第一而不是"新模块"**：因为这是信息密度最高的实验类型。如果发现 `mid_weight: 0.30` 比 `mid_weight: 0.15` 的下游好了 0.005，这比引入任何新模块都更有价值——它告诉我们"方向没错，只是还没找到最优工作点"。

#### 第二优先级：加 prefix-conditional entropy loss

在 v2 配置上，加一个轻量的 predictability regularizer：

```python
# 在 training loop 的 forward 之后
with torch.no_grad():
    indices = model.get_indices(batch_embeddings, use_sk=False)
# 计算 batch-level H(c2|c1) 和 H(c3|c1,c2)
pred_loss = batch_conditional_entropy(indices)
loss_total += pred_weight * pred_loss
```

`pred_weight` 从 0.01 开始试。

**这直接对齐了 ReSID 的核心洞察**，且实现成本很低（~20 行新代码）。

#### 第三优先级：CIR 边权

在 `build_coarse_graph` 中，把简单的距离衰减权重替换为 CIR-weighted 版本。这只改离线图构建，不改任何训练代码。

#### 不建议做的

- 更多 retention / anchor 变体（R401e/f/g）→ 已经证明 representation-level retention 不能精确控制 prefix stability
- warm-start control experiment → 学术价值高但时间成本高，且不直接改善下游结果
- v2_uniform ablation → 对论文 claim 重要，但现在不紧急

### D3. 下一阶段优先级排序

```
P0  推 R401b/R401d 到 SFT evaluate           ← 必须做，消解当前最大的信息缺口
P1  图/loss 配比 sweep（6个 tokenizer 变体）    ← 高信息密度，低成本
P2  prefix-conditional entropy loss           ← 直接对齐 ReSID 核心洞察
P3  CIR 边权升级 G_coarse                     ← 低侵入图设计升级
P4  完整 FaGSP 级联滤波                       ← 中侵入图设计升级
P5  v2_uniform ablation                       ← 论文 claim 验证
```

---

## E. 最终判断

### E1. 当前最可能的 3 个真正瓶颈

1. **超参数探索严重不足。** 五组核心图正则权重从未被扫过。我们不知道当前配比是否在图正则的最优工作区间内。所有后续的"方向性"判断都可能被这个基本事实 confound。

2. **图正则 loss 只有拉力没有推力。** `MSE(H, A@H)` 让邻居靠近，但不推远非邻居。RQ-VAE 的离散量化需要判别性才能产生好的 codebook 布局，而当前 loss 不提供这种判别性。这是"结构好但下游不涨"的最可能的 mechanism-level 解释。

3. **G_mid 构建过于简化。** 当前的频谱切片只是 FaGSP 的最粗糙近似，丢失了级联处理中的非线性/判别性步骤。而 G_mid 是三张图中信号最重要的一张（权重最高 0.15），它的质量直接限制了方法的天花板。

### E2. 最建议的 3 个下一步方向

1. **在 v2 基础上做基础 loss 配比 sweep**（~6 个 tokenizer 变体）→ 回答"当前配比是不是最大瓶颈"
2. **实现 prefix-conditional entropy 作为训练目标**（~20 行新代码）→ 直接缩小 tokenizer 优化目标和下游解码需求之间的 gap
3. **用 CIR 重构 G_coarse 的边权**（改 graph bank，不改训练循环）→ 让图正则的输入信号更干净

### E3. 最看好的 2 个可直接借鉴/复用的论文模块

1. **ReSID 的 prefix-conditional entropy 目标 `H(c_l | C_{<l})`**
   - 不需要搬整个 GAOQ，只需要把这一项加到 loss 中
   - 直接回应了 R304 的发现（b|a 和 c|ab 的可预测性和下游相关）
   - 是当前方法论中最明显的"知道应该做但还没做"的缺口

2. **CAGCN 的 Common Interacted Ratio (CIR) 边权计算**
   - 不需要搬整个 GNN，只需要用 CIR 重新加权 G_coarse 的边
   - 直接回应了 G_coarse 构建方式粗糙的问题
   - 实现成本极低（离线预计算，改 ~30 行图构建代码）

### E4. 当前 narrative 的潜在误导

1. **"层级感知"的 claim 还没有被严格验证。** v1_hierarchy SFT (0.09360) 不如 v1_baseline SFT (0.09430)。v2 的改善可能主要来自 ambiguity weighting 和 semantic retention，不是来自三图三层分配。在论文中需要一个明确的 ablation（v2_uniform vs v2）来支撑这个 claim。

2. **"FaGSP 风格的中频图"这个说法可能 oversell。** 当前实现只是对称归一化矩阵的特征值切片，和 FaGSP 的完整方法有显著差距。建议在论文中用"spectral band-pass view inspired by FaGSP"而非暗示这是 FaGSP 的实现。

3. **"warm-start + retention 是 mechanism validation"这个 framing 可能掩盖了一个更重要的事实：** R401b/R401d 的结构改善可能主要来自 warm-start 本身（从一个已经不错的 v2 checkpoint 继续训练 10000 epochs），而不是来自 retention loss。如果是这样，那 retention loss 的 contribution 就被 overstate 了。

4. **过度关注 Industrial 这一个数据集。** 3686 items 是一个很小的测试场。所有的诊断、结构指标、和下游结果都可能有很高的 variance。在写论文之前，至少在 Office Products (4866 items) 上复现核心结论是必要的。但这不是当前最紧急的事。

---

## 附：一句话总结

> 当前 MGR-SID 的大方向（graph-informed SID construction）是对的，但我们在基础功（图设计质量、loss 配比、loss 表达力）上的投入严重不足，同时把过多精力放在了诊断指标的优化和新 loss 组件的堆叠上。下一步最高 ROI 的工作不是继续造新变体，而是回到基本面：调好已有的 loss 配比，加一个真正对齐下游需求的 predictability 目标，升级图的边权质量。
