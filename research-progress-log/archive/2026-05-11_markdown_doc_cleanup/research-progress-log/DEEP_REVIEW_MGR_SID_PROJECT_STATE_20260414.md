# MGR-SID 项目深度复盘与方向再审视

Status（状态）: `snapshot（快照）`

Snapshot date: `2026-04-14`

这份文档保留为一次深度 review snapshot（深度复盘快照）。

它不是 current-state document（当前状态文档），也不应该和 [CURRENT_STATE.md](/home/leejt/OneRec/research-progress-log/CURRENT_STATE.md) 竞争权威入口。

适合在这两种场景再读：

- 需要回看 `2026-04-14` 时点的系统性判断
- 需要提炼论文叙事或 postmortem（复盘）材料

如果你想先同步最新主线，请读：

1. [CURRENT_STATE.md](/home/leejt/OneRec/research-progress-log/CURRENT_STATE.md)
2. [experiment_results.csv](/home/leejt/OneRec/experiment_results.csv)

**日期**: 2026-04-14  
**定位**: 在 Stage-3 首轮 tokenizer 候选空间（R401b、R401d）全部完成、即将推 SFT 之前，对整个项目的方法论、实验证据、论文借鉴关系、和下一步方向做一次系统性的深度审视。

---

## 1. 思维链起点：我们到底在做什么

### 1.1 项目的一句话定义

> 利用层级感知的图结构协同信息，改善 MiniOneRec 的 SID 码本空间构建，使下游 LLM 能更好地学习推荐任务。

### 1.2 这句话中每个词的含义

- **层级感知**：SID 是三层 RQ-VAE 编码 `<a, b, c>`。不同层应该接收不同粒度的协同信号。这是方法论的核心 claim。
- **图结构协同信息**：用户行为中的协同关系（共现、转移、频谱中频）被编码为 item-item 图。图是信息的载体，不是最终表示。
- **改善 SID 码本空间**：不是改善图本身，不是改善 LLM 本身，而是改善 tokenizer 产出的离散码本空间。
- **下游 LLM 能更好地学习**：最终评判标准是 evaluate 结果（NDCG@k, HR@k），不是 tokenizer 侧的结构指标。

### 1.3 为什么这个定义现在特别重要

因为在过去两周的实验中，我们曾经偏离过这个定义。具体来说：

- Stage-2 的 interface diagnostics 引导我们把"prefix stability"从诊断工具升级成了优化目标。
- Stage-3 的实验计划一度被 frame 为"保守地让 SID 靠近 v2 baseline"。
- 我们差点把"codebook drift"和"pair retention"当成 hard gate 来决定是否推 SFT。

这些都是错误的。每次 SFT 都是从 base model 出发学一个全新的码本空间。SID 空间离 baseline 远不远，和下游 LLM 学得好不好，**不是**单调关系。

---

## 2. 全量实验证据盘点

### 2.1 所有已有下游 evaluate 结果

这是目前唯一的"硬性证据"。按 NDCG@10 排序：

| 排名 | 系统 | Tokenizer | Recipe | Stage | NDCG@10 | HR@10 | NDCG@1 | HR@1 |
|---:|---|---|---|---|---:|---:|---:|---:|
| 1 | strongest original RL | original semantic | off+p05 | RL | **0.10726** | **0.15133** | 0.07324 | 0.07324 |
| 2 | v2_on_p05 RL | mgr_v2 | on+p05 | RL | **0.10432** | 0.14185 | **0.07434** | **0.07434** |
| 3 | strongest original SFT | original semantic | off+p05 | SFT | 0.10372 | **0.15089** | 0.06706 | 0.06706 |
| 4 | v2_on_p05 SFT | mgr_v2 | on+p05 | SFT | 0.10271 | 0.14626 | **0.07059** | **0.07059** |
| 5 | v2_on SFT | mgr_v2 | on+off | SFT | 0.10082 | 0.14251 | 0.07037 | 0.07037 |
| 6 | R202a SFT (R208) | stage2_r202a | on+p05 | SFT | 0.09974 | 0.14251 | 0.06552 | 0.06552 |
| 7 | original on+off SFT | original semantic | on+off | SFT | 0.09872 | 0.14714 | 0.06243 | 0.06243 |
| 8 | original on+p05 SFT | original semantic | on+p05 | SFT | 0.09870 | 0.14207 | 0.06353 | 0.06353 |
| 9 | original refactor SFT | original semantic | off+off | SFT | 0.09930 | 0.14207 | 0.06618 | 0.06618 |
| 10 | v1_hierarchy SFT | mgr_v1_hierarchy | on+off | SFT | 0.09360 | 0.13038 | 0.06265 | 0.06265 |
| 11 | v1_baseline SFT | mgr_v1_baseline | on+off | SFT | 0.09430 | 0.13435 | 0.06309 | 0.06309 |
| 12 | v2_off_off SFT | mgr_v2 | off+off | SFT | 0.09125 | 0.13391 | 0.05890 | 0.05890 |
| 13 | v2_desc_align SFT | mgr_v2 | off+p05 | SFT | 0.08993 | 0.13082 | 0.05912 | 0.05912 |

### 2.2 从这张表中能读出什么

**发现 1：v2 的 SID 空间确实不同于 original semantic SID。**

- original semantic 在 `title_history2sid_off` 下最强（NDCG@10 = 0.10372）
- v2 在 `title_history2sid_on` 下最强（NDCG@10 = 0.10271）
- v2 在 `title_history2sid_off` 下反而大幅恶化（0.08993）

这说明 v2 的码本空间改变了 LLM 学习 SID 的方式——它需要显式的 SID 结构消费（title→SID 映射）才能发挥优势。这是一个有意义的 mechanism finding。

**发现 2：v2 的 top-1 / top-3 优势是真实的。**

- v2_on_p05 SFT 的 NDCG@1 = 0.07059，比 strongest original SFT 的 0.06706 高 5.3%
- v2_on_p05 RL 的 NDCG@1 = 0.07434，比 strongest original RL 的 0.07324 高 1.5%

这意味着 v2 的码本空间在"最精确的推荐"上更好。LLM 更容易学到 top-1 命中。

**发现 3：v2 的 mid-beam（@5/@10）仍有 gap。**

- v2_on_p05 RL NDCG@10 = 0.10432 vs strongest original RL 0.10726 (−2.7%)
- v2_on_p05 RL HR@10 = 0.14185 vs strongest original RL 0.15133 (−6.3%)

这个 gap 是我们一直在试图解决的。

**发现 4：R202a 不仅没解决 mid-beam gap，反而损失了 top-1 优势。**

- R202a SFT NDCG@1 = 0.06552 vs v2_on_p05 SFT 0.07059 (−7.2%)
- R202a SFT NDCG@10 = 0.09974 vs v2_on_p05 SFT 0.10271 (−2.9%)

这是唯一一个 stage-2 tokenizer 推到 SFT 的结果，而且是负面的。

**发现 5：v1（hierarchy / baseline）的下游表现显著低于 v2 和 original。**

- v1_hierarchy SFT NDCG@10 = 0.09360，比 v2_on_p05 SFT 低 8.8%
- 这说明 v1 的图正则方式虽然在 tokenizer 侧有一定效果，但下游转化很差

### 2.3 尚未推到 SFT 的 tokenizer 候选

| Tokenizer | Collision | Status | Mean l2 leaves | H(l3\|l1,l2) |
|---|---:|---|---:|---:|
| original semantic | 16 (0.43%) | rl_evaluated | — | — |
| mgr_v2 | 13 (0.35%) | rl_evaluated | 4.3422 | 1.1001 |
| R202a (stage2) | 13 (0.35%) | sft_evaluated | 3.6148 | 1.0308 |
| R401b (stage3) | 11 (0.30%) | **sft_ready** | **2.6967** | **0.7373** |
| R401d (stage3) | 11 (0.30%) | **sft_ready** | **2.5711** | **0.7156** |

R401b 和 R401d 在结构指标上**远超**所有前代 tokenizer。但结构指标不是最终判断标准——它们需要推到 SFT 才知道是否真的更好。

### 2.4 Learnability Probe 对比（3-seed mean）

| Variant | a acc | b\|a acc | c\|ab acc |
|---|---:|---:|---:|
| v2 | 0.0908 | **0.2424** | 0.4365 |
| R202a | 0.0978 | 0.2118 | 0.4159 |
| R401b | 0.0780 | 0.2484 | **0.4712** |
| R401d | 0.0799 | 0.2455 | **0.4829** |

**这是非常重要的发现**：

- R401b 和 R401d 的 `b|a` learnability **回到了甚至超过了 v2 水平**（0.2484 / 0.2455 vs 0.2424），而 R202a 是 0.2118
- R401b 和 R401d 的 `c|ab` learnability **大幅超过所有前代**（0.4712 / 0.4829 vs v2 的 0.4365）
- 但 `a` learnability 降低了（0.0780 / 0.0799 vs v2 的 0.0908）

这意味着：
1. R401b/R401d 的 SID 空间在条件可预测性上**优于** v2 和 R202a
2. 但 level-a 的预测变难了——可能因为 a-token 的分配方式改变了
3. R202a 的核心问题（b|a 可预测性下降）在 R401b/R401d 中被修复了

---

## 3. 方法设计的诚实审视：我们借鉴的论文真的有用吗？

### 3.1 论文→方法的实际映射

| 来源 | 我们说的借鉴 | 实际在代码中的体现 | 真实贡献度 |
|---|---|---|---|
| **FaGSP** | 中频带通图构建 → G_mid | `paper_transplants.py` 中的 `fagsp_mid_base` 构建 | **高** — 这是我们方法中最独特的组件之一。G_mid 的中频谱设计直接来自 FaGSP 的 band-pass 思想，且 G_mid 在实验中被确认为信号最强的图视角。 |
| **ReSID** | prefix-conditional entropy / 层级可预测性 | 未直接实现。R304 learnability probe 受其启发设计作为诊断工具。 | **中低** — 思想层面的启发，但未进入训练目标。Stage-3 plan 中的 predictability regularizer（Block 4 contingency）是它的潜在落地点，但目前还是 DEFERRED。 |
| **HiD-VAE** | hierarchical uniqueness / 层级分离 | 未直接实现。stop-grad 是一个更简单的层级隔离机制。 | **低** — HiD-VAE 的 uniqueness loss 概念被讨论过但从未实现。stop-grad 在 R202a 中用了，但它不是来自 HiD-VAE，而是一个更通用的梯度隔离技巧。 |
| **LETTER** | tokenization quality / code assignment | 未直接实现。collision rate 和 codebook utilization 是标准的 VQ 诊断，不特指 LETTER。 | **低** — 更像是 related work 中的背景引用，而非方法设计的直接来源。 |
| **ACLR（项目早期内部方法线，不是论文）** | local collaborative repair / ambiguity framing | 主要体现在历史问题 framing：它帮助我们更早注意到 local ambiguity、选择性干预和 ambiguity gate 的价值。当前 `v2` 的实现并不是 ACLR 的移植版。 | **中低** — 这是内部历史主线带来的 conceptual carry-over，不应被写成外部文献借鉴。当前 `offline_combined` prior 和 item-level reweighting 仍然是后续 tokenizer 线里重新长出来的实现。 |

### 3.2 诚实结论

**真正在代码中发挥作用的外部论文借鉴，最直接的其实主要是 FaGSP（G_mid 构建）。**

其余论文更多是在 related work 叙事和实验设计思路上提供了参考，而非在方法实现中直接使用。  
另外，`ACLR` 不是外部论文，而是项目早期的一条内部 alternative line；它最多算历史上的 conceptual predecessor，不能写成论文来源。

这不一定是坏事——说明我们的方法不是论文拼凑，而是基于自己的问题定义和实验发现逐步构建的。但在写论文时需要诚实地描述借鉴关系：

- FaGSP：方法层面的直接借鉴（图构建）
- ReSID：评估设计的启发（learnability probe）
- HiD-VAE / LETTER：related work 对比，非方法来源
- ACLR：项目内部早期方案，对问题 framing 有历史影响，但不是外部论文来源

### 3.3 我们方法中真正原创的部分

1. **三图三层的 role assignment**：G_coarse → L1, G_mid → L2, G_local → L3。这个设计不来自任何单一论文。
2. **Ambiguity-aware graph reweighting**：用 offline prior 对 item 级别的图正则强度做差异化加权。这个具体实现是我们自己的。
3. **Graph smoothness as structural supervision**：用 `MSE(H, A@H)` 而非对比学习或图编码器来注入协同信息。这个选择使得方法极其轻量，且与 RQ-VAE 的量化训练自然兼容。
4. **Semantic retention**：用语义 kNN 相似度分布的 KL 散度来保持语义结构不被图正则破坏。

---

## 4. 方法演化的诚实回顾：每一步的 motivation 对不对？

### 4.1 v1 → v2：ambiguity-aware weighting

**Motivation**：v1 的均匀图正则对所有 item 一视同仁，但有些 item 的语义位置已经很好了（easy），强推它们反而会破坏。

**实验验证**：v2 在 tokenizer 结构上优于 v1，且 v2 SFT 结果远好于 v1 SFT（0.10082 vs 0.09360）。但需要注意，v2 用了不同的 recipe（on+off vs on+off），所以这个对比是公平的。

**判断**：**Motivation 成立，且有实验支撑。**

### 4.2 v2 → R202a (stop-grad)：层级隔离

**Motivation**：v2 的三层图正则的梯度会互相干扰——L2 的图 loss 会影响 L1 的 codebook，导致层间不一致。

**实验验证**：R202a 的 tokenizer 结构确实更好（mean l2 leaves 4.34→3.61），但 SFT 结果变差（0.10271→0.09974）。interface diagnostics 显示 99.65% 的 l1 prefix 改变了。

**判断**：**Motivation 部分成立**（层级隔离确实改善了局部结构），**但 solution 有严重副作用**（全局 SID 重排）。问题在于 stop-grad 是一个太强的干预——它同时解决了层间梯度干扰和破坏了层间条件可预测性。

### 4.3 R202a → R401b/R401d (teacher-guided retention + codebook anchor)：保守修复

**Motivation**：R202a 的问题是 SID 空间变化太大导致下游学不好。所以用 teacher retention 约束上层表示不要偏离太远。

**实验验证**：R401b/R401d 的 pair retention 并没有如预期般提高（l1: 0.41 vs R202a 0.41, l2: 0.45 vs R202a 0.61）。但结构指标进一步大幅改善（mean l2 leaves 2.70/2.57），且 learnability probe 的 b|a 和 c|ab **回升甚至超过了 v2**。

**判断**：**原始 motivation（保持 prefix 稳定）没有达成，但意外地产生了一个结构更强、learnability 更好的新码本空间。**

这是一个关键的认知修正：retention loss 的价值可能不在于"保持离 v2 近"，而在于它作为额外的正则化约束，帮助训练找到了一个更好的码本空间均衡点。

### 4.4 演化路径的整体判断

```
v1 (uniform graph reg)
  ↓ motivation: 区分 easy/hard item → 正确
v2 (ambiguity-aware graph reg)
  ↓ motivation: 解决层间梯度干扰 → 部分正确，但 solution 太强
R202a (stop-grad isolation)
  ↓ motivation: 约束 SID 空间不要偏离太远 → 目标未达成，但意外发现更好的空间
R401b/R401d (teacher retention ± codebook anchor)
  ↓ ??? 下游 evaluate 还没跑
```

每一步的 motivation 都有一定道理，但实际效果往往不完全符合预期。**这很正常**——研究就是这样的。但重要的是：

1. 我们每次都有清晰的 motivation → design → experiment → evaluation 链条
2. 负面结果（R202a downstream regression）被正确地用来修正方向
3. 诊断工具（R301-R304）提供了超越 evaluate 数字的 mechanism understanding

---

## 5. 当前设计的根本性审问

### 5.1 "图正则"这个注入方式是不是天然有上限？

当前的 graph smoothness loss 是：

```
L_graph = MSE(H, A @ H)
```

这个 loss 的含义是：让图邻居的表示更接近。

**本质上这是一个平滑约束**。它能做到的事情是：
- 让共现/转移/频谱中频邻居在表示空间中更接近
- 从而让这些邻居更可能被分配到相同或相近的 SID 前缀

**但它做不到的事情是**：
- 告诉 RQ-VAE 哪些 item 应该被区分开（没有推远力）
- 控制 codebook 的利用率和分布均匀性
- 直接影响下游 LLM 的 token 学习难度

**这意味着**：图正则能改善 SID 的"局部一致性"（邻居靠近），但不能直接改善"全局判别性"（非邻居推远）和"下游可学习性"（token 序列好预测）。

当前的 learnability 改善（R401b/R401d 的 b|a、c|ab 回升）更可能来自 warm-start + retention 的正则化效应，而不是来自图正则本身。

**思考**：如果我们想让图信息更深入地影响码本空间的质量（而不只是局部平滑），可能需要考虑：
- 图感知的对比 loss（推远非邻居）
- 图感知的 codebook 初始化（用图社区结构预定义 codebook 布局）
- 图感知的量化决策（在量化时考虑图邻居的分配）

但这些都是**下一个研究阶段**的方向，不是当前应该做的。当前应该先让 R401b/R401d 推到 SFT，看硬性结果。

### 5.2 warm-start 是不是真正的功臣？

一个需要认真考虑的假说：

> R401b/R401d 的结构改善，主要来自 warm-start（从 v2 checkpoint 出发继续训练），而不是来自 teacher-guided retention 或 codebook anchor。

证据：
- R401b 的 retention loss 很轻（γ=0.05），且 pair retention 并没有提高
- R401d 的 codebook anchor 虽然减小了 drift，但也没有改善 pair retention
- 两者都从 v2 checkpoint warm-start，然后继续训练了 ~10000 epochs

这意味着可能的情况是：
1. v2 的 codebook 已经是一个不错的初始化点
2. 从这个点继续训练（无论有没有 retention），都会收敛到一个结构更好的解
3. retention / anchor 的作用主要是作为正则化器，防止训练跑偏

**如何验证这个假说**：跑一个 **R401-control**——从 v2 warm-start，但**不加任何 retention / anchor loss**，只保留原来的 v2 训练目标。如果这个 control 也能达到类似的结构改善，那 retention / anchor 的贡献就需要重新评估。

但**现在不应该跑这个 control**。优先级最高的是推 R401b/R401d 到 SFT evaluate。

### 5.3 我们的层级感知 claim 到底有多强？

核心 claim 是："不同 SID level 应该感知不同粒度的协同结构"。

这个 claim 的实验支撑链是：
1. v1_hierarchy 的 tokenizer 结构比 v1_baseline 好 → ✅ 但 v1_hierarchy 的下游比 v1_baseline 差（0.09360 vs 0.09430），所以**层级感知在 v1 中没有下游价值**
2. v2 在 v1 基础上加了 ambiguity-aware weighting → v2 下游比 v1 好很多 → ✅ 但这个改善可能主要来自 ambiguity weighting，不是来自层级感知本身
3. v2 下游比 original semantic 好 → ✅ 但 original semantic 完全没有图信息，所以这证明的是"图信息有用"，不一定是"层级感知有用"

**诚实地说**：我们目前还没有严格证明"层级感知"比"均匀图正则"在下游更好。v1_hierarchy 的下游表现甚至不如 v1_baseline。v2 的改善可能来自 ambiguity weighting 而非层级分配。

要严格验证层级感知的 claim，需要：
- **v2_uniform**：用 v2 的所有技术（ambiguity weighting, semantic retention, warm-start），但三层都用同一张图（比如都用 G_mid），看是否不如当前的三图三层方案
- 或者更简单的 ablation：在 R401b 的训练中把三张图换成同一张，看结构和下游是否退化

**但同样，这不是当前最高优先级**。当前最高优先级是推 SFT。

---

## 6. 我们现在真正需要的下一步

### 6.1 最高优先级：推 R401b 和 R401d 到 SFT evaluate

理由：
1. 它们的 tokenizer 侧结构指标是所有候选中最强的
2. 它们的 learnability 在 b|a 和 c|ab 上超过了 v2
3. 它们是目前唯一的 `sft_ready_not_run` 候选
4. **只有 evaluate 结果能告诉我们这些码本空间是否真的更好**

具体执行：
- recipe：`title_history2sid_on + desc_align_p05`（与 v2_on_p05 对齐）
- hyperparameters：完全对齐 v2_on_p05 SFT
- 单 seed first（seed=42）
- 如果结果正向且 magnitude 显著（>= 0.003 on NDCG@10），考虑多 seed 验证

### 6.2 根据 SFT 结果的三种场景

**场景 A：R401b 或 R401d 超过 v2_on_p05 SFT**
→ 我们得到了一个新的最强 SFT 候选
→ 下一步：推到 RL，看是否能缩小与 strongest original RL 的 mid-beam gap
→ 同时：如果两者都超过了，取更好的那个

**场景 B：R401b / R401d 与 v2_on_p05 SFT 持平**
→ 结构指标远好、learnability 更高、但下游没有改善
→ 这意味着：tokenizer 侧的结构和可预测性改善**不是**下游 mid-beam gap 的瓶颈
→ 下一步方向应该转向：
  - RL 目标/解码策略（但这不在当前研究范围内）
  - 或者下游 SFT recipe 的进一步探索
  - 或者在论文中把"结构更好但下游持平"定位为"结构改善的必要但不充分条件"

**场景 C：R401b / R401d 低于 v2_on_p05 SFT**
→ 又一次结构好但下游差
→ 需要认真考虑：是否我们的图正则方式从根本上无法产出对 LLM 更友好的码本空间？
→ 可能的方向转向：
  - 图信息改变注入方式（对比 loss / codebook 初始化 / quantization-time 干预）
  - 或者承认：图信息的主要价值在于 top-1/top-3 的改善（v2 已经实现），mid-beam gap 需要其他手段

### 6.3 关于是否需要新的 tokenizer 候选

**现在不需要**。理由：
1. R401b 和 R401d 还没推到 SFT，在不知道结果之前启动新 tokenizer 训练是浪费资源
2. 我们已经有 10 个工业线 tokenizer，7 个有下游 SFT 结果。证据密度已经很高了
3. 如果 R401b/R401d 的 SFT 结果正向，我们有足够的证据推进到 RL 和论文写作
4. 如果 R401b/R401d 的 SFT 结果负向，我们需要的是方向性思考，不是更多 tokenizer 变体

---

## 7. 如果现在要写论文，story 是什么？

### 7.1 已有的 evidence chain

1. **Semantic SID 有局部歧义问题**：同一 l2 prefix 下过多 item 共享，导致 LLM 的 beam search 在叶子级别困难 → 由 v2 vs baseline 的结构诊断支撑
2. **图结构协同信息可以改善这个问题**：三张不同粒度的图分别作用于三层 SID → 由 v2 的 tokenizer 结构改善支撑
3. **但改善需要 ambiguity-aware**：均匀的图正则会破坏已经稳定的语义结构 → 由 v2 vs v1 的对比支撑
4. **v2 的改善能传导到下游**：v2_on_p05 SFT 在 top-1/top-3 上超过 original SFT → 由 evaluate 结果支撑
5. **v2 的改善能存活到 RL**：v2_on_p05 RL 在 NDCG@1 上超过 strongest original RL → 由 RL evaluate 结果支撑
6. **但 v2 在 mid-beam 上仍有 gap**：top-5/top-10/top-20 上落后 → 由 evaluate 结果支撑
7. **进一步的结构改善（R401b/R401d）在 tokenizer 侧非常显著** → 由结构诊断和 learnability probe 支撑
8. **7 的改善是否传导到下游** → **待验证**

### 7.2 如果 R401b/R401d SFT 正向

Story 变成：

> 层级感知的图协同信息注入 + ambiguity-aware 加权 + warm-start refinement = 一个既结构更好又下游更强的 SID 码本空间。

这个 story 很完整：问题定义 → 方法设计 → tokenizer 侧验证 → 下游验证 → ablation（v1 vs v2, R202a vs R401b）。

### 7.3 如果 R401b/R401d SFT 持平或负向

Story 需要调整为：

> 层级感知的图协同信息注入可以显著改善 SID 的局部结构和条件可预测性，但这些 tokenizer 侧的改善不会自动转化为下游 ranking 的提升。结合 v2 的 top-1/top-3 改善和 mid-beam gap，我们认为图信息在 SID 构建中的价值主要体现在精确推荐（head hit）而非多样性检索（beam diversity）。

这个 story 仍然有价值——它是一个 mechanism understanding contribution，而非单纯的 performance improvement contribution。

---

## 8. 长期方向的坦率思考

### 8.1 当前方法的天花板在哪里？

如果 R401b/R401d 仍然无法在 mid-beam 上超过 original SID，那可能意味着：

1. **图正则作为 smoothness constraint 的表达力不够**。它只能拉近邻居，不能推远非邻居，不能控制 codebook 布局。更强的注入方式（对比学习、图感知量化）可能有更高的天花板。

2. **SID 码本空间的质量可能不是 mid-beam gap 的主要瓶颈**。mid-beam 需要的是 beam diversity 而非 top-1 precision。beam diversity 可能更多取决于 LLM 的 token-level generation 策略，而非 tokenizer 的码本布局。

3. **数据集规模限制**。Industrial & Scientific 只有 3686 items。在这个尺度上，图信息的价值可能有天然上限。更大的数据集可能让图信息发挥更大作用。

### 8.2 如果有更多时间，最值得探索的方向

按优先级排序：

1. **推 R401b/R401d 到 SFT**（当前最高优先级，不需要更多时间思考）
2. **v2_uniform ablation**：严格验证层级感知 vs 均匀图正则的差异
3. **图感知对比 loss**：在图正则中加入推远力（非邻居在同一 l2 下应被推开）
4. **更大数据集验证**：在 Office Products（4866 items）上复现当前最强方案
5. **ReSID-style predictability regularizer**：在训练中显式优化 prefix-conditional entropy

---

## 9. 总结

### 9.1 我们做对了什么

1. **问题定义清晰**：从一开始就锁定了"图结构 → SID 层级构建"这个方向
2. **实验-反馈循环紧密**：每个负面结果都及时修正了方向（R202a → interface diagnostics → R401）
3. **诊断工具完善**：R301-R304 + local ambiguity analysis 提供了超越 evaluate 数字的理解
4. **evidence chain 连贯**：从 tokenizer 结构 → SFT evaluate → RL evaluate，逐层验证
5. **方向纠偏及时**：用户正确地指出"prefix stability 不是目标"，避免了进一步的方向偏离

### 9.2 我们做错了什么或可以改进的

1. **过早地把诊断指标当成优化目标**：pair retention、codebook drift 被提升为 hard gate，消耗了设计空间
2. **论文借鉴的深度不够**：ReSID 和 HiD-VAE 的核心思想（predictability regularization、uniqueness loss）没有真正实现，只停留在叙事层面
3. **v1 baseline 的下游对比不够重视**：v1_hierarchy 下游不如 v1_baseline 这个事实暗示层级感知的 claim 需要更严格的验证
4. **warm-start 的贡献没有被隔离**：不清楚 R401b/R401d 的改善有多少来自 warm-start，多少来自 retention/anchor

### 9.3 当下最重要的一件事

**推 R401b 和 R401d 到 SFT evaluate。**

所有其他的思考——论文 story、方法改进、ablation 设计——都应该等这个结果出来之后再做决定。

因为这个结果会告诉我们一个根本性的问题：

> 在 tokenizer 侧的结构改善和 learnability 改善都达到了前所未有的水平之后，下游 LLM 到底能不能从中受益？

如果能，我们的方向是对的，继续深化。
如果不能，我们需要从根本上重新思考"更好的 SID 结构"和"更好的下游推荐"之间的关系。
