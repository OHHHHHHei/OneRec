# Experiment Plan（实验计划）

Status（状态）: `reference（参考）`
Last updated（更新日期）: `2026-04-16`

**Problem（问题）**: 当前 `graph carrier`（图载体）对一批 dense semantic variant families（稠密语义变体家族）存在 blind spot（盲区）。这些物品在语义空间中非常接近，但在当前直接 `item-item graph`（物品-物品图）里往往没有足够强的协同边，于是 tokenizer（分词器）训练拿不到有效的协同监督。

**Method Thesis（方法主张）**: 借鉴 `Seq2Graph`（序列到图增广）的核心思想，但不搬运整套 `SGRec`（推荐器）模块，而是把“跨序列前驱上下文共享”压缩成一张 offline（离线） pure-collaborative（纯协同）的 high-order rescue graph（高阶补盲图），先增强 `G_coarse`（粗图），再从新 coarse（粗图）重建 `G_mid`（中图），同时保持当前 `v2` 的 ambiguity-aware graph supervision（歧义感知图监督）和 semantic-structure retention（语义结构保持）主干不变。

**Date（日期）**: `2026-04-16`

## Why This Plan Exists（为什么需要这个计划）

当前最具体、最可信的问题已经收敛为：

- 一批最差热点来自 `3D filament`（3D 打印耗材）这类 dense semantic variant families（稠密语义变体家族）
- 它们在当前 `coarse / mid / local`（粗图 / 中图 / 局部图）里经常几乎没有直接边
- 但 posterior output analysis（后验输出分析）和 case study（病例分析）已经显示：
  - 它们并不是完全没有协同信息
  - 它们常常共享非常相似的 predecessor context（前驱上下文）
  - 也就是：不同用户在不同序列里，会被相似的历史物品引导到这些不同变体 item（物品）

这说明当前问题更像：

> 现有直接 `item-item` 图只能看见 direct transition / direct co-occurrence（直接转移 / 直接共现），但看不见 cross-sequence predecessor sharing（跨序列前驱共享）形成的 high-order collaboration（高阶协同）。

`Seq2Graph`（序列到图增广）最值得借的部分正是：

- 不把每条序列看成彼此孤立
- 允许一个目标节点吸收“来自其他序列里相关前驱”的协同上下文

但我们当前研究对象是 tokenizer（分词器）前端图载体，而不是下游 sequential recommender（序列推荐器）本身，所以本计划只借它的 graph augmentation principle（图增广原则），不借它的 full recommender head（完整推荐头）。

## Design Principles（设计原则）

1. **图必须保持 pure collaborative（纯协同）**
   - 新图只能使用 train-only behavior log（训练集行为日志）
   - 不允许用 semantic similarity（语义相似度）直接构边
   - semantic information（语义信息）只能保留在 retention（保持）支路和 posterior analysis（后验分析）里

2. **先改 graph carrier（图载体），不先加新 loss（损失）**
   - 当前导师反馈和现有实验都说明：loss（损失）项不能继续膨胀
   - 本轮只改 `graph bank`（图库），不改训练目标主干

3. **补盲，不重写**
   - 新图不是要替换当前所有 direct collaborative structure（直接协同结构）
   - 它只应该补当前直接图看不见、但 high-order context（高阶上下文）明确存在的边

4. **优先改 `G_coarse`（粗图），再重建 `G_mid`（中图）**
   - 当前 `G_mid` 严重依赖 `G_coarse`
   - 如果 coarse（粗图）没看见这些高阶协同，mid（中图）很难凭空补出来

5. **下游 `SFT -> evaluate`（监督微调到评测）仍然是唯一裁决**
   - offline graph summary（离线图摘要）和 tokenizer-side generate check（分词器侧生成检查）只能做 catastrophic failure filter（灾难性失败过滤）
   - 不再把任何 prior diagnostic（前验诊断）当作科学证据或 promotion gate（推进门槛）

## Claim Map（主张映射）

| Claim（主张） | Why It Matters（重要性） | Minimum Convincing Evidence（最小可信证据） | Linked Blocks（关联模块） |
|---|---|---|---|
| C1. 当前 dense variant family（稠密变体家族）上的主要 blind spot（盲区）来自 direct graph（直接图）漏掉了 cross-sequence predecessor sharing（跨序列前驱共享）形成的 high-order collaboration（高阶协同）。 | 这是当前 graph-carrier（图载体）方向最具体、最可执行的问题定义。 | 一个纯协同的 predecessor-sharing graph（前驱共享图）能够显著改变这些热点物品的图可见性，并在 downstream `SFT -> evaluate` 上至少改善 `HR@10 / NDCG@10` 或 candidate retention（候选保留）模式。 | B0, B1, B2, B3 |
| C2. 真正有用的不是简单 densification（稠密化），而是 reliability-aware rescue（可靠性感知补盲）。 | 需要防止 reviewer（审稿人）把结果解释成“只是加了更多边”。 | full variant（完整变体）优于 naive context-only（朴素上下文版），或者至少表现出更健康的 failure pattern（失败模式）。 | B1, B2 |
| Anti-claim（反主张）: gains（增益）只是来自 another noisy graph（另一张噪声图）或 indirect semantic leakage（间接语义泄露）。 | 这是最自然的质疑。 | 图构建严格只用 train behavior（训练行为）；与当前 direct graph（直接图）的差异可解释；naive densification 对照不更好。 | B0, B1, B2 |

## Core Module Design（核心模块设计）

### 1. Base Objects（基础对象）

Let（记）:

- `L_raw` = 当前由 [graph_bank.py](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/graph_bank.py:194) `build_local_graph`（构建局部图）得到的原始历史到目标转移矩阵
- 其中 `L_raw[q, i]` 表示：历史物品 `q` 以某个 recency-aware weight（时近权重）导向目标物品 `i` 的强度

我们把它重新解释为：

- 对于目标物品 `i`，列向量 `L_raw[:, i]` 就是它的 predecessor context signature（前驱上下文签名）

### 2. Seq2Graph-lite Context Graph（Seq2Graph-lite 上下文图）

#### 2.1 Context Signature Normalization（上下文签名归一化）

先对 `L_raw` 的列做 `l2 normalization`（`l2` 归一化），得到：

```text
L_ctx = l2_col_normalize(L_raw)
```

这样不同 popularity（流行度）和频次规模的目标物品可以更公平地比较上下文方向，而不是简单比谁观察次数更多。

#### 2.2 High-Order Context Similarity（高阶上下文相似度）

构造：

```text
H_ctx = L_ctx^T @ L_ctx
```

并把对角线清零。

这一步的含义是：

- 如果两个目标物品 `i, j` 经常被相似的一批前驱物品触发出来
- 那么它们在 `H_ctx[i, j]` 上就会获得较高的 high-order collaborative affinity（高阶协同亲和）

它与当前 direct `item-item` graph（直接物品图）的关键差异在于：

- 当前图看 `q -> i` 或 `i <-> j` 是否直接出现
- `H_ctx` 看的是“`i` 和 `j` 是否共享相似前驱”

这正是 `Seq2Graph`（序列到图增广）里“跨序列相关前驱为目标节点补上下文”的离线 item-level（物品级）版本。

#### 2.3 Reliability Reweighting（可靠性重加权）

为了避免单纯 context similarity（上下文相似度）引入太多噪声边，对 `H_ctx` 的候选边再乘一个 shared-support reliability（共享支撑可靠性）：

```text
r(i, j) = sum_q min(L_raw[q, i], L_raw[q, j]) / (sum_q max(L_raw[q, i], L_raw[q, j]) + eps)
```

这本质上是 predecessor support（前驱支撑）上的 weighted Jaccard（加权 Jaccard）。

设计理由：

- 如果两个物品只是“方向很像”，但实际共享前驱很少，`r(i, j)` 会低
- 如果它们确实反复由相同或相似前驱触发，`r(i, j)` 会高

这一步是 `CAGCN`（协同感知图卷积网络）那类“不是所有邻居都值得相信”的思想，在我们这里的简化移植版。

#### 2.4 Direct-Weak Rescue Mask（直接弱连接补盲掩码）

我们不想用新图重写已经健康的直接边，因此只对 direct graph（直接图）较弱的 pair（物品对）开放补盲：

```text
d(i, j) = coarse_raw[i, j] + coarse_raw[j, i] + local_raw[i, j] + local_raw[j, i]
m(i, j) = 1[d(i, j) < tau_direct]
```

然后定义完整 rescue affinity（补盲亲和）：

```text
G_rescue(i, j) = H_ctx(i, j) * r(i, j) * m(i, j)
```

设计理由：

- 如果 `i, j` 已经有很强 direct support（直接支撑），不需要 `Seq2Graph-lite` 去重写
- 如果 direct graph（直接图）几乎看不见它们，但 predecessor sharing（前驱共享）很强，那它们正是最值得补的 blind spot（盲区）

#### 2.5 Sparsification（稀疏化）

对 `G_rescue` 做：

- row top-`k`（每行前 `k` 保留）
- row normalize（行归一化）
- optional symmetric keep（可选对称保留）

得到：

```text
G_seq2g_rescue
```

### 3. Carrier Integration（图载体整合）

#### 3.1 New Coarse Graph（新粗图）

```text
G_coarse_seq2g = rownorm((1 - alpha) * G_coarse_purified + alpha * G_seq2g_rescue)
```

这是一张 residual rescue coarse graph（残差补盲粗图），不是 full replacement（完全替换）。

#### 3.2 New Mid Graph（新中图）

从 `G_coarse_seq2g` 按当前 `FaGSP-style`（`FaGSP` 风格）流程重建：

```text
G_mid_seq2g = FaGSP(G_coarse_seq2g)
```

#### 3.3 Local Graph（局部图）

`G_local` 暂时保持不变：

```text
G_local = local_purified
```

原因：

- 当前问题更像 coarse / mid routing（粗层 / 中层路由）盲区
- 不要在第一轮同时改 coarse（粗图）和 local（局部图），否则难以归因

### 4. Training Interface（训练接口）

本轮**不改**：

- `ambiguity prior`（歧义先验）
- graph loss（图损失）形式
- semantic retention（语义保持）形式
- selective separation（选择性分离）分支

也就是说，当前 [train_v2.py](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/train_v2.py:430) 里的训练骨架保持不变，只通过新 `view_name`（视图名）替换：

- `coarse_view_name = coarse_seq2g_rescue`
- `mid_view_name = fagsp_mid_seq2g_rescue`
- `local_view_name = local_purified`

## Experiment Blocks（实验模块）

### Block B0: Offline Graph Construction Audit（离线构图审计）

- Claim tested（检验主张）:
  - `Seq2Graph-lite`（轻量 `Seq2Graph`）确实补出了当前 direct graph（直接图）看不见的高阶协同边
- Why this block exists（存在原因）:
  - 这一步只负责确认 carrier（载体）真的变了，且变得合理
  - 它不是 scientific verdict（科学裁决）
- Dataset / split / task（数据 / 划分 / 任务）:
  - `Industrial_and_Scientific`
  - train CSV only（只用训练集）
- Compared systems（对比系统）:
  - `coarse_purified`
  - `coarse_seq2g_ctx_only`
  - `coarse_seq2g_rel`
  - `coarse_seq2g_rel_masked`
- Metrics（指标）:
  - connected rate（连通率）
  - density / avg degree（密度 / 平均度）
  - overlap with baseline coarse（与 baseline 粗图重叠）
  - rescue-edge ratio（补盲边比例）
  - on hotspot families（热点家族上）:
    - semantic-near pairs（语义近邻对）中被新 coarse（粗图）看见的比例
    - predecessor-sharing pairs（前驱共享对）中 direct-zero but rescue-positive（直接为零但补盲为正）的比例
- Success criterion（成功标准）:
  - 结构稳定，不塌缩
  - 能显著提升 dense variant hotspots（稠密变体热点）上的 graph visibility（图可见性）
- Failure interpretation（失败解释）:
  - 如果只是全局加边，但热点 visibility（可见性）不提升，说明方法没有对准 blind spot（盲区）
- Priority（优先级）: `MUST-RUN（必跑）`

### Block B1: Tokenizer Catastrophic-Failure Filter（分词器灾难性失败过滤）

- Claim tested（检验主张）:
  - 新 carrier（载体）至少不会像 `R530a` 那样一开始就明显破坏 tokenizer（分词器）
- Why this block exists（存在原因）:
  - 这里只做 engineering filter（工程过滤），不做科学证据
- Dataset / split / task（数据 / 划分 / 任务）:
  - `Industrial_and_Scientific`
  - tokenizer train + sid-generate（分词器训练 + SID 生成）
- Compared systems（对比系统）:
  - current `v2` tokenizer graph bank
  - `R640a`: context-only rescue coarse（仅上下文补盲粗图）
  - `R640b`: context + reliability（上下文 + 可靠性）
  - `R640c`: context + reliability + direct-weak mask（上下文 + 可靠性 + 直接弱连接掩码）
- Metrics（指标）:
  - generated collision（生成后冲突率）
  - catastrophic prefix collapse（灾难性前缀塌缩）检查
  - basic graph-usage summary（基础图使用摘要）
- Setup details（设置）:
  - freeze all non-carrier choices（冻结所有非载体选择）
  - keep current `v2` loss weights（损失权重） and ambiguity settings（歧义设置）
- Success criterion（成功标准）:
  - candidate（候选）没有出现明显灾难性退化
- Failure interpretation（失败解释）:
  - 如果最简单版本都显著破坏 tokenizer（分词器），说明 carrier（载体）写法还不对，不要推进下游
- Priority（优先级）: `MUST-RUN（必跑）`

### Block B2: Main Downstream Verdict（主下游裁决）

- Claim tested（检验主张）:
  - `Seq2Graph-lite` 高阶协同补盲能够改善 current `v2` 的 beam retention（候选束保留）/ family recall（家族召回）短板
- Why this block exists（存在原因）:
  - 最终只有 downstream `SFT -> evaluate`（下游 `SFT -> evaluate`）能回答这个问题
- Dataset / split / task（数据 / 划分 / 任务）:
  - `Industrial_and_Scientific`
  - `title_history2sid_on + desc_align_p05`
- Compared systems（对比系统）:
  - current `v2_on_p05`
  - strongest original SFT（原版最强 `SFT`）作为外部参考
  - only one promoted `R640*` candidate（只推进一个 `R640*` 候选）
- Metrics（指标）:
  - `NDCG@1/3/5/10`
  - `HR@1/3/5/10`
  - pairwise output diagnosis（成对输出诊断）:
    - target retention from `top10 -> 11-50`（目标从 `top10` 掉到 `11-50`）
    - family-level hit pattern（家族级命中模式）
- Success criterion（成功标准）:
  - 至少 match（追平）或超过 current `v2_on_p05` 的 `NDCG@10 / HR@10`
  - 或者显著改善 beam retention（候选束保留）模式，值得进一步推进
- Failure interpretation（失败解释）:
  - 如果 graph visibility（图可见性）变好，但 downstream（下游）仍不改善，说明 carrier blind spot（载体盲区）不是唯一瓶颈
- Priority（优先级）: `MUST-RUN（必跑）`

### Block B3: Focused Posterior Analysis on Variant Families（面向变体家族的聚焦后验分析）

- Claim tested（检验主张）:
  - 如果方法有效，改善应优先出现在 dense variant families（稠密变体家族）上，而不是随机散落
- Why this block exists（存在原因）:
  - 这是当前问题定义是否真的被命中的最好解释性证据
- Dataset / split / task（数据 / 划分 / 任务）:
  - promoted candidate（已推进候选） vs `v2_on_p05`
- Metrics（指标）:
  - family-weighted `top10` gain / loss（家族加权 `top10` 增益 / 损失）
  - hotspot item recovery（热点物品恢复）
  - target-side l2 fanout / neighborhood retention（目标侧 `l2` 分叉 / 邻域保留）
- Success criterion（成功标准）:
  - 改善集中在当前最差的 dense variant hotspots（稠密变体热点）上
- Failure interpretation（失败解释）:
  - 如果 gain（增益）很分散，说明方法虽然可能有效，但未必真解决了我们当前定义的问题
- Priority（优先级）: `MUST-RUN（必跑）`

### Block B4: Optional Dynamic Seq2Graph Variant（可选动态 `Seq2Graph` 变体）

- Claim tested（检验主张）:
  - `Seq2Graph` 风格的 stochastic neighbor sampling（随机邻居采样）是否能进一步减少噪声边副作用
- Why this block exists（存在原因）:
  - 这是最接近原论文动态增广精神的附加实验
  - 但当前不是主线所必需
- Compared systems（对比系统）:
  - best static `R640*`
  - `R641`: epoch-wise sampled rescue graph（按 epoch 采样的补盲图）
- Priority（优先级）: `NICE-TO-HAVE（可选）`

## Proposed Runs（建议运行）

### D640: `Seq2Graph-lite` Graph Audit（图审计）

- Purpose（目的）:
  - 导出 `H_ctx` / reliability（可靠性） / masked rescue（掩码补盲）三步变化
  - 看热点家族的 visibility（可见性）是否真的提升

### R640a: Context-Only Residual Rescue（仅上下文残差补盲）

- Definition（定义）:

```text
G_rescue = topk(rownorm(H_ctx))
G_coarse_new = rownorm((1 - alpha) * coarse_purified + alpha * G_rescue)
G_mid_new = FaGSP(G_coarse_new)
```

- Role（角色）:
  - 最朴素的 `Seq2Graph-lite` 基线
  - 回答“光靠共享前驱上下文值不值得”

### R640b: Reliability-Aware Rescue（可靠性感知补盲）

- Definition（定义）:

```text
G_rescue = topk(rownorm(H_ctx * R))
```

- Role（角色）:
  - 检验“邻居可靠性”是否必要

### R640c: Full Rescue with Direct-Weak Mask（带直接弱连接掩码的完整补盲）

- Definition（定义）:

```text
G_rescue = topk(rownorm(H_ctx * R * M))
```

- Role（角色）:
  - 当前最完整、最贴问题定义的正式方案
  - 理论上最有希望修 blind spot（盲区），同时不过度重写已有健康 direct edges（直接边）

### R645: Best Candidate -> `SFT -> evaluate`（最佳候选推进到 `SFT -> evaluate`）

- Recipe（配方）:
  - `title_history2sid_on + desc_align_p05`
- Decision rule（决策规则）:
  - 不用 prior diagnostic（前验诊断）排排序
  - 只排除 catastrophic failure（灾难性失败）
  - 在非灾难性候选中，优先推进 `R640c`

### R646: Optional RL Promotion（可选 `RL` 推进）

- Only if（仅当）:
  - `R645` 至少 match（追平）或超过 `v2_on_p05` 的 `SFT` 表现

## Run Order and Milestones（执行顺序与里程碑）

| Milestone（里程碑） | Goal（目标） | Runs（运行） | Decision Gate（决策门） | Cost（成本） | Risk（风险） |
|---|---|---|---|---|---|
| M0 | carrier sanity（载体合理性） | `D640` | 图不塌缩，热点 visibility（可见性）提升 | CPU / low（低） | 误把全局加边当成有意义补盲 |
| M1 | static tokenizer screen（静态分词器筛选） | `R640a`, `R640b`, `R640c` | 只过滤 catastrophic failure（灾难性失败） | 3 tokenizer runs（3 次分词器运行） | naive context graph（朴素上下文图）可能太噪 |
| M2 | main downstream verdict（主下游裁决） | `R645` | 是否能 match / beat `v2_on_p05` | 1 `SFT -> evaluate` chain（1 条 `SFT -> evaluate` 链） | tokenizer-side change（分词器侧变化）不一定能存活到下游 |
| M3 | optional dynamic appendix（可选动态附录） | `R641` | 只在 static 版有正信号时运行 | 1 tokenizer run（1 次分词器运行） | 提高复杂度但不一定有收益 |
| M4 | optional RL（可选 `RL`） | `R646` | 只有 `R645` 为正才推进 | 1 `RL` chain（1 条 `RL` 链） | 不必要地消耗算力 |

## Compute and Data Budget（算力与数据预算）

- Total estimated GPU-hours（总 GPU 小时）:
  - `D640`: CPU-only（纯 CPU）
  - `R640a/b/c`: `3` tokenizer runs（`3` 次分词器运行）
  - `R645`: `1` downstream `SFT -> evaluate`（下游 `SFT -> evaluate`）
  - optional `R641 / R646`: `1-2` extra runs（额外 `1-2` 次运行）
- Data preparation needs（数据准备需求）:
  - `L_raw` column-normalization（列归一化）
  - top-`M` candidate extraction（前 `M` 候选提取）
  - reliability computation（可靠性计算）
  - direct-strength mask（直接强度掩码）
- Biggest bottleneck（最大瓶颈）:
  - 需要在不引入语义泄露的前提下，把 predecessor-sharing（前驱共享）写成稳定、稀疏、可训练的 coarse graph（粗图）

## Risks and Mitigations（风险与缓解）

- Risk（风险）: `H_ctx` 只是另一种 noisy densification（噪声稠密化）。
  - Mitigation（缓解）:
    - 设置 `R640a` 作为 naive baseline（朴素基线）
    - 用 `R640b / R640c` 显式检验 reliability（可靠性）与 direct-weak mask（直接弱连接掩码）的必要性

- Risk（风险）: 新 coarse（粗图）改动过大，破坏现有稳定 routing（路由）。
  - Mitigation（缓解）:
    - 使用 residual mixing（残差混合），不做 pure replacement（纯替换）
    - `alpha` 从小值开始

- Risk（风险）: 只改 coarse（粗图）仍不足以改善 downstream retention（下游保留）。
  - Mitigation（缓解）:
    - 保持 `B3` 的 posterior family analysis（后验家族分析）
    - 如果 `R645` 为负，再决定是否需要把 high-order signal（高阶信号）只接到 `G_mid`（中图）作为 appendix（附录）分支

- Risk（风险）: 重新走回 diagnostics-first（诊断优先）老路。
  - Mitigation（缓解）:
    - 明确规定 `B0 / B1` 仅做 engineering filter（工程过滤）
    - scientific verdict（科学裁决）只能来自 `B2`

## Implementation Mapping（实现映射）

建议新增：

- in [graph_bank.py](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/graph_bank.py)
  - `build_seq2graph_context_matrix(...)`
  - `build_seq2graph_reliability(...)`
  - `build_seq2graph_rescue_graph(...)`

- in [transplanted_graph_bank.py](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/transplanted_graph_bank.py)
  - register:
    - `coarse_seq2g_ctx_only`
    - `coarse_seq2g_rel`
    - `coarse_seq2g_rel_masked`
    - `fagsp_mid_seq2g_ctx_only`
    - `fagsp_mid_seq2g_rel`
    - `fagsp_mid_seq2g_rel_masked`

建议新增配置项：

- `seq2g_mix_alpha`
- `seq2g_context_topk`
- `seq2g_candidate_topm`
- `seq2g_direct_tau`
- `seq2g_use_reliability`
- `seq2g_use_direct_weak_mask`

## Final Checklist（最终检查表）

- [ ] Graph remains pure collaborative（图保持纯协同）
- [ ] No new training loss is introduced（不引入新训练损失）
- [ ] `G_coarse` changes first, then `G_mid` is rebuilt（先改 `G_coarse`，再重建 `G_mid`）
- [ ] `R640a / R640b / R640c` isolate the key mechanism（隔离关键机制）
- [ ] Downstream `SFT -> evaluate` remains the only scientific verdict（下游 `SFT -> evaluate` 仍是唯一科学裁决）
