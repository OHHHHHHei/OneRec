# Experiment Plan（实验计划）

Status（状态）: `reference（参考）`
Last updated（更新日期）: `2026-04-18`

**Problem（问题）**: 当前 `MGR-SID`（图监督 `SID`）的主损失接口仍然以 attraction-only graph smoothness（仅吸引式图平滑）为主。它能表达“哪些物品应该更近”，但不能直接表达 `semantic-close but collaboratively inconsistent`（语义接近但协同不一致）物品应该被显式分开。`R670a` 还进一步说明：如果把 `L1`（第一层）改成过强的语义凝聚，再叠加 `stop-gradient prefix`（前缀停梯度），前缀空间可能直接塌缩。

**Method Thesis（方法主张）**: 下一步最干净的验证，不是继续堆新的 graph carrier（图载体）或继续增强 `L1` 语义锚定，而是回到一个更简洁的层级分工：

- `L1`（第一层）保留 `coarse graph smoothness`（粗图平滑）
- `L2`（第二层）改成 `multi-hop transition`（多跳转移）上的 pairwise contrastive pull-push（成对对比式拉近 / 推远）
- `L3`（第三层）保留 `local graph smoothness`（局部图平滑）
- 打开 `stop-gradient`（停止梯度），让每层图损失只优化本层向量，而不无保护地改写前缀

**Date（日期）**: `2026-04-18`

## Why This Plan Exists（为什么需要这个计划）

这份计划对应一个很具体的收敛判断：

1. `R530a` 说明：把 `local_multihop`（局部多跳图）直接放到 `L3`（第三层）并不能带来好结果；因此多跳结构本身不是“自动正信号”。
2. `R670a` 说明：`L1` 语义高置信凝聚 + `L2` push-pull（推远拉近）+ `stop-gradient`（停止梯度）在当前形态下会把前缀空间压扁；因此下一步不能继续沿“更强 `L1` 语义入口”方向推进。
3. 当前最有价值的新假设是：问题不一定只在 graph carrier（图载体）本身，也可能在 `L2`（第二层）监督接口。当前 `L2` smoothness（平滑）只会把 graph neighbors（图邻居）做平均拉近，不会直接学习“该推远谁”。

因此，下一步应该做一次最小而明确的接口替换：

> 保持 `L1/L3`（第一层 / 第三层）仍用 smoothness（平滑）稳定层级结构，只把 `L2`（第二层）改成更直接表达协同判别性的 contrastive-style loss（对比式损失）。

## Design Principles（设计原则）

1. **只替换一个关键接口**
   - 不同时重写三层损失
   - 不同时再换新的 `G_coarse`（粗图）或 `G_local`（局部图）来源
   - 本轮唯一新接口是 `L2 contrastive`（第二层对比式损失）

2. **每层只做一件事**
   - `L1`：coarse grouping（粗粒度分组）
   - `L2`：collaborative discrimination（协同判别）
   - `L3`：local refinement（局部细化）

3. **停止梯度不是“断掉前缀语义”，而是“只让本层吃梯度”**
   - `L2` 仍然看到 `L1` 前缀位置
   - 但 `L2` loss（损失）不再直接反传改写 `q1`

4. **多跳图只放到 `L2`，不再放 `L3`**
   - `R530a` 已经说明：`local_multihop`（局部多跳图）直接替换 `L3` 是明确负结果
   - 本轮改法是：`L2 <- local_multihop`（第二层 <- 局部多跳图），`L3 <- local_purified`（第三层 <- 净化局部图）

5. **不把前验 diagnostics（前验诊断）当作推进门槛**
   - tokenizer-side（分词器侧）只允许用 catastrophic failure filter（灾难性失败过滤）排除明显坏实验
   - 真正裁决仍然是 `SFT -> evaluate`（监督微调到评测）

## Core Hypothesis（核心假设）

当前最值得验证的假设是：

\[
\text{`L2`（第二层）的问题，不只是 } G_{\mathrm{mid}} \text{（中图）来源不够好，}
\]

\[
\text{也可能是当前 } \mathcal{L}_{\mathrm{mid}}^{\mathrm{smooth}} \text{（第二层平滑损失）接口过于“只拉不推”。}
\]

因此，如果我们改成：

\[
L1 \text{ smooth} + L2 \text{ contrastive} + L3 \text{ smooth}
\]

并同时使用：

\[
G_{\mathrm{mid}} = \texttt{local\_multihop}
\]

\[
\texttt{hierarchy\_stopgrad\_previous\_levels} = \texttt{true}
\]

那么我们有机会得到一个更符合层级分工直觉的 tokenizer（分词器）空间：

- `L1`（第一层）不被 `L2` 辅助目标拉散
- `L2`（第二层）可以显式学会“该靠近谁、该远离谁”
- `L3`（第三层）继续承接局部转移细化，而不是承担判别主力

## Method Definition（方法定义）

### 1. Layer Representations（层表示）

令三层残差量化向量为：

\[
q^{(1)}, q^{(2)}, q^{(3)}
\]

本计划中的监督表示定义为：

\[
\hat H^{(1)} = q^{(1)}
\]

\[
\hat H^{(2)} = \operatorname{sg}\!\left(q^{(1)}\right) + q^{(2)}
\]

\[
\hat H^{(3)} = \operatorname{sg}\!\left(q^{(1)} + q^{(2)}\right) + q^{(3)}
\]

其中：

\[
\operatorname{sg}(\cdot)
\]

表示 `stop-gradient`（停止梯度）。

### 2. Graph Assignment（图分配）

本计划使用：

\[
G_{\mathrm{coarse}} = \texttt{coarse\_purified}
\]

\[
G_{\mathrm{mid}} = \texttt{local\_multihop}
\]

\[
G_{\mathrm{local}} = \texttt{local\_purified}
\]

其中 `local_multihop`（局部多跳图）定义为：

\[
A = \operatorname{RowNorm}\!\left(G_{\mathrm{local}}\right)
\]

\[
G_{\mathrm{mid}}
=
\operatorname{RowNorm}
\left(
A + \alpha A^2 + \alpha^2 A^3 + \cdots + \alpha^{K-1} A^K
\right)
\]

本轮主实验先使用最小稳定版本：

\[
G_{\mathrm{mid}}
=
\operatorname{RowNorm}\left(A + \alpha A^2\right)
\]

并设置：

\[
\alpha = 0.35,\quad K = 2
\]

### 3. Loss Interface（损失接口）

#### 3.1 `L1` Smoothness（第一层平滑）

\[
\mathcal L_{1,\mathrm{smooth}}
=
\frac{
\sum_i a_i
\left\|
\hat h_i^{(1)} - \sum_j G^{\mathrm{coarse}}_{ij}\hat h_j^{(1)}
\right\|_2^2
}{
\sum_i a_i
}
\]

#### 3.2 `L2` Contrastive Pull（第二层对比式拉近）

从 `local_multihop`（局部多跳图）中抽取正样本集合：

\[
P_2 = \{(i,j) \mid G^{\mathrm{mid}}_{ij} \text{ is strong}\}
\]

定义：

\[
\mathcal L_{2,\mathrm{pull}}
=
\frac{1}{|P_2|}
\sum_{(i,j)\in P_2}
w_{ij}
\left(
1 - \cos\left(\hat h_i^{(2)}, \hat h_j^{(2)}\right)
\right)
\]

其中：

\[
w_{ij} = G^{\mathrm{mid}}_{ij}
\]

或其 top-k（前 k）归一化权重。

#### 3.3 `L2` Contrastive Push（第二层对比式推远）

负样本集合定义为：

\[
N_2 = \{(i,k) \mid \text{semantic-near and multi-hop-weak}\}
\]

也就是：

- 语义相似度高
- 但在 `local_multihop`（局部多跳图）里连接弱或接近零

定义：

\[
\mathcal L_{2,\mathrm{push}}
=
\frac{1}{|N_2|}
\sum_{(i,k)\in N_2}
u_{ik}
\left[
\cos\left(\hat h_i^{(2)}, \hat h_k^{(2)}\right) - m_2
\right]_+^2
\]

其中：

\[
m_2 \in (0,1)
\]

是 margin（间隔），

\[
u_{ik}
\]

是 pair reliability（物品对可靠性）或统一权重。

#### 3.4 `L3` Smoothness（第三层平滑）

\[
\mathcal L_{3,\mathrm{smooth}}
=
\frac{
\sum_i a_i
\left\|
\hat h_i^{(3)} - \sum_j G^{\mathrm{local}}_{ij}\hat h_j^{(3)}
\right\|_2^2
}{
\sum_i a_i
}
\]

### 4. Total Objective（总目标）

\[
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{recon}}
+
\mathcal L_{\mathrm{rq}}
+
\lambda_1 \mathcal L_{1,\mathrm{smooth}}
+
\lambda_2^{+}\mathcal L_{2,\mathrm{pull}}
+
\lambda_2^{-}\mathcal L_{2,\mathrm{push}}
+
\lambda_3 \mathcal L_{3,\mathrm{smooth}}
\]

本轮 primary run（主运行）建议初始权重：

\[
\lambda_1 = 0.05
\]

\[
\lambda_2^{+} = 0.15
\]

\[
\lambda_2^{-} = 0.01
\]

\[
\lambda_3 = 0.05
\]

并关闭：

\[
\texttt{semantic\_coarse\_weight} = 0,\quad
\texttt{semantic\_mid\_weight} = 0
\]

理由是：第一轮先做 clean attribution（干净归因），避免再把结果解释成“其实还是 semantic retention（语义保持）起作用”。

## Pair Construction（物品对构造）

### Positive Pairs for `L2`（第二层正样本）

正样本来自 `local_multihop`（局部多跳图）：

1. 取每个 item（物品）在 `local_multihop` 中的 top-k neighbors（前 k 邻居）
2. 只保留：

\[
G^{\mathrm{mid}}_{ij} > \tau_{\mathrm{pos}}
\]

的边

3. 用图权重或其归一化结果作为：

\[
w_{ij}
\]

建议首轮：

- `topk = 32`
- `tau_pos`（正边阈值）使用行内非零边的中位数或固定较小阈值

### Negative Pairs for `L2`（第二层负样本）

负样本来自：

\[
\text{semantic-near} \cap \text{multi-hop-weak}
\]

即：

1. 在语义图中取 top semantic neighbors（前语义邻居）
2. 过滤出：

\[
G^{\mathrm{mid}}_{ik} \le \tau_{\mathrm{weak}}
\]

的 pair（物品对）

3. 保留可靠性最高的一部分 pair（物品对）

建议首轮：

- semantic top-k（语义前 k）继续沿用现有 `82596` 级别的全量候选导出逻辑
- `tau_{\mathrm{weak}}` 使用 semantic-near 候选对上的 `G_mid`（中图）低分位点
- 不新增复杂 reliability model（可靠性模型）；首轮直接用现有 `pair reliability`（物品对可靠性）导出方式或统一权重

## Planned Run Matrix（计划运行矩阵）

### `R680a` Primary Run（主运行）

目标：

- 精确验证用户刚刚确认的主假设

设置：

- `L1 smooth + L2 contrastive + L3 smooth`
- `G_mid <- local_multihop`
- `hierarchy_stopgrad_previous_levels = true`
- 不加 semantic retention（语义保持）
- 不改 `G_coarse / G_local`（粗图 / 局部图）来源

配置差分（相对 stable `v2`）：

- `coarse_view_name = coarse_purified`
- `mid_view_name = local_multihop`
- `local_view_name = local_purified`
- `hierarchy_stopgrad_previous_levels = true`
- `coarse_weight = 0.05`
- `local_weight = 0.05`
- `semantic_coarse_weight = 0.0`
- `semantic_mid_weight = 0.0`
- 新增：
  - `l2_contrastive_pull_weight = 0.15`
  - `l2_contrastive_push_weight = 0.01`
  - `l2_contrastive_margin = 0.15`
  - `l2_positive_view_name = local_multihop`
  - `l2_negative_rule = semantic_near_multihop_weak`

### `R680b` Fallback Run（回退运行，仅在 `R680a` 明显塌缩时启用）

目标：

- 判断失败是否来自“完全去掉 semantic anchor（语义锚定）”而不是来自 `L2 contrastive`（第二层对比式损失）本身

相对 `R680a` 的唯一区别：

- `semantic_coarse_weight = 0.025`

这不是默认并行主实验；只有当 `R680a` 出现明显 prefix collapse（前缀塌缩）时才启用。

## Success / Failure Criteria（成功 / 失败标准）

### Tokenizer Catastrophic Filter（分词器灾难性过滤）

允许使用，但**仅用于排除明显坏实验**，不作为科学结论：

- generated collision（生成后冲突）是否灾难性爆炸
- active L1（活跃第一层码）是否塌到异常低
- unique L2 pairs（唯一第二层前缀数）是否异常低

如果出现类似 `R670a`：

- `active L1 << 100`
- `unique L2 pairs << 1000`

则判为 prefix collapse（前缀塌缩），不推进下游。

### Downstream Decision Rule（下游决策规则）

如果 `R680a` tokenizer/generate（分词器训练与生成）非灾难性，则**直接推进**：

\[
\texttt{title\_history2sid\_on + desc\_align\_p05}
\]

并以 `SFT -> evaluate`（监督微调到评测）作为唯一有效裁决。

主要观察：

- `NDCG@1/3/5/10`
- `HR@1/3/5/10`
- 相对 `R650a / R660a / v2_on_p05` 的差异模式

## Risks（风险）

1. **False negatives（伪负样本）**
   - `semantic-near + multihop-weak`（语义近 + 多跳弱连接）里仍可能有真实协同近邻
   - 因此首轮 push（推远）权重必须保持小

2. **Pair sparsity（物品对稀疏）**
   - 如果 `local_multihop`（局部多跳图）太稀，`L2 pull`（第二层拉近）会退化成弱监督

3. **Stop-gradient under-coupling（停止梯度导致层间脱耦过度）**
   - `R670a` 说明 stop-gradient（停止梯度）不是天然安全的
   - 但本轮比 `R670a` 更稳的一点是：`L1` 回到 `coarse smoothness`（粗图平滑），不再使用强 `L1` 语义凝聚

4. **Plan degenerates to generic contrastive learning（计划退化成通用对比学习）**
   - 必须坚持：
     - 只改 `L2`
     - 正样本来自协同 `multi-hop graph`（多跳图）
     - 负样本来自 `semantic-near but graph-weak`（语义近但图弱）
   - 不把它写成通用 representation learning（表征学习）故事

## Implementation Scope（实现范围）

本计划需要新增或修改的仅包括：

1. `L2 contrastive pull`（第二层对比式拉近）实现
2. `mid weak pair source`（中图弱连接物品对）导出逻辑切到 `local_multihop`
3. config（配置）支持：
   - `mid_smoothness_mode = contrastive` 或等价开关
   - `l2_contrastive_*` 参数
4. `CURRENT_STATE.md`（当前状态文档）同步下一步决策

不在本计划范围内：

- 再改 `G_coarse`（粗图）
- 再改 `L3`（第三层）为 contrastive（对比式）
- 再引入完整 `InfoNCE`（信息噪声对比估计）或 memory bank（记忆库）
- 再做新的 graph-carrier 大搜索

## Run Order（执行顺序）

1. 实现 `R680a` 所需代码和配置
2. 导出 `local_multihop`（局部多跳图）对应的 `L2` 正 / 负 pair（物品对）
3. 运行 `R680a tokenizer -> sid-generate`（分词器训练到 SID 生成）
4. 只做 catastrophic filter（灾难性过滤）
5. 如果非灾难性，直接推进 `SFT -> evaluate`（监督微调到评测）
6. 只有当 `R680a` 明显塌缩时，才启用 `R680b`

## Expected Decision Value（预期决策价值）

这份计划的价值不在于“保证下一条一定赢”，而在于它能非常干净地回答下面这个问题：

\[
\text{当前瓶颈，究竟主要在 } G_{\mathrm{mid}} \text{（中图）本身，还是在 } L2 \text{（第二层）监督接口？}
\]

如果 `R680a` 失败，我们能得到更清楚的负结论：

- 不是所有 `stopgrad + push-pull`（停止梯度 + 推远拉近）都值得继续
- 也不是 `multi-hop`（多跳）一放到 `L2` 就会变好

如果 `R680a` 成功，则它会成为当前最有信息量的一条 clean interface test（干净接口测试）主线。
