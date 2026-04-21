# MGR-SID Research Synthesis Report（研究综合报告）

Status（状态）: `snapshot（阶段快照）`

Date（日期）: `2026-04-21`

Last Synced（最近同步）: `2026-04-21`（含当日代码核查与诊断结论）

Scope（范围）: Industrial_and_Scientific（工业与科学用品）上围绕 MiniOneRec SID tokenizer（语义标识分词器）的 graph collaborative injection（图协同注入）研究阶段总结。

Primary Source（主要来源）:

- `research-progress-log/CURRENT_STATE.md`
- `research-progress-log/experiment_registry/tokenizer_registry.csv`
- `research-progress-log/experiment_registry/sft_registry.csv`
- `research-progress-log/experiment_registry/rl_registry.csv`
- `research-progress-log/experiment_registry/downstream_scoreboard.csv`
- 最近一轮 `original_l2/l3 ambiguity-aware`（原版第二/第三层歧义感知）实验 README（阶段说明）

## 1. Executive Summary（执行摘要）

本阶段研究围绕一个核心问题展开：原版 MiniOneRec 的 SID（Semantic ID，语义标识）tokenizer（分词器）主要由 semantic embedding（语义嵌入）驱动，缺乏 collaborative signal（协同信号），因此容易把“语义相近但协同关系不同”的 item（物品）放得太近。我们的目标是构造一个更好的 SID codebook space（语义标识码本空间），让 downstream SFT（下游监督微调）和 RL（强化学习）更容易在 `@1/@3/@5/@10`（主要截断）上排出正确推荐。

早期 `v1/v2`（第一/第二版）证明了 graph collaborative signal（图协同信号）确实可以影响 SID tokenizer（语义标识分词器），其中 `v2` 的 ambiguity-aware graph supervision（歧义感知图监督）与 semantic-structure retention（语义结构保持）是当前最强正证据。更严格地说，这些结果支持“图协同信号具备可测影响和潜在价值”，但尚未严格证明它能相对 strongest original MiniOneRec（最强原版 MiniOneRec）带来端到端净收益。`v2_on_p05 -> RL` 的 `NDCG@10 = 0.10431921`，超过 strongest original SFT（最强原版监督微调）的 `0.10372025`，但仍低于 strongest original RL（最强原版强化学习）的 `0.10726345`。

后续大量实验进一步说明：直接在 RQ-VAE（Residual Quantized Variational Autoencoder，残差量化变分自编码器）内部继续叠加 graph smoothness（图平滑）、pull-push（拉推）、contrastive loss（对比损失）、ranking loss（排序损失）或更复杂 graph carrier（图载体），并没有稳定转化成 `@1/@3/@5/@10`（主要截断）收益。很多 tokenizer-side proxy（分词器侧代理指标）变好时，downstream ranking（下游排序）反而变差。

当前最重要的经验结论是：collaborative signal（协同信号）有价值，但 graph loss（图损失）进入 SID tokenizer（语义标识分词器）的接口必须非常克制。强行改变原版 semantic routing（语义路由）会伤害 LLM（Large Language Model，大语言模型）的 learnability（可学习性）和 top-k calibration（前 k 排序校准）。

同步到当日最新口径后，需要区分两个层次：`v2_on_p05 -> RL` 仍是 strongest validated line（最强已验证线，作为参考基准），但 active execution line（当前执行线）已经收敛到 diagnostic-driven minimal-edit collaborative injection（诊断驱动的最小编辑协同注入）检查，不再继续横向扩展大规模 graph-loss（图损失）组合。

## 2. Baseline Policy（基线口径）

主 baseline（主基线）必须是 strongest original MiniOneRec（最强原版 MiniOneRec），而不是 recipe-aligned original（配方对齐原版）或 internal control（内部对照）。

| Baseline（基线） | Tokenizer（分词器） | Recipe（配方） | Stage（阶段） | NDCG@10 | HR@10 |
|---|---|---|---|---:|---:|
| strongest original SFT（最强原版监督微调） | original semantic（原版语义） | `title_history2sid_off + desc_align_p05` | SFT | `0.10372025` | `0.15089345` |
| strongest original RL（最强原版强化学习） | original semantic（原版语义） | `title_history2sid_off + desc_align_p05` | RL | `0.10726345` | `0.15133466` |
| recipe-aligned original（配方对齐原版） | original semantic（原版语义） | `title_history2sid_on + desc_align_p05` | SFT | `0.10182815` | `0.14626075` |
| internal control（内部对照） | `mgr_upstream_baseline / mgr_upstream_hierarchy` | legacy（历史链路） | SFT | not main baseline（非主基线） | not main baseline（非主基线） |

结论口径：

- 与 strongest original SFT/RL（最强原版监督微调/强化学习）相比，才是最终有效性判断。
- recipe-aligned original（配方对齐原版）只能用于 recipe isolation（配方隔离）或机制分析。
- internal control（内部对照）只能用于早期链路检查，不能作为论文主 baseline（主基线）。

## 3. Core Motivation（核心动机）

原版 MiniOneRec 的 SID tokenizer（语义标识分词器）本质上是 semantic-first（语义优先）的 RQ-VAE（残差量化变分自编码器）：item title / description（物品标题 / 描述）经过 embedding（嵌入）后进入 encoder（编码器），再通过 residual quantization（残差量化）得到三层 SID（语义标识）。

这个设计有一个关键缺口：

> 语义相近不等于协同相近。

典型问题是：两个商品在 title / description（标题 / 描述）里非常相似，但购买人群、使用场景或用户序列上下文完全不同。纯 semantic tokenizer（语义分词器）容易把它们放进相近 SID prefix（语义标识前缀），导致 LLM（大语言模型）在生成推荐时难以区分。

因此，我们希望 graph（图）承载 collaborative signal（协同信号），让 SID codebook space（语义标识码本空间）不仅保留 semantic structure（语义结构），还显式反映 user-item behavior（用户-物品行为）中的协同结构。

## 4. Research Idea（研究构想）

最初构想是 hierarchy-aware collaborative tokenizer（层级感知协同分词器）：

| SID Level（语义标识层级） | Intended Role（预期角色） | Graph Signal（图信号） |
|---|---|---|
| L1（第一层） | coarse routing（粗粒度路由） | coarse collaborative community（粗粒度协同社区） |
| L2（第二层） | collaborative branching（协同分叉） | mid-level collaborative relation（中层协同关系） |
| L3（第三层） | local refinement（局部细化） | local collaborative neighbor（局部协同邻居） |

随着实验推进，核心想法被进一步收敛为：

> 在语义相近的候选中，collaborative-positive item（协同正样本）应该比 semantic-near but collaborative-weak negative（语义近但协同弱负样本）更接近。

对应的 L2 ranking objective（第二层排序目标）是：

$$
s_{ip}^{(2)}
\ge
s_{in}^{(2)}
+
m,
$$

其中 \(p\) 是 collaborative-positive item（协同正样本），\(n\) 是 semantic-near collaborative-weak hard negative（语义近但协同弱困难负样本），\(m\) 是 margin（间隔）。

这个问题定义是当前最有价值的理论收敛：我们不再只是做 graph smoothness（图平滑），而是尝试解决 semantic-collaborative mismatch（语义-协同错配）。

## 5. Experiment Inventory（实验盘点）

截至本报告，split registry（分表总账）中已有：

| Registry（总账） | Count（数量） | Meaning（含义） |
|---|---:|---|
| tokenizer registry（分词器总账） | `42` | SID tokenizer（语义标识分词器）生成与结构结果 |
| SFT registry（监督微调总账） | `31` | SFT/evaluate（监督微调/评测）结果 |
| RL registry（强化学习总账） | `9` | RL/evaluate（强化学习/评测）结果 |
| downstream scoreboard（下游榜单） | `40` | 下游结果快速比较 |

主要实验家族如下：

| Family（实验家族） | Goal（目标） | Current Reading（当前解读） |
|---|---|---|
| original semantic（原版语义） | MiniOneRec baseline（基线） | 仍是最强主 baseline（主基线） |
| v1 upstream（第一版上游） | 验证 graph hierarchy（图层级）方向可行 | 支持 graph signal（图信号）能改变 SID 结构（语义标识结构），但不是主 baseline（主基线） |
| v2 ambiguity-aware（第二版歧义感知） | selective graph supervision（选择性图监督） + semantic retention（语义保持） | 当前最强正证据 |
| stage-2 retention（第二阶段保持） | 修正语义保持 / 前缀保持 | tokenizer proxy（分词器代理）好看，但下游未赢 |
| stage-3 prefix search（第三阶段前缀搜索） | 搜索 codebook space（码本空间） | 结构指标更好不等于下游更好 |
| graph carrier（图载体） | TAGCF / FaGSP / MGDCF / Seq2Graph 等图构建 | 大多未带来稳定下游收益 |
| push-pull / contrastive（拉推 / 对比） | 拉近协同正样本、推开协同弱负样本 | 理论贴合，但实现容易压坏路由 |
| collab-ranking R720（协同排序 R720） | L2 ranking（第二层排序）主线 | `local_multihop`（局部多跳）优于 `fagsp_mid_base`，但没赢主 baseline |
| minimal-edit original（原版最小编辑） | 原版 RQ-VAE 上只动 L2/L3 | 比大框架更稳，但仍未超过 strongest original |
| ambiguity-aware minimal edit（歧义感知最小编辑） | L2/L3 加歧义缩放 | 当前缩放方式过强，导致压缩或塌缩 |

## 6. Key Downstream Results（关键下游结果）

主指标必须看 `@1/@3/@5/@10`（主要截断）。

| Run（运行） | Stage（阶段） | Tokenizer（分词器） | Recipe（配方） | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | HR@1 | HR@3 | HR@5 | HR@10 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| strongest original RL（最强原版强化学习） | RL | original semantic | off + p05 | `0.07324068` | `0.08903190` | `0.09704467` | `0.10726345` | `0.07324068` | `0.10037503` | `0.11978822` | `0.15133466` |
| v2 RL（第二版强化学习） | RL | `mgr_tokenizer_v2_offline` | on + p05 | `0.07434370` | `0.09053678` | `0.09629833` | `0.10431921` | `0.07434370` | `0.10280168` | `0.11692036` | `0.14184867` |
| strongest original SFT（最强原版监督微调） | SFT | original semantic | off + p05 | `0.06706375` | `0.08500848` | `0.09315326` | `0.10372025` | `0.06706375` | `0.09838959` | `0.11824399` | `0.15089345` |
| v2_on_p05 SFT（第二版监督微调） | SFT | `mgr_tokenizer_v2_offline` | on + p05 | `0.07059343` | `0.08451223` | `0.09253300` | `0.10270767` | `0.07059343` | `0.09508052` | `0.11471432` | `0.14626075` |
| recipe-aligned original（配方对齐原版） | SFT | original semantic | on + p05 | `0.06860799` | `0.08416813` | `0.09054808` | `0.10182815` | `0.06860799` | `0.09574233` | `0.11118465` | `0.14626075` |
| original_l2_multihop_ranking（原版第二层多跳排序） | SFT | L2 ranking | on + p05 | `0.06618134` | `0.08283132` | `0.09144148` | `0.10165136` | `0.06618134` | `0.09485992` | `0.11581734` | `0.14736378` |
| original_l3_collab_local（原版第三层局部协同） | SFT | L3 local pull | on + p05 | `0.06684315` | `0.08445174` | `0.09226315` | `0.10159264` | `0.06684315` | `0.09772777` | `0.11692036` | `0.14604015` |
| R720e inverse ambiguity（R720e 逆歧义） | SFT | collab-ranking | on + p05 | `0.06397529` | `0.08397369` | `0.09185278` | `0.10094471` | `0.06397529` | `0.09883080` | `0.11802338` | `0.14604015` |

关键解读：

- `v2_on_p05 SFT`（第二版监督微调）接近 strongest original SFT（最强原版监督微调），但 `NDCG@10` 仍低 `0.00101`。
- `v2 RL`（第二版强化学习）超过 strongest original SFT（最强原版监督微调），但仍低于 strongest original RL（最强原版强化学习）。
- 最近 minimal-edit（最小编辑）系列都没有超过 strongest original SFT（最强原版监督微调）。
- `HR@50`（命中率@50）不能作为晋级依据；我们现在只看 `@1/@3/@5/@10`（主要截断）。

## 7. Method Evolution（方法演进）

### 7.1 v1（第一版）

`v1` 使用固定权重的 hierarchical graph regularization（层级图正则）。它证明 graph collaborative signal（图协同信号）可以改变 SID code assignment（语义标识分配），降低部分 collision（冲突）与局部歧义。但它没有证明 graph-supervised tokenizer（图监督分词器）能超过 strongest original tokenizer（最强原版分词器）。

局限是它不区分 item（物品）是否真的需要协同修正。对于语义本来清晰、routeability（可路由性）已经较好的 item（物品），图监督可能成为干扰。

### 7.2 v2（第二版）

`v2` 的核心机制是：

- ambiguity-aware graph supervision（歧义感知图监督）: 高歧义 item（物品）更依赖 collaborative graph（协同图）。
- semantic-structure retention（语义结构保持）: 避免协同图把语义路由打散。
- hierarchy-aware supervision（层级感知监督）: 让不同 SID level（语义标识层级）承接不同尺度的图信号。

`v2` 是当前最重要的正证据，但需要注意：`v2` 的成功是组合机制的成功。semantic retention（语义保持）的独立必要性还没有被完全隔离证明，不能过度宣称。

### 7.3 后续扩展

后续方向包括 stage-2（第二阶段）、stage-3（第三阶段）、graph carrier upgrade（图载体升级）、push-pull（拉推）、contrastive loss（对比损失）、ranking loss（排序损失）、minimal-edit（最小编辑）等。

这些实验共同给出一个负结论：**更强或更复杂的 tokenizer-side graph loss（分词器侧图损失）不自动带来更好的 downstream ranking（下游排序）。**

## 8. Graph Construction Lessons（构图经验）

我们尝试过多类 graph carrier（图载体）：

- `coarse_purified`（净化粗图）: 从用户历史共现构造粗粒度协同关系。
- `fagsp_mid_base`（基础中层频域图）: 用 spectral / frequency idea（谱 / 频域思想）构造中层图。
- `local_purified`（净化局部图）: 局部邻接图。
- `local_multihop`（局部多跳图）: 利用多跳协同关系补充直接边缺失。
- TAGCF-style semantic-to-topology（TAGCF 风格语义转拓扑）: 试图把属性或语义拓扑作为图载体。
- Seq2Graph-style carrier（Seq2Graph 风格载体）: 从序列上下文构造高阶协同。
- MGDCF / FaGSP / prism anchor（MGDCF / FaGSP / 棱镜锚点）等。

关键经验：

- `local_multihop`（局部多跳图）比 `fagsp_mid_base`（基础中层图）更适合作为 L2（第二层）协同载体。
- `prism_anchor_coarse`（语义锚定粗图）在当前实现下会导致 L1（第一层）灾难性塌缩，不能继续作为 L1 carrier（第一层载体）。
- 只换 graph carrier（图载体）不能解决“图结构如何转化为 LLM ranking benefit（大语言模型排序收益）”的问题。

## 9. R720 Collab-Ranking Family（R720 协同排序家族）

R720 系列尝试把核心问题显式写成 L2 ranking（第二层排序）：

$$
\mathcal L_{\mathrm{collab\_ranking}}
=
\mathcal L_{\mathrm{rec}}
+
\mathcal L_{\mathrm{rq}}
+
0.05\,\mathcal L_{\mathrm{pull}}^{(1)}
+
0.03\,\mathcal L_{\mathrm{rank}}^{(2)}
+
0.03\,\mathcal L_{\mathrm{pull}}^{(3)}.
$$

主要结果：

| Run（运行） | Change（变化） | Tokenizer Result（分词器结果） | Downstream Reading（下游解读） |
|---|---|---|---|
| R720a | `fagsp_mid_base` as L2 graph（基础中层图） | collision（冲突）低，但 L2 compression（第二层压缩）明显 | SFT 负，`NDCG@10 = 0.09235` |
| R720b | 换成 `local_multihop`（局部多跳图） | active L1（活跃第一层码）`247`, unique L2（唯一第二层前缀）`2889` | 明显优于 R720a，但仍低于 strongest original |
| R720d | `prism_anchor_coarse`（语义锚定粗图）替换 L1 图 | collision（冲突）`1755 / 3686`, active L1（活跃第一层码）`2` | 灾难性塌缩 |
| R720e | L1 inverse ambiguity（第一层逆歧义） | collision（冲突）`13 / 3686`, active L1（活跃第一层码）`190` | collab-ranking 家族内较强，但仍未超 baseline |
| R720f | K1=128 hard capacity reduction（硬容量压缩） | collision（冲突）低 | SFT 负，`NDCG@10 = 0.08955` |

信息量：

- `local_multihop`（局部多跳）是比 `fagsp_mid_base` 更合理的 L2 carrier（第二层载体）。
- L1（第一层）不能硬压缩，也不能用不稳定 coarse graph（粗图）强推。
- L1 compactness proxy（第一层紧凑性代理指标）不能作为晋级指标。
- R720 家族没有超过 strongest original SFT（最强原版监督微调），不应推进 RL（强化学习）。

## 10. Minimal-Edit Original RQ-VAE（原版最小编辑）

为了避免完整框架过度扰动，我们回到 original RQ-VAE base（原版残差量化变分自编码器基座），只做最小图监督改动。

### 10.1 original_l3_collab_local（原版第三层局部协同）

只在 L3（第三层）加入 local collaborative pull（局部协同拉近）。

结果：

- generated collision（生成冲突）: `13 / 3686 = 0.0035268584`
- SFT `NDCG@10 = 0.10159264`
- SFT `HR@10 = 0.14604015`

解读：

- 结构安全。
- 下游没有超过 strongest original SFT（最强原版监督微调）。
- L3（第三层）只能做局部消歧，无法修复 L1/L2（第一/第二层）已经剪掉的候选。

### 10.2 original_l2_multihop_ranking（原版第二层多跳排序）

只在 L2（第二层）加入 local_multihop ranking loss（局部多跳排序损失），并启用 previous-level stop-gradient（前层停梯度）：

$$
p_i^{(2)}
=
\mathrm{sg}(q_i^{(1)}) + q_i^{(2)}.
$$

结果：

- generated collision（生成冲突）: `15 / 3686 = 0.0040694520`
- active L1（活跃第一层码）: `88`
- unique L2（唯一第二层前缀）: `2449`
- SFT `NDCG@10 = 0.10165136`
- SFT `HR@10 = 0.14736378`

诊断结论：

- `HR@50`（命中率@50）有稳定信号，但这只是 secondary diagnostic signal（次级诊断信号）。
- primary cutoff（主要截断）`@1/@3/@5/@10` 没有稳定正收益。
- final Top10 proxy（最终前 10 代理诊断）没有显示 target L2 prefix survival（目标第二层前缀存活）改善。
- 所以这条线不是 primary downstream win（主要下游胜利）。

## 11. Ambiguity-Aware Minimal-Edit Results（歧义感知最小编辑结果）

最近三条实验专门检验 ambiguity-aware scaling（歧义感知缩放）是否能增强 L2/L3 minimal edit（第二/第三层最小编辑）。

| Experiment（实验） | Design（设计） | Collision（冲突） | Active L1（活跃第一层码） | Unique L2（唯一第二层前缀） | Verdict（裁决） |
|---|---|---:|---:|---:|---|
| `original_l3_ambiguity_aware` | L3 local pull + ambiguity scaling（第三层局部拉近 + 歧义缩放） | `657 / 3686` | `18` | `256` | clear no-go（明确停止） |
| `original_l2_ambiguity_aware` | L2 smoothness + ambiguity scaling（第二层平滑 + 歧义缩放） | `13 / 3686` | `50` | `1693` | non-catastrophic but over-compressed（非灾难性但过度压缩） |
| `original_l2_ranking_ambiguity_aware` | L2 ranking + ambiguity scaling（第二层排序 + 歧义缩放） | `15 / 3686` | `77` | `1649` | non-catastrophic but over-compressed（非灾难性但过度压缩） |

关键观察：

- L3 ambiguity-aware（第三层歧义感知）会直接导致 global routing collapse（全局路由塌缩）。
- L2 ambiguity-aware smoothness（第二层歧义感知平滑）不塌，但把 L2（第二层）严重压粗。
- L2 ranking ambiguity-aware（第二层排序歧义感知）逻辑上最贴合 push-pull motivation（推拉动机），但当前版本仍然把 mid-level structure（中层结构）压得过粗。
- offline ambiguity prior（离线歧义先验）整体偏高，`0.5 -> 1.5` 的 graph scale（图缩放）没有真正做到 sparse selective trigger（稀疏选择性触发）。

## 12. Tokenizer Distribution Findings（分词器分布发现）

与 strongest original tokenizer（最强原版分词器）相比，`L2 + ambiguity-aware`（第二层 + 歧义感知）并不是 L1（第一层）全局塌缩。

| Tokenizer（分词器） | Collision（冲突） | Active L1（活跃第一层码） | Unique L2（唯一第二层前缀） | L1 Max Bucket（第一层最大桶） | L2 Mean Bucket（第二层平均桶大小） |
|---|---:|---:|---:|---:|---:|
| strongest original（最强原版） | `16 / 3686` | `48` | `2295` | `247` | `1.606` |
| L2 ambiguity-aware smooth（第二层歧义感知平滑） | `13 / 3686` | `50` | `1693` | `291` | `2.177` |
| L2 ranking ambiguity-aware（第二层排序歧义感知） | `15 / 3686` | `77` | `1649` | `236` | not primary（非主统计） |

分布层面的信息：

- 原版最大 L1 bucket（第一层最大桶）主要是 3D printing ecosystem（3D 打印生态），语义较干净。
- L2 ambiguity-aware smooth（第二层歧义感知平滑）把 3D printing cluster（3D 打印簇）进一步扩成 super-bucket（超级大桶）。
- L2 ranking ambiguity-aware（第二层排序歧义感知）最大桶是 heterogeneous mixed industrial bucket（异质工业混合桶），包括 switches（开关）、tapes（胶带）、rods（杆件）、tweezers（镊子）、bottles（瓶子）、filters（滤芯）、adhesives（粘接用品），语义纯度较差。
- 这说明当前 ambiguity-aware ranking（歧义感知排序）没有稳定形成更好的 collaborative cluster（协同簇），反而可能制造中层混合桶。

## 13. What Evidence Supports（证据支持什么）

1. Graph collaborative signal（图协同信号）对 SID tokenizer（语义标识分词器）有可测影响，但尚未证明相对 strongest original MiniOneRec（最强原版 MiniOneRec）的端到端净收益。

`v1/v2` 证明了协同图信号能够改变 SID 码本空间，并让下游表现接近 strongest original SFT（最强原版监督微调）。但所有已评估的 graph-loss tokenizer（图损失分词器）目前都没有超过 strongest original SFT/RL（最强原版监督微调/强化学习），因此不能写成“图协同信号已经严格有效”。更严谨的结论是：图协同信号可能有价值，当前主要瓶颈在融合接口（fusion interface，融合接口）和 routeability（可路由性）保护。

2. v2 的 selective design（选择性设计）比全局粗暴图监督更稳。

v2 的 ambiguity-aware（歧义感知）和 semantic-structure retention（语义结构保持）是当前最强正证据，但它支持的是“选择性低扰动协同注入更稳”，而不是“任意图损失都有用”。

3. 更好的 tokenizer-side structure（分词器侧结构）不等于更好的 downstream ranking（下游排序）。

stage-2（第二阶段）、stage-3（第三阶段）、R650/R690/R720 多个实验都出现过结构好看但下游不赢的情况。

4. `local_multihop`（局部多跳图）比 `fagsp_mid_base`（基础中层图）更适合 L2（第二层）。

R720a -> R720b 给出清楚信息。

5. `HR@50`（命中率@50）不能作为主目标。

我们已经明确主评测是 `@1/@3/@5/@10`（主要截断），尤其 `NDCG@10` 和 `HR@10`。

## 14. What Has Not Been Proven（尚未证明）

1. 还没有证明任何新 tokenizer（分词器）超过 strongest original MiniOneRec SFT（最强原版 MiniOneRec 监督微调）。

`v2_on_p05 SFT` 接近但未超过；minimal-edit（最小编辑）系列也未超过。

2. 还没有证明新 tokenizer（分词器）超过 strongest original MiniOneRec RL（最强原版 MiniOneRec 强化学习）。

`v2 RL` 在 `NDCG@1/@3`（前 1 / 前 3）强，但 `NDCG@10` 和 `HR@10` 低于 strongest original RL（最强原版强化学习）。

3. 还不能宣称 end-to-end overall best（端到端整体最优）。

当前不能写成 overall best（整体最优），只能说 v2 提供了强正证据，后续实验提供了大量机制约束。

4. 还没有证明 push-pull ranking（推拉排序）接口能带来 top-k（前 k）提升。

逻辑动机成立，但当前实现过度压缩 L2（第二层）。

5. 还没有证明现有 ambiguity prior（歧义先验）校准正确。

offline ambiguity prior（离线歧义先验）整体偏高，当前 scaling（缩放）太像全局加压。

## 15. Current Core Diagnosis（当前核心诊断）

当前最可能的问题不是“图信息完全没用”，而是：

> 图协同信号进入 RQ-VAE（残差量化变分自编码器）的当前接口，会改变 SID codebook（语义标识码本），但这种改变不能稳定转化为 LLM（大语言模型）在 `@1/@3/@5/@10`（主要截断）上的排序收益。

具体表现：

- graph loss（图损失）一旦过强，容易压缩 L1/L2 prefix space（第一/第二层前缀空间）。
- L2 ranking（第二层排序）虽然贴合 semantic-collaborative mismatch（语义-协同错配），但会制造粗糙或异质的大桶。
- L3 local repair（第三层局部修复）安全但上限低，因为前缀阶段剪掉的候选无法靠 L3（第三层）救回。
- 原版 strongest tokenizer（最强原版分词器）虽然纯语义，但对 LLM（大语言模型）非常 routeable（可路由）。

## 16. Current Position（当前定位）

当前 strongest validated line（最强已验证线）仍然是：

> `v2_on_p05 -> RL`

但它不是 overall best（整体最优）。真正的 strongest baseline（最强基线）仍是 original MiniOneRec strongest RL（原版 MiniOneRec 最强强化学习）。

当前不应再把大量资源投入以下方向：

- 继续换 graph carrier（图载体）
- 继续叠 L1/L2/L3 graph loss（第一/第二/第三层图损失）
- 继续把 `HR@50`（命中率@50）当正信号
- 继续用 tokenizer proxy（分词器代理指标）直接决定 SFT（监督微调）
- 继续加重 naive ambiguity scaling（朴素歧义缩放）

## 17. Recommended Next Direction（推荐下一步）

建议下一步收敛为三条以内：

1. 以 `@1/@3/@5/@10`（主要截断）作为唯一推进门槛，停止把 `HR@50`（命中率@50）当作晋级依据。

当前 `original_l2_multihop_ranking`（原版第二层多跳排序）在 secondary diagnostic signal（次级诊断信号）上有 `HR@50` 优势，但在 primary cutoff（主要截断）上没有稳定正收益，因此按主目标口径应视为 no-go（停止）。

2. 若继续实验，只做 tightly controlled L2-only repair（严格受控的仅第二层小修复）。

优先限制在 `lambda_2`（第二层损失权重）、teacher pair quality（教师样本对质量）、触发稀疏度和 Top10 proxy（前 10 代理诊断）对齐，不直接叠加 `L2 + L3`（第二层 + 第三层）或引入新的大框架图载体。

3. 保留 v2（第二版）作为 strongest validated reference（最强已验证参考），并把图信号优先迁移到 downstream interface（下游接口）。

可优先尝试 hard negative mining（困难负样本挖掘）、SFT curriculum（监督微调课程学习）、candidate reranking（候选重排）或 beam-level prior（束搜索层先验），避免继续重度改写 tokenizer internal objective（分词器内部目标）。

## 18. Final Takeaway（最终结论）

这一阶段不是失败，而是把问题边界大幅收窄了：

- 我们证明了 collaborative graph（协同图）能够显著改变 SID tokenizer（语义标识分词器）并产生接近强基线的信号；但尚未证明它相对 strongest original MiniOneRec（最强原版 MiniOneRec）带来严格端到端净收益。
- 我们也证明了直接在 RQ-VAE（残差量化变分自编码器）里继续加图损失，不是稳定超过 strongest original MiniOneRec（最强原版 MiniOneRec）的路径。
- 当前最有价值的 idea（想法）不是“更强图监督”，而是“语义近但协同远的 item（物品）需要被选择性分开”。
- 但这个 idea（想法）需要更好的接口：更稀疏的 ambiguity trigger（歧义触发）、更低扰动的 codebook update（码本更新），或转移到 SFT / decoding（监督微调 / 解码）阶段。

因此，当前研究应从 broad graph-loss exploration（广泛图损失探索）转向 interface calibration（接口校准）和 v2-centered controlled improvement（围绕 v2 的受控小改进）。

## 19. 2026-04-21 Sync Delta（当日同步增量）

### 19.1 Decision Delta（决策增量）

- strongest validated line（最强已验证线）仍是 `v2_on_p05 -> RL`，但它当前是 reference baseline（参考基线），不是 active execution line（当前执行线）。
- active execution line（当前执行线）应保持为 minimal-edit diagnostic（最小编辑诊断）框架，且以 `@1/@3/@5/@10`（主要截断）作为唯一 go / no-go gate（推进 / 停止门槛）。
- `original_l3_ambiguity_aware`（原版第三层歧义感知）是 clear no-go（明确停止）；`original_l2_ranking_ambiguity_aware`（原版第二层排序歧义感知）是 non-catastrophic but over-compressed（非灾难性但过度压缩），当前不应优先推进到 SFT（监督微调）。

### 19.2 Mechanism Clarification from Code Audit（代码核查机制澄清）

1. 原生 RQ-VAE（原版残差量化变分自编码器）是 semantic-first tokenizer（语义优先分词器），不显式建模 collaborative signal（协同信号）；协同项来自后续 MGR-SID（多粒度关系感知语义标识）训练扩展（如 graph contrastive objective，图对比目标）。
2. SFT（监督微调）是 multi-task mixture（多任务混合）训练，`history SID -> next SID`（历史语义标识到下一语义标识）是主要子任务之一，但不是唯一监督来源。
3. 训练集中存在 `history_item_sid` 长度为 `1` 的样本是 prefix-window construction（前缀滑窗构造）导致的自然结果，不是数据错误；当前统计约占 `19.4%`。
4. 词表扩展时新增 token embedding（标记嵌入）在当前 `transformers==5.4.0` 下默认采用 mean_resizing（均值重缩放）初始化，不是全零初始化；因此“新增 SID token（语义标识标记）可学习”在初始化层面是成立的，但是否带来下游收益仍取决于训练信号与覆盖质量。

### 19.3 Practical Implication（实践含义）

- 下一步不应继续扩大 tokenizer-side complexity（分词器侧复杂度）；应优先做 low-disturbance interface design（低扰动接口设计）与 primary-cutoff diagnostics（主要截断诊断）闭环。
