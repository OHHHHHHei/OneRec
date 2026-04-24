# MGR-SID 当前方法公式说明（与当前代码实现严格对齐）

**日期**：2026-04-12  
**状态**：archived code-aligned reference（已归档代码对齐参考）  
**用途**：这份文档用于把仓库里已经实现并实际跑过实验的 `MGR-SID v1 / v2 / R720 / QCR` 方法写成一版严格 code-aligned（代码对齐）的公式说明，方便后续复盘或方法追溯。  
**重要说明**：本文档只描述**当前代码真实实现**，不描述尚未接入训练的理想化扩展。凡是这里写出的公式，都应当能在当前代码中找到直接对应。

**2026-04-24 归档说明**：MGR-SID / ACLR / QCR 研究线已经归档为 negative-result stage（负结果阶段）。本文档保留为 code provenance（代码追溯）和 method reference（方法参考），不再作为新实验的默认出发点。归档入口见 [MGR-SID Negative Research Archive](/home/leejt/OneRec/research-progress-log/archive/2026-04-24_mgr_sid_negative_research_archive/README.md)。

**2026-04-18 历史主线收口说明**：当时 active mainline（活跃主线）曾切换为第 11 节的 `R720a`。前面关于 `v1/v2/R690` 的内容保留为 code provenance（代码追溯）和 baseline reference（基线参考）。

## 1. 对应代码范围

当前文档主要对应以下实现：

- `src/onerec/sid/models/rqvae.py`
- `src/onerec/sid/models/rq.py`
- `src/onerec/sid/models/vq.py`
- `src/onerec/experiments/mgr_sid/graph_bank.py`
- `src/onerec/experiments/mgr_sid/transplanted_graph_bank.py`
- `src/onerec/experiments/mgr_sid/paper_transplants.py`
- `src/onerec/experiments/mgr_sid/train_v1.py`
- `src/onerec/experiments/mgr_sid/train_v2.py`
- `scripts/experiment_mgr_sid_v2_proxy_sanity.py`

本文档的目标是回答一个非常具体的问题：

> 如果我们只看当前代码，`MGR-SID v1` 和 `MGR-SID v2` 到底在优化什么？

## 2. 记号与问题设置

设 item 集合为

$$
\mathcal I = \{1,\dots,N\}.
$$

每个 item $i$ 对应一个输入语义向量

$$
\mathbf{x}_i \in \mathbb{R}^d.
$$

当前 tokenizer 需要为每个 item 学习一个三层 semantic ID：

$$
sid(i)=\big[z_i^{(1)}, z_i^{(2)}, z_i^{(3)}\big].
$$

当前 MGR-SID 不替换 MiniOneRec 的 semantic tokenizer，而是在其训练目标之上加入 graph-based regularization。

## 3. Semantic RQ-VAE Backbone

### 3.1 Encoder

当前代码先通过 encoder 把输入 embedding 映射到量化空间：

$$
\mathbf{u}_i = f_{\mathrm{enc}}(\mathbf{x}_i).
$$

### 3.2 Residual Quantization

设第 $l$ 层 codebook 为

$$
\mathcal C^{(l)}=\{\mathbf{c}^{(l)}_1,\dots,\mathbf{c}^{(l)}_{K_l}\}.
$$

当前实验中总共有三层量化器，即 $L=3$，且当前配置为

$$
K_1=K_2=K_3=256.
$$

初始化残差：

$$
\mathbf r_i^{(1)}=\mathbf u_i.
$$

由于当前 tokenizer 实验配置使用

$$
\texttt{sk\_epsilons}=[0,0,0],
$$

所以训练时每一层都使用硬最近邻分配。第 $l$ 层的离散 token 为

$$
z_i^{(l)}=\arg\min_{k\in[K_l]}\left\|\mathbf r_i^{(l)}-\mathbf c_k^{(l)}\right\|_2^2,
$$

对应的量化向量为

$$
\mathbf q_i^{(l)}=\mathbf c_{z_i^{(l)}}^{(l)}.
$$

残差按代码更新为

$$
\mathbf r_i^{(l+1)}=\mathbf r_i^{(l)}-\mathbf q_i^{(l)}.
$$

累计量化表示定义为

$$
\mathbf h_i^{(l)}=\sum_{t=1}^{l}\mathbf q_i^{(t)}.
$$

最终第三层累计表示为

$$
\mathbf h_i^{(3)}=\mathbf q_i^{(1)}+\mathbf q_i^{(2)}+\mathbf q_i^{(3)}.
$$

### 3.3 Decoder

当前 decoder 从最终累计表示恢复输入向量：

$$
\hat{\mathbf x}_i = f_{\mathrm{dec}}(\mathbf h_i^{(3)}).
$$

## 4. Reconstruction 与 Quantization Loss

### 4.1 单层 Vector Quantizer 损失

当前 `VectorQuantizer` 每层使用的损失为

$$
\mathcal L_{\mathrm{vq}}^{(l)}(i)
=
\left\|\mathbf q_i^{(l)}-\mathrm{sg}\!\left[\mathbf r_i^{(l)}\right]\right\|_2^2
+
\beta
\left\|\mathrm{sg}\!\left[\mathbf q_i^{(l)}\right]-\mathbf r_i^{(l)}\right\|_2^2,
$$

其中 $\mathrm{sg}[\cdot]$ 表示 stop-gradient，当前配置中

$$
\beta=0.25.
$$

### 4.2 多层 Residual Quantizer 损失

当前 `ResidualVectorQuantizer` 直接对三层量化损失取平均：

$$
\mathcal L_{\mathrm{rq}}
=
\frac{1}{3}\sum_{l=1}^{3}\mathcal L_{\mathrm{vq}}^{(l)}.
$$

### 4.3 Reconstruction Loss

当前 tokenizer 训练使用 MSE 重建损失：

$$
\mathcal L_{\mathrm{rec}}
=
\frac{1}{Nd}
\sum_{i=1}^{N}
\left\|\hat{\mathbf x}_i-\mathbf x_i\right\|_2^2.
$$

### 4.4 Semantic Backbone 基础目标

当前代码中的基本 semantic objective 为

$$
\mathcal L_{\mathrm{sem}}
=
\mathcal L_{\mathrm{rec}}+\lambda_{\mathrm{rq}}\mathcal L_{\mathrm{rq}}.
$$

当前实验配置里 `quant_loss_weight=1.0`，因此真实使用的是

$$
\mathcal L_{\mathrm{sem}}=\mathcal L_{\mathrm{rec}}+\mathcal L_{\mathrm{rq}}.
$$

## 5. Graph Bank：当前训练实际使用的三张图

当前训练实际使用的图视图只有三张：

- coarse collaborative graph：`coarse_purified`
- middle-resolution spectral graph：`fagsp_mid_base`
- local transition graph：`local_purified`

虽然 graph-bank 代码还会生成 `prism_anchor_coarse`、`prism_anchor_local`、`fagsp_mid_prism`、`gsprec_mid_prism` 等额外视图，但当前 `v1` 和 `v2` 训练都没有直接把它们放进 loss。

### 5.1 Coarse Collaborative Graph

对每个训练样本 $(H_t,y_t)$，先截取最近 $K_h$ 个历史 item：

$$
H_t^{(K_h)}=[i_{t-m_t},\dots,i_{t-1}],
$$

当前配置中

$$
K_h = 10.
$$

然后构造带目标的序列并去重：

$$
S_t=\operatorname{uniq}(H_t^{(K_h)} \oplus [y_t]).
$$

若 $u,v \in S_t$，且在 $S_t$ 中位置分别为 $p<q$，则加入对称边：

$$
w_{uv}^{\mathrm{coarse}} \mathrel{+}= \frac{1}{q-p},
\qquad
w_{vu}^{\mathrm{coarse}} \mathrel{+}= \frac{1}{q-p}.
$$

汇总所有训练样本后得到原始 coarse 图 $\mathbf W^{\mathrm{coarse}}$。

之后代码按下面 3 步处理：

第一步，support pruning：

$$
\bar W_{uv}^{\mathrm{coarse}}
=
W_{uv}^{\mathrm{coarse}}
\cdot
\mathbf 1\!\left[W_{uv}^{\mathrm{coarse}}\ge \tau_c\right],
$$

当前

$$
\tau_c = 2.0.
$$

第二步，popularity debias：

$$
\tilde W_{uv}^{\mathrm{coarse}}
=
\frac{\bar W_{uv}^{\mathrm{coarse}}}
{\max(\mathrm{pop}(u),1)^{1/2}\max(\mathrm{pop}(v),1)^{1/2}}.
$$

第三步，row normalization：

$$
A_{uv}^{\mathrm{coarse}}
=
\frac{\tilde W_{uv}^{\mathrm{coarse}}}
{\sum_{v'}\tilde W_{uv'}^{\mathrm{coarse}}}.
$$

于是当前训练使用的 coarse 图是

$$
\mathbf A_{\mathrm{coarse}}.
$$

### 5.2 Local Transition Graph

对每个训练样本 $(H_t,y_t)$，遍历最近 $K_h$ 个历史 item。若 $u$ 是距离 target 第 $a$ 个历史 item，则加入有向边

$$
w_{u\to y_t}^{\mathrm{local}} \mathrel{+}= \frac{1}{a},
\qquad a=1,2,\dots,K_h.
$$

汇总得到原始 local 图 $\mathbf W^{\mathrm{local}}$。

随后代码执行：

第一步，support pruning：

$$
\bar W_{uv}^{\mathrm{local}}
=
W_{uv}^{\mathrm{local}}
\cdot
\mathbf 1\!\left[W_{uv}^{\mathrm{local}}\ge \tau_l\right],
$$

当前

$$
\tau_l = 1.0.
$$

第二步，target popularity correction：

$$
\tilde W_{uv}^{\mathrm{local}}
=
\frac{\bar W_{uv}^{\mathrm{local}}}
{\sqrt{\max(\mathrm{pop}(v),1)}}.
$$

第三步，row normalization：

$$
A_{uv}^{\mathrm{local}}
=
\frac{\tilde W_{uv}^{\mathrm{local}}}
{\sum_{v'}\tilde W_{uv'}^{\mathrm{local}}}.
$$

于是当前训练使用的 local 图是

$$
\mathbf A_{\mathrm{local}}.
$$

### 5.3 Middle-Resolution Spectral Graph

当前中尺度图不是 heuristic middle-view，而是 `fagsp_mid_base`。

先对 coarse purified graph 做对称归一化：

$$
\mathbf S
=
\mathbf D^{-1/2}\mathbf A_{\mathrm{coarse}}\mathbf D^{-1/2},
$$

其中 $\mathbf D$ 是 $\mathbf A_{\mathrm{coarse}}$ 的行和对角矩阵。

然后取前 $r$ 个最大特征成分：

$$
\mathbf S \approx \mathbf U\mathbf \Lambda \mathbf U^\top,
$$

当前配置中

$$
r = 48.
$$

代码只保留中间一段谱分量。令

$$
\ell = \lfloor r\rho_{\min}\rfloor,
\qquad
h = \lceil r\rho_{\max}\rceil,
$$

当前参数为

$$
\rho_{\min}=0.25,
\qquad
\rho_{\max}=0.65.
$$

于是重构矩阵为

$$
\mathbf M
=
\mathbf U_{[:,\ell:h]}
\,
\max(\mathbf\Lambda_{\ell:h,\ell:h},0)
\,
\mathbf U_{[:,\ell:h]}^\top.
$$

随后代码继续执行：

- 逐元素取非负；
- 清零对角线；
- row normalization。

因此当前训练中真正使用的 middle-resolution graph 为

$$
\mathbf A_{\mathrm{mid}}
=
\operatorname{RowNorm}\!\left(
\operatorname{OffDiag}\!\big([\mathbf M]_+\big)
\right).
$$

### 5.4 逐行 Top-k 裁剪

当前进入训练之前，每张图还会做一次逐行 top-k 保留。若 $k_g=32$，记操作为

$$
\operatorname{TopKRow}(\mathbf A, k_g).
$$

当前配置里 `graph_topk=32`，因此训练实际使用的是

$$
\mathbf A_{\mathrm{coarse}}^\star
=
\operatorname{TopKRow}(\mathbf A_{\mathrm{coarse}},32),
$$

$$
\mathbf A_{\mathrm{mid}}^\star
=
\operatorname{TopKRow}(\mathbf A_{\mathrm{mid}},32),
$$

$$
\mathbf A_{\mathrm{local}}^\star
=
\operatorname{TopKRow}(\mathbf A_{\mathrm{local}},32).
$$

## 6. V1：Hierarchy-Aware Graph Regularization

### 6.1 子图抽取

对一个 mini-batch $\mathcal B$，当前代码会从整图中抽取 batch 内 item 对应的子图：

$$
\mathbf A_{\mathcal B}^{(l)}
=
\mathbf A^{(l)\star}[\mathcal B,\mathcal B].
$$

### 6.2 当前代码中的 Graph Smoothness Loss

设

$$
\mathbf H_{\mathcal B}^{(l)}
=
[\mathbf h_i^{(l)}]_{i\in\mathcal B}
\in
\mathbb R^{|\mathcal B|\times d_l}.
$$

当前 `train_v1.py` 中的 graph smoothness loss 实际对应

$$
\mathcal L_{\mathrm{graph}}^{(l)}
=
\frac{1}{|\mathcal B|d_l}
\left\|
\mathbf H_{\mathcal B}^{(l)}
-
\mathbf A_{\mathcal B}^{(l)}\mathbf H_{\mathcal B}^{(l)}
\right\|_F^2.
$$

这里之所以是这个形式，是因为代码直接使用了

$$
\operatorname{MSE}(\mathbf H_{\mathcal B}^{(l)},\ \mathbf A_{\mathcal B}^{(l)}\mathbf H_{\mathcal B}^{(l)}).
$$

### 6.3 Level-to-Graph Assignment

当前 v1 的 hierarchy-aware assignment 是固定的：

$$
\mathbf A^{(1)\star}\leftarrow \mathbf A_{\mathrm{coarse}}^\star,
\qquad
\mathbf A^{(2)\star}\leftarrow \mathbf A_{\mathrm{mid}}^\star,
\qquad
\mathbf A^{(3)\star}\leftarrow \mathbf A_{\mathrm{local}}^\star.
$$

### 6.4 V1 总目标

因此，当前 v1 训练目标为

$$
\mathcal L_{\mathrm{v1}}
=
\mathcal L_{\mathrm{sem}}
+
\lambda_c \mathcal L_{\mathrm{graph}}^{(1)}
+
\lambda_m \mathcal L_{\mathrm{graph}}^{(2)}
+
\lambda_l \mathcal L_{\mathrm{graph}}^{(3)}.
$$

当前训练配置中

$$
\lambda_c = 0.05,
\qquad
\lambda_m = 0.15,
\qquad
\lambda_l = 0.05.
$$

## 7. V2：外部读取的 Ambiguity Prior

当前 `train_v2.py` 并不会在训练过程中在线计算 ambiguity，而是直接从 CSV 读取每个 item 的标量 prior：

$$
a_i \in [0,1].
$$

当前真正用到的列是：

$$
\texttt{ambiguity\_column}=\texttt{offline\_combined}.
$$

这个 `offline_combined` 来自 `scripts/experiment_mgr_sid_v2_proxy_sanity.py`。

### 7.1 Semantic Density

先对 semantic embedding 做 L2 normalization。对 item $i$ 的 semantic top-k 邻居，定义

$$
\rho_i^{\mathrm{sem}}
=
\frac{1}{k_s}
\sum_{j\in\mathcal N_{k_s}^{\mathrm{sem}}(i)}
\max\!\big(0,\cos(\mathbf x_i,\mathbf x_j)\big).
$$

### 7.2 Semantic-Collaborative Disagreement

记

- $\mathcal N_{k_s}^{\mathrm{sem}}(i)$：semantic kNN；
- $\mathcal N_{k_g}^{\mathrm{mid}}(i)$：mid graph 上的 top-k neighbor set。

则当前代码中的 disagreement 是

$$
\delta_i^{\mathrm{sc}}
=
1-
\frac{
\left|
\mathcal N_{k_s}^{\mathrm{sem}}(i)\cap \mathcal N_{k_g}^{\mathrm{mid}}(i)
\right|
}{
\left|
\mathcal N_{k_s}^{\mathrm{sem}}(i)\cup \mathcal N_{k_g}^{\mathrm{mid}}(i)
\right|
}.
$$

### 7.3 Graph Competition

若 $p_{ij}^{\mathrm{mid}}$ 是 item $i$ 在 mid graph 的 top-k 邻居权重归一化分布，则当前代码中的 graph competition 实际是归一化熵：

$$
\gamma_i^{\mathrm{comp}}
=
-\frac{1}{\log |\mathcal N_{k_g}^{\mathrm{mid}}(i)|}
\sum_{j\in\mathcal N_{k_g}^{\mathrm{mid}}(i)}
p_{ij}^{\mathrm{mid}}\log p_{ij}^{\mathrm{mid}}.
$$

### 7.4 Offline Combined Prior

记 $\operatorname{Norm}(\cdot)$ 为 min-max normalization 到 $[0,1]$。则当前训练中真实使用的 prior 为

$$
a_i
=
\frac{
\operatorname{Norm}(\rho_i^{\mathrm{sem}})
+
\operatorname{Norm}(\delta_i^{\mathrm{sc}})
+
\operatorname{Norm}(\gamma_i^{\mathrm{comp}})
}{3}.
$$

这就是当前 `offline_combined` 的精确形式。

## 8. V2 的 Semantic kNN Graph

当前 v2 额外构造了一张 semantic graph。对每个 item $i$，保留其 top-k semantic neighbors，并用非负 cosine similarity 作为边权：

$$
w_{ij}^{\mathrm{sem}}
=
\max\!\big(0,\cos(\mathbf x_i,\mathbf x_j)\big),
\qquad
j\in \mathcal N_{k_s}^{\mathrm{sem}}(i),\ j\neq i.
$$

再做 row normalization，得到

$$
\mathbf A_{\mathrm{sem}}.
$$

进入训练前，同样做逐行 top-k 裁剪：

$$
\mathbf A_{\mathrm{sem}}^\star
=
\operatorname{TopKRow}(\mathbf A_{\mathrm{sem}}, k_s).
$$

当前配置中 `semantic_graph_topk=32`。

### 8.1 Experimental External Semantic Graph Hook（实验性外部语义图接口）

从 `R670a` 开始，`train_v2.py` 增加了 `semantic_external_graph_path`（外部语义图路径）配置项。

如果该项为空，代码仍按上面的方式从 semantic embedding（语义嵌入）构造普通 semantic kNN graph（语义近邻图）。

如果该项非空，代码会直接读取外部 sparse graph（稀疏图）：

$$
\mathbf A_{\mathrm{sem}}
=
\operatorname{RowNorm}
\left(
\operatorname{LoadSparse}
(\texttt{semantic\_external\_graph\_path})
\right).
$$

随后仍然执行逐行 top-k 裁剪：

$$
\mathbf A_{\mathrm{sem}}^\star
=
\operatorname{TopKRow}(\mathbf A_{\mathrm{sem}}, k_s).
$$

这只是替换 semantic graph（语义图）来源，不改变 `_weighted_graph_smoothness_loss`（加权图平滑损失）的计算形式。
`R670a` 使用该接口把 `L1`（第一层）的 semantic-side smoothness（语义侧平滑）限制在 high-confidence semantic pairs（高置信语义物品对）上。

## 9. V2：当前代码真实使用的加权 Smoothness Loss

当前 v2 没有使用 KL semantic retention，也没有在训练时使用 online uncertainty。  
它真实做的是：使用 ambiguity prior 对 graph-side 与 semantic-side 两类 smoothness loss 做 item-wise reweighting。

### 9.1 Graph-side 权重

对 batch $\mathcal B$ 内 item $i$，当前 graph-side 权重为

$$
g_i
=
g_{\min}+(g_{\max}-g_{\min})a_i.
$$

当前配置中

$$
g_{\min}=0.5,
\qquad
g_{\max}=1.5.
$$

### 9.2 Semantic-side 权重

当前 semantic-side 权重为

$$
s_i
=
s_{\min}+(s_{\max}-s_{\min})(1-a_i).
$$

当前配置中

$$
s_{\min}=0.5,
\qquad
s_{\max}=1.5.
$$

这意味着：

- ambiguity 高的 item：graph-side 权重更大；
- ambiguity 低的 item：semantic-side 权重更大。

### 9.3 当前代码中的 Weighted Smoothness Loss

对任意表示矩阵 $\mathbf H$、子图 $\mathbf A$、以及 batch 内的 item 权重向量 $\mathbf w$，当前代码真实计算的是

$$
\mathcal L_{\mathrm{ws}}(\mathbf H,\mathbf A,\mathbf w)
=
\frac{
\sum_{i\in\mathcal B}
w_i
\cdot
\frac{1}{d}
\left\|
\mathbf h_i-(\mathbf A\mathbf H)_i
\right\|_2^2
}{
\sum_{i\in\mathcal B} w_i
}.
$$

这与 `train_v2.py` 中 `_weighted_graph_smoothness_loss` 完全对应。

## 10. V2：当前代码的精确总目标

当前 v2 一共使用 5 个正则项：

1. coarse graph on level 1  
2. mid graph on level 2  
3. local graph on level 3  
4. semantic graph on level 1  
5. semantic graph on level 2

注意：当前 semantic graph **不作用于 level 3**。

记 batch 子图分别为

$$
\mathbf A_{\mathcal B}^{c}=\mathbf A_{\mathrm{coarse}}^\star[\mathcal B,\mathcal B],
$$

$$
\mathbf A_{\mathcal B}^{m}=\mathbf A_{\mathrm{mid}}^\star[\mathcal B,\mathcal B],
$$

$$
\mathbf A_{\mathcal B}^{l}=\mathbf A_{\mathrm{local}}^\star[\mathcal B,\mathcal B],
$$

$$
\mathbf A_{\mathcal B}^{s}=\mathbf A_{\mathrm{sem}}^\star[\mathcal B,\mathcal B].
$$

则当前 v2 在代码中真实优化的是

$$
\mathcal L_{\mathrm{v2}}
=
\mathcal L_{\mathrm{sem}}
+
\lambda_c\,\mathcal L_{\mathrm{ws}}(\mathbf H_{\mathcal B}^{(1)},\mathbf A_{\mathcal B}^{c},\mathbf g_{\mathcal B})
+
\lambda_m\,\mathcal L_{\mathrm{ws}}(\mathbf H_{\mathcal B}^{(2)},\mathbf A_{\mathcal B}^{m},\mathbf g_{\mathcal B})
+
\lambda_l\,\mathcal L_{\mathrm{ws}}(\mathbf H_{\mathcal B}^{(3)},\mathbf A_{\mathcal B}^{l},\mathbf g_{\mathcal B})
$$

$$
\qquad
+
\mu_c\,\mathcal L_{\mathrm{ws}}(\mathbf H_{\mathcal B}^{(1)},\mathbf A_{\mathcal B}^{s},\mathbf s_{\mathcal B})
+
\mu_m\,\mathcal L_{\mathrm{ws}}(\mathbf H_{\mathcal B}^{(2)},\mathbf A_{\mathcal B}^{s},\mathbf s_{\mathcal B}).
$$

当前配置中系数为

$$
\lambda_c=0.05,
\qquad
\lambda_m=0.15,
\qquad
\lambda_l=0.05,
$$

$$
\mu_c=0.05,
\qquad
\mu_m=0.025.
$$

因此，若把系数全部代入，当前实验中真实优化的是

$$
\mathcal L_{\mathrm{v2}}
=
\mathcal L_{\mathrm{sem}}
+
0.05\,\mathcal L_{\mathrm{ws}}(\mathbf H^{(1)},\mathbf A^{c},\mathbf g)
+
0.15\,\mathcal L_{\mathrm{ws}}(\mathbf H^{(2)},\mathbf A^{m},\mathbf g)
+
0.05\,\mathcal L_{\mathrm{ws}}(\mathbf H^{(3)},\mathbf A^{l},\mathbf g)
$$

$$
\qquad
+
0.05\,\mathcal L_{\mathrm{ws}}(\mathbf H^{(1)},\mathbf A^{s},\mathbf s)
+
0.025\,\mathcal L_{\mathrm{ws}}(\mathbf H^{(2)},\mathbf A^{s},\mathbf s).
$$

### 10.1 R670a Experimental Objective（R670a 实验目标）

`R670a` 不是当前 v2 的替代定义，而是一个正在验证的 clean hierarchy（干净层级分工）实验变体。

它把 level representation（层级表示）改成 stop-gradient prefix（前缀停梯度）版本：

$$
\tilde{\mathbf H}^{(1)} = \mathbf Q^{(1)},
$$

$$
\tilde{\mathbf H}^{(2)}
=
\operatorname{sg}(\mathbf Q^{(1)})+\mathbf Q^{(2)}.
$$

对应的训练目标为

$$
\mathcal L_{\mathrm{R670a}}
=
\mathcal L_{\mathrm{sem}}
+
0.08\,\mathcal L_{\mathrm{ws}}
(\tilde{\mathbf H}^{(1)},\mathbf A_{\mathrm{sem\_hc}},\mathbf s)
+
0.15\,\mathcal L_{\mathrm{ws}}
(\tilde{\mathbf H}^{(2)},\mathbf A_{\mathrm{mid}},\mathbf g)
+
0.01\,\mathcal L_{\mathrm{sep}}
(\tilde{\mathbf H}^{(2)},\mathbf P_{\mathrm{weak}}).
$$

其中：

- $\mathbf A_{\mathrm{sem\_hc}}$ 是外部 high-confidence semantic graph（高置信语义图）。
- $\mathbf A_{\mathrm{mid}}$ 使用 `fagsp_mid_base`（基础中层图）。
- $\mathbf P_{\mathrm{weak}}$ 是 semantic-near + mid-weak pairs（语义近但中图弱连接物品对）。
- `coarse_weight = 0.0`，`local_weight = 0.0`，`semantic_mid_weight = 0.0`。
- 这个目标不限制 active L1 code count（活跃第一层码数量），只改变 `L1`（第一层）和 `L2`（第二层）的训练信号分工。

### 10.2 R690 Experimental Objective（R690 实验目标）

`R690a / R690b` 不是当前 v2 的替代定义，而是一个 `CoST-inspired`（受 CoST 启发）的 contrastive quantization（对比式量化）实验分支。

从这一版开始，`train_v2.py` 增加了三类新的可选项：

- `l1_contrastive_pull_weight`（第一层成对拉近权重）
- `l2_contrastive_mode`（第二层对比模式），支持：
  - `pairwise_pull`
  - `graph_infonce`
- `l3_contrastive_pull_weight`（第三层成对拉近权重）

当 `l2_contrastive_mode = graph_infonce` 时，第二层目标可写成

$$
\mathcal L_{\mathrm{InfoNCE}}^{(2)}
=
- \frac{1}{|\mathcal B'|}
\sum_{i\in \mathcal B'}
\log
\frac{
\sum_{j} A^{m}_{ij}\exp(\mathrm{sim}(\mathbf H^{(2)}_i,\mathbf H^{(2)}_j)/\tau)
}{
\sum_{j} A^{m}_{ij}\exp(\mathrm{sim}(\mathbf H^{(2)}_i,\mathbf H^{(2)}_j)/\tau)
+
\sum_{k} P^{\mathrm{weak}}_{ik}\exp(\mathrm{sim}(\mathbf H^{(2)}_i,\mathbf H^{(2)}_k)/\tau)
},
$$

其中：

- $A^{m}$ 是 `mid_view_name`（中层图视图）对应的 batch subgraph（batch 子图），作为第二层正样本权重。
- $P^{\mathrm{weak}}$ 是 semantic-near + mid-weak pairs（语义近但中图弱连接物品对）构成的负样本矩阵。
- $\tau$ 是 `l2_infonce_temperature`（第二层对比温度）。

`R690a` 的实验目标是

$$
\mathcal L_{\mathrm{R690a}}
=
\mathcal L_{\mathrm{sem}}
+
0.10\,
\mathcal L_{\mathrm{InfoNCE}}^{(2)}.
$$

`R690b` 则在同一第二层对比目标外，再加轻量前后层保护：

$$
\mathcal L_{\mathrm{R690b}}
=
\mathcal L_{\mathrm{sem}}
+
0.03\,
\mathcal L_{\mathrm{pull}}^{(1)}
+
0.10\,
\mathcal L_{\mathrm{InfoNCE}}^{(2)}
+
0.02\,
\mathcal L_{\mathrm{pull}}^{(3)}.
$$

其中：

- $\mathcal L_{\mathrm{pull}}^{(1)}$ 使用 semantic graph（语义图）上的 pairwise pull（成对拉近）。
- $\mathcal L_{\mathrm{pull}}^{(3)}$ 使用 local graph（局部图）上的 pairwise pull（成对拉近）。
- `R690b` 额外设置 `hierarchy_stopgrad_previous_levels = true`，使 `L2/L3`（第二层/第三层）辅助损失不再直接反传改写前层前缀。

## 11. R720a：历史主线候选的 L2 排序对比实现

`R720a` 是当前已经接入代码的主线候选，不再是只停留在讨论里的方法。它对应的核心配置是：

- `config/experiments/sid_train_industrial_mgr_sid_collab_ranking_mainline.yaml`
- `src/onerec/experiments/mgr_sid/train_collab_ranking_sid.py`
- `scripts/experiment_mgr_sid_collab_ranking_train.py`
- `scripts/experiment_mgr_sid_collab_ranking_pair_source.py`
- `scripts/experiment_mgr_sid_collab_ranking_train_generate.sh`

它保留 semantic RQ-VAE backbone（语义残差量化骨干）：

$$
\mathcal L_{\mathrm{base}}
=
\mathcal L_{\mathrm{rec}}
+
\mathcal L_{\mathrm{rq}}.
$$

三层职责被固定为：

$$
L1:\ \text{coarse collaborative routing},
\qquad
L2:\ \text{collaborative ranking branch},
\qquad
L3:\ \text{local refinement}.
$$

其中 `L2`（第二层）不再使用 graph smoothness（图平滑），而是使用 ranking contrastive loss（排序对比损失）：对每个 anchor item（锚点物品） $i$，从 `fagsp_mid_base`（基础中层图）中取 positive collaborative items（协同正样本） $p$，从 semantic-near but mid-weak pairs（语义近但中图弱连接物品对）中取 hard negative items（困难负样本） $n$，要求：

$$
s_{ip}^{(2)}
\ge
s_{in}^{(2)}
+
m.
$$

当前代码中的具体损失为：

$$
\mathcal L_{\mathrm{rank}}^{(2)}
=
\frac{
\sum_i
\sum_{p \in \operatorname{TopK}^{+}(i)}
\sum_{n \in \operatorname{TopK}^{-}(i)}
w_{ip}^{+} w_{in}^{-}
\left[
m+s_{in}^{(2)}-s_{ip}^{(2)}
\right]_+
}{
\sum_i
\sum_{p \in \operatorname{TopK}^{+}(i)}
\sum_{n \in \operatorname{TopK}^{-}(i)}
w_{ip}^{+} w_{in}^{-}
}.
$$

这里：

$$
s_{ij}^{(2)}
=
\operatorname{cos}
\left(
\tilde{\mathbf h}_i^{(2)},
\tilde{\mathbf h}_j^{(2)}
\right),
$$

并且在 `hierarchy_stopgrad_previous_levels = true` 时：

$$
\tilde{\mathbf h}_i^{(2)}
=
\operatorname{sg}\left[\mathbf q_i^{(1)}\right]
+
\mathbf q_i^{(2)}.
$$

因此 `L2` 排序对比主要更新第二层分叉，而不是直接改写第一层粗路由。

当前 `R720a` 的总目标是：

$$
\mathcal L_{\mathrm{R720a}}
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

其中：

- $\mathcal L_{\mathrm{pull}}^{(1)}$ 使用 `coarse_purified`（净化粗图）。
- $\mathcal L_{\mathrm{rank}}^{(2)}$ 使用 `fagsp_mid_base`（基础中层图）作为正样本图。
- hard negatives（困难负样本）由 `semantic-near + mid-weak`（语义近 + 中图弱连接）规则离线生成。
- $\mathcal L_{\mathrm{pull}}^{(3)}$ 使用 `local_purified`（净化局部图）。
- `semantic_coarse_weight = 0` 且 `semantic_mid_weight = 0`，当时主线不再额外堆 semantic retention（语义保持）项。
- `mid_weight = 0`，当时主线不同时叠加 `L2` graph smoothness（第二层图平滑）。

当前初始超参数是：

$$
m=0.1,\qquad
\operatorname{TopK}^{+}=8,\qquad
\operatorname{TopK}^{-}=16.
$$

## 12. 当前代码没有做什么

为了防止后续写作时把“想做的版本”和“已经实现的版本”混在一起，这里明确记录：当前代码**没有**做下面这些事。

### 12.1 没有在训练中接入 online uncertainty

虽然 `experiment_mgr_sid_v2_proxy_sanity.py` 里分析过 online uncertainty，但当前 `train_v2.py` 训练只读取 `offline_combined`，没有把 online uncertainty 接入 loss。

### 12.2 没有定义 level-wise ambiguity $a_i^{(l)}$

当前代码中 ambiguity 是 item-level scalar

$$
a_i,
$$

而不是 level-dependent 的

$$
a_i^{(l)}.
$$

### 12.3 没有使用 KL / neighborhood distribution matching 作为 semantic retention

当前所谓 semantic retention，真实实现不是 KL 保邻域分布，而是**semantic graph 上的 weighted smoothness loss**。

### 12.4 没有在 level 3 上施加 semantic retention

当前 semantic graph loss 只加在 level 1 和 level 2。

### 12.5 没有 learned gate

当前 graph-role assignment 仍然是固定的：

$$
L1 \leftarrow \text{coarse},
\qquad
L2 \leftarrow \text{mid},
\qquad
L3 \leftarrow \text{local}.
$$

## 13. 一句话总结

如果只用一句话总结当前代码版本的方法，那么最准确的表述是：

> 当前 `MGR-SID v2` 保留 MiniOneRec 的三层 semantic RQ-VAE tokenizer，不改 graph role assignment；它通过一个外部预计算的 item-level ambiguity prior，分别对 coarse/mid/local graph smoothness 与 semantic kNN graph smoothness 做 item-wise reweighting，从而在高歧义 item 上强化 graph supervision，在低歧义 item 上强化 semantic structure preservation。

这就是当前代码层面与实验层面真正对齐的 method 公式。
