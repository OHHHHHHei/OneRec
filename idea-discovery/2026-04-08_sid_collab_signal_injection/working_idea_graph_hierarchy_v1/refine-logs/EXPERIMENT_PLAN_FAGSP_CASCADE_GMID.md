# Experiment Plan: FaGSP Cascade `G_mid`

## 动机

当前 `MGR-SID v2` 里最关键的中尺度图视图是 `fagsp_mid_base`，但它其实只是一个很粗的频段切片近似：

- 从 `coarse_purified` 出发
- 做对称归一化
- 取一段中频特征值
- 重构出一个 `band-pass view`（带通视图）

这个做法有用，但它没有真正移植 `FaGSP` 最有判别性的设计：

1. `high-pass filter`（高通滤波）先突出独特边
2. `support selection`（支撑选择）保留真正值得强调的关系
3. `low-pass filter`（低通滤波）再在增强后的信号上形成更稳的预测视图

因此，这条实验线要回答的问题非常明确：

> 当前 `G_mid` 的天花板，是不是受限于 `FaGSP` 借鉴过浅；如果把 item-side cascade 做完整一点，能不能得到更好的中尺度图。

## 方法定义

我们引入一个新的中尺度图视图：

- 名称：`fagsp_mid_cascade`

它的构造分三步：

### Step 1: 高通支撑发现

在 `coarse_purified` 的对称化图上做谱分解，取一小段高频子空间，得到一个高通支撑分数矩阵。

直觉：

- 高频不代表“更好”
- 但更可能对应那些把某个 item（物品）和别的 item 区分开的独特关系

### Step 2: 支撑选择

对每个目标 item（物品）按列做分位数筛选，只保留当前列里最值得强调的一小部分已有边。

直觉：

- 不是所有高频边都可靠
- 支撑选择的作用是避免高频噪声直接污染图结构

### Step 3: 增强后低通重构

把被选中的边在原图上做加权增强，然后在增强后的图上做低通重构，得到最终的 `fagsp_mid_cascade`。

直觉：

- 高通阶段负责“找判别性”
- 低通阶段负责“把这些判别性编织回一个更稳的中尺度视图”

## 当前实现版本

第一版参数：

- `fagsp_cascade_high_rank = 16`
- `fagsp_cascade_low_rank = 32`
- `fagsp_cascade_support_quantile = 0.8`
- `fagsp_cascade_boost_alpha = 0.5`

这是一版保守起点，目标不是一次找到最优点，而是先验证：

> 级联机制本身是不是值得继续做。

## 首个实验

### `R520`

系统定义：

- `L1 <- coarse_purified`
- `L2 <- fagsp_mid_cascade`
- `L3 <- local_purified`

保持不变：

- ambiguity-aware weighting（歧义感知加权）
- semantic retention（语义保持）
- 其余 tokenizer 训练 recipe（训练配方）

不叠加：

- stage-3 prefix retention（前缀保持）
- codebook anchor（码本锚定）
- TAGCF 属性图替换

原因：

这轮只想回答一个问题：

> 单纯把 `G_mid` 从粗糙频段切片升级成更像 `FaGSP` 的 cascade，会发生什么？

## 判读标准

### 第一层：tokenizer-side（分词器侧）

- train-side / generate-side `collision`（训练侧 / 生成侧冲突率）
- 相对 `v2` 的 `local ambiguity`（局部歧义）变化
- 是否出现明显更好的中尺度结构画像

### 第二层：是否值得推下游

如果 `R520` 至少满足：

- 生成后 `collision` 不差于 `v2`
- 且 `local ambiguity` 不是明显回退

就值得继续进入完整 `SFT -> evaluate`（监督微调到评测）。

## 后续自然分支

如果 `R520` 有正信号，下一步最自然的是：

1. `R521`
   - `fagsp_mid_cascade_prism`
   - 看语义锚定版 cascade 是否更稳

2. `R522`
   - 小范围扫：
     - `high_rank`
     - `support_quantile`
     - `boost_alpha`

3. `R523`
   - 对比 `gsprec_mid_prism`
   - 回答“更完整的 FaGSP 级联”和“引入转移信息的 GSPRec 中图”谁更值

## 结论预期

这条线不是为了证明“FaGSP 论文本身适合我们”，而是为了更具体地回答：

> 在 MGR-SID 里，`G_mid` 到底缺的是一个真正的判别性中图，还是缺别的东西。

如果 `R520` 没有给出正信号，我们就能更有把握地说：

- 问题未必在 `FaGSP` 借鉴过浅
- 可能更该回到图源或损失配比本身

如果 `R520` 有正信号，这条线就有机会成为当前主线里最值得继续推进的图设计升级方向。
