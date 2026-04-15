# FaGSP Cascade Mid 支链

这条支链专门处理一个已经很明确的问题：

> 我们当前的 `fagsp_mid_base` 只是 `FaGSP`（Frequency-aware Graph Signal Processing for Collaborative Filtering，面向协同过滤的频率感知图信号处理）的粗浅频段切片近似，并没有真正复用它最关键的 `cascaded filter module`（级联滤波模块）。

## 这条支链想回答什么

1. 当前 `G_mid`（中尺度图）是不是因为实现得太粗，才限制了图信号的上限。
2. 如果把 `FaGSP` 的核心思路真正移植进来：
   - 先做 `high-pass`（高通）找判别性强的边
   - 再做 `support selection`（支撑选择）筛出值得强调的关系
   - 最后做 `low-pass`（低通）形成更稳的中尺度图
   会不会比现在的 `fagsp_mid_base` 更适合做 SID 监督。

## 当前落地版本

第一版不会完整移植整篇 `FaGSP`，而只移植最关键的 **item-side cascade（物品侧级联）**：

- 输入：`coarse_purified`
- 输出：`fagsp_mid_cascade`

不直接动：

- `user-side parallel filter`（用户侧并行滤波）
- 原论文整体推荐主干

原因很简单：我们当前要升级的是 `graph carrier`（图载体），不是把 MiniOneRec tokenizer（分词器）改成传统图协同过滤模型。

## 关键文件

- 方案文档：
  - [EXPERIMENT_PLAN_FAGSP_CASCADE_GMID.md](../../refine-logs/EXPERIMENT_PLAN_FAGSP_CASCADE_GMID.md)
- 代码实现：
  - [paper_transplants.py](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/paper_transplants.py)
  - [transplanted_graph_bank.py](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/transplanted_graph_bank.py)
  - [train_v2.py](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/train_v2.py)
- 首个实验配置：
  - [sid_train_industrial_mgr_sid_fagsp_r520_mid_cascade.yaml](/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_fagsp_r520_mid_cascade.yaml)
- 启动脚本：
  - [experiment_mgr_sid_fagsp_r520_mid_cascade_train.sh](/home/leejt/OneRec/scripts/experiment_mgr_sid_fagsp_r520_mid_cascade_train.sh)

## 当前判断

这条支链不是为了“再发明一个全新方向”，而是为了先补齐一个已经存在的明显缺口：

- 我们一直在说 `FaGSP-inspired`（受 `FaGSP` 启发）
- 但真正最值钱的那段级联机制并没有进代码

所以这条支链的研究价值很务实：

> 先回答“更认真地移植 `FaGSP` 的 item-side cascade，能不能直接提升 `G_mid` 的质量”。
