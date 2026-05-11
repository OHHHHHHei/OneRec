# 2026-04-15 FaGSP Cascade 支链 `R520`

## 目的

这是 `FaGSP` 深化支链的第一轮 tokenizer（分词器）实验。

它回答的问题是：

> 如果把当前粗糙的 `fagsp_mid_base` 换成一个更接近 `FaGSP` item-side cascade（物品侧级联）机制的 `fagsp_mid_cascade`，能不能得到更好的 `G_mid`（中尺度图）候选。

## 实验定义

- 运行编号：`R520`
- 变体：
  - `L1 <- coarse_purified`
  - `L2 <- fagsp_mid_cascade`
  - `L3 <- local_purified`
- 训练方式：
  - 从头训练
  - 不用 `warm-start`（热启动）
  - 不加 stage-3 的 `retention`（保持）或 `anchor`（锚定）

## 关键实现

`fagsp_mid_cascade` 的核心步骤：

1. `high-pass`（高通）支撑发现
2. `support selection`（支撑选择）
3. 增强后 `low-pass`（低通）重构

实现文件：

- [paper_transplants.py](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/paper_transplants.py)
- [transplanted_graph_bank.py](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/transplanted_graph_bank.py)
- [train_v2.py](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/train_v2.py)

## 配置

- [sid_train_industrial_mgr_sid_fagsp_r520_mid_cascade.yaml](/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_fagsp_r520_mid_cascade.yaml)

## 启动脚本

- [experiment_mgr_sid_fagsp_r520_mid_cascade_train.sh](/home/leejt/OneRec/scripts/experiment_mgr_sid_fagsp_r520_mid_cascade_train.sh)

## Sanity 状态

- 1-epoch sanity：`PASSED`
- sanity 输出：
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_fagsp_cascade_20260415/sanity_r520_mid_cascade/Apr-15-2026_09-17-49/summary.json`
- sanity 读数：
  - `total_loss = 3.3568`
  - `collision = 0.038253`

## 正式训练状态

- 日期：`2026-04-15`
- tmux：`mgr_fagsp_r520_mid_cascade`
- GPU：`7`
- 当前状态：`COMPLETED`
- 日志：
  - `/home/leejt/OneRec/logs/experiment_mgr_sid_fagsp_r520_mid_cascade_20260415.log`

## 当前判断

这轮不是为了证明 “FaGSP 整篇论文适合我们”，而是为了更具体地回答：

> `G_mid` 的质量瓶颈，是不是来自我们之前对 `FaGSP` 的借鉴过浅。

## 训练结果

- 运行目录：
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_fagsp_cascade_20260415/industrial_r520_mid_cascade/Apr-15-2026_09-19-12`
- 最佳 train-side `collision`（训练侧冲突率）：
  - `0.1280520890`
- 最佳轮次：
  - `epoch = 9849`
- 最佳 checkpoint（检查点）：
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_fagsp_cascade_20260415/industrial_r520_mid_cascade/Apr-15-2026_09-19-12/best_collision_model.pth`
- 训练 summary（摘要）：
  - [summary.json](/data/leejt/OneRec/output_weights/experiments/mgr_sid_fagsp_cascade_20260415/industrial_r520_mid_cascade/Apr-15-2026_09-19-12/summary.json)

## Generate 结果

- 生成索引：
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_fagsp_cascade_20260415/generated_indices/Industrial_and_Scientific.fagsp_r520_mid_cascade.index.json`
- 生成 summary（摘要）：
  - [R520_generate_summary.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-15_mgr_sid_fagsp_r520_mid_cascade/R520_generate_summary.json)
- 最终 generated `collision`（生成后冲突率）：
  - `0.0037981552`
  - 即 `14 / 3686`
- `max_conflict`（最大冲突簇大小）：
  - `2`
- `collision_rounds_used`（冲突修补轮数）：
  - `20`

## 当前判断

`R520` 的生成后结果没有达到当前更强分支的水平：

- 比 `v2` 的 `13 / 3686` 略差
- 明显不如 `R510` 的 `11 / 3686`

所以到目前为止，更完整的 `FaGSP cascade`（FaGSP 级联）实现还没有在最终 SID 质量上给出正信号。
