# 2026-04-15 `R530a` Local Multi-Hop Tokenizer Screen（局部多跳分词器筛选）

## 目的

这是 coarse/local graph-carrier（粗图 / 局部图载体）计划里第一个真正推进到 tokenizer（分词器）训练的候选。

它回答的问题是：

> 如果不继续换 `G_mid`（中尺度图），而只把 `L3 <- local_multihop`（局部多跳图），
> 能不能得到比当前 `v2` 更有希望的 tokenizer-side（分词器侧）证据。

## 候选来源

- 计划：
  - [EXPERIMENT_PLAN_COARSE_LOCAL_GRAPH_CARRIERS.md](/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/refine-logs/EXPERIMENT_PLAN_COARSE_LOCAL_GRAPH_CARRIERS.md)
- 诊断：
  - [2026-04-15_mgr_sid_coarse_local_graph_diagnostics_industrial/README.md](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-15_mgr_sid_coarse_local_graph_diagnostics_industrial/README.md)

## 实验定义

- 运行编号：
  - `R530a`
- 变体：
  - `L1 <- coarse_purified`
  - `L2 <- fagsp_mid_base`
  - `L3 <- local_multihop`
- `local_multihop` 定义：
  - `A + αA^2`
  - `α = 0.35`
  - `max_hop = 2`

## 配置

- [sid_train_industrial_mgr_sid_r530a_local_multihop.yaml](/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_r530a_local_multihop.yaml)

## 启动脚本

- [experiment_mgr_sid_r530a_local_multihop_train_generate.sh](/home/leejt/OneRec/scripts/experiment_mgr_sid_r530a_local_multihop_train_generate.sh)

## Runtime（运行时）

- launch date（启动日期）:
  - `2026-04-15`
- tmux（终端复用）:
  - `mgr_r530a_local_multihop`
- GPU（图形处理器）:
  - `2`
- status（状态）:
  - `COMPLETED`

## Logs（日志）

- combined train/generate log（训练 / 生成合并日志）:
  - `/home/leejt/OneRec/logs/experiment_mgr_sid_r530a_local_multihop_20260415.log`

## Output Targets（输出目标）

- train output（训练输出）:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_coarse_local_20260415/industrial_r530a_local_multihop`
- generated index（生成索引）:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_coarse_local_20260415/generated_indices/Industrial_and_Scientific.r530a_local_multihop.index.json`
- generate summary（生成摘要）:
  - [R530a_generate_summary.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-15_mgr_sid_r530_local_multihop_industrial/R530a_generate_summary.json)

## 结果

- train-stage（训练阶段）best collision（最佳冲突率）:
  - `0.5784047748`
- generated collision（生成后冲突率）:
  - `0.0290287575`
  - `107 / 3686`
- max conflict（最大冲突大小）:
  - `17`
- collision rounds used（使用的冲突修复轮数）:
  - `20`

## 当前判断

`R530a` 是一个**明确负结果**。

它说明：

- shallow multi-hop `G_local`（浅层多跳局部图）虽然确实大幅改图；
- 但当前这版 `L3 <- A + αA^2` 没有带来更好的 tokenizer-side（分词器侧）结构；
- 相反，它在 final generated SID（最终生成 SID）上显著退化，远弱于当前 `v2` 和 stage-3 tokenizer（阶段 3 分词器）候选。

所以这条实验当前支持的结论是：

- “只改 `G_local`（局部图），而且按当前 multi-hop（多跳）写法去扩邻域” 并不是下一步主线；
- 下一步更合理的重点应转回 `G_coarse`（粗粒度图）重构，而不是直接推进 `R530b`（更深的 `A + αA^2 + α^2A^3`）。
