# 2026-04-16 `R542a` MGDCF Coarse Tokenizer Screen（`MGDCF` 粗图分词器筛选）

## 目的

这是 coarse reconstruction（粗图重构）主线的第一个正式 tokenizer（分词器）实验。

它回答的问题是：

> 如果把当前基于局部序列共现的 `coarse_purified`（净化粗图）换成
> `MGDCF` 风格的 reconstructed coarse graph（重构粗图），
> 并同时让 `G_mid`（中尺度图）从这个新 coarse（粗图）重新导出，
> 能不能得到比当前 `v2` 更有希望的 tokenizer-side（分词器侧）结果。

## 候选来源

- 诊断：
  - [2026-04-16_mgr_sid_mgdcf_coarse_diagnostics_industrial/README.md](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_mgdcf_coarse_diagnostics_industrial/README.md)
- 关键结论：
  - `mgdcf_keep_ratio = 0.20` 是当前最合理的第一个正式 tokenizer（分词器）推进点

## 实验定义

- 运行编号：
  - `R542a`
- 变体：
  - `L1 <- coarse_mgdcf`
  - `L2 <- fagsp_mid_mgdcf`
  - `L3 <- local_purified`
- `MGDCF` 关键参数：
  - `mgdcf_keep_ratio = 0.20`
  - `mgdcf_binarize_edges = true`

## 配置

- [sid_train_industrial_mgr_sid_r542a_mgdcf_coarse.yaml](/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_r542a_mgdcf_coarse.yaml)

## 启动脚本

- [experiment_mgr_sid_r542a_mgdcf_coarse_train_generate.sh](/home/leejt/OneRec/scripts/experiment_mgr_sid_r542a_mgdcf_coarse_train_generate.sh)

## Runtime（运行时）

- launch date（启动日期）:
  - `2026-04-16`
- tmux（终端复用）:
  - `mgr_r542a_mgdcf_coarse`
- GPU（图形处理器）:
  - `2`
- status（状态）:
  - `COMPLETED`

## Logs（日志）

- combined train/generate log（训练 / 生成合并日志）:
  - `/home/leejt/OneRec/logs/experiment_mgr_sid_r542a_mgdcf_coarse_20260416.log`

## Output Targets（输出目标）

- train output（训练输出）:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_mgdcf_coarse_20260416/industrial_r542a_mgdcf_coarse_r020`
- generated index（生成索引）:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_mgdcf_coarse_20260416/generated_indices/Industrial_and_Scientific.r542a_mgdcf_coarse_r020.index.json`
- generate summary（生成摘要）:
  - [R542a_generate_summary.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_r542_mgdcf_coarse_industrial/R542a_generate_summary.json)

## 当前判断

`R542a` 是当前最值得先跑的 coarse（粗图）重构实验，因为它同时满足：

- 它不是“几乎不改图”的弱重加权版本；
- 它直接重构 `G_coarse`（粗粒度图）这张母图；
- 它会连带改变 `L1` 和 `L2`，而不是只改一层。

所以这条 run（运行）比当前继续深挖 local multi-hop（局部多跳）更有信息量。

## 结果

- train-stage（训练阶段）best collision（最佳冲突率）:
  - `0.3182311449`
- generated collision（生成后冲突率）:
  - `0.0113944655`
  - `42 / 3686`
- `max_conflict`（最大冲突大小）:
  - `4`
- `collision_rounds_used`（冲突修补轮数）:
  - `20`

## 当前判断

`R542a` 的结果是：

- **明显好于** `R530a` 这类失败的 local-only（仅局部）扩散分支；
- 但**仍然明显弱于**当前 `v2`、`R510`、`R520` 和 stage-3 tokenizer（阶段 3 分词器）候选。

所以当前最准确的结论是：

- `MGDCF` 风格 coarse reconstruction（粗图重构）证明了“重构 `G_coarse`（粗粒度图）”这条线确实能产生强改图效果；
- 但当前这版 `R542a` 还没有给出足够强的 tokenizer-side（分词器侧）正信号，不值得直接推进到下游 `SFT -> evaluate`（监督微调到评测）。
