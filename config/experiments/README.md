# Active Experiment Configs（活跃实验配置）

Status（状态）: `navigation（导航）`

Last updated（更新日期）: `2026-04-18`

## Mainline（主线）

当前主线 tokenizer（分词器）训练配置只有一个：

- `sid_train_industrial_mgr_sid_collab_ranking_mainline.yaml`

它对应当前主线方法：`ambiguity-aware stop-gradient L2 ranking contrastive SID`（歧义感知停止梯度中层排序对比 SID）。

## Temporary Legacy（临时历史保留）

以下 `R690b` 文件暂时保留在原位，因为 `mgr_r690b_sft_eval_4gpu` 这个 tmux（终端复用器）会话仍在运行，后续 evaluate（评测）阶段还会读取这些配置：

- `sid_train_industrial_mgr_sid_r690b_hier_cost_guided.yaml`
- `sft_industrial_mgr_r690b_title_on_desc_p05.yaml`
- `evaluate_industrial_mgr_r690b_title_on_desc_p05.yaml`

等该运行结束并完成结果登记后，这三个配置也应归档。

## Archive（归档）

旧分支配置已经移动到：

- `/home/leejt/OneRec/config/archive/2026-04-18_pre_r720_legacy_experiments/`

这些配置只用于 provenance（追溯）和历史复现，不应作为新实验起点。
