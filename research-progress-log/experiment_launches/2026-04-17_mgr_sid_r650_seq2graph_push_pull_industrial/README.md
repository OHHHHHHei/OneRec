# 2026-04-17 `R650` Seq2Graph Push-Pull（Seq2Graph 推远拉近）

Status（状态）: `snapshot（快照）`
Snapshot date（快照日期）: `2026-04-17`

## 目的

`R650a` 是对当前核心判断的直接实验化：

- `R640c` 的 `Seq2Graph-lite rel_masked`（轻量 `Seq2Graph` 可靠性感知加掩码版）不能作为 standalone carrier-only smoothness（独立的仅图载体加平滑监督）方法继续推进。
- 但它仍然可以作为 `push-pull`（推远拉近）的 graph carrier substrate（图载体基底）。
- 因此本实验把 `R640c` 的 high-order graph（高阶图）放入 `R630c` 的 mid-only `pull + push`（仅中层拉近加推远）框架。

## 设计

`R650a = Seq2Graph-mid pull + Seq2Graph-mid weak push`（Seq2Graph 中层拉近 + Seq2Graph 中层弱连接推远）。

- `pull`（拉近）：
  - `L2 <- fagsp_mid_seq2g_rel_masked`
  - `mid_weight = 0.15`
- `push`（推远）：
  - pair source（物品对来源）来自 semantic-near + `fagsp_mid_seq2g_rel_masked` weak（语义近 + Seq2Graph 中图弱连接）
  - `selective_separation_weight = 0.01`
  - `selective_separation_margin = 0.15`
  - `selective_separation_levels = [2]`
- 为了保持归因清楚，本实验不额外启用 coarse/local pull（粗层/局部层拉近）或 semantic retention（语义保持）：
  - `coarse_weight = 0.0`
  - `local_weight = 0.0`
  - `semantic_coarse_weight = 0.0`
  - `semantic_mid_weight = 0.0`

## 对应文件

- tokenizer config（分词器配置）：
  - [sid_train_industrial_mgr_sid_r650a_seq2graph_mid_pull_push.yaml](/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_r650a_seq2graph_mid_pull_push.yaml)
- pair source script（物品对来源脚本）：
  - [experiment_mgr_sid_mid_pull_push_pair_source.py](/home/leejt/OneRec/scripts/experiment_mgr_sid_mid_pull_push_pair_source.py)
- train/generate script（训练生成脚本）：
  - [experiment_mgr_sid_r650a_seq2graph_mid_pull_push_train_generate.sh](/home/leejt/OneRec/scripts/experiment_mgr_sid_r650a_seq2graph_mid_pull_push_train_generate.sh)
- launch script（启动脚本）：
  - [launch_mgr_sid_r650_seq2graph_push_pull_tmux.sh](/home/leejt/OneRec/scripts/launch_mgr_sid_r650_seq2graph_push_pull_tmux.sh)

## 当前状态

当前阶段：`FINISHED_TOKENIZER_GENERATED`（分词器训练与生成已完成）。

- `tmux`（终端复用器） session（会话）：
  - `mgr_r650a_seq2graph_push_pull`，已退出
- runtime GPU（运行显卡）：
  - `CUDA_VISIBLE_DEVICES=3`
- pair source summary（物品对来源摘要）：
  - `semantic_pair_count = 82596`
  - `weak_pair_count = 1190`
  - `weak_pair_item_coverage_rate = 0.29896907216494845`
  - `weak_threshold = 0.0016943709051702172`

## 结果

- train best collision（训练最佳冲突率）：`0.1142159523`
- train best epoch（训练最佳轮次）：`9949`
- best loss（最佳损失）：`0.2820739150`
- generated collision（生成后冲突）：`11 / 3686 = 0.0029842648`
- max conflict（最大冲突簇）：`2`
- collision rounds used（冲突修复轮数）：`20`

## 当前结论

`R650a` 没有出现 catastrophic failure（灾难性失败），generated collision（生成后冲突）与 `R630c` 的 `11 / 3686` 持平，略好于 `R640c` 的 `12 / 3686`，也略好于当前 `v2` 的 `13 / 3686`。

但这仍然只是 tokenizer-side first gate（分词器侧第一关），不是 downstream verdict（下游裁决）。尤其 `R630c` 曾经也达到 `11 / 3686`，但后续 `SFT/evaluate`（监督微调/评测）为负，因此 `R650a` 下一步必须接 `title_history2sid_on + desc_align_p05` 下游验证，不能只凭 collision（冲突）判断成功。
