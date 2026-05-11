# 2026-04-17 `R650a` SFT/Evaluate（监督微调/评测）

Status（状态）: `snapshot（快照）`
Snapshot date（快照日期）: `2026-04-17`

## 目的

`R650a` tokenizer/generate（分词器训练与生成）已经完成且没有 catastrophic failure（灾难性失败）。本阶段把它接入当前 graph-aware（图感知）最合理下游 recipe（配方）：

- `title_history2sid_on`
- `desc_align_p05`

这一步只先启动 SFT（监督微调），不自动连跑 evaluate（评测）。训练完成后再用同目录下的 evaluate config（评测配置）做 downstream verdict（下游裁决）。

## 对应输入

- tokenizer index（分词器索引）：
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_seq2graph_push_pull_20260417/generated_indices/Industrial_and_Scientific.r650a_seq2graph_mid_pull_push.index.json`
- prepared data root（已准备数据根目录）：
  - `/home/leejt/OneRec/data_experiment/Amazon/r650a_seq2graph_mid_pull_push`

## 对应文件

- SFT config（监督微调配置）：
  - [sft_industrial_mgr_r650a_title_on_desc_p05.yaml](/home/leejt/OneRec/config/experiments/sft_industrial_mgr_r650a_title_on_desc_p05.yaml)
- evaluate config（评测配置）：
  - [evaluate_industrial_mgr_r650a_title_on_desc_p05.yaml](/home/leejt/OneRec/config/experiments/evaluate_industrial_mgr_r650a_title_on_desc_p05.yaml)
- SFT script（监督微调脚本）：
  - [experiment_mgr_sid_r650a_sft_chain.sh](/home/leejt/OneRec/scripts/experiment_mgr_sid_r650a_sft_chain.sh)
- launch script（启动脚本）：
  - [launch_mgr_sid_r650a_sft_tmux.sh](/home/leejt/OneRec/scripts/launch_mgr_sid_r650a_sft_tmux.sh)

## 当前状态

当前阶段：`COMPLETED_NEGATIVE`（已完成，负结果）。

- SFT tmux（监督微调终端复用器） session（会话）：
  - `mgr_r650a_sft`，已结束
- SFT runtime GPUs（监督微调运行显卡）：
  - `CUDA_VISIBLE_DEVICES=2,3,4,5`
- evaluate runtime GPUs（评测运行显卡）：
  - `CUDA_VISIBLE_DEVICES=3,4,5,7`
- SFT output（监督微调输出）：
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r650a_sft_eval_20260417/title_on_desc_p05/sft`
- SFT log（监督微调日志）：
  - `/home/leejt/OneRec/logs/experiment_mgr_sid_r650a_sft_20260417.log`
- evaluate log（评测日志）：
  - `/home/leejt/OneRec/logs/experiment_mgr_sid_r650a_eval_20260417.log`
- evaluate result（评测结果）：
  - `/home/leejt/OneRec/results/experiments/mgr_sid_r650a_sft_eval_20260417/final_result_sft_mgr_r650a_title_on_desc_p05_Industrial_and_Scientific.json`

## Final Metrics（最终指标）

- `NDCG@1/3/5/10/20/50`
  - `0.06530 / 0.08132 / 0.08778 / 0.09518 / 0.10491 / 0.11654`
- `HR@1/3/5/10/20/50`
  - `0.06530 / 0.09354 / 0.10920 / 0.13236 / 0.17075 / 0.22987`
- constraint invalid total（约束失配总数）
  - `0`
- SFT stop epoch（监督微调停止轮次）
  - `5.5`
- SFT final eval loss（监督微调最终验证损失）
  - `1.6138453483581543`
- SFT final train loss（监督微调最终训练损失）
  - `0.4548963058436887`

## Verdict（裁决）

这一步是 `R650a` 的第一个 downstream test（下游测试）。由于 `R630c` 曾经 tokenizer-side collision（分词器侧冲突）同样达到 `11 / 3686` 但下游为负，本实验的关键不是看 tokenizer metric（分词器指标），而是看 SFT/evaluate（监督微调/评测）是否真正提升 `NDCG@10` 和 `HR@10`。

结论是：`R650a` 为负结果。

- 相比 `R640c` carrier-only smoothness（仅图载体加平滑监督），`R650a` 有小幅恢复：
  - `NDCG@10`: `0.09306 -> 0.09518`
  - `HR@10`: `0.13126 -> 0.13236`
- 相比 `R630c` mid-only pull-push（仅中层拉近推远），`R650a` 也略好：
  - `NDCG@10`: `0.09261 -> 0.09518`
  - `HR@10`: `0.12972 -> 0.13236`
- 但相比 current `v2_on_p05`（当前 `v2_on_p05`）SFT，仍然明显落后：
  - `NDCG@10`: `0.10271 -> 0.09518`
  - `HR@10`: `0.14626 -> 0.13236`
- 相比 strongest original SFT（原版最强 SFT），也明显落后：
  - `NDCG@10`: `0.10372 -> 0.09518`
  - `HR@10`: `0.15089 -> 0.13236`

因此：

> `R650a` 不能推进到 `RL`（强化学习）。它说明 Seq2Graph-lite high-order carrier（轻量 Seq2Graph 高阶载体）放进 explicit push-pull（显式拉近推远）后，比 carrier-only（仅图载体）和原始 `R630c` 略有恢复，但恢复幅度远远不够，当前实现不能成为新的 strongest line（最强主线）。
