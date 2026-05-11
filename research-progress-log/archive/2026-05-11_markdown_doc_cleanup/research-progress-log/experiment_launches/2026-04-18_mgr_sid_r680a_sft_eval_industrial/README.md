# 2026-04-18 `R680a` SFT/Evaluate（监督微调/评测）

Status（状态）: `completed_negative（已完成，负结果）`
Launch date（启动日期）: `2026-04-18`

## 目的

`R680a` tokenizer/generate（分词器训练与生成）已经完成，并且当前最重要的问题不是再做新的 tokenizer-side proxy（分词器侧代理指标），而是看它在当前最强下游 recipe（配方）下能不能真正转化成更好的 evaluate（评测）结果。

因此这一步把 `R680a` 接入当前固定 recipe（配方）：

- `title_history2sid_on`
- `desc_align_p05`

本阶段已经完成 `SFT -> evaluate`（监督微调到评测）全链路，并给出正式 downstream verdict（下游裁决）。

## 对应输入

- tokenizer index（分词器索引）：
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r680_l1_smooth_l2_contrastive_multihop_20260418/generated_indices/Industrial_and_Scientific.r680a_l1_smooth_l2_contrastive_multihop.index.json`
- prepared data root（已准备数据根目录）：
  - `/home/leejt/OneRec/data_experiment/Amazon/r680a_l1_smooth_l2_contrastive_multihop`

## 对应文件

- SFT config（监督微调配置）：
  - [sft_industrial_mgr_r680a_title_on_desc_p05_2gpu.yaml](/home/leejt/OneRec/config/experiments/sft_industrial_mgr_r680a_title_on_desc_p05_2gpu.yaml)
- evaluate config（评测配置）：
  - [evaluate_industrial_mgr_r680a_title_on_desc_p05_2gpu.yaml](/home/leejt/OneRec/config/experiments/evaluate_industrial_mgr_r680a_title_on_desc_p05_2gpu.yaml)
- SFT chain script（监督微调链路脚本）：
  - [experiment_mgr_sid_r680a_sft_chain.sh](/home/leejt/OneRec/scripts/experiment_mgr_sid_r680a_sft_chain.sh)
- launch script（启动脚本）：
  - [launch_mgr_sid_r680a_sft_tmux.sh](/home/leejt/OneRec/scripts/launch_mgr_sid_r680a_sft_tmux.sh)

## 2 卡批大小对齐说明

这次不是简单把 `4` 卡训练缩成 `2` 卡，而是显式保持 `effective batch`（有效批大小）不变。

当前 `SFT pipeline`（监督微调流水线）在 `DDP`（分布式数据并行）下使用：

```text
gradient_accumulation_steps = batch_size // micro_batch_size // world_size
```

因此：

- 原 `4` 卡配置：
  - `batch_size = 1024`
  - `micro_batch_size = 2`
  - `world_size = 4`
  - `gradient_accumulation_steps = 128`
  - `effective batch = 2 x 4 x 128 = 1024`
- 当前 `2` 卡配置：
  - `batch_size = 1024`
  - `micro_batch_size = 2`
  - `world_size = 2`
  - `gradient_accumulation_steps = 256`
  - `effective batch = 2 x 2 x 256 = 1024`

也就是说，这次主要变化是 runtime topology（运行拓扑）和 wall-clock speed（墙钟速度），不是优化时看到的有效批大小。

## 当前状态

- `tmux`（终端复用器） session（会话）：
  - `mgr_r680a_sft_2gpu`，已结束
- runtime GPUs（运行显卡）：
  - `CUDA_VISIBLE_DEVICES=5,7`
- prepared data variant（已准备数据变体）：
  - `r680a_l1_smooth_l2_contrastive_multihop`
- SFT output（监督微调输出）：
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r680a_sft_eval_20260418/title_on_desc_p05_2gpu/sft`
- SFT log（监督微调日志）：
  - `/home/leejt/OneRec/logs/experiment_mgr_sid_r680a_sft_20260418.log`
- evaluate log（评测日志）：
  - `/home/leejt/OneRec/logs/experiment_mgr_sid_r680a_eval_20260418.log`
- evaluate result（评测结果）：
  - `/home/leejt/OneRec/results/experiments/mgr_sid_r680a_sft_eval_20260418/final_result_sft_mgr_r680a_title_on_desc_p05_2gpu_Industrial_and_Scientific.json`

## SFT Summary（监督微调摘要）

- W&B run（实验追踪）：
  - `sft_mgr_r680a_title_on_desc_p05_2gpu_industrial`
  - `run_id = mduahhd0`
- best checkpoint（最佳检查点）：
  - `checkpoint-456`
- best eval loss（最佳验证损失）：
  - `1.522908329963684`
- final logged eval loss（最终记录验证损失）：
  - `1.575154423713684`
- final train loss（最终训练损失）：
  - `0.4730283639006067`
- stop epoch（停止轮次）：
  - `5.5`

## Final Metrics（最终指标）

- `NDCG@1/3/5/10/20/50`
  - `0.06883 / 0.08497 / 0.09038 / 0.09864 / 0.10776 / 0.11990`
- `HR@1/3/5/10/20/50`
  - `0.06883 / 0.09707 / 0.11008 / 0.13567 / 0.17207 / 0.23362`
- constraint invalid total（约束失配总数）
  - `0`

## Verdict（裁决）

结论是：`R680a` 比近期多个负分支都更强，但仍然没有超过 current `v2_on_p05`（当前 `v2_on_p05`），因此这次 downstream verdict（下游裁决）仍然是负结果，不推进到 `RL`（强化学习）。

- 相比 `R650a`，`R680a` 全面更好：
  - `NDCG@10`: `0.09518 -> 0.09864`
  - `HR@10`: `0.13236 -> 0.13567`
- 相比 `R640c`，提升也更明显：
  - `NDCG@10`: `0.09306 -> 0.09864`
  - `HR@10`: `0.13126 -> 0.13567`
- 但相比 current `v2_on_p05`，仍然落后：
  - `NDCG@10`: `0.10271 -> 0.09864`
  - `HR@10`: `0.14626 -> 0.13567`
- 相比 recipe-aligned original baseline（配方对齐原版基线）`title_history2sid_on + desc_align_p05`，也仍然略低：
  - `NDCG@10`: `0.10183 -> 0.09864`
  - `HR@10`: `0.14626 -> 0.13567`

这说明：

- `L2 contrastive interface`（第二层对比式接口）这条线不是无效的，它确实比 `R640c / R650a` 这类近期负分支更有竞争力。
- 但当前这版 `R680a` 还不足以支撑“找到了更好的 SID space（SID 空间）”这个更强结论。
- 更细地看，`R680a` 在更浅的 top-k（前 k）位置恢复得不错，例如：
  - `NDCG@1 = 0.06883`
  - `NDCG@3 = 0.08497`
  - `HR@3 = 0.09707`
- 但这种改善没有延续到 `@10`，说明它可能改善了更早期的 routing（路由）选择，却没有把更深层次的 candidate quality（候选质量）一起拉起来。
