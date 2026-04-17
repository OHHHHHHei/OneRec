# 2026-04-17 `R640c` SFT/Evaluate（监督微调与评测）

Status（状态）: `snapshot（快照）`
Snapshot date（快照日期）: `2026-04-17`

## 目的

这是 `R640c = Seq2Graph-lite rel_masked`（轻量 `Seq2Graph` 可靠性感知加掩码版）通过 tokenizer-side（分词器侧）灾难性失败筛选后的正式 downstream adjudication（下游裁决）。

对应 `R645` 的执行对象就是：

- `L1 <- coarse_seq2g_rel_masked`
- `L2 <- fagsp_mid_seq2g_rel_masked`
- `L3 <- local_purified`

下游 recipe（配方）保持当前 `v2` strongest graph-aware recipe（最强图感知配方）不变：

- `title_history2sid_on + desc_align_p05`

## 对应文件

- 数据转换：
  - [experiment_mgr_sid_prepare_data.py](/home/leejt/OneRec/scripts/experiment_mgr_sid_prepare_data.py)
- 链路脚本：
  - [experiment_mgr_sid_r640c_sft_eval_chain.sh](/home/leejt/OneRec/scripts/experiment_mgr_sid_r640c_sft_eval_chain.sh)
- `SFT` 配置：
  - [sft_industrial_mgr_r640c_title_on_desc_p05.yaml](/home/leejt/OneRec/config/experiments/sft_industrial_mgr_r640c_title_on_desc_p05.yaml)
- `evaluate`（评测）配置：
  - [evaluate_industrial_mgr_r640c_title_on_desc_p05.yaml](/home/leejt/OneRec/config/experiments/evaluate_industrial_mgr_r640c_title_on_desc_p05.yaml)

## 结果

当前链路已经完成：

- status（状态）：
  - `FINISHED_NEGATIVE`
- `SFT`（监督微调）：
  - final eval loss（最终验证损失）: `1.5987722873687744`
  - final train loss（最终训练损失）: `0.4760145429028659`
  - stop epoch（停止轮次）: `5.5`
- `evaluate`（评测）：
  - `NDCG@1 = 0.06265167`
  - `NDCG@3 = 0.07810151`
  - `NDCG@5 = 0.08528820`
  - `NDCG@10 = 0.09305728`
  - `NDCG@20 = 0.10191194`
  - `NDCG@50 = 0.11386779`
  - `HR@1 = 0.06265167`
  - `HR@3 = 0.08956541`
  - `HR@5 = 0.10699316`
  - `HR@10 = 0.13125965`
  - `HR@20 = 0.16633576`
  - `HR@50 = 0.22678138`
- constraint invalid total（约束非法总数）:
  - `0`

## 结论

- `R640c` 低于 current `v2_on_p05`（当前 `v2_on_p05`）SFT：`0.09306` vs `0.10271` on `NDCG@10`。
- `R640c` 也低于 strongest original SFT（原版最强 SFT）：`0.09306` vs `0.10372` on `NDCG@10`。
- 因此 `Seq2Graph-lite rel_masked`（轻量 `Seq2Graph` 可靠性感知加掩码版）作为 carrier-only smoothness（仅图载体加平滑监督）不应推进到 `RL`（强化学习）。
- 这个结果不直接否定 high-order carrier + explicit push-pull（高阶载体 + 显式推远拉近）方向；它只说明“只换图并继续 attraction-only graph smoothness（仅吸引式图平滑）”不够。
