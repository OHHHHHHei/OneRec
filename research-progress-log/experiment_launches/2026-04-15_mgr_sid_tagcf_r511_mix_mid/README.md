# 2026-04-15 TAGCF 支链 `R511`：`G_mid <- mix(fagsp_mid_base, G_attr_fused)`

## 目的

这是 `TAGCF` 支链在 `R510` 之后的第二轮 tokenizer（分词器）实验。

它回答的问题是：

> 属性拓扑图更适合作为中尺度图的 **补充信号**，还是更适合作为 **完全替换信号**？

## 实验定义

- 运行编号：`R511`
- 变体：
  - `L1 <- coarse_purified`
  - `L2 <- 0.5 * fagsp_mid_base + 0.5 * G_attr_fused`
  - `L3 <- local_purified`
- 训练方式：
  - 从头训练
  - 不用 `warm-start`（热启动）
  - 不加额外 `retention`（保持）或 `anchor`（锚定）项

## 为什么这样设计

`R510` 的结果说明：

- 纯属性图替换版不是废分支
- 但结构结果明显是 mixed（混合）

因此，下一步最自然的不是直接继续推纯替换，而是先回答：

> 把属性拓扑当作一个增量图视图，而不是彻底替代原中图，会不会更稳。

## 配置

- [sid_train_industrial_mgr_sid_tagcf_r511_mix_mid.yaml](/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_tagcf_r511_mix_mid.yaml)

## 属性图来源

- `R501 fused` 输出：
  [item_attribute_graph.npz](/home/leejt/OneRec/results/tagcf_branch/20260415_m0_attribute_graphs/R501_attr_fused_textphrase/item_attribute_graph.npz)

## 训练脚本

- [experiment_mgr_sid_tagcf_r511_mix_mid_train.sh](/home/leejt/OneRec/scripts/experiment_mgr_sid_tagcf_r511_mix_mid_train.sh)

## 输出根目录

- `/data/leejt/OneRec/output_weights/experiments/mgr_sid_tagcf_branch_20260415/industrial_r511_mix_mid`

## 启动状态

- 日期：`2026-04-15`
- tmux：`mgr_tagcf_r511_mix_mid`
- GPU：`2`
- 当前状态：`COMPLETED`
- 日志：
  - `/home/leejt/OneRec/logs/experiment_mgr_sid_tagcf_r511_mix_mid_20260415.log`

## Sanity 状态

- 1-epoch sanity：`PASSED`
- sanity 输出：
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_tagcf_branch_20260415/sanity_r511_mix_mid/Apr-15-2026_09-32-44/summary.json`
- sanity 读数：
  - `total_loss = 3.3436`
  - `collision = 0.038253`
  - 说明：
    - 混合后的 `G_mid`（中尺度图）已成功接入训练
    - 当前配置和设备映射都正常

## 当前判断

`R511` 是一枪非常关键的分支实验，因为它比 `R510` 更直接回答：

- 属性拓扑是否有价值
- 以及它的最佳角色到底是 replacement（替换）还是 additive graph signal（增量图信号）

## 训练结果

- 运行目录：
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_tagcf_branch_20260415/industrial_r511_mix_mid/Apr-15-2026_09-33-42`
- 最佳 train-side `collision`（训练侧冲突率）：
  - `0.1825827455`
- 最佳轮次：
  - `epoch = 9649`
- 最佳 checkpoint（检查点）：
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_tagcf_branch_20260415/industrial_r511_mix_mid/Apr-15-2026_09-33-42/best_collision_model.pth`
- 训练 summary（摘要）：
  - [summary.json](/data/leejt/OneRec/output_weights/experiments/mgr_sid_tagcf_branch_20260415/industrial_r511_mix_mid/Apr-15-2026_09-33-42/summary.json)

## Generate 结果

- 生成索引：
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_tagcf_branch_20260415/generated_indices/Industrial_and_Scientific.tagcf_r511_mix_mid.index.json`
- 生成 summary（摘要）：
  - [R511_generate_summary.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-15_mgr_sid_tagcf_r511_mix_mid/R511_generate_summary.json)
- 最终 generated `collision`（生成后冲突率）：
  - `0.0048833424`
  - 即 `18 / 3686`
- `max_conflict`（最大冲突簇大小）：
  - `3`
- `collision_rounds_used`（冲突修补轮数）：
  - `20`

## 当前判断

`R511` 没有验证出“混合版比纯替换版更稳”：

- `R510`：`11 / 3686`
- `R511`：`18 / 3686`

所以在最直接的生成后 SID 质量上，`0.5 / 0.5` 的中图混合版当前是明显退步的。
