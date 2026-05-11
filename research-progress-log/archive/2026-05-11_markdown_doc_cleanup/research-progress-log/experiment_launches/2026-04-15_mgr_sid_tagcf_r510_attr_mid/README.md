# 2026-04-15 TAGCF 支链 `R510`：`G_mid <- G_attr_fused`

## 目的

这是 `TAGCF` 支链的第一轮 tokenizer（分词器）训练实验。

它回答的问题非常直接：

> 如果我们把当前 `G_mid`（中尺度图）直接替换成 `R501_attr_fused_textphrase` 构造出的属性拓扑图，能不能形成一个值得继续推下游的 `SID codebook space`（SID 码本空间）？

## 实验定义

- 运行编号：`R510`
- 变体：
  - `L1 <- coarse_purified`
  - `L2 <- G_attr_fused`
  - `L3 <- local_purified`
- 训练方式：
  - 从头训练
  - 不用 `warm-start`（热启动）
  - 不加额外 `retention`（保持）或 `anchor`（锚定）项

## 为什么这样设计

这轮的目标是尽量干净地回答：

> 图载体本身换掉以后，会发生什么？

所以它故意不叠加：

- stage-3 的 prefix 控制
- teacher-guided retention（教师引导保持）
- codebook anchor（码本锚定）

## 配置

- [sid_train_industrial_mgr_sid_tagcf_r510_attr_mid.yaml](/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_tagcf_r510_attr_mid.yaml)

## 属性图来源

- `R501 fused` 输出：
  [item_attribute_graph.npz](/home/leejt/OneRec/results/tagcf_branch/20260415_m0_attribute_graphs/R501_attr_fused_textphrase/item_attribute_graph.npz)

## 训练脚本

- [experiment_mgr_sid_tagcf_r510_attr_mid_train.sh](/home/leejt/OneRec/scripts/experiment_mgr_sid_tagcf_r510_attr_mid_train.sh)

## 输出根目录

- `/data/leejt/OneRec/output_weights/experiments/mgr_sid_tagcf_branch_20260415/industrial_r510_attr_mid`

## 启动状态

- 日期：`2026-04-15`
- tmux：`mgr_tagcf_r510_attr_mid`
- GPU：`7`
- 当前状态：`COMPLETED`
- train pid：
  - `88152`
- 日志：
  - `/home/leejt/OneRec/logs/experiment_mgr_sid_tagcf_r510_attr_mid_20260415.log`

## 启动修正说明

第一次正式启动时曾经立即失败，原因不是方法问题，而是设备编号配置错误：

- 启动脚本设置的是：
  - `CUDA_VISIBLE_DEVICES=7`
- 但配置里误写成了：
  - `device: cuda:7`

在这种情况下，进程内部只能看到一张卡，它的内部编号应该是 `cuda:0`。  
这个问题已经修正，并已重新启动正式训练。

## Sanity 状态

- 1-epoch sanity：`PASSED`
- sanity 输出：
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_tagcf_branch_20260415/sanity_r510_attr_mid/Apr-15-2026_00-19-04/summary.json`
- sanity 读数：
  - `total_loss = 3.3391`
  - `collision = 0.038253`
  - 说明：
    - 外部属性图成功接入 `L2`
    - 训练和评估都能正常走通

## 训练结果

- 运行目录：
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_tagcf_branch_20260415/industrial_r510_attr_mid/Apr-15-2026_00-21-49`
- 最佳 train-side `collision`（训练侧冲突率）：
  - `0.1134020619`
- 最佳轮次：
  - `epoch = 9549`
- 最佳 checkpoint（检查点）：
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_tagcf_branch_20260415/industrial_r510_attr_mid/Apr-15-2026_00-21-49/best_collision_model.pth`
- 训练 summary（摘要）：
  - [summary.json](/data/leejt/OneRec/output_weights/experiments/mgr_sid_tagcf_branch_20260415/industrial_r510_attr_mid/Apr-15-2026_00-21-49/summary.json)

## Generate 结果

- 生成索引：
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_tagcf_branch_20260415/generated_indices/Industrial_and_Scientific.tagcf_r510_attr_mid.index.json`
- 生成 summary（摘要）：
  - [R510_generate_summary.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-15_mgr_sid_tagcf_r510_attr_mid/R510_generate_summary.json)
- 最终 generated `collision`（生成后冲突率）：
  - `0.0029842648`
  - 即 `11 / 3686`
- `max_conflict`（最大冲突簇大小）：
  - `2`
- `collision_rounds_used`（冲突修补轮数）：
  - `20`

## 结构诊断

基于 `v2 offline_combined` 的同口径 `local ambiguity analysis`（局部歧义分析）：

- 对比文件：
  - [R512_v2_vs_r510_attr_mid_local_ambiguity.md](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-15_mgr_sid_tagcf_r510_attr_mid/R512_v2_vs_r510_attr_mid_local_ambiguity.md)
  - [R512_v2_vs_r510_attr_mid_local_ambiguity.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-15_mgr_sid_tagcf_r510_attr_mid/R512_v2_vs_r510_attr_mid_local_ambiguity.json)

核心读数：

- mean target `l2` leaf count（测试目标平均 `l2` 叶子数）：
  - `4.3422 -> 3.6848`
- fraction targets in multi-leaf `same_l2`（测试目标落在多叶 `same_l2` 的比例）：
  - `0.4873 -> 0.5277`
- fraction targets in `l2 >= 4`（测试目标落在深拥挤 `l2` 的比例）：
  - `0.2228 -> 0.2285`
- mean target `l3` entropy under `l2`（测试目标在 `l2` 下的平均 `l3` 熵）：
  - `1.1001 -> 1.0898`

## 当前判断

`R510` 的画像是一个**混合结果**：

- 训练侧 `collision`（冲突率）不理想；
- 但 generate 后最终 `collision` 回到了当前支线常见的 `11 / 3686`；
- 相对 `v2`，它把部分目标 item（物品）的 `same_l2` 叶子数压低了，
  但 multi-leaf `same_l2` 和深拥挤 `l2` 的比例没有一起变好。

因此，到目前为止，`R510` 不能被读成一个清晰的 tokenizer-side（分词器侧）正结果。
更准确地说，它是一个：

- `generate collision`（生成后冲突率）可接受，
- 结构表现混合，
- 仍需谨慎决定是否值得继续推下游的属性图替换分支。
