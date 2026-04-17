# 2026-04-18 `R670a` Clean L1 Semantic + L2 Push-Pull（干净 L1 语义 + L2 推远拉近）

Status（状态）: `completed_negative（已完成负结果）`
Launch date（启动日期）: `2026-04-18`

## 目的

`R670a` 用来验证一个更干净的层级分工假设：

- `L1`（第一层）应该主要承担 semantic-dominant coarse routing（语义主导的粗路由）。
- `L2`（第二层）再承担 collaborative refinement（协同细分），包括协同图上的 pull（拉近）和语义近但协同弱物品对的 push（推远）。
- `L2` 的辅助 loss（损失）不应该无保护地反传去改写 `L1` 的语义入口，所以本实验打开 stop-gradient prefix（前缀停梯度）。

这个实验**不是**直接限制 active L1 code count（活跃第一层码数量）。我们不告诉 tokenizer（分词器）必须用多少个 `L1` code（第一层码），而是让它通过更清晰的训练信号自主学习一个更适合下游 LLM（大语言模型）学习的粗层组织。

## 设计

`R670a = RQ + L1 high-confidence semantic pull + L2 collaborative pull/push + stop-gradient prefix`
（残差量化 + 第一层高置信语义拉近 + 第二层协同拉近/推远 + 前缀停梯度）。

核心 loss（损失）为：

```text
L_total =
  L_recon + L_rq
  + lambda_l1_sem * L_smooth(h1, G_l1_sem_high_conf)
  + lambda_l2_pull * L_smooth(stopgrad(q1) + q2, G_mid_base)
  + alpha_l2_push * L_sep(stopgrad(q1) + q2, P_mid_weak)
```

其中：

- `h1 = q1`
- `h2 = stopgrad(q1) + q2`
- `G_l1_sem_high_conf`（第一层高置信语义图）只连接同品牌、同类型或语义强近邻的高置信物品对。
- `G_mid_base`（基础中层协同图）使用 `fagsp_mid_base`，不使用 `Seq2Graph-lite`（轻量 Seq2Graph），让这个实验先回到更干净的 v2 图基座。
- `P_mid_weak`（中层弱协同推远物品对）使用 semantic-near + `fagsp_mid_base` weak（语义近 + 基础中图弱连接）规则生成。

## 配置

- `coarse_weight = 0.0`
- `mid_weight = 0.15`
- `local_weight = 0.0`
- `semantic_coarse_weight = 0.08`
- `semantic_mid_weight = 0.0`
- `semantic_external_graph_path = R670a_l1_high_conf_semantic_graph.npz`
- `mid_view_name = fagsp_mid_base`
- `hierarchy_stopgrad_previous_levels = true`
- `selective_separation_weight = 0.01`
- `selective_separation_margin = 0.15`
- `selective_separation_levels = [2]`
- `selective_separation_use_pair_reliability = true`
- `selective_separation_use_ambiguity_scaling = false`

## 对应文件

- tokenizer config（分词器配置）：
  - [sid_train_industrial_mgr_sid_r670a_clean_l1_semantic_l2_push_pull.yaml](/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_r670a_clean_l1_semantic_l2_push_pull.yaml)
- L1 semantic graph builder（第一层语义图构建脚本）：
  - [experiment_mgr_sid_r670a_l1_semantic_graph.py](/home/leejt/OneRec/scripts/experiment_mgr_sid_r670a_l1_semantic_graph.py)
- train/generate script（训练生成脚本）：
  - [experiment_mgr_sid_r670a_clean_l1_semantic_l2_push_pull_train_generate.sh](/home/leejt/OneRec/scripts/experiment_mgr_sid_r670a_clean_l1_semantic_l2_push_pull_train_generate.sh)
- launch script（启动脚本）：
  - [launch_mgr_sid_r670_clean_l1_semantic_l2_push_pull_tmux.sh](/home/leejt/OneRec/scripts/launch_mgr_sid_r670_clean_l1_semantic_l2_push_pull_tmux.sh)

## 运行产物

- `tmux`（终端复用器） session（会话）：
  - `mgr_r670a_clean_l1_semantic_l2_push_pull`，已结束
- runtime GPU（运行显卡）：
  - `CUDA_VISIBLE_DEVICES=7`
- output root（输出根目录）：
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r670_clean_l1_semantic_l2_push_pull_20260418`
- generated SID index（生成语义标识索引）：
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r670_clean_l1_semantic_l2_push_pull_20260418/generated_indices/Industrial_and_Scientific.r670a_clean_l1_semantic_l2_push_pull.index.json`
- log（日志）：
  - `/home/leejt/OneRec/logs/experiment_mgr_sid_r670a_clean_l1_semantic_l2_push_pull_20260418.log`

## Launch Artifacts（启动产物）

- L1 high-confidence semantic graph（第一层高置信语义图）：
  - `nnz_after_symmetry = 36753`
  - `positive_rows_before_symmetry = 2797`
  - `self_loop_rows_before_symmetry = 889`
  - `unique_non_self_pairs = 17932`
- L2 push pair source（第二层推远物品对来源）：
  - `mid_view_name = fagsp_mid_base`
  - `semantic_pair_count = 82596`
  - `weak_pair_count = 1211`
  - `weak_pair_item_coverage_rate = 0.2797069995`
  - `weak_threshold = 0.0016112356`

## Early Training Health（早期训练健康检查）

- 训练已进入主循环，GPU（显卡）7 已被占用。
- 早期 collision（冲突率）仍然很高，但在下降：
  - epoch（轮次）49: `0.998644`
  - epoch（轮次）199: `0.996744`
  - epoch（轮次）399: `0.962561`
  - epoch（轮次）499: `0.914542`
  - epoch（轮次）949: `0.817417`
- 最终结果已经证明：这不是短期抖动，而是一路恢复不足的真实负信号。

## Final Tokenizer Result（最终分词器结果）

- run_dir（运行目录）：
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r670_clean_l1_semantic_l2_push_pull_20260418/industrial_r670a_clean_l1_semantic_l2_push_pull/Apr-18-2026_01-16-51`
- best train collision（训练最佳冲突率）：
  - `0.4850786761`
- best epoch（最佳轮次）：
  - `8549`
- best loss（最佳损失）：
  - `0.4683470130`
- generated collision（生成后冲突）：
  - `162 / 3686 = 0.0439500814`
- max conflict（最大冲突簇）：
  - `35`
- active L1（活跃第一层码）：
  - `19`
- mean / median / max L1 bucket size（第一层桶大小均值 / 中位数 / 最大值）：
  - `194.0 / 203 / 310`
- unique L2 pairs（唯一第二层前缀数）：
  - `375`

## Interpretation（结果解读）

- 这是一次明确的 tokenizer collapse（分词器塌缩），不是边缘负结果。
- 最关键的结构信号不是单独的 `collision`（冲突率），而是前缀空间被压扁了：
  - `active L1`（活跃第一层码）从 `R660a` 的 `181` 直接掉到 `19`
  - `unique L2 pairs`（唯一第二层前缀数）从 `2598` 掉到 `375`
- 这说明 `L1` 高置信语义拉近 + stop-gradient prefix（前缀停梯度）的组合把前两层码本过度凝聚了，`L2` 没有真正获得健康的协同细分空间。
- 因为 tokenizer（分词器）层面已经严重退化，这个分支**不建议推进到 SFT（监督微调）**。

## Family Coverage Diagnosis（物品族覆盖诊断）

新增分析产物：

- `R670A_FAMILY_L1_SPREAD_COMPARISON.csv`
- `R670A_L1_BUCKET_FAMILY_COMPOSITION.csv`
- `R670A_FOCUS_FAMILY_L1_BUCKETS.csv`

核心现象：

- 之前被拆得过散的家族，确实明显回并了。
- 但不是“刚好回到合理粗粒度”，而是“回并过头并且开始互相混桶”。

重点家族对比：

- `3d_filament`（3D 打印耗材）：
  - `orig / r650 / r660 / r670` 的 `L1` 数量为 `21 / 39 / 38 / 14`
  - 其中 `240 / 386 = 62.2%` 的物品集中到一个几乎纯净的 `L1` 桶 `<a_192>`，该桶内 `3d_filament` 占比 `95.2%`
  - 说明这类高度同质家族确实被拉回来了
- `tape`（胶带）：
  - `22 / 47 / 41 / 16`
  - `163 / 217 = 75.1%` 集中在 `<a_41>`，该桶内 `tape` 占比 `89.6%`
  - 也说明“过散 -> 回并”这个方向对这类家族是有效的
- `adhesive_epoxy`（胶黏剂/环氧）：
  - `14 / 32 / 28 / 11`
  - `96 / 125 = 76.8%` 集中到 `<a_69>`
  - 但 `<a_69>` 这个桶本身只有 `41.9%` 是 `adhesive_epoxy`，其余混入了很多 `other`
- `connector_fitting`（连接件）：
  - `19 / 49 / 43 / 14`
  - 虽然数量回到了原版附近，但主要分散在两个大桶：`107` 个在 `<a_122>`，`91` 个在 `<a_115>`
  - 其中 `<a_115>` 只有 `43.5%` 是 `connector_fitting`，混桶已经比较明显
- `gauge_meter`（仪表/测量器）：
  - `29 / 85 / 72 / 16`
  - 数量收得最猛，但并没有形成干净主桶，最大的单桶只承载 `50 / 338 = 14.8%`
  - 说明它不是“被重新组织好了”，而是被压进了多个混合大桶里

整体纯度结论：

- `R670a` 的 `active L1 = 19`，但 weighted top-family purity（按桶大小加权的头号家族纯度）只有 `0.693`
- 这低于：
  - original（原版）`0.748`
  - `v2` `0.797`
  - `r650` `0.779`
  - `r660` `0.768`

所以最准确的结论是：

- `R670a` 证明了“让 L1（第一层）更凝聚”这个方向本身是有效信号
- 但它同时证明了：当前这套做法把很多家族**一起压进了少数超大 L1 桶**，导致覆盖变粗了，却不够干净

## 判读方式

- 如果 `R670a` 的 tokenizer/generate（分词器训练与生成）不灾难性塌缩，并且 `SFT -> evaluate`（监督微调到评测）明显优于 `R650a/R660a`，说明“L1 语义入口 + L2 协同推远拉近 + 前缀停梯度”比继续堆全套约束更有希望。
- 如果 tokenizer（分词器）侧可用但下游仍差，说明 `push-pull`（推远拉近）物品对选择或 L1 高置信语义图仍需要重新设计。
- 如果 tokenizer（分词器）侧直接塌缩，优先怀疑 `semantic_coarse_weight = 0.08` 或 stop-gradient prefix（前缀停梯度）破坏了层间协调。

本次实验已经落在第三种情况：tokenizer（分词器）侧直接塌缩，因此不进入后续 `SFT -> evaluate`（监督微调到评测）。
