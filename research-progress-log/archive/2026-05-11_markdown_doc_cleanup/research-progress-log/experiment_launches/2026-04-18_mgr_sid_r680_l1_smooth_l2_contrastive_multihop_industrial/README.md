# 2026-04-18 `R680a` L1 Smooth + L2 Contrastive + L3 Smooth（第一层平滑 + 第二层对比式 + 第三层平滑）

Status（状态）: `tokenizer_generated（分词器已生成）`
Launch date（启动日期）: `2026-04-18`

## 目的

`R680a` 是当前最干净的一次 `L2` interface test（第二层接口测试）：

- `L1`（第一层）回到 `coarse_purified`（净化粗图） smoothness（平滑）监督
- `L2`（第二层）不再使用 graph smoothness（图平滑），而是改为：
  - `local_multihop`（局部多跳图） positive pull（正样本拉近）
  - `semantic-near + multihop-weak`（语义近 + 多跳弱连接） negative push（负样本推远）
- `L3`（第三层）继续使用 `local_purified`（净化局部图） smoothness（平滑）
- 打开 stop-gradient prefix（前缀停梯度），让 `L2` 的对比式监督主要塑造 `q2`，不无保护地改写 `q1`

核心问题是验证：

- 当前瓶颈是否主要在 `L2` supervision interface（第二层监督接口）
- 而不只是继续改 graph carrier（图载体）来源

## 设计

`R680a = RQ + L1 coarse smooth + L2 multihop pull/push + L3 local smooth + stop-gradient prefix`
（残差量化 + 第一层粗图平滑 + 第二层多跳拉近/推远 + 第三层局部平滑 + 前缀停梯度）。

训练目标可写为：

```text
L_total =
  L_recon + L_rq
  + lambda_c * L_smooth(h1, G_coarse_purified)
  + lambda_l2_pull * L_pull(stopgrad(q1) + q2, G_local_multihop)
  + lambda_l2_push * L_sep(stopgrad(q1) + q2, P_semantic_near_multihop_weak)
  + lambda_l * L_smooth(stopgrad(q1 + q2) + q3, G_local_purified)
```

其中：

- `h1 = q1`
- `h2 = stopgrad(q1) + q2`
- `h3 = stopgrad(q1 + q2) + q3`
- `L_pull`（拉近）直接使用 `local_multihop`（局部多跳图）边权
- `L_sep`（分离）复用现有 selective separation（选择性分离）模块，只在 `L2` 生效

## 配置

- `coarse_view_name = coarse_purified`
- `mid_view_name = local_multihop`
- `local_view_name = local_purified`
- `hierarchy_stopgrad_previous_levels = true`
- `coarse_weight = 0.05`
- `mid_weight = 0.0`
- `local_weight = 0.05`
- `l2_contrastive_pull_weight = 0.15`
- `semantic_coarse_weight = 0.0`
- `semantic_mid_weight = 0.0`
- `selective_separation_weight = 0.01`
- `selective_separation_margin = 0.15`
- `selective_separation_levels = [2]`
- `selective_separation_use_pair_reliability = true`
- `selective_separation_use_ambiguity_scaling = false`

## 对应文件

- tokenizer config（分词器配置）：
  - [sid_train_industrial_mgr_sid_r680a_l1_smooth_l2_contrastive_multihop.yaml](/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_r680a_l1_smooth_l2_contrastive_multihop.yaml)
- pair source builder（物品对来源脚本）：
  - [experiment_mgr_sid_mid_pull_push_pair_source.py](/home/leejt/OneRec/scripts/experiment_mgr_sid_mid_pull_push_pair_source.py)
- train/generate script（训练生成脚本）：
  - [experiment_mgr_sid_r680a_l1_smooth_l2_contrastive_multihop_train_generate.sh](/home/leejt/OneRec/scripts/experiment_mgr_sid_r680a_l1_smooth_l2_contrastive_multihop_train_generate.sh)
- launch script（启动脚本）：
  - [launch_mgr_sid_r680a_l1_smooth_l2_contrastive_multihop_tmux.sh](/home/leejt/OneRec/scripts/launch_mgr_sid_r680a_l1_smooth_l2_contrastive_multihop_tmux.sh)

## 运行产物

- `tmux`（终端复用器） session（会话）：
  - `mgr_r680a_l1_smooth_l2_contrastive_multihop`
- runtime GPU（运行显卡）：
  - `CUDA_VISIBLE_DEVICES=7`
- output root（输出根目录）：
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r680_l1_smooth_l2_contrastive_multihop_20260418`
- generated SID index（生成语义标识索引）：
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r680_l1_smooth_l2_contrastive_multihop_20260418/generated_indices/Industrial_and_Scientific.r680a_l1_smooth_l2_contrastive_multihop.index.json`
- log（日志）：
  - `/home/leejt/OneRec/logs/experiment_mgr_sid_r680a_l1_smooth_l2_contrastive_multihop_20260418.log`

## Launch Artifacts（启动产物）

- pair source（物品对来源）：
  - `mid_view_name = local_multihop`
  - `semantic_pair_count = 82596`
  - `weak_pair_count = 1738`
  - `weak_pair_item_coverage_rate = 0.4880629409`
  - `weak_threshold = 0.0028070429`
  - `semantic_sim_mean = 0.0323920565`
  - `mid_affinity_mean = 0.0016472911`
  - `reliability_mean = 0.0133802006`
  - summary file（摘要文件）：
    - [R680a_pair_source_summary.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-18_mgr_sid_r680_l1_smooth_l2_contrastive_multihop_industrial/R680a_pair_source_summary.json)

## Early Training Health（早期训练健康检查）

- `tmux`（终端复用器） session 已创建并保持运行中
- 训练日志已进入持续训练阶段：
  - [experiment_mgr_sid_r680a_l1_smooth_l2_contrastive_multihop_20260418.log](/home/leejt/OneRec/logs/experiment_mgr_sid_r680a_l1_smooth_l2_contrastive_multihop_20260418.log)
- 当前可见 `collision`（冲突率）轨迹：
  - epoch（轮次）49: `0.999457`
  - epoch（轮次）99: `0.996744`
  - epoch（轮次）149: `0.999186`
  - epoch（轮次）199: `0.997830`
  - epoch（轮次）249: `0.992404`
- 当前解读：
  - 还没有出现像 `R670a` 那样一眼可判的 prefix collapse（前缀塌缩）终局
  - 但早期 collision recovery（冲突恢复）明显偏慢，后续需要继续盯 generate（生成）结果

## Final Tokenizer Result（最终分词器结果）

- run_dir（运行目录）：
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r680_l1_smooth_l2_contrastive_multihop_20260418/industrial_r680a_l1_smooth_l2_contrastive_multihop/Apr-18-2026_02-23-43`
- best train collision（训练最佳冲突率）：
  - `0.0984807379`
- best epoch（最佳轮次）：
  - `8749`
- best loss（最佳损失）：
  - `0.3416493237`
- final epoch collision（最终轮次冲突率）：
  - `0.1025501899`
- generated collision（生成后冲突）：
  - `11 / 3686 = 0.0029842648`
- max conflict（最大冲突簇）：
  - `2`
- active L1（活跃第一层码）：
  - `226`
- mean / median / max L1 bucket size（第一层桶大小均值 / 中位数 / 最大值）：
  - `16.31 / 15 / 117`
- unique L2 pairs（唯一第二层前缀数）：
  - `2833`
- generated SID index（生成语义标识索引）：
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r680_l1_smooth_l2_contrastive_multihop_20260418/generated_indices/Industrial_and_Scientific.r680a_l1_smooth_l2_contrastive_multihop.index.json`

## Tokenizer-Side Comparison（分词器侧对比）

| tokenizer（分词器） | active L1（活跃第一层码） | unique L2 pairs（唯一第二层前缀数） | generated collision（生成后冲突） | max conflict（最大冲突簇） |
| --- | ---: | ---: | ---: | ---: |
| original MiniOneRec | 48 | 2295 | 16 / 3686 | 3 |
| v2 | 203 | 2680 | 13 / 3686 | 2 |
| R650a | 199 | 2782 | 11 / 3686 | 2 |
| R660a | 181 | 2598 | 12 / 3686 | 2 |
| R680a | 226 | 2833 | 11 / 3686 | 2 |

## 当前判读

- `R680a` 没有复现 `R670a` 的 prefix collapse（前缀塌缩）；它是一个明确的 non-catastrophic tokenizer（非灾难性分词器）。
- 单看 generated collision（生成后冲突），`R680a` 已经达到和 `R650a` 相同的 `11 / 3686`，并优于 `R660a` 的 `12 / 3686`。
- 但它的 `active L1`（活跃第一层码）和 `unique L2 pairs`（唯一第二层前缀数）都更高，说明这条线更像是在“把中层分辨力拉开”，而不是在“收紧粗前缀”。
- 这给出的最稳妥结论不是“已经更好”，而是：
  - `L2 contrastive interface`（第二层对比式接口）至少没有把 tokenizer（分词器）直接打坏
  - 是否真的带来更好的 SID space（SID 空间），还要看 downstream `SFT -> evaluate`（监督微调到评测）

## Decision Rule（决策规则）

- 当前状态下，`R680a` 已满足“非灾难性 tokenizer/generate（非灾难性分词器训练与生成）”条件，可以推进 `title_history2sid_on + desc_align_p05` 的 `SFT -> evaluate`（监督微调到评测）。
- 若后续下游为负，优先怀疑：
  - `local_multihop`（局部多跳图）作为 `L2` 正样本载体仍然不够精准
  - `semantic-near + multihop-weak`（语义近 + 多跳弱连接）负样本构造仍含伪负样本
  - stop-gradient prefix（前缀停梯度）虽避免塌缩，但也可能让层间协商不足
