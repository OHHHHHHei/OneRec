# 2026-04-18 `R693a` Hierarchical Collaboration-Only Multihop（层级纯协同多跳版）

Status（状态）: `tokenizer_finalized（分词器已定稿）`

Launch date（启动日期）: `2026-04-18`

## 目的

`R693a` 是对当前 `R690b`（受 `CoST` 启发的分层对比式量化）主骨架的一次收束版重写：

- 不再把显式 semantic graph（语义图）监督作为额外辅助项
- 让 `RQ-VAE`（残差量化变分自编码器）的输入语义嵌入和 reconstruction（重建）目标自然承担“保语义”职责
- 三层辅助项只负责注入 collaborative graph signal（协同图信号）

当前要验证的问题是：

- `L1`（第一层）如果只吃高置信 `coarse collaborative graph`（粗协同图）正边，能否形成更干净的粗入口
- `L2`（第二层）如果改用 `local_multihop`（局部多跳图）做 `InfoNCE`（对比学习损失）正样本来源，是否比 `R690b` 当前的 `fagsp_mid_base`（基础中图）更适合承担中层协同细分

## 设计

这次的总目标是：

```text
L_total
= L_recon + L_rq
+ lambda_1 * L1_coarse_pull
+ lambda_2 * L2_graph_guided_infonce
+ lambda_3 * L3_local_pull
```

其中：

- `L1 <- coarse_purified`（净化粗图），但不是整张图直接平滑，而是先构造 high-confidence positive graph（高置信正边图）
- `L2 <- local_multihop`（局部多跳图）作为正样本来源
- `L2` 负样本来自 `coarse candidate + multihop weak`（粗图候选 + 多跳弱连接）
- `L3 <- local_purified`（净化局部图）
- `hierarchy_stopgrad_previous_levels = true`

## 图源构造

本阶段新增两份训练输入：

1. `L1` 高置信粗协同图：
   - 文件：
     - [R693a_l1_coarse_highconf_graph.npz](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-18_mgr_sid_r693_hier_collab_only_multihop_industrial/R693a_l1_coarse_highconf_graph.npz)
   - 规则：
     - 基于 `coarse_purified`（净化粗图）
     - `topk = 8`
     - `quantile = 0.75`
     - mutual high-confidence edges（双向高置信边）保留

2. `L2` 负样本对：
   - 文件：
     - [R693a_all_mid_graph_weak_pairs.csv](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-18_mgr_sid_r693_hier_collab_only_multihop_industrial/R693a_all_mid_graph_weak_pairs.csv)
   - 规则：
     - `coarse candidate topk = 16`
     - `mid weak quantile = 0.25`
     - rule（规则）=`coarse_candidate_mid_graph_weak`

图源摘要：

- [R693a_graph_source_summary.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-18_mgr_sid_r693_hier_collab_only_multihop_industrial/R693a_graph_source_summary.json)
- 当前统计：
  - `l1_graph_undirected_edge_count = 2191`
  - `l1_graph_item_coverage_rate = 0.7982`
  - `weak_pair_count = 4519`
  - `weak_pair_item_coverage_rate = 0.8304`
  - `weak_threshold_mean = 0.04627`

## 配置

- tokenizer config（分词器配置）：
  - [sid_train_industrial_mgr_sid_r693a_hier_collab_only_multihop.yaml](/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_r693a_hier_collab_only_multihop.yaml)
- 关键参数：
  - `l1_contrastive_pull_weight = 0.02`
  - `l2_contrastive_pull_weight = 0.10`
  - `l2_contrastive_mode = graph_infonce`
  - `l2_infonce_temperature = 0.10`
  - `l3_contrastive_pull_weight = 0.02`
  - `coarse_weight = 0.0`
  - `mid_weight = 0.0`
  - `local_weight = 0.0`
  - `coarse_view_name = coarse_purified`
  - `mid_view_name = local_multihop`
  - `local_view_name = local_purified`
  - `hierarchy_stopgrad_previous_levels = true`

## 对应文件

- graph source builder（图源脚本）：
  - [experiment_mgr_sid_hier_collab_graph_sources.py](/home/leejt/OneRec/scripts/experiment_mgr_sid_hier_collab_graph_sources.py)
- train/generate script（训练生成脚本）：
  - [experiment_mgr_sid_r693a_hier_collab_only_multihop_train_generate.sh](/home/leejt/OneRec/scripts/experiment_mgr_sid_r693a_hier_collab_only_multihop_train_generate.sh)
- launch script（启动脚本）：
  - [launch_mgr_sid_r693_hier_collab_only_multihop_tmux.sh](/home/leejt/OneRec/scripts/launch_mgr_sid_r693_hier_collab_only_multihop_tmux.sh)

## 运行产物

- `tmux`（终端复用器） session（会话）：
  - `mgr_r693a_hier_collab_only_multihop`
- runtime GPU（运行显卡）：
  - `CUDA_VISIBLE_DEVICES=3`
- output root（输出根目录）：
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r693_hier_collab_only_20260418`
- log（日志）：
  - [experiment_mgr_sid_r693a_hier_collab_only_multihop_20260418.log](/home/leejt/OneRec/logs/experiment_mgr_sid_r693a_hier_collab_only_multihop_20260418.log)

## 启动检查

- 启动前 smoke test（冒烟测试）已通过：
  - 新 `L1` 图、`L2` 负样本、训练主循环都能完成 `1` 个 epoch（轮次）跑通

## First Run Status（第一次运行状态）

第一次正式长跑已经结束，但**不是有效 tokenizer（分词器）结果**，而是一次数值崩溃。

- run_dir（运行目录）：
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r693_hier_collab_only_20260418/industrial_r693a_hier_collab_only_multihop/Apr-18-2026_12-58-12`
- 训练日志：
  - [experiment_mgr_sid_r693a_hier_collab_only_multihop_20260418.log](/home/leejt/OneRec/logs/experiment_mgr_sid_r693a_hier_collab_only_multihop_20260418.log)
- 关键现象：
  - epoch（轮次）`0` 正常：
    - `total = 3.344863`
    - `recon = 3.326673`
    - `rq = 0.007210`
    - `l1_pull = 0.099298`
    - `l2_pull = 0.061775`
    - `l3_pull = 0.140801`
  - 从 epoch（轮次）`1` 开始，`total / recon / rq / l1_pull / l3_pull` 全部变成 `NaN`（非数）
  - `l2_pull` 从 epoch（轮次）`1` 起固定为 `0.0`
  - `summary.json` 中：
    - `best.loss = Infinity`
    - `best.collision_rate = 0.9997287032`
    - `best.epoch = -1`
- generate（生成）阶段也失败：
  - `experiment_mgr_sid_v1_generate.py` 在量化阶段触发：
    - `AssertionError: amplitude > 0`
  - 因此没有可用的 generated SID index（生成 SID 索引）

## 失败定位

- 这不是“训练完了但指标差”，而是**训练在第一个优化步后就进入了数值不稳定状态**。
- 更具体地说：
  - epoch（轮次）`0` 正常，说明图源文件、配置读取、前向和首轮反向传播都能走通
  - epoch（轮次）`1` 立刻全量 `NaN`，说明问题更像是：
    - 新的纯协同 `loss`（损失）组合在一次参数更新后把表示/码本打坏
    - 而不是训练后期慢慢发散
- 根因已定位到 `L2 graph-guided InfoNCE`（第二层图引导对比损失）的实现：
  - 文件：
    - [train_v2.py](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/train_v2.py)
  - 关键位置：
    - [train_v2.py](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/train_v2.py:338)
  - 已确认事实：
    - 单独对 `recon + rq`（重建 + 量化）、`L1 pull`（第一层拉近）、`L3 pull`（第三层拉近）做 backward（反向传播）时，梯度全部有限
    - 单独对 `L2 InfoNCE` 做 backward（反向传播）时，encoder（编码器）所有层立刻出现非有限梯度
    - `R693a` 当前 batch（批次）里有 `40` 个 item（物品）在 `L2` 正负样本图上都没有 active pairs（有效正负样本对）
- 具体机制：
    - 这些全空行在 `_weighted_graph_guided_infonce_loss` 中被 `masked_fill(..., -1e9)` 后，`row_max` 仍是有限值 `-1e9`
    - 随后代码执行 `exp(similarity - row_max) * active_mask`
    - 对全空行来说，这会形成 `exp(大数) * 0 = inf * 0 = NaN`
    - loss（损失）表面上仍然是有限的，但梯度已经被污染，首个 optimizer step（优化器更新）后 encoder 参数变成 `NaN`
- 因此这次运行**不能进入 registry（总账）作为有效 tokenizer 结果**，也不能推进 `SFT -> evaluate`（监督微调到评测）

## Bug Fix Status（缺陷修复状态）

根因已经在代码中修复：

- 修复文件：
  - [train_v2.py](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/train_v2.py)
- 修复方式：
  - 对 `_weighted_graph_guided_infonce_loss` 中的空行（既无正样本也无负样本）做数值安全处理
  - 不再执行 `exp(大数) * 0`
  - 改为只在 active pairs（有效样本对）上做稳定化指数计算，空行保持全零

修复后验证结果：

- 单独对 `L2 InfoNCE`（第二层对比学习损失）做 backward（反向传播）：
  - encoder（编码器）非有限梯度参数数目：`0`
- `2 epoch` smoke test（两轮冒烟测试）：
  - epoch（轮次）`0` 正常
  - epoch（轮次）`1` 仍保持有限值，没有再出现 `NaN`（非数）
  - collision（冲突率）已能正常计算：`0.066739`

当前状态：

- 上一轮 `R693a` 正式长跑仍然是无效结果，不写入 registry（总账）
- 但实现层面的数值稳定性缺陷已经修复，`R693a` 现在**可以重新启动正式训练**

## Valid Rerun Result（修复后有效重跑结果）

修复 `L2 graph-guided InfoNCE`（第二层图引导对比学习损失）的空行数值稳定性问题后，`R693a` 已重新完整跑完 tokenizer/generate（分词器训练与生成），这次是有效结果。

- rerun run_dir（重跑运行目录）：
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r693_hier_collab_only_20260418/industrial_r693a_hier_collab_only_multihop/Apr-18-2026_14-13-42`
- train summary（训练摘要）：
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r693_hier_collab_only_20260418/industrial_r693a_hier_collab_only_multihop/Apr-18-2026_14-13-42/summary.json`
- best checkpoint（最佳检查点）：
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r693_hier_collab_only_20260418/industrial_r693a_hier_collab_only_multihop/Apr-18-2026_14-13-42/best_collision_model.pth`
- generated SID index（生成 SID 索引）：
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r693_hier_collab_only_20260418/generated_indices/Industrial_and_Scientific.r693a_hier_collab_only_multihop.index.json`
- generate summary（生成摘要）：
  - [R693a_generate_summary.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-18_mgr_sid_r693_hier_collab_only_multihop_industrial/R693a_generate_summary.json)

核心结果：

| metric（指标） | value（数值） |
| --- | ---: |
| best train collision（训练最佳冲突率） | `0.1009224091` |
| best epoch（最佳轮次） | `9499` |
| generated collision（生成后冲突） | `12 / 3686 = 0.0032555616` |
| max conflict（最大冲突簇） | `2` |
| active L1（活跃第一层码） | `90` |
| unique L2 pairs（唯一第二层前缀数） | `2274` |
| unique SID（唯一 SID 数） | `3674` |

与相邻分支的粗略对比：

| variant（变体） | generated collision（生成后冲突） | active L1（活跃第一层码） | unique L2 pairs（唯一第二层前缀数） |
| --- | ---: | ---: | ---: |
| `R690a` | `11 / 3686` | `118` | `2527` |
| `R690b` | `14 / 3686` | `33` | `1989` |
| `R693a` | `12 / 3686` | `90` | `2274` |

当前判读：

- `R693a` 没有复现第一次运行的 `NaN`（非数）崩溃，说明 `InfoNCE`（对比学习损失）空行修复有效。
- 相比 `R690b`，`R693a` 的 `L1`（第一层）没有被压得那么狠：`active L1` 从 `33` 回升到 `90`，同时 `unique L2 pairs` 也从 `1989` 回升到 `2274`。
- 相比 `R690a`，`R693a` 略微更收紧：`active L1` 从 `118` 降到 `90`，但 generated collision（生成后冲突）从 `11` 变为 `12`。
- 因此它是一个 non-catastrophic tokenizer candidate（非灾难性分词器候选），但是否真的优于 `R690a/R690b/R680a` 只能通过 downstream SFT -> evaluate（下游监督微调到评测）裁决。

## Focus Family SID Diagnosis（重点物品族 SID 诊断）

诊断产物：

- [R693A_FOCUS_FAMILY_SID_ANALYSIS.md](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-18_mgr_sid_r693_hier_collab_only_multihop_industrial/R693A_FOCUS_FAMILY_SID_ANALYSIS.md)
- [R693A_TOKENIZER_FAMILY_OVERALL_COMPARISON.csv](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-18_mgr_sid_r693_hier_collab_only_multihop_industrial/R693A_TOKENIZER_FAMILY_OVERALL_COMPARISON.csv)
- [R693A_FAMILY_L1_SPREAD_COMPARISON.csv](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-18_mgr_sid_r693_hier_collab_only_multihop_industrial/R693A_FAMILY_L1_SPREAD_COMPARISON.csv)
- [R693A_FOCUS_FAMILY_L1_BUCKETS.csv](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-18_mgr_sid_r693_hier_collab_only_multihop_industrial/R693A_FOCUS_FAMILY_L1_BUCKETS.csv)
- [R693A_L1_BUCKET_FAMILY_COMPOSITION.csv](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-18_mgr_sid_r693_hier_collab_only_multihop_industrial/R693A_L1_BUCKET_FAMILY_COMPOSITION.csv)

核心观察：

- `R693a` 的 `active L1 = 90`，位于 `R690a` 的 `118` 和 `R690b` 的 `33` 之间；这说明它没有回到 `R680a/R690a` 的偏碎状态，也没有像 `R690b` 那样 prefix over-compression（前缀过度压缩）。
- `3d_filament`（3D 打印耗材）主桶 `<a_178>` 覆盖 `210 / 386 = 54.4%`，桶纯度 `99.1%`；这是较合理的粗入口，但仍比 original（原版）的 `21` 个 `L1` 更分散，当前为 `33` 个 `L1`。
- `tape`（胶带）最理想：主桶 `<a_31>` 覆盖 `155 / 217 = 71.4%`，桶纯度 `96.9%`，基本符合“同族物品先进入共同粗入口”的预期。
- `connector_fitting`（连接件）明显比 `R690b` 干净：主桶覆盖 `35.8%`，桶纯度 `90.7%`；不是单一大桶，但前几大桶组织合理。
- `adhesive_epoxy`（胶黏剂/环氧）没有形成单一主桶，但主桶纯度从 `R690b` 的 `48.5%` 提升到 `87.2%`；考虑该族内部语义跨度较大，拆成多个较纯子入口可能比强行合并更合理。
- `gauge_meter`（仪表/测量器）仍然是主要问题族：`44` 个 `L1`，最大桶只覆盖 `13.6%`，且桶纯度只有 `54.8%`。这说明 `R693a` 对高度混杂的仪表类仍没有形成稳定粗入口。

## Downstream SFT Verdict（下游监督微调裁决）

`R693a -> title_history2sid_on + desc_align_p05` 的 `SFT -> evaluate`（监督微调到评测）已经完成。

- SFT config（监督微调配置）：
  - [sft_industrial_mgr_r693a_title_on_desc_p05.yaml](/home/leejt/OneRec/config/experiments/sft_industrial_mgr_r693a_title_on_desc_p05.yaml)
- evaluate config（评测配置）：
  - [evaluate_industrial_mgr_r693a_title_on_desc_p05.yaml](/home/leejt/OneRec/config/experiments/evaluate_industrial_mgr_r693a_title_on_desc_p05.yaml)
- SFT output（监督微调输出）:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r693a_sft_eval_20260418/title_on_desc_p05/sft/final_checkpoint`
- result JSON（结果文件）:
  - `/home/leejt/OneRec/results/experiments/mgr_sid_r693a_sft_eval_20260418/final_result_sft_mgr_r693a_title_on_desc_p05_Industrial_and_Scientific.json`
- W&B run（实验追踪）:
  - `sft_mgr_r693a_title_on_desc_p05_industrial`, run id `z8plj1wj`

结果：

| metric（指标） | @1 | @3 | @5 | @10 | @20 | @50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `NDCG`（归一化折损累计增益） | `0.06309287` | `0.07971921` | `0.08678189` | `0.09730760` | `0.10626678` | `0.11924144` |
| `HR`（命中率） | `0.06309287` | `0.09155085` | `0.10875800` | `0.14162806` | `0.17736598` | `0.24310611` |

训练摘要：

- best checkpoint（最佳检查点）: `checkpoint-456`
- best eval loss（最佳验证损失）: `1.5220661163`
- final eval loss（最终验证损失）: `1.5941315889`
- train loss（训练损失）: `0.4985252046`
- stop epoch（停止轮次）: `5.5`

裁决：

- `R693a` 是负结果，不推进 `RL`（强化学习）。
- 相比 `R680a`：
  - `NDCG@10`: `0.09863899 -> 0.09730760`
  - `HR@10`: `0.13567174 -> 0.14162806`
  - 也就是说 `HR@10`（命中率@10）更高，但 `NDCG@1/3/5/10`（归一化折损累计增益）全线更低，排序质量没有变好。
- 相比 current `v2_on_p05`（当前 v2_on_p05）：
  - `NDCG@10`: `0.10270767 -> 0.09730760`
  - `HR@10`: `0.14626075 -> 0.14162806`
  - 仍明显落后。

## 当前判读

- 这是目前最接近“语义由 `RQ-VAE` 主干保留、显式辅助项只注入协同信息”的 clean hierarchy（干净层级）版本。
- 它比 `R690b` 更符合当前收束后的主方法口径：
  - `L1` 负责粗协同入口
  - `L2` 负责中层协同细分
  - `L3` 负责局部细化
- 第一次完整运行暴露的是 `InfoNCE`（对比学习损失）空行数值稳定性问题，而不是方法本身的有效负结果。
- 修复后重跑已经得到有效 tokenizer/generate（分词器训练与生成）结果：
  - generated collision（生成后冲突）=`12 / 3686`
  - active L1（活跃第一层码）=`90`
  - unique L2 pairs（唯一第二层前缀数）=`2274`
- 下一步如继续验证，应接 `title_history2sid_on + desc_align_p05` 的 `SFT -> evaluate`（监督微调到评测），用下游指标判断这版 clean collaboration-only hierarchy（干净纯协同层级）是否优于 `R690a/R690b/R680a`。
