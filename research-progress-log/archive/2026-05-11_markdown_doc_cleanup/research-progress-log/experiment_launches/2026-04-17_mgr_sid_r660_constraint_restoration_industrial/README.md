# 2026-04-17 `R660a` Constraint Restoration（约束恢复）

Status（状态）: `tokenizer_generated（分词器已生成）`
Launch date（启动日期）: `2026-04-17`

## 目的

`R660a` 用来验证一个很具体的问题：`R650a` 的负结果是否主要来自移除了 `L1/L3/semantic`（第一层/第三层/语义）直接约束，而不是来自 `Seq2Graph-lite + push-pull`（轻量 Seq2Graph + 推远拉近）本身。

## 设计

`R660a = R650a + v2-style full constraints`（R650a 加回 v2 风格全套约束）。

- 保持 `R650a` 不变的部分：
  - `coarse_view_name = coarse_seq2g_rel_masked`
  - `mid_view_name = fagsp_mid_seq2g_rel_masked`
  - `selective_separation_weight = 0.01`
  - `selective_separation_margin = 0.15`
  - `selective_separation_levels = [2]`
  - pair source（物品对来源）仍为 semantic-near + `fagsp_mid_seq2g_rel_masked` weak（语义近 + Seq2Graph 中图弱连接）
- 相对 `R650a` 恢复的约束：
  - `coarse_weight: 0.0 -> 0.05`
  - `local_weight: 0.0 -> 0.05`
  - `semantic_coarse_weight: 0.0 -> 0.05`
  - `semantic_mid_weight: 0.0 -> 0.025`

## 对应文件

- tokenizer config（分词器配置）：
  - [sid_train_industrial_mgr_sid_r660a_seq2graph_push_pull_full_constraints.yaml](/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_r660a_seq2graph_push_pull_full_constraints.yaml)
- train/generate script（训练生成脚本）：
  - [experiment_mgr_sid_r660a_seq2graph_push_pull_full_constraints_train_generate.sh](/home/leejt/OneRec/scripts/experiment_mgr_sid_r660a_seq2graph_push_pull_full_constraints_train_generate.sh)
- launch script（启动脚本）：
  - [launch_mgr_sid_r660_constraint_restoration_tmux.sh](/home/leejt/OneRec/scripts/launch_mgr_sid_r660_constraint_restoration_tmux.sh)

## 当前状态

当前阶段：`TOKENIZER_GENERATED_PENDING_SFT`（分词器已生成，等待监督微调）。

- `tmux`（终端复用器） session（会话）：
  - `mgr_r660a_constraint_restoration`，已结束
- runtime GPU（运行显卡）：
  - `CUDA_VISIBLE_DEVICES=7`
- output root（输出根目录）：
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r660_constraint_restoration_20260417`
- log（日志）：
  - `/home/leejt/OneRec/logs/experiment_mgr_sid_r660a_seq2graph_push_pull_full_constraints_20260417.log`
- pair source summary（物品对来源摘要）：
  - `semantic_pair_count = 82596`
  - `weak_pair_count = 1190`
  - `weak_pair_item_coverage_rate = 0.29896907216494845`
  - `weak_threshold = 0.0016943709051702172`
  - `semantic_sim_mean = 0.03232755345926315`
  - `mid_affinity_mean = 0.0014584285848355312`
  - `reliability_mean = 0.004500024733162467`

## Tokenizer Result（分词器结果）

- train best collision（训练最佳冲突率）：
  - `0.1323928378`
- best loss（最佳损失）：
  - `0.2442536950`
- best epoch（最佳轮次）：
  - `9799`
- generated collision（生成后冲突）：
  - `12 / 3686 = 0.0032555616`
- max conflict（最大冲突簇）：
  - `2`
- active L1（活跃第一层码）：
  - `181`
- mean / median / max L1 bucket size（第一层桶大小均值 / 中位数 / 最大值）：
  - `20.36 / 19 / 64`
- unique L2 pairs（唯一第二层前缀数）：
  - `2598`
- generated SID index（生成语义标识索引）：
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r660_constraint_restoration_20260417/generated_indices/Industrial_and_Scientific.r660a_seq2graph_push_pull_full_constraints.index.json`

## Tokenizer-Side Comparison（分词器侧对比）

| tokenizer（分词器） | active L1（活跃第一层码） | unique L2 pairs（唯一第二层前缀数） | generated collision（生成后冲突） | max conflict（最大冲突簇） |
| --- | ---: | ---: | ---: | ---: |
| original MiniOneRec | 48 | 2295 | 16 / 3686 | 3 |
| v2 | 203 | 2680 | 13 / 3686 | 2 |
| R640c | 209 | 2737 | 12 / 3686 | 3 |
| R650a | 199 | 2782 | 11 / 3686 | 2 |
| R660a | 181 | 2598 | 12 / 3686 | 2 |

## 当前判读

- 如果 `R660a` 明显优于 `R650a`，说明 `R650a` 的负结果很可能被 L1/L3/semantic（第一层/第三层/语义）约束缺失放大，`push-pull`（推远拉近）仍值得在受保护层级结构内继续验证。
- 如果 `R660a` 仍明显低于 `v2_on_p05`，说明问题不只是移除约束，`Seq2Graph-lite`（轻量 Seq2Graph）载体或当前 `push-pull`（推远拉近）物品对设计本身仍有问题。
- 本实验已经完成 tokenizer/generate（分词器训练与生成），且不是 catastrophic failure（灾难性失败）。
- 恢复 `L1/L3/semantic`（第一层/第三层/语义）约束后，active L1（活跃第一层码）从 `R650a` 的 `199` 降到 `181`，说明 L1（第一层）确实有所收紧。
- 但 generated collision（生成后冲突）从 `R650a` 的 `11 / 3686` 变成 `12 / 3686`，没有给出 tokenizer-side（分词器侧）更强信号。
- 由于 retired prior diagnostics（已退役前验诊断）不能继续充当 promotion gate（推进门槛），当前不能仅凭 tokenizer-side collision（分词器侧冲突）给出方法结论。
- 下一步若继续该分支，应把 `SFT -> evaluate`（监督微调到评测）定位为 diagnostic downstream check（诊断性下游检查），而不是强候选推进。
