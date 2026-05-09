# L2 Local Multihop Rescue Tokenizers（第二层局部多跳救援分词器）

Status（状态）: `navigation + stage-snapshot（导航 + 阶段快照）`

Last updated（更新日期）: `2026-05-09`

## Reading Order（阅读顺序）

This branch is now organized around the current mainline（当前主线） and near-term candidates（近期候选）.

Read in this order（按这个顺序阅读）:

1. [MAINLINE.md](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/MAINLINE.md): current mainline（当前主线）.
2. [mainline/](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/mainline): clean mainline view（干净主线视图）.
3. [active_candidates/](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/active_candidates): candidates still worth running or interpreting（仍值得运行或解释的候选）.
4. [ablations/](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/ablations): completed/lower-priority ablations（已完成或低优先级消融）.
5. [archive/](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/archive): superseded helpers（已替代辅助脚本） and old queue wrappers（旧队列脚本）.

Implementation note（实现备注）:

- The organized directories use symlinks（软链接） first.
- Original paths（原始路径） are preserved for reproducibility（可复现性） and running tmux jobs（运行中会话任务）.
- Physical moves（物理移动） should happen only after active SFT/eval（监督微调/评测） jobs finish.

## Role（角色）

This branch（分支） tests whether a refreshed local multihop graph（刷新后的局部多跳图） can rescue L2 collaborative hierarchy information（第二层协同层级信息） in SID tokenizer construction（语义标识分词器构建） without returning to heavy graph propagation（重图传播）.

The current mainline（当前主线） is:

> `r690b_lmh_l2_contrastive_pull_weight001`

## Motivation（动机）

Earlier `fagsp_mid_base` / broad mid-graph（宽中图） variants looked unreliable, and the old local graph prior（旧局部图先验） plus oversized graph weights（过大的图权重） could destabilize tokenizer training（分词器训练）.

This branch isolates a more conservative hypothesis（保守假设）:

- use a refreshed local multihop graph（刷新后的局部多跳图）;
- keep graph pressure weak（保持弱图压力）;
- focus on L2 prefix structure（第二层前缀结构）;
- judge by downstream SFT / RL（下游监督微调 / 强化学习）, not tokenizer proxy（分词器代理指标） alone.

## Current Mainline Variant（当前主线变体）

Variant（变体）:

- `r690b_lmh_l2_contrastive_pull_weight001`

Tokenizer checkpoint（分词器检查点）:

- `/data/leejt/OneRec/output_weights/experiments/mgr_sid_l2_lmh_sweep_20260507/r690b_lmh_l2_contrastive_pull_weight001/May-07-2026_02-18-34/best_collision_model.pth`

Generated index（生成索引）:

- `/data/leejt/OneRec/output_weights/experiments/mgr_sid_l2_lmh_sweep_20260507/generated_indices/Industrial_and_Scientific.r690b_lmh_l2_contrastive_pull_weight001.index.json`

Prepared data root（准备后的数据根目录）:

- `/home/leejt/OneRec/data_experiment/Amazon/r690b_lmh_l2_contrastive_pull_weight001`

Tokenizer-side diagnostics（分词器侧诊断）:

- generated collision（生成碰撞）: `16 / 3686`
- generated collision rate（生成碰撞率）: `0.004340748779164406`
- max conflict（最大冲突簇）: `2`
- active L1（活跃第一层码）: `60`
- unique L12（唯一一二层前缀）: `2330`

## SFT Result（监督微调结果）

SFT/evaluate（监督微调/评测） config（配置）:

- [sft_industrial_r690b_lmh_l2_contrastive_pull_weight001_title_on_desc_p05_4gpu.yaml](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/configs/sft_industrial_r690b_lmh_l2_contrastive_pull_weight001_title_on_desc_p05_4gpu.yaml)
- [evaluate_industrial_r690b_lmh_l2_contrastive_pull_weight001_title_on_desc_p05_4gpu.yaml](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/configs/evaluate_industrial_r690b_lmh_l2_contrastive_pull_weight001_title_on_desc_p05_4gpu.yaml)

Result JSON（结果文件）:

- [final_result_sft_mgr_r690b_lmh_l2_contrastive_pull_weight001_title_on_desc_p05_4gpu_Industrial_and_Scientific.json](/home/leejt/OneRec/results/experiments/mgr_sid_l2_lmh_sweep_sft_eval_20260507/final_result_sft_mgr_r690b_lmh_l2_contrastive_pull_weight001_title_on_desc_p05_4gpu_Industrial_and_Scientific.json)

Metrics（指标）:

| Metric（指标） | @1 | @3 | @5 | @10 | @20 | @50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NDCG（归一化折损累计增益） | 0.070593 | 0.088131 | 0.094889 | 0.104383 | 0.112558 | 0.124803 |
| HR（命中率） | 0.070593 | 0.100816 | 0.117362 | 0.146923 | 0.179351 | 0.241121 |

Constraint invalid total（约束解码无效数）:

- `0`

Verdict（裁决）:

- Promising positive SFT signal（有希望的监督微调正向信号）.
- Beats strongest original SFT（原版最强监督微调） on `NDCG@1/@3/@5/@10`.
- Beats `v2_on_p05` SFT（监督微调） on all primary `NDCG/HR@1/@3/@5/@10`.
- Still below strongest original SFT on `HR@10`（命中率@10）.

## Error Analysis（错误分析）

Artifacts（产物）:

- [report.md](/home/leejt/OneRec/results/analysis/r690b_lmh_pull001_error_analysis_20260507/report.md)
- [summary.json](/home/leejt/OneRec/results/analysis/r690b_lmh_pull001_error_analysis_20260507/summary.json)
- [slice_comparison.csv](/home/leejt/OneRec/results/analysis/r690b_lmh_pull001_error_analysis_20260507/slice_comparison.csv)

Key finding（关键发现）:

- The method improves rank quality（排序质量） more than hit coverage（命中覆盖）.
- Against strongest original SFT（原版最强监督微调）, it has fewer `Hit@10`（命中@10） examples but a higher `NDCG@10`（归一化折损累计增益@10） because correct items move earlier.
- Against `v2_on_p05`, the `same_L2_seen`（历史中出现同第二层前缀） slice has a large gain:
  - `NDCG@10 +0.044051`
  - `HR@10 +0.066845`

Main weakness（主要弱点）:

- `no_same_L1`（历史中无同第一层前缀） remains weak.
- Fine-grained sibling items（同族兄弟物品） are sometimes confused.

## Running RL/Eval（正在运行的强化学习/评测）

RL/evaluate（强化学习/评测） was launched as the current decisive experiment（当前裁决实验）.

Launch script（启动脚本）:

- [launch_r690b_lmh_pull001_rl_eval_tmux.sh](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/scripts/launch_r690b_lmh_pull001_rl_eval_tmux.sh)

Chain script（串联脚本）:

- [experiment_r690b_lmh_pull001_rl_eval_chain.sh](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/scripts/experiment_r690b_lmh_pull001_rl_eval_chain.sh)

Configs（配置）:

- [rl_industrial_r690b_lmh_l2_contrastive_pull_weight001_title_on_desc_p05_4gpu.yaml](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/configs/rl_industrial_r690b_lmh_l2_contrastive_pull_weight001_title_on_desc_p05_4gpu.yaml)
- [evaluate_industrial_r690b_lmh_l2_contrastive_pull_weight001_title_on_desc_p05_rl_4gpu.yaml](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/configs/evaluate_industrial_r690b_lmh_l2_contrastive_pull_weight001_title_on_desc_p05_rl_4gpu.yaml)

Runtime（运行信息）:

- tmux session（会话）: `mgr_r690b_lmh_pull001_rl_eval_0507`
- GPUs（显卡）: `2,3,4,5`
- W&B run（实验追踪）: `fug78gw7`
- log（日志）: [mgr_r690b_lmh_pull001_rl_eval_0507.log](/home/leejt/OneRec/logs/l2_lmh_rl/mgr_r690b_lmh_pull001_rl_eval_0507.log)

Expected RL output（预期强化学习输出）:

- `/data/leejt/OneRec/output_weights/experiments/mgr_sid_l2_lmh_sweep_rl_eval_20260507/r690b_lmh_l2_contrastive_pull_weight001_title_on_desc_p05_4gpu/rl/final_checkpoint`

Expected result JSON（预期结果文件）:

- `/home/leejt/OneRec/results/experiments/mgr_sid_l2_lmh_sweep_rl_eval_20260507/final_result_rl_mgr_r690b_lmh_l2_contrastive_pull_weight001_title_on_desc_p05_4gpu_Industrial_and_Scientific.json`

## Decision Rule（决策规则）

If RL（强化学习） beats or clearly approaches the strongest original RL（原版最强强化学习） while preserving the SFT ranking gain（监督微调排序增益）, this branch should become the strongest result line（最强结果线）.

If RL is mixed（结果混合）, inspect whether it:

- recovers `HR@10`（命中率@10） coverage（覆盖）;
- amplifies sibling confusion（兄弟物品混淆）;
- preserves `same_L2_seen`（同第二层前缀历史） gains.

Nearby follow-ups（近邻后续）:

- pull weight ablation（拉近权重消融）: `0.005 / 0.02`
- weak sibling separation（弱兄弟物品分离）
- rerun with the same graph but alternative RL reward weighting（同图不同强化学习奖励权重）

## Stage 1 L2 Sweep（第一阶段第二层权重扫描）

The active L2 hyperparameter sweep（第二层超参数扫描） is tracked in:

- [STAGE1_L2_SWEEP.md](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/STAGE1_L2_SWEEP.md)
- [stage1_l2_sweep_tracker.csv](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/stage1_l2_sweep_tracker.csv)
