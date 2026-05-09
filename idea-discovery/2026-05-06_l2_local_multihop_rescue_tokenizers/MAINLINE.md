# Current Mainline（当前主线）

Status（状态）: `navigation（导航）`

Last updated（更新日期）: `2026-05-09`

## Role（角色）

This file is the entry point for the active L2 local-multihop SID tokenizer line（第二层局部多跳语义标识分词器主线）.

The current mainline（当前主线） is:

- tokenizer（分词器）: `r690b_lmh_l2_contrastive_pull_weight001`
- dataset（数据集）: `Industrial_and_Scientific`
- downstream recipe（下游配方）: `title_history2sid_on + desc_align_p05`
- SFT setting（监督微调设置）: `4gpu`, `batch_size=1024`, `micro_batch_size=2`, `seed=42`

## Method Idea（方法思路）

The core goal（核心目标） is to inject hierarchical collaborative information（层级协同信息） into SID construction（语义标识构建） while preserving a semantic coarse route（语义粗路由）.

Current interpretation（当前解释）:

- L1（第一层） mostly preserves semantic coarse grouping（语义粗分类）.
- L2（第二层） receives local multihop collaborative signal（局部多跳协同信号） through a weak graph contrastive objective（弱图对比目标）.
- L3（第三层） keeps local fine-grained refinement（局部细粒度修正）.
- Downstream SFT/RL（监督微调/强化学习） is the final judge（最终裁决），not tokenizer proxy（分词器代理指标） alone.

## Main Artifacts（主线产物）

Tokenizer index（分词器索引）:

- `/data/leejt/OneRec/output_weights/experiments/mgr_sid_l2_lmh_sweep_20260507/generated_indices/Industrial_and_Scientific.r690b_lmh_l2_contrastive_pull_weight001.index.json`

Organized configs（整理后的配置入口）:

- [SFT config](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/mainline/configs/sft_eval/sft_title_on_desc_p05_4gpu.yaml)
- [SFT eval config](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/mainline/configs/sft_eval/evaluate_title_on_desc_p05_4gpu.yaml)
- [RL config](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/mainline/configs/rl_eval/rl_title_on_desc_p05_4gpu.yaml)
- [RL eval config](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/mainline/configs/rl_eval/evaluate_rl_title_on_desc_p05_4gpu.yaml)

Organized scripts（整理后的脚本入口）:

- [SFT/eval chain](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/mainline/scripts/sft_eval/run_sft_eval_chain.sh)
- [RL/eval chain](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/mainline/scripts/rl_eval/run_rl_eval_chain.sh)

Reports and diagnostics（报告与诊断）:

- [Advisor report](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/mainline/reports/advisor_report_main.tex)
- [Structural diagnostic](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/mainline/diagnostics/structural_diagnostic)
- [Codebook reasonableness](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/mainline/diagnostics/codebook_reasonableness)

## Current Results（当前结果）

SFT（监督微调）:

| Metric（指标） | @1 | @3 | @5 | @10 |
| --- | ---: | ---: | ---: | ---: |
| NDCG（归一化折损累计增益） | 0.070593 | 0.088131 | 0.094889 | 0.104383 |
| HR（命中率） | 0.070593 | 0.100816 | 0.117362 | 0.146923 |

RL（强化学习）:

| Metric（指标） | @1 | @3 | @5 | @10 |
| --- | ---: | ---: | ---: | ---: |
| NDCG（归一化折损累计增益） | 0.073020 | 0.087362 | 0.094663 | 0.105132 |
| HR（命中率） | 0.073020 | 0.097948 | 0.115597 | 0.148026 |

Interpretation（解释）:

- SFT improves the strongest original SFT baseline（原始最强监督微调基线） on all NDCG @1/@3/@5/@10.
- SFT does not improve all HR（命中率） positions, especially @5/@10.
- RL improves over this tokenizer's SFT at @1 and @10, but is still below the strongest original RL baseline（原始最强强化学习基线）.

## Active Candidates（活跃候选）

Near-term candidates（近期候选） are separated from the mainline:

- [L2 square（第二层平方图）](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/active_candidates/l2_square/README.md)
- [L3 ranking（第三层排序损失）](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/active_candidates/l3_ranking/README.md)
- [L1 sweep（第一层权重扫描）](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/active_candidates/l1_sweep/README.md)
- [Office transfer（Office 数据集迁移）](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/active_candidates/office_transfer/README.md)
- [Recipe ablation（训练配方消融）](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/active_candidates/recipe_ablation/README.md)

## Notes（备注）

This organization uses symlinks（软链接） for safety. Original paths（原始路径） are preserved so existing logs（日志）, tmux jobs（会话任务）, and registries（总账） remain reproducible（可复现）.
