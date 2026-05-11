# Current Mainline（当前主线）

Status（状态）: `mainline-detail（主线细节）`

Last updated（更新日期）: `2026-05-11`

## Role（角色）

This file records the main evidence line（主要证据线） for the L2 local-multihop SID tokenizer branch（第二层局部多跳语义标识分词器分支）.

The canonical current-state summary（权威当前状态总结） is:

- [CURRENT_STATE.md](/home/leejt/OneRec/research-progress-log/CURRENT_STATE.md)

The current mainline（当前主线） is:

- method name（方法名）: `LMH-HCSID`（局部多跳层级协同语义标识）
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

- [Tokenizer config](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/configs/hcsid/sid_train_industrial_lmh_hcsid.yaml)
- [SFT config](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/mainline/configs/sft_eval/sft_title_on_desc_p05_4gpu.yaml)
- [SFT eval config](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/mainline/configs/sft_eval/evaluate_title_on_desc_p05_4gpu.yaml)
- [RL config](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/mainline/configs/rl_eval/rl_title_on_desc_p05_4gpu.yaml)
- [RL eval config](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/mainline/configs/rl_eval/evaluate_rl_title_on_desc_p05_4gpu.yaml)

Organized scripts（整理后的脚本入口）:

- tokenizer trainer（分词器训练器）: [trainer.py](/home/leejt/OneRec/src/onerec/experiments/hcsid/trainer.py)
- tokenizer CLI（分词器命令行入口）: [train_entry.py](/home/leejt/OneRec/src/onerec/experiments/hcsid/train_entry.py)
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

## Archived Candidate Notes（已归档候选笔记）

Non-mainline candidate notes（非主线候选笔记） were archived to keep this branch readable（保持分支可读）:

- [moved_files.txt](/home/leejt/OneRec/research-progress-log/archive/2026-05-11_markdown_doc_cleanup/moved_files.txt)

## Notes（备注）

This file is intentionally kept as the only active Markdown（活跃文档） inside this method branch（方法分支）.
