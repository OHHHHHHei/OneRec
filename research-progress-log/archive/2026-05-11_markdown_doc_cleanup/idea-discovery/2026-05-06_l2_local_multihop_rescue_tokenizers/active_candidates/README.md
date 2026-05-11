# Candidate Parking Lot（候选暂存区）

Status（状态）: `parking-lot + evidence-index（暂存区 + 证据索引）`

Last updated（更新日期）: `2026-05-11`

This directory keeps near-term ideas（近期想法）, completed ablations（已完成消融）, and transfer checks（迁移检查） that are related to the current L2 local-multihop SID tokenizer line（第二层局部多跳语义标识分词器线）.

It is not the canonical current-state source（不是权威当前状态源）. Use:

- [CURRENT_STATE.md](/home/leejt/OneRec/research-progress-log/CURRENT_STATE.md)
- [MAINLINE.md](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/MAINLINE.md)

## Current Entries（当前条目）

| Entry（条目） | Status（状态） | Use（用途） |
| --- | --- | --- |
| [l1_sweep/](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/active_candidates/l1_sweep/README.md) | `evaluated-negative（已评测负向）` | Evidence that stronger L1 semantic pull（更强第一层语义拉近） hurts downstream SFT（下游监督微调）. |
| [l2_square/](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/active_candidates/l2_square/README.md) | `mixed / partially-finalized（混合 / 部分定稿）` | Candidate using A-local squared graph（局部图平方） at L2; keep separate until registry status is finalized（总账状态定稿）. |
| [l3_ranking/](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/active_candidates/l3_ranking/README.md) | `evaluated-negative（已评测负向）` | Evidence against replacing L3 local pull（第三层局部拉近） with ranking loss（排序损失） under the tested setting. |
| [office_transfer/](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/active_candidates/office_transfer/README.md) | `supporting-evidence（支持证据）` | Transfer check（迁移检查） on Office. |
| [recipe_ablation/](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/active_candidates/recipe_ablation/README.md) | `evaluated-negative（已评测负向）` | Evidence that `title_history2sid_off + desc_align_p05` is a bad recipe（坏配方） for the current tokenizer（当前分词器）. |

## Rule（规则）

- Keep launch-only state（仅启动状态） in logs（日志） or conversation（对话）.
- Keep finalized results（定稿结果） in the split registry（分表总账）.
- Keep this directory as a lightweight navigation layer（轻量导航层）, not a second registry（第二总账）.
