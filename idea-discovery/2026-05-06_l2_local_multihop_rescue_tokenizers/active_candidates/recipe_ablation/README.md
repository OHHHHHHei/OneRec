# Recipe Ablation（训练配方消融）

Status（状态）: `active-candidate（活跃候选）`

This branch tests the downstream recipe（下游训练配方） while keeping the tokenizer（分词器） fixed to the mainline anchor（主线锚点）.

Current representative（当前代表）:

- tokenizer（分词器）: `r690b_lmh_l2_contrastive_pull_weight001`
- recipe（配方）: `title_history2sid_off + desc_align_p05`

Use this branch only to interpret downstream training effects（下游训练影响）, not tokenizer quality（分词器质量）.
