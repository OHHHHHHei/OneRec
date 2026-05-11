# Recipe Ablation（训练配方消融）

Status（状态）: `evaluated-negative（已评测负向）`

Last updated（更新日期）: `2026-05-11`

This branch tests the downstream recipe（下游训练配方） while keeping the tokenizer（分词器） fixed to the current evidence-line anchor（当前证据线锚点）:

- tokenizer（分词器）: `r690b_lmh_l2_contrastive_pull_weight001`
- ablated recipe（消融配方）: `title_history2sid_off + desc_align_p05`

SFT result（监督微调结果）:

| Metric（指标） | @1 | @3 | @5 | @10 |
| --- | ---: | ---: | ---: | ---: |
| NDCG（归一化折损累计增益） | 0.051621 | 0.071323 | 0.079629 | 0.090581 |
| HR（命中率） | 0.051621 | 0.085595 | 0.105890 | 0.140084 |

Verdict（裁决）:

- Strongly negative versus current tokenizer with `title_history2sid_on + desc_align_p05`.
- Use this only as downstream recipe evidence（下游配方证据）, not as tokenizer quality evidence（分词器质量证据）.

Interpretation（解释）:

- For this tokenizer（分词器）, keeping title-history-to-SID supervision（标题历史到语义标识监督） appears important for downstream learnability（下游可学习性）.

Registry row（总账记录）:

- `sft_industrial_mgr_r690b_lmh_l2_contrastive_pull_weight001_title_off_desc_p05_4gpu_20260509_102800`
