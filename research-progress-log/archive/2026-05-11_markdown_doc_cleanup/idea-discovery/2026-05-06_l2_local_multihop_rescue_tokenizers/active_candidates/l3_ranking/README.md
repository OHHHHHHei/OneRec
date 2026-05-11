# L3 Ranking（第三层排序损失）

Status（状态）: `evaluated-negative（已评测负向）`

Last updated（更新日期）: `2026-05-11`

This ablation（消融） tested replacing L3 local pull（第三层局部拉近） with ranking loss（排序损失） while keeping the current L1/L2 anchor（当前第一/二层锚点）.

Variant（变体）:

- `r690b_lmh_l2_weight001_l3_ranking002`

SFT result（监督微调结果）:

| Metric（指标） | @1 | @3 | @5 | @10 |
| --- | ---: | ---: | ---: | ---: |
| NDCG（归一化折损累计增益） | 0.064417 | 0.078748 | 0.085432 | 0.094551 |
| HR（命中率） | 0.064417 | 0.089786 | 0.105890 | 0.134128 |

Verdict（裁决）:

- Clearly negative versus current mainline（相对当前主线明显负向）.
- Clearly negative versus strongest original SFT baseline（相对原版最强监督微调基线明显负向）.
- Do not promote to RL（不要推进强化学习）.

Interpretation（解释）:

- The tested ranking loss（排序损失） at L3 likely disrupts fine-grained local refinement（细粒度局部修正） more than it helps collaborative separation（协同拆分）.

Registry row（总账记录）:

- `sft_industrial_mgr_r690b_lmh_l2_weight001_l3_ranking002_title_on_desc_p05_4gpu_20260509_165700`
