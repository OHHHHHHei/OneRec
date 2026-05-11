# L1 Sweep（第一层权重扫描）

Status（状态）: `evaluated-negative（已评测负向）`

Last updated（更新日期）: `2026-05-11`

This candidate tested whether increasing L1 semantic pull strength（第一层语义拉近强度） improves the balance between semantic routing（语义路由） and downstream learnability（下游可学习性）.

Representative variant（代表变体）:

- `r690b_lmh_l1_weight040_l2_weight001_l3_weight002`

SFT result（监督微调结果）:

| Metric（指标） | @1 | @3 | @5 | @10 |
| --- | ---: | ---: | ---: | ---: |
| NDCG（归一化折损累计增益） | 0.065078 | 0.080428 | 0.087976 | 0.098104 |
| HR（命中率） | 0.065078 | 0.091330 | 0.109640 | 0.140966 |

Verdict（裁决）:

- Negative versus current L1=0.030 mainline（相对当前第一层 0.030 主线为负）.
- Negative versus strongest original SFT baseline（相对原版最强监督微调基线为负）.
- Interpretation（解释）: stronger L1 pull likely over-strengthens coarse semantic routing（过度强化粗语义路由） and hurts downstream ranking（下游排序）.

Registry row（总账记录）:

- `sft_industrial_r690b_lmh_l1_weight040_l2_weight001_l3_weight002_title_on_desc_p05_20260509_032424`
