# Office Transfer（Office 数据集迁移）

Status（状态）: `supporting-evidence（支持证据）`

Last updated（更新日期）: `2026-05-11`

This branch tests whether the current Industrial idea（Industrial 当前思路） transfers to `Office_Products`.

Variant（变体）:

- `office_r690b_lmh_l1w030_l2w010_l3w020`

SFT comparison（监督微调对比）:

| Method（方法） | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | HR@1 | HR@3 | HR@5 | HR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Office baseline（Office 基线） | 0.081176 | 0.099118 | 0.106344 | 0.114869 | 0.081176 | 0.112002 | 0.129470 | 0.155364 |
| Current method transfer（当前方法迁移） | 0.081998 | 0.100994 | 0.107274 | 0.114991 | 0.081998 | 0.114879 | 0.130086 | 0.153925 |

Verdict（裁决）:

- Small positive NDCG（归一化折损累计增益） transfer.
- Mixed HR（命中率） transfer because HR@10 drops.
- Supporting evidence only（仅支持证据）, not a clean transfer win（不是干净迁移胜利）.

Registry row（总账记录）:

- `sft_office_r690b_lmh_l1w030_l2w010_l3w020_title_on_desc_p05_20260509_065300`
