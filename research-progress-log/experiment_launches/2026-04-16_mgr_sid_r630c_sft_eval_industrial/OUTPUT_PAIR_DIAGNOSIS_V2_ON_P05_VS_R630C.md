# Output Pair Diagnosis（输出成对诊断）

## Scope（范围）

This note compares `v2_on_p05` against `R630c` using the aligned per-example top-k comparison rows（逐样本对齐 top-k 对比行）.

## Headline（摘要）

- `top1`: `R630c` gains on `52` examples but loses on `91` examples.
- `top10`: `R630c` gains on `118` examples but loses on `193` examples.
- Among `top1` losses, `71.4%` stay within `2-10`, `11.0%` fall to `11-50`, and `17.6%` collapse beyond `50`.
- Among `top10` losses, `53.4%` only fall behind `10` but stay in `11-50`, while `46.6%` disappear beyond `50`.

## Retention Diagnosis（保留诊断）

- For `top1` losses, the exact target is still inside `R630c` `top10` on `71.4%` of lost examples.
- For `top10` losses, the exact target is still inside `R630c` `top50` on `53.4%` of lost examples.
- On the remaining `top10` losses, `10.4%` keep only a same-`l2` neighbor（同 `l2` 邻居）, `18.1%` keep only a same-`l1` neighbor（同 `l1` 邻居）, and `18.1%` lose the whole local neighborhood（局部邻域）.
- This separates rank-drop（名次下掉） from neighborhood-collapse（邻域坍塌）: if the exact target is still in `top50`, the main problem is beam retention（候选束保留） rather than total routing failure（整体路由失败）.

## Structure vs Output（结构与输出）

- On `top1` losses, baseline mean target `l2` fanout is `3.98901` vs hierarchy `3.54945`.
- On `top10` losses, baseline mean target `l2` fanout is `8.38860` vs hierarchy `5.05699`.
- On `top10` losses, hierarchy `l2` fanout does not increase on `79.3%` of examples.
- On `top10` gains, hierarchy `l2` fanout does not increase on `67.8%` of examples.
- So a cleaner local structure（更干净的局部结构） can appear on both gains and losses; tokenizer-side crowding reduction（分词器侧拥挤度降低） alone is not sufficient to explain downstream behavior（下游行为）.

## History Length（历史长度）

- `top10` losses by history bucket: `1-3=37.8%`, `4-7=46.6%`, `8+=15.5%`.
- `top10` gains by history bucket: `1-3=41.5%`, `4-7=46.6%`, `8+=11.9%`.

## Item Hotspots（物品热点）

### Worst `top10` deltas（最差 `top10` 差值）

| count | delta_top10 | delta_top1 | baseline_top10 | hierarchy_top10 | baseline_mean_l2 | hierarchy_mean_l2 | item |
|---:|---:|---:|---:|---:|---:|---:|---|
| 5 | -1.000 | +0.000 | 1.000 | 0.000 | 17.0 | 20.0 | 2934: Inland 1.75mm Yellow PLA 3D Printer Filament - 1kg Spool (2.2 lbs) |
| 11 | -0.818 | +0.000 | 0.818 | 0.000 | 25.0 | 9.0 | 3475: 3D Solutech Real Orange 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) |
| 6 | -0.667 | +0.000 | 0.667 | 0.000 | 2.0 | 4.0 | 560: Litmus pH Test Strips, Universal Application (pH 1-14), 2 Packs of 100 Strips |
| 14 | -0.643 | +0.000 | 0.643 | 0.000 | 25.0 | 3.0 | 3522: 3D Solutech Real Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - ST176BLPLA |
| 10 | -0.600 | +0.000 | 0.700 | 0.100 | 25.0 | 12.0 | 3494: 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) |
| 5 | -0.600 | +0.000 | 0.600 | 0.000 | 1.0 | 3.0 | 98: Industrial & Scientific" /> |
| 9 | -0.556 | +0.000 | 0.556 | 0.000 | 25.0 | 9.0 | 3442: 3D Solutech Real Purple 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) |
| 8 | -0.500 | +0.000 | 0.500 | 0.000 | 1.0 | 1.0 | 201: ColorConnex Coupler & Plug Kit (7 Piece), Industrial Type D, 1/4 in. NPT, Red - A73457D |
| 4 | -0.500 | -0.250 | 0.500 | 0.000 | 1.0 | 2.0 | 42: Precision Brand M6S Micro Seal, Miniature All Stainless Worm Gear Hose Clamp, 5/16" - 7/8" (Pack of 10) |
| 7 | -0.429 | -0.429 | 0.429 | 0.000 | 2.0 | 1.0 | 416: PRO 1 Fuel Line Hose 1/4 Inch Inside Diameter X 25 Feet Length NRB/PVCC SAE30R6 |

### Best `top10` deltas（最好 `top10` 差值）

| count | delta_top10 | delta_top1 | baseline_top10 | hierarchy_top10 | baseline_mean_l2 | hierarchy_mean_l2 | item |
|---:|---:|---:|---:|---:|---:|---:|---|
| 6 | +0.667 | +0.000 | 0.000 | 0.667 | 17.0 | 20.0 | 2718: Inland 1.75mm Pink PLA 3D Printer Filament - 1kg Spool (2.2 lbs) |
| 4 | +0.500 | +0.000 | 0.000 | 0.500 | 2.0 | 3.0 | 1123: Smooth-On XTC-3D High Performance 3D Print Coating - 24oz. Unit |
| 5 | +0.400 | +0.000 | 0.000 | 0.400 | 3.0 | 1.0 | 970: Wixey WR25 Mini Digital Height Gauge |
| 5 | +0.400 | +0.000 | 0.400 | 0.800 | 10.0 | 20.0 | 3114: Inland 1.75mm Purple PLA 3D Printer Filament - 1kg Spool (2.2 lbs) |
| 32 | +0.375 | +0.188 | 0.000 | 0.375 | 25.0 | 3.0 | 3112: 3D Solutech Real Black 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - PLA175RBLK |
| 8 | +0.375 | +0.125 | 0.375 | 0.750 | 21.0 | 21.0 | 444: HATCHBOX ABS 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, White |
| 8 | +0.375 | +0.125 | 0.000 | 0.375 | 1.0 | 2.0 | 2544: SainSmart Clear Flexible TPU 3D Printing Filament, 1.75 mm, 0.8 kg, Dimensional Accuracy +/- 0.05 mm |
| 8 | +0.250 | +0.000 | 0.000 | 0.250 | 1.0 | 1.0 | 278: T-Rex 241309 Shurtech Ferociously Strong Tape, 12 Yd L X 1.88 in W, 1-Roll, 12 Yards, Gunmetal Gray |
| 4 | +0.250 | +0.000 | 0.000 | 0.250 | 1.0 | 1.0 | 331: Stanley TRA700BN Heavy-Duty Staple & Brad Assortment, 2500-Pack |
| 4 | +0.250 | +0.000 | 0.250 | 0.500 | 10.0 | 10.0 | 975: X-Treme Tape TPE-XZLB Silicone Rubber Self Fusing Tape, 1" x 10', Triangular, Black |

## Takeaways（结论）

- If `R630c` loses mostly by dropping exact targets from `top10` to `11-50` or `>50`, the core failure is beam retention（候选束保留） rather than simple local reranking（局部重排）.
- If many losses happen even when hierarchy-side `l2` fanout does not increase, tokenizer-side structural cleanup（分词器侧结构清理） is not a reliable explanation for downstream improvement（下游提升）.
- The item hotspot table（物品热点表） helps distinguish systematic category failures（系统性类别失败） from random per-example noise（逐样本随机噪声）.
