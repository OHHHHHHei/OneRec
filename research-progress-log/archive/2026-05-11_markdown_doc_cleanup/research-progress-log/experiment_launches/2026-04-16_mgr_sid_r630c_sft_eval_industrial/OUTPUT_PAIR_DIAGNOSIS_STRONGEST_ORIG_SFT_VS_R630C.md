# Output Pair Diagnosis（输出成对诊断）

## Scope（范围）

This note compares `strongest_original_sft` against `R630c` using the aligned per-example top-k comparison rows（逐样本对齐 top-k 对比行）.

## Headline（摘要）

- `top1`: `R630c` gains on `54` examples but loses on `77` examples.
- `top10`: `R630c` gains on `106` examples but loses on `202` examples.
- Among `top1` losses, `68.8%` stay within `2-10`, `20.8%` fall to `11-50`, and `10.4%` collapse beyond `50`.
- Among `top10` losses, `56.4%` only fall behind `10` but stay in `11-50`, while `43.6%` disappear beyond `50`.

## Retention Diagnosis（保留诊断）

- For `top1` losses, the exact target is still inside `R630c` `top10` on `68.8%` of lost examples.
- For `top10` losses, the exact target is still inside `R630c` `top50` on `56.4%` of lost examples.
- On the remaining `top10` losses, `12.9%` keep only a same-`l2` neighbor（同 `l2` 邻居）, `13.9%` keep only a same-`l1` neighbor（同 `l1` 邻居）, and `16.8%` lose the whole local neighborhood（局部邻域）.
- This separates rank-drop（名次下掉） from neighborhood-collapse（邻域坍塌）: if the exact target is still in `top50`, the main problem is beam retention（候选束保留） rather than total routing failure（整体路由失败）.

## Structure vs Output（结构与输出）

- On `top1` losses, baseline mean target `l2` fanout is `11.16883` vs hierarchy `6.36364`.
- On `top10` losses, baseline mean target `l2` fanout is `10.07426` vs hierarchy `5.57426`.
- On `top10` losses, hierarchy `l2` fanout does not increase on `85.1%` of examples.
- On `top10` gains, hierarchy `l2` fanout does not increase on `83.0%` of examples.
- So a cleaner local structure（更干净的局部结构） can appear on both gains and losses; tokenizer-side crowding reduction（分词器侧拥挤度降低） alone is not sufficient to explain downstream behavior（下游行为）.

## History Length（历史长度）

- `top10` losses by history bucket: `1-3=30.2%`, `4-7=54.0%`, `8+=15.8%`.
- `top10` gains by history bucket: `1-3=33.0%`, `4-7=53.8%`, `8+=13.2%`.

## Item Hotspots（物品热点）

### Worst `top10` deltas（最差 `top10` 差值）

| count | delta_top10 | delta_top1 | baseline_top10 | hierarchy_top10 | baseline_mean_l2 | hierarchy_mean_l2 | item |
|---:|---:|---:|---:|---:|---:|---:|---|
| 5 | -1.000 | +0.000 | 1.000 | 0.000 | 19.0 | 20.0 | 2934: Inland 1.75mm Yellow PLA 3D Printer Filament - 1kg Spool (2.2 lbs) |
| 10 | -0.800 | -0.200 | 0.900 | 0.100 | 24.0 | 12.0 | 3494: 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) |
| 10 | -0.700 | +0.000 | 0.700 | 0.000 | 24.0 | 12.0 | 2660: 3D Solutech Real Grey 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) |
| 12 | -0.583 | +0.000 | 0.583 | 0.000 | 24.0 | 12.0 | 3453: 3D Solutech Real Gold 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) |
| 6 | -0.500 | +0.000 | 0.500 | 0.000 | 24.0 | 9.0 | 3631: 3D Solutech Skin 3D Printer PLA Filament 1.75MM Filament 2.2 LBS (1.0KG) |
| 4 | -0.500 | +0.000 | 0.500 | 0.000 | 2.0 | 2.0 | 42: Precision Brand M6S Micro Seal, Miniature All Stainless Worm Gear Hose Clamp, 5/16" - 7/8" (Pack of 10) |
| 4 | -0.500 | +0.000 | 0.500 | 0.000 | 2.0 | 1.0 | 2204: Proto-pasta CFP11705 The Original Carbon Fiber Spool , PLA 1.75 mm, 500 g , Black |
| 7 | -0.429 | -0.286 | 0.429 | 0.000 | 2.0 | 1.0 | 416: PRO 1 Fuel Line Hose 1/4 Inch Inside Diameter X 25 Feet Length NRB/PVCC SAE30R6 |
| 5 | -0.400 | +0.000 | 0.600 | 0.200 | 8.0 | 1.0 | 125: Gorilla 6100101 Tape Handy Roll, 1-Pack, Black |
| 8 | -0.375 | +0.000 | 0.500 | 0.125 | 24.0 | 9.0 | 3557: 3D Solutech Chocolate Brown 3D Printer PLA Filament 1.75MM Filament 2.2 LBS (1.0KG) |

### Best `top10` deltas（最好 `top10` 差值）

| count | delta_top10 | delta_top1 | baseline_top10 | hierarchy_top10 | baseline_mean_l2 | hierarchy_mean_l2 | item |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | +0.500 | +0.000 | 0.000 | 0.500 | 10.0 | 10.0 | 450: X-Treme Tape TPE-X36ZLB Silicone Rubber Self Fusing Tape, 1" x 36', Triangular, Black |
| 4 | +0.500 | +0.000 | 0.000 | 0.500 | 1.0 | 3.0 | 1123: Smooth-On XTC-3D High Performance 3D Print Coating - 24oz. Unit |
| 5 | +0.400 | +0.000 | 0.000 | 0.400 | 1.0 | 1.0 | 970: Wixey WR25 Mini Digital Height Gauge |
| 5 | +0.400 | +0.000 | 0.400 | 0.800 | 19.0 | 20.0 | 3114: Inland 1.75mm Purple PLA 3D Printer Filament - 1kg Spool (2.2 lbs) |
| 17 | +0.294 | +0.000 | 0.118 | 0.412 | 3.0 | 2.0 | 3016: HATCHBOX 3D Printer Filament, Dimensional Accuracy +/- 0.03mm, 1.75 mm, 1 kg Spool, Wood |
| 7 | +0.286 | +0.000 | 0.000 | 0.286 | 1.0 | 1.0 | 54: Industrial & Scientific" /> |
| 7 | +0.286 | +0.000 | 0.000 | 0.286 | 1.0 | 1.0 | 2059: MG Chemicals Wood 3D Printer Filament, 1.75mm, 0.5 Kg (1.1 lbs.) - Wood |
| 15 | +0.267 | -0.067 | 0.667 | 0.933 | 47.0 | 24.0 | 1850: HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, Red |
| 8 | +0.250 | +0.250 | 0.250 | 0.500 | 8.0 | 1.0 | 175: Gorilla Crystal Clear Duct Tape, 1.88&rdquo; x 9 yd, Clear, (Pack of 1) |
| 8 | +0.250 | +0.000 | 0.000 | 0.250 | 1.0 | 1.0 | 278: T-Rex 241309 Shurtech Ferociously Strong Tape, 12 Yd L X 1.88 in W, 1-Roll, 12 Yards, Gunmetal Gray |

## Takeaways（结论）

- If `R630c` loses mostly by dropping exact targets from `top10` to `11-50` or `>50`, the core failure is beam retention（候选束保留） rather than simple local reranking（局部重排）.
- If many losses happen even when hierarchy-side `l2` fanout does not increase, tokenizer-side structural cleanup（分词器侧结构清理） is not a reliable explanation for downstream improvement（下游提升）.
- The item hotspot table（物品热点表） helps distinguish systematic category failures（系统性类别失败） from random per-example noise（逐样本随机噪声）.
