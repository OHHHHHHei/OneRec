# Output Pair Diagnosis（输出成对诊断）

## Scope（范围）

This note compares `v2_on_p05` against `R650a` using the aligned per-example top-k comparison rows（逐样本对齐 top-k 对比行）.

## Headline（摘要）

- `top1`: `R650a` gains on `55` examples but loses on `79` examples.
- `top10`: `R650a` gains on `143` examples but loses on `206` examples.
- Among `top1` losses, `65.8%` stay within `2-10`, `24.1%` fall to `11-50`, and `10.1%` collapse beyond `50`.
- Among `top10` losses, `63.1%` only fall behind `10` but stay in `11-50`, while `36.9%` disappear beyond `50`.

## Retention Diagnosis（保留诊断）

- For `top1` losses, the exact target is still inside `R650a` `top10` on `65.8%` of lost examples.
- For `top10` losses, the exact target is still inside `R650a` `top50` on `63.1%` of lost examples.
- On the remaining `top10` losses, `4.9%` keep only a same-`l2` neighbor（同 `l2` 邻居）, `14.1%` keep only a same-`l1` neighbor（同 `l1` 邻居）, and `18.0%` lose the whole local neighborhood（局部邻域）.
- This separates rank-drop（名次下掉） from neighborhood-collapse（邻域坍塌）: if the exact target is still in `top50`, the main problem is beam retention（候选束保留） rather than total routing failure（整体路由失败）.

## Structure vs Output（结构与输出）

- On `top1` losses, baseline mean target `l2` fanout is `4.20253` vs hierarchy `3.98734`.
- On `top10` losses, baseline mean target `l2` fanout is `6.81553` vs hierarchy `5.51942`.
- On `top10` losses, hierarchy `l2` fanout does not increase on `73.3%` of examples.
- On `top10` gains, hierarchy `l2` fanout does not increase on `78.3%` of examples.
- So a cleaner local structure（更干净的局部结构） can appear on both gains and losses; tokenizer-side crowding reduction（分词器侧拥挤度降低） alone is not sufficient to explain downstream behavior（下游行为）.

## History Length（历史长度）

- `top10` losses by history bucket: `1-3=38.3%`, `4-7=48.5%`, `8+=13.1%`.
- `top10` gains by history bucket: `1-3=36.4%`, `4-7=50.3%`, `8+=13.3%`.

## Item Hotspots（物品热点）

### Worst `top10` deltas（最差 `top10` 差值）

| count | delta_top10 | delta_top1 | baseline_top10 | hierarchy_top10 | baseline_mean_l2 | hierarchy_mean_l2 | item |
|---:|---:|---:|---:|---:|---:|---:|---|
| 11 | -0.727 | +0.000 | 0.818 | 0.091 | 25.0 | 18.0 | 3475: 3D Solutech Real Orange 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) |
| 5 | -0.600 | +0.000 | 0.600 | 0.000 | 1.0 | 1.0 | 98: Industrial & Scientific" /> |
| 31 | -0.516 | -0.065 | 0.581 | 0.065 | 1.0 | 3.0 | 2909: 3D Solutech Silver Metal 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - PLA175TCMS |
| 14 | -0.500 | +0.000 | 0.643 | 0.143 | 25.0 | 3.0 | 3522: 3D Solutech Real Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - ST176BLPLA |
| 6 | -0.500 | +0.000 | 0.667 | 0.167 | 2.0 | 1.0 | 560: Litmus pH Test Strips, Universal Application (pH 1-14), 2 Packs of 100 Strips |
| 4 | -0.500 | +0.000 | 0.500 | 0.000 | 10.0 | 10.0 | 450: X-Treme Tape TPE-X36ZLB Silicone Rubber Self Fusing Tape, 1" x 36', Triangular, Black |
| 9 | -0.444 | +0.000 | 0.556 | 0.111 | 25.0 | 18.0 | 3442: 3D Solutech Real Purple 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) |
| 8 | -0.375 | +0.000 | 0.500 | 0.125 | 1.0 | 1.0 | 201: ColorConnex Coupler & Plug Kit (7 Piece), Industrial Type D, 1/4 in. NPT, Red - A73457D |
| 8 | -0.375 | -0.250 | 0.875 | 0.500 | 1.0 | 1.0 | 2807: VIVOSUN 6 Inch 440 CFM Inline Duct Ventilation Fan with Variable Speed Controller |
| 14 | -0.357 | -0.143 | 0.429 | 0.071 | 7.0 | 8.0 | 2703: eSUN 3D 1.75mm PETG Black Filament 1kg (2.2lb), PETG 3D Printer Filament, 1.75mm Solid Opaque Black |

### Best `top10` deltas（最好 `top10` 差值）

| count | delta_top10 | delta_top1 | baseline_top10 | hierarchy_top10 | baseline_mean_l2 | hierarchy_mean_l2 | item |
|---:|---:|---:|---:|---:|---:|---:|---|
| 11 | +0.818 | +0.636 | 0.091 | 0.909 | 25.0 | 18.0 | 3466: 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - RA-I00I-DQUY |
| 5 | +0.600 | +0.000 | 0.000 | 0.600 | 1.0 | 2.0 | 125: Gorilla 6100101 Tape Handy Roll, 1-Pack, Black |
| 12 | +0.500 | +0.000 | 0.000 | 0.500 | 25.0 | 18.0 | 3453: 3D Solutech Real Gold 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) |
| 5 | +0.400 | +0.200 | 0.000 | 0.400 | 3.0 | 3.0 | 970: Wixey WR25 Mini Digital Height Gauge |
| 5 | +0.400 | +0.000 | 0.200 | 0.600 | 4.0 | 3.0 | 675: J-B Weld 8265S Original Cold-Weld Steel Reinforced Epoxy - 2 oz. |
| 7 | +0.286 | +0.000 | 0.000 | 0.286 | 3.0 | 3.0 | 51: HATCHBOX 1 Spool 3D Printer Filament Tabletop Wall Mount Rack |
| 32 | +0.250 | +0.031 | 0.000 | 0.250 | 25.0 | 3.0 | 3112: 3D Solutech Real Black 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - PLA175RBLK |
| 16 | +0.250 | +0.062 | 0.000 | 0.250 | 1.0 | 1.0 | 57: Gorilla 2 Part Epoxy, 5 Minute Set, .85 ounce Syringe, Clear |
| 8 | +0.250 | +0.000 | 0.250 | 0.500 | 10.0 | 16.0 | 1553: Inland 1.75mm Red PLA 3D Printer Filament - 1kg Spool (2.2 lbs) |
| 8 | +0.250 | +0.000 | 0.000 | 0.250 | 10.0 | 16.0 | 1554: Inland 1.75mm Peak Green PLA 3D Printer Filament - 1kg Spool (2.2 lbs) |

## Takeaways（结论）

- If `R650a` loses mostly by dropping exact targets from `top10` to `11-50` or `>50`, the core failure is beam retention（候选束保留） rather than simple local reranking（局部重排）.
- If many losses happen even when hierarchy-side `l2` fanout does not increase, tokenizer-side structural cleanup（分词器侧结构清理） is not a reliable explanation for downstream improvement（下游提升）.
- The item hotspot table（物品热点表） helps distinguish systematic category failures（系统性类别失败） from random per-example noise（逐样本随机噪声）.
