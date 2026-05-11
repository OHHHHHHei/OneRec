# Baseline vs Hierarchy Final SID Local Ambiguity Analysis

## Conclusion

- `same_l2` 有改善：测试目标 item 的平均 `l2` 叶子数从 `4.3422` 降到 `4.1266`。
- 测试目标落在多叶 `same_l2` bucket 的比例从 `0.4873` 变到 `0.5831`。
- 测试目标落在深拥挤 `l2` bucket (`>=4` leaves) 的比例从 `0.2228` 变到 `0.2585`。
- 测试样本里，有 `0.1015` 的目标 item 从 `same_l2` 多叶 bucket 被移到单叶 bucket；只有 `0.1972` 被移入更拥挤的 `same_l2` bucket。

## Catalog-Level

| Metric | Baseline | Hierarchy | Delta |
|---|---:|---:|---:|
| Item-weighted mean l2 leaf count | 2.394466 | 2.574878 | +0.180412 |
| Fraction items in multi-leaf l2 | 0.423494 | 0.519262 | +0.095768 |
| Fraction items in l2 with >=4 leaves | 0.130765 | 0.174986 | +0.044221 |
| Weighted H(level3|level1,level2) | 0.715048 | 0.869628 | +0.154580 |

## Test-Weighted

| Metric | Baseline | Hierarchy | Delta |
|---|---:|---:|---:|
| Mean target l2 leaf count | 4.342158 | 4.126627 | -0.215531 |
| Fraction targets in multi-leaf l2 | 0.487315 | 0.583058 | +0.095742 |
| Fraction targets in l2 with >=4 leaves | 0.222811 | 0.258548 | +0.035738 |
| Mean target l3 entropy under l2 | 1.100115 | 1.212795 | +0.112681 |

## Movement Summary

- `SID` changed on `100.00\%` of catalog items.
- Test-weighted targets with reduced `l2` leaf count: `25.50\%`.
- Test-weighted targets with increased `l2` leaf count: `30.73\%`.
- Test-weighted targets moved out of multi-leaf `same_l2`: `10.15\%`.
- Test-weighted targets moved into multi-leaf `same_l2`: `19.72\%`.
- Test-weighted mean delta of `l2` leaf count: `-0.215531`.

## Top Improved Examples

- `2161` | `25 -> 2` | 3D Solutech Natural Clear 1.75mm 3D Printer PLA Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG)
- `2697` | `25 -> 2` | 3D Solutech Real White 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG)
- `1851` | `23 -> 2` | HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, Yellow
- `3435` | `25 -> 7` | 3D Solutech Real Black 1.75mm Flexible 3D Printer Filament 2.2 LBS (1.0KG)
- `3474` | `25 -> 7` | 3D Solutech See Through Red 1.75mm PETG 3D Printer Filament 2.2 LBS (1.0KG)
- `3547` | `25 -> 7` | 3D Solutech See Through Blue 1.75mm PETG 3D Printer Filament 2.2 LBS (1.0KG)
- `3557` | `25 -> 7` | 3D Solutech Chocolate Brown 3D Printer PLA Filament 1.75MM Filament 2.2 LBS (1.0KG)
- `3681` | `25 -> 7` | 3D Solutech Real Orange 1.75mm ABS 3D Printer Filament 2.2 LBS (1.0KG)
- `3382` | `9 -> 1` | E-Projects 100EP514200R 200 Ohm Resistors, 1/4 W, 5% (Pack of 100)
- `730` | `7 -> 1` | Crest Pro-Health Advanced Extra Deep Clean Toothpaste Twin Pack, 5.1 Ounce
