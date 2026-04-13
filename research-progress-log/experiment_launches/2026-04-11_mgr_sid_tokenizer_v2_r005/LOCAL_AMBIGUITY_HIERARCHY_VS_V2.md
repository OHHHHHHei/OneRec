# Baseline vs Hierarchy Final SID Local Ambiguity Analysis

## Conclusion

- `same_l2` 有改善：测试目标 item 的平均 `l2` 叶子数从 `4.4498` 降到 `4.3422`。
- 测试目标落在多叶 `same_l2` bucket 的比例从 `0.6131` 变到 `0.4873`。
- 测试目标落在深拥挤 `l2` bucket (`>=4` leaves) 的比例从 `0.2828` 变到 `0.2228`。
- 测试样本里，有 `0.2332` 的目标 item 从 `same_l2` 多叶 bucket 被移到单叶 bucket；只有 `0.1074` 被移入更拥挤的 `same_l2` bucket。

## Catalog-Level

| Metric | Baseline | Hierarchy | Delta |
|---|---:|---:|---:|
| Item-weighted mean l2 leaf count | 2.659251 | 2.394466 | -0.264786 |
| Fraction items in multi-leaf l2 | 0.495388 | 0.423494 | -0.071894 |
| Fraction items in l2 with >=4 leaves | 0.179056 | 0.130765 | -0.048291 |
| Weighted H(level3|level1,level2) | 0.866407 | 0.715048 | -0.151359 |

## Test-Weighted

| Metric | Baseline | Hierarchy | Delta |
|---|---:|---:|---:|
| Mean target l2 leaf count | 4.449812 | 4.342158 | -0.107655 |
| Fraction targets in multi-leaf l2 | 0.613060 | 0.487315 | -0.125745 |
| Fraction targets in l2 with >=4 leaves | 0.282815 | 0.222811 | -0.060004 |
| Mean target l3 entropy under l2 | 1.293525 | 1.100115 | -0.193411 |

## Movement Summary

- `SID` changed on `100.00\%` of catalog items.
- Test-weighted targets with reduced `l2` leaf count: `38.52\%`.
- Test-weighted targets with increased `l2` leaf count: `21.69\%`.
- Test-weighted targets moved out of multi-leaf `same_l2`: `23.32\%`.
- Test-weighted targets moved into multi-leaf `same_l2`: `10.74\%`.
- Test-weighted mean delta of `l2` leaf count: `-0.107655`.

## Top Improved Examples

- `3096` | `26 -> 1` | HATCHBOX 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, Bronze
- `2701` | `26 -> 2` | HATCHBOX 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, Transparent Black
- `3016` | `26 -> 2` | HATCHBOX 3D Printer Filament, Dimensional Accuracy +/- 0.03mm, 1.75 mm, 1 kg Spool, Wood
- `3572` | `15 -> 3` | Inland 1.75mm Black ABS 3D Printer Filament - 1kg Spool (2.2 lbs)
- `3645` | `15 -> 3` | Inland 1.75mm Natural PLA 3D Printer Filament - 1kg Spool (2.2 lbs)
- `1203` | `17 -> 7` | eSUN 3D 1.75mm PETG Natural Filament 1kg (2.2lb), PETG 3D Printer Filament, Semi-Transparent 1.75mm Natural
- `2157` | `17 -> 7` | eSUN 3D 1.75mm PETG Blue Filament 1kg (2.2lb), PETG 3D Printer Filament, 1.75mm Semi-Transparent Blue
- `2158` | `17 -> 7` | eSUN 3D 1.75mm PETG Green Filament 1kg (2.2lb), PETG 3D Printer Filament, Semi-Transparent 1.75mm Green
- `2703` | `17 -> 7` | eSUN 3D 1.75mm PETG Black Filament 1kg (2.2lb), PETG 3D Printer Filament, 1.75mm Solid Opaque Black
- `2893` | `17 -> 7` | eSUN 3D 1.75mm PETG White Filament 1kg (2.2lb), PETG 3D Printer Filament, 1.75mm Solid Opaque White
