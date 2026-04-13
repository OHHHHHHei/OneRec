# Baseline vs Hierarchy Final SID Local Ambiguity Analysis

## Conclusion

- `same_l2` 没改善：测试目标 item 的平均 `l2` 叶子数从 `3.6148` 变到 `4.9572`。
- 测试目标落在多叶 `same_l2` bucket 的比例从 `0.4988` 变到 `0.5449`。
- 测试目标落在深拥挤 `l2` bucket (`>=4` leaves) 的比例从 `0.1994` 变到 `0.2621`。
- 测试样本里，有 `0.1227` 的目标 item 从 `same_l2` 多叶 bucket 被移到单叶 bucket；只有 `0.1688` 被移入更拥挤的 `same_l2` bucket。

## Catalog-Level

| Metric | Baseline | Hierarchy | Delta |
|---|---:|---:|---:|
| Item-weighted mean l2 leaf count | 2.340206 | 2.814162 | +0.473956 |
| Fraction items in multi-leaf l2 | 0.444655 | 0.520890 | +0.076234 |
| Fraction items in l2 with >=4 leaves | 0.125068 | 0.190993 | +0.065925 |
| Weighted H(level3|level1,level2) | 0.728211 | 0.923560 | +0.195349 |

## Test-Weighted

| Metric | Baseline | Hierarchy | Delta |
|---|---:|---:|---:|
| Mean target l2 leaf count | 3.614825 | 4.957203 | +1.342378 |
| Fraction targets in multi-leaf l2 | 0.498787 | 0.544893 | +0.046106 |
| Fraction targets in l2 with >=4 leaves | 0.199426 | 0.262078 | +0.062652 |
| Mean target l3 entropy under l2 | 1.030772 | 1.262250 | +0.231479 |

## Movement Summary

- `SID` changed on `100.00\%` of catalog items.
- Test-weighted targets with reduced `l2` leaf count: `17.54\%`.
- Test-weighted targets with increased `l2` leaf count: `37.30\%`.
- Test-weighted targets moved out of multi-leaf `same_l2`: `12.27\%`.
- Test-weighted targets moved into multi-leaf `same_l2`: `16.88\%`.
- Test-weighted mean delta of `l2` leaf count: `+1.342378`.

## Top Improved Examples

- `1206` | `11 -> 1` | eSUN 1.75mm Black PLA PRO (PLA+) 3D Printer Filament 1KG Spool (2.2lbs), Black
- `2284` | `11 -> 1` | HICTOP 1.75mm Black PLA 3D Printer Filament - 1kg Spool (2.2 lbs) - Dimensional Accuracy +/- 0.02mm
- `2740` | `11 -> 1` | 1.75mm White PLA 3D Printer Filament - 1kg Spool (2.2 lbs) - Dimensional Accuracy +/- 0.03mm
- `319` | `12 -> 3` | 6061 Aluminum Rectangular Bar, Unpolished (Mill) Finish, Extruded, T6511 Temper, ASTM B221, 1/8" Thickness, 1" Width, 36" Length
- `323` | `12 -> 3` | 6101 Aluminum Rectangular Bar, Unpolished (Mill) Finish, T61 Temper, ASTM B317, 1/4" Thickness, 2" Width, 12" Length
- `2868` | `12 -> 3` | 6061 Aluminum Rectangular Bar, Unpolished (Mill) Finish, Extruded, T6511 Temper, ASTM B211/ASTM B221, 1-1/2" Thickness, 1-1/2" Width, 12" Length
- `1146` | `8 -> 1` | SummitLink 218 Pcs Black Assorted Heat Shrink Tube 8 Sizes Tubing Wrap Sleeve Set Combo
- `1290` | `8 -> 1` | SummitLink 306 Pcs Red Black Assorted Heat Shrink Tube 8 Sizes Tubing Wrap Sleeve Set Combo
- `2719` | `11 -> 4` | MeltInk3D Silver PLA 3D Printer Filament &Oslash; 1.75mm, 1Kg (2.2 Lb), MADE in U.S.A, Dimensional Accuracy: &plusmn; 0.05mm
- `2739` | `11 -> 4` | 1.75mm Black PLA 3D Printer Filament - 1kg Spool (2.2 lbs) - Dimensional Accuracy +/- 0.03mm
