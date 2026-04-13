# Baseline vs Hierarchy Final SID Local Ambiguity Analysis

## Conclusion

- `same_l2` 有改善：测试目标 item 的平均 `l2` 叶子数从 `4.3422` 降到 `3.6148`。
- 测试目标落在多叶 `same_l2` bucket 的比例从 `0.4873` 变到 `0.4988`。
- 测试目标落在深拥挤 `l2` bucket (`>=4` leaves) 的比例从 `0.2228` 变到 `0.1994`。
- 测试样本里，有 `0.1257` 的目标 item 从 `same_l2` 多叶 bucket 被移到单叶 bucket；只有 `0.1372` 被移入更拥挤的 `same_l2` bucket。

## Catalog-Level

| Metric | Baseline | Hierarchy | Delta |
|---|---:|---:|---:|
| Item-weighted mean l2 leaf count | 2.394466 | 2.340206 | -0.054259 |
| Fraction items in multi-leaf l2 | 0.423494 | 0.444655 | +0.021161 |
| Fraction items in l2 with >=4 leaves | 0.130765 | 0.125068 | -0.005697 |
| Weighted H(level3|level1,level2) | 0.715048 | 0.728211 | +0.013163 |

## Test-Weighted

| Metric | Baseline | Hierarchy | Delta |
|---|---:|---:|---:|
| Mean target l2 leaf count | 4.342158 | 3.614825 | -0.727333 |
| Fraction targets in multi-leaf l2 | 0.487315 | 0.498787 | +0.011471 |
| Fraction targets in l2 with >=4 leaves | 0.222811 | 0.199426 | -0.023384 |
| Mean target l3 entropy under l2 | 1.100115 | 1.030772 | -0.069343 |

## Movement Summary

- `SID` changed on `100.00\%` of catalog items.
- Test-weighted targets with reduced `l2` leaf count: `24.93\%`.
- Test-weighted targets with increased `l2` leaf count: `21.42\%`.
- Test-weighted targets moved out of multi-leaf `same_l2`: `12.57\%`.
- Test-weighted targets moved into multi-leaf `same_l2`: `13.72\%`.
- Test-weighted mean delta of `l2` leaf count: `-0.727333`.

## Top Improved Examples

- `2161` | `25 -> 1` | 3D Solutech Natural Clear 1.75mm 3D Printer PLA Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG)
- `3474` | `25 -> 1` | 3D Solutech See Through Red 1.75mm PETG 3D Printer Filament 2.2 LBS (1.0KG)
- `3557` | `25 -> 1` | 3D Solutech Chocolate Brown 3D Printer PLA Filament 1.75MM Filament 2.2 LBS (1.0KG)
- `3435` | `25 -> 2` | 3D Solutech Real Black 1.75mm Flexible 3D Printer Filament 2.2 LBS (1.0KG)
- `3681` | `25 -> 2` | 3D Solutech Real Orange 1.75mm ABS 3D Printer Filament 2.2 LBS (1.0KG)
- `1851` | `23 -> 1` | HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, Yellow
- `3442` | `25 -> 3` | 3D Solutech Real Purple 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG)
- `3475` | `25 -> 3` | 3D Solutech Real Orange 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG)
- `3631` | `25 -> 3` | 3D Solutech Skin 3D Printer PLA Filament 1.75MM Filament 2.2 LBS (1.0KG)
- `3547` | `25 -> 4` | 3D Solutech See Through Blue 1.75mm PETG 3D Printer Filament 2.2 LBS (1.0KG)
