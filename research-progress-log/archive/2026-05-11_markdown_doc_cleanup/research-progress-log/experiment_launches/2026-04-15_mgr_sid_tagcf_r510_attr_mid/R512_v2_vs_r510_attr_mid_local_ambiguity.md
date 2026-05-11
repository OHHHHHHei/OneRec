# Baseline vs Hierarchy Final SID Local Ambiguity Analysis

## Conclusion

- `same_l2` 有改善：测试目标 item 的平均 `l2` 叶子数从 `4.3422` 降到 `3.6848`。
- 测试目标落在多叶 `same_l2` bucket 的比例从 `0.4873` 变到 `0.5277`。
- 测试目标落在深拥挤 `l2` bucket (`>=4` leaves) 的比例从 `0.2228` 变到 `0.2285`。
- 测试样本里，有 `0.1202` 的目标 item 从 `same_l2` 多叶 bucket 被移到单叶 bucket；只有 `0.1606` 被移入更拥挤的 `same_l2` bucket。

## Catalog-Level

| Metric | Baseline | Hierarchy | Delta |
|---|---:|---:|---:|
| Item-weighted mean l2 leaf count | 2.394466 | 2.398806 | +0.004341 |
| Fraction items in multi-leaf l2 | 0.423494 | 0.473684 | +0.050190 |
| Fraction items in l2 with >=4 leaves | 0.130765 | 0.134292 | +0.003527 |
| Weighted H(level3|level1,level2) | 0.715048 | 0.776531 | +0.061482 |

## Test-Weighted

| Metric | Baseline | Hierarchy | Delta |
|---|---:|---:|---:|
| Mean target l2 leaf count | 4.342158 | 3.684756 | -0.657401 |
| Fraction targets in multi-leaf l2 | 0.487315 | 0.527686 | +0.040371 |
| Fraction targets in l2 with >=4 leaves | 0.222811 | 0.228546 | +0.005736 |
| Mean target l3 entropy under l2 | 1.100115 | 1.089844 | -0.010271 |

## Movement Summary

- `SID` changed on `100.00\%` of catalog items.
- Test-weighted targets with reduced `l2` leaf count: `23.47\%`.
- Test-weighted targets with increased `l2` leaf count: `26.98\%`.
- Test-weighted targets moved out of multi-leaf `same_l2`: `12.02\%`.
- Test-weighted targets moved into multi-leaf `same_l2`: `16.06\%`.
- Test-weighted mean delta of `l2` leaf count: `-0.657401`.

## Top Improved Examples

- `2659` | `25 -> 2` | 3D Solutech Apple Green 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG)
- `3683` | `25 -> 2` | 3D Solutech Purple 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG)
- `3547` | `25 -> 3` | 3D Solutech See Through Blue 1.75mm PETG 3D Printer Filament 2.2 LBS (1.0KG)
- `3112` | `25 -> 4` | 3D Solutech Real Black 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - PLA175RBLK
- `3435` | `25 -> 4` | 3D Solutech Real Black 1.75mm Flexible 3D Printer Filament 2.2 LBS (1.0KG)
- `3474` | `25 -> 4` | 3D Solutech See Through Red 1.75mm PETG 3D Printer Filament 2.2 LBS (1.0KG)
- `3522` | `25 -> 4` | 3D Solutech Real Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - ST176BLPLA
- `3681` | `25 -> 4` | 3D Solutech Real Orange 1.75mm ABS 3D Printer Filament 2.2 LBS (1.0KG)
- `1851` | `23 -> 3` | HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, Yellow
- `3453` | `25 -> 5` | 3D Solutech Real Gold 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG)
