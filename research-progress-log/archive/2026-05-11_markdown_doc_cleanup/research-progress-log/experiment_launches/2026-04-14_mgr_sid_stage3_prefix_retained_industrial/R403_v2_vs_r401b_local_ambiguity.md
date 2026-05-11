# Baseline vs Hierarchy Final SID Local Ambiguity Analysis

## Conclusion

- `same_l2` 有改善：测试目标 item 的平均 `l2` 叶子数从 `4.3422` 降到 `2.6967`。
- 测试目标落在多叶 `same_l2` bucket 的比例从 `0.4873` 变到 `0.3880`。
- 测试目标落在深拥挤 `l2` bucket (`>=4` leaves) 的比例从 `0.2228` 变到 `0.1522`。
- 测试样本里，有 `0.1796` 的目标 item 从 `same_l2` 多叶 bucket 被移到单叶 bucket；只有 `0.0803` 被移入更拥挤的 `same_l2` bucket。

## Catalog-Level

| Metric | Baseline | Hierarchy | Delta |
|---|---:|---:|---:|
| Item-weighted mean l2 leaf count | 2.394466 | 1.853228 | -0.541237 |
| Fraction items in multi-leaf l2 | 0.423494 | 0.304666 | -0.118828 |
| Fraction items in l2 with >=4 leaves | 0.130765 | 0.071894 | -0.058871 |
| Weighted H(level3|level1,level2) | 0.715048 | 0.470394 | -0.244654 |

## Test-Weighted

| Metric | Baseline | Hierarchy | Delta |
|---|---:|---:|---:|
| Mean target l2 leaf count | 4.342158 | 2.696669 | -1.645489 |
| Fraction targets in multi-leaf l2 | 0.487315 | 0.388043 | -0.099272 |
| Fraction targets in l2 with >=4 leaves | 0.222811 | 0.152217 | -0.070593 |
| Mean target l3 entropy under l2 | 1.100115 | 0.737341 | -0.362774 |

## Movement Summary

- `SID` changed on `100.00\%` of catalog items.
- Test-weighted targets with reduced `l2` leaf count: `32.91\%`.
- Test-weighted targets with increased `l2` leaf count: `11.85\%`.
- Test-weighted targets moved out of multi-leaf `same_l2`: `17.96\%`.
- Test-weighted targets moved into multi-leaf `same_l2`: `8.03\%`.
- Test-weighted mean delta of `l2` leaf count: `-1.645489`.

## Top Improved Examples

- `3522` | `25 -> 1` | 3D Solutech Real Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - ST176BLPLA
- `2993` | `25 -> 2` | 3D Solutech Blue 3D Printer Ultra PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG)
- `3112` | `25 -> 2` | 3D Solutech Real Black 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - PLA175RBLK
- `3435` | `25 -> 2` | 3D Solutech Real Black 1.75mm Flexible 3D Printer Filament 2.2 LBS (1.0KG)
- `3493` | `25 -> 2` | 3D Solutech Printer Filament, Real Blue PLA, 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 1.1 LBS (0.5KG)
- `3592` | `25 -> 2` | 3D Solutech Printer Filament, Real Black PLA, 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 1.1 LBS (0.5KG)
- `3599` | `25 -> 2` | 3D Solutech Aqua Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG)
- `3681` | `25 -> 2` | 3D Solutech Real Orange 1.75mm ABS 3D Printer Filament 2.2 LBS (1.0KG)
- `1851` | `23 -> 1` | HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, Yellow
- `1888` | `25 -> 4` | 3D Solutech Hot Pink 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG)
