# Baseline vs Hierarchy Final SID Local Ambiguity Analysis

## Conclusion

- `same_l2` 有改善：测试目标 item 的平均 `l2` 叶子数从 `4.3422` 降到 `2.5711`。
- 测试目标落在多叶 `same_l2` bucket 的比例从 `0.4873` 变到 `0.3816`。
- 测试目标落在深拥挤 `l2` bucket (`>=4` leaves) 的比例从 `0.2228` 变到 `0.1463`。
- 测试样本里，有 `0.1840` 的目标 item 从 `same_l2` 多叶 bucket 被移到单叶 bucket；只有 `0.0783` 被移入更拥挤的 `same_l2` bucket。

## Catalog-Level

| Metric | Baseline | Hierarchy | Delta |
|---|---:|---:|---:|
| Item-weighted mean l2 leaf count | 2.394466 | 1.812263 | -0.582203 |
| Fraction items in multi-leaf l2 | 0.423494 | 0.314433 | -0.109061 |
| Fraction items in l2 with >=4 leaves | 0.130765 | 0.076506 | -0.054259 |
| Weighted H(level3|level1,level2) | 0.715048 | 0.471414 | -0.243634 |

## Test-Weighted

| Metric | Baseline | Hierarchy | Delta |
|---|---:|---:|---:|
| Mean target l2 leaf count | 4.342158 | 2.571145 | -1.771013 |
| Fraction targets in multi-leaf l2 | 0.487315 | 0.381646 | -0.105670 |
| Fraction targets in l2 with >=4 leaves | 0.222811 | 0.146261 | -0.076550 |
| Mean target l3 entropy under l2 | 1.100115 | 0.715644 | -0.384471 |

## Movement Summary

- `SID` changed on `100.00\%` of catalog items.
- Test-weighted targets with reduced `l2` leaf count: `35.80\%`.
- Test-weighted targets with increased `l2` leaf count: `10.30\%`.
- Test-weighted targets moved out of multi-leaf `same_l2`: `18.40\%`.
- Test-weighted targets moved into multi-leaf `same_l2`: `7.83\%`.
- Test-weighted mean delta of `l2` leaf count: `-1.771013`.

## Top Improved Examples

- `3453` | `25 -> 2` | 3D Solutech Real Gold 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG)
- `3466` | `25 -> 2` | 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - RA-I00I-DQUY
- `3493` | `25 -> 2` | 3D Solutech Printer Filament, Real Blue PLA, 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 1.1 LBS (0.5KG)
- `3592` | `25 -> 2` | 3D Solutech Printer Filament, Real Black PLA, 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 1.1 LBS (0.5KG)
- `1851` | `23 -> 1` | HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, Yellow
- `1552` | `25 -> 3` | 3D Solutech Teal Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG)
- `2659` | `25 -> 3` | 3D Solutech Apple Green 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG)
- `2660` | `25 -> 3` | 3D Solutech Real Grey 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG)
- `3112` | `25 -> 3` | 3D Solutech Real Black 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - PLA175RBLK
- `3522` | `25 -> 3` | 3D Solutech Real Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - ST176BLPLA
