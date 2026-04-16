# Baseline vs Hierarchy Final SID Local Ambiguity Analysis

## Conclusion

- `same_l2` 有改善：测试目标 item 的平均 `l2` 叶子数从 `4.3422` 降到 `4.1282`。
- 测试目标落在多叶 `same_l2` bucket 的比例从 `0.4873` 变到 `0.6389`。
- 测试目标落在深拥挤 `l2` bucket (`>=4` leaves) 的比例从 `0.2228` 变到 `0.2994`。
- 测试样本里，有 `0.0927` 的目标 item 从 `same_l2` 多叶 bucket 被移到单叶 bucket；只有 `0.2442` 被移入更拥挤的 `same_l2` bucket。

## Catalog-Level

| Metric | Baseline | Hierarchy | Delta |
|---|---:|---:|---:|
| Item-weighted mean l2 leaf count | 2.394466 | 2.942485 | +0.548020 |
| Fraction items in multi-leaf l2 | 0.423494 | 0.606348 | +0.182854 |
| Fraction items in l2 with >=4 leaves | 0.130765 | 0.233044 | +0.102279 |
| Weighted H(level3|level1,level2) | 0.715048 | 1.066234 | +0.351186 |

## Test-Weighted

| Metric | Baseline | Hierarchy | Delta |
|---|---:|---:|---:|
| Mean target l2 leaf count | 4.342158 | 4.128171 | -0.213986 |
| Fraction targets in multi-leaf l2 | 0.487315 | 0.638871 | +0.151555 |
| Fraction targets in l2 with >=4 leaves | 0.222811 | 0.299360 | +0.076550 |
| Mean target l3 entropy under l2 | 1.100115 | 1.312902 | +0.212788 |

## Movement Summary

- `SID` changed on `100.00\%` of catalog items.
- Test-weighted targets with reduced `l2` leaf count: `22.52\%`.
- Test-weighted targets with increased `l2` leaf count: `39.58\%`.
- Test-weighted targets moved out of multi-leaf `same_l2`: `9.27\%`.
- Test-weighted targets moved into multi-leaf `same_l2`: `24.42\%`.
- Test-weighted mean delta of `l2` leaf count: `-0.213986`.

## Top Improved Examples

- `3474` | `25 -> 1` | 3D Solutech See Through Red 1.75mm PETG 3D Printer Filament 2.2 LBS (1.0KG)
- `3592` | `25 -> 2` | 3D Solutech Printer Filament, Real Black PLA, 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 1.1 LBS (0.5KG)
- `3557` | `25 -> 3` | 3D Solutech Chocolate Brown 3D Printer PLA Filament 1.75MM Filament 2.2 LBS (1.0KG)
- `3631` | `25 -> 3` | 3D Solutech Skin 3D Printer PLA Filament 1.75MM Filament 2.2 LBS (1.0KG)
- `3681` | `25 -> 3` | 3D Solutech Real Orange 1.75mm ABS 3D Printer Filament 2.2 LBS (1.0KG)
- `2419` | `19 -> 1` | HATCHBOX 3D ABS-1KG1.75-BLU ABS 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, Blue
- `3547` | `25 -> 9` | 3D Solutech See Through Blue 1.75mm PETG 3D Printer Filament 2.2 LBS (1.0KG)
- `3505` | `17 -> 2` | Taulman 3D 618 Natural Nylon Filament for 3D Printer 1.75mm 2 1lb Spool Bundle MADE IN USA
- `3112` | `25 -> 10` | 3D Solutech Real Black 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - PLA175RBLK
- `3522` | `25 -> 10` | 3D Solutech Real Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - ST176BLPLA
