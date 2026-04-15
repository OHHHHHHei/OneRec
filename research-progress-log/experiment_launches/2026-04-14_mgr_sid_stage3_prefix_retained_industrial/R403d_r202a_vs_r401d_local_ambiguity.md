# Baseline vs Hierarchy Final SID Local Ambiguity Analysis

## Conclusion

- `same_l2` 有改善：测试目标 item 的平均 `l2` 叶子数从 `3.6148` 降到 `2.5711`。
- 测试目标落在多叶 `same_l2` bucket 的比例从 `0.4988` 变到 `0.3816`。
- 测试目标落在深拥挤 `l2` bucket (`>=4` leaves) 的比例从 `0.1994` 变到 `0.1463`。
- 测试样本里，有 `0.1963` 的目标 item 从 `same_l2` 多叶 bucket 被移到单叶 bucket；只有 `0.0792` 被移入更拥挤的 `same_l2` bucket。

## Catalog-Level

| Metric | Baseline | Hierarchy | Delta |
|---|---:|---:|---:|
| Item-weighted mean l2 leaf count | 2.340206 | 1.812263 | -0.527944 |
| Fraction items in multi-leaf l2 | 0.444655 | 0.314433 | -0.130222 |
| Fraction items in l2 with >=4 leaves | 0.125068 | 0.076506 | -0.048562 |
| Weighted H(level3|level1,level2) | 0.728211 | 0.471414 | -0.256797 |

## Test-Weighted

| Metric | Baseline | Hierarchy | Delta |
|---|---:|---:|---:|
| Mean target l2 leaf count | 3.614825 | 2.571145 | -1.043680 |
| Fraction targets in multi-leaf l2 | 0.498787 | 0.381646 | -0.117141 |
| Fraction targets in l2 with >=4 leaves | 0.199426 | 0.146261 | -0.053166 |
| Mean target l3 entropy under l2 | 1.030772 | 0.715644 | -0.315128 |

## Movement Summary

- `SID` changed on `100.00\%` of catalog items.
- Test-weighted targets with reduced `l2` leaf count: `37.44\%`.
- Test-weighted targets with increased `l2` leaf count: `10.43\%`.
- Test-weighted targets moved out of multi-leaf `same_l2`: `19.63\%`.
- Test-weighted targets moved into multi-leaf `same_l2`: `7.92\%`.
- Test-weighted mean delta of `l2` leaf count: `-1.043680`.

## Top Improved Examples

- `1849` | `23 -> 3` | HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, Purple
- `2419` | `21 -> 2` | HATCHBOX 3D ABS-1KG1.75-BLU ABS 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, Blue
- `3645` | `15 -> 1` | Inland 1.75mm Natural PLA 3D Printer Filament - 1kg Spool (2.2 lbs)
- `3453` | `16 -> 2` | 3D Solutech Real Gold 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG)
- `3466` | `16 -> 2` | 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - RA-I00I-DQUY
- `3493` | `16 -> 2` | 3D Solutech Printer Filament, Real Blue PLA, 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 1.1 LBS (0.5KG)
- `3592` | `16 -> 2` | 3D Solutech Printer Filament, Real Black PLA, 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 1.1 LBS (0.5KG)
- `1552` | `16 -> 3` | 3D Solutech Teal Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG)
- `2659` | `16 -> 3` | 3D Solutech Apple Green 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG)
- `2660` | `16 -> 3` | 3D Solutech Real Grey 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG)
