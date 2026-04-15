# Baseline vs Hierarchy Final SID Local Ambiguity Analysis

## Conclusion

- `same_l2` 有改善：测试目标 item 的平均 `l2` 叶子数从 `3.6148` 降到 `2.6967`。
- 测试目标落在多叶 `same_l2` bucket 的比例从 `0.4988` 变到 `0.3880`。
- 测试目标落在深拥挤 `l2` bucket (`>=4` leaves) 的比例从 `0.1994` 变到 `0.1522`。
- 测试样本里，有 `0.1800` 的目标 item 从 `same_l2` 多叶 bucket 被移到单叶 bucket；只有 `0.0693` 被移入更拥挤的 `same_l2` bucket。

## Catalog-Level

| Metric | Baseline | Hierarchy | Delta |
|---|---:|---:|---:|
| Item-weighted mean l2 leaf count | 2.340206 | 1.853228 | -0.486978 |
| Fraction items in multi-leaf l2 | 0.444655 | 0.304666 | -0.139989 |
| Fraction items in l2 with >=4 leaves | 0.125068 | 0.071894 | -0.053174 |
| Weighted H(level3|level1,level2) | 0.728211 | 0.470394 | -0.257817 |

## Test-Weighted

| Metric | Baseline | Hierarchy | Delta |
|---|---:|---:|---:|
| Mean target l2 leaf count | 3.614825 | 2.696669 | -0.918156 |
| Fraction targets in multi-leaf l2 | 0.498787 | 0.388043 | -0.110743 |
| Fraction targets in l2 with >=4 leaves | 0.199426 | 0.152217 | -0.047209 |
| Mean target l3 entropy under l2 | 1.030772 | 0.737341 | -0.293431 |

## Movement Summary

- `SID` changed on `100.00\%` of catalog items.
- Test-weighted targets with reduced `l2` leaf count: `35.12\%`.
- Test-weighted targets with increased `l2` leaf count: `9.60\%`.
- Test-weighted targets moved out of multi-leaf `same_l2`: `18.00\%`.
- Test-weighted targets moved into multi-leaf `same_l2`: `6.93\%`.
- Test-weighted mean delta of `l2` leaf count: `-0.918156`.

## Top Improved Examples

- `2419` | `21 -> 1` | HATCHBOX 3D ABS-1KG1.75-BLU ABS 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, Blue
- `1849` | `23 -> 3` | HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, Purple
- `3522` | `16 -> 1` | 3D Solutech Real Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - ST176BLPLA
- `2993` | `16 -> 2` | 3D Solutech Blue 3D Printer Ultra PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG)
- `3112` | `16 -> 2` | 3D Solutech Real Black 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - PLA175RBLK
- `3493` | `16 -> 2` | 3D Solutech Printer Filament, Real Blue PLA, 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 1.1 LBS (0.5KG)
- `3592` | `16 -> 2` | 3D Solutech Printer Filament, Real Black PLA, 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 1.1 LBS (0.5KG)
- `3599` | `16 -> 2` | 3D Solutech Aqua Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG)
- `1888` | `16 -> 4` | 3D Solutech Hot Pink 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG)
- `2659` | `16 -> 4` | 3D Solutech Apple Green 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG)
