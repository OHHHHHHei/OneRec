# Baseline vs Hierarchy Final SID Local Ambiguity Analysis

## Conclusion

- `same_l2` 有改善：测试目标 item 的平均 `l2` 叶子数从 `4.7999` 降到 `4.4498`。
- 测试目标落在多叶 `same_l2` bucket 的比例从 `0.6894` 变到 `0.6131`。
- 测试目标落在深拥挤 `l2` bucket (`>=4` leaves) 的比例从 `0.3283` 变到 `0.2828`。
- 测试样本里，有 `0.1732` 的目标 item 从 `same_l2` 多叶 bucket 被移到单叶 bucket；只有 `0.0968` 被移入更拥挤的 `same_l2` bucket。

## Catalog-Level

| Metric | Baseline | Hierarchy | Delta |
|---|---:|---:|---:|
| Item-weighted mean l2 leaf count | 3.186923 | 2.659251 | -0.527672 |
| Fraction items in multi-leaf l2 | 0.600651 | 0.495388 | -0.105263 |
| Fraction items in l2 with >=4 leaves | 0.258003 | 0.179056 | -0.078947 |
| Weighted H(level3|level1,level2) | 1.111955 | 0.866407 | -0.245548 |

## Test-Weighted

| Metric | Baseline | Hierarchy | Delta |
|---|---:|---:|---:|
| Mean target l2 leaf count | 4.799912 | 4.449812 | -0.350099 |
| Fraction targets in multi-leaf l2 | 0.689389 | 0.613060 | -0.076329 |
| Fraction targets in l2 with >=4 leaves | 0.328259 | 0.282815 | -0.045445 |
| Mean target l3 entropy under l2 | 1.453294 | 1.293525 | -0.159769 |

## Movement Summary

- `SID` changed on `100.00\%` of catalog items.
- Test-weighted targets with reduced `l2` leaf count: `36.77\%`.
- Test-weighted targets with increased `l2` leaf count: `25.04\%`.
- Test-weighted targets moved out of multi-leaf `same_l2`: `17.32\%`.
- Test-weighted targets moved into multi-leaf `same_l2`: `9.68\%`.
- Test-weighted mean delta of `l2` leaf count: `-0.350099`.

## Top Improved Examples

- `2419` | `25 -> 1` | HATCHBOX 3D ABS-1KG1.75-BLU ABS 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, Blue
- `3631` | `22 -> 1` | 3D Solutech Skin 3D Printer PLA Filament 1.75MM Filament 2.2 LBS (1.0KG)
- `249` | `25 -> 6` | UP! ABS Plastic Filament, 1.75 mm Diameter, 1.54 lbs Spool, Black
- `2909` | `22 -> 4` | 3D Solutech Silver Metal 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - PLA175TCMS
- `3112` | `22 -> 4` | 3D Solutech Real Black 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - PLA175RBLK
- `3592` | `22 -> 4` | 3D Solutech Printer Filament, Real Black PLA, 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 1.1 LBS (0.5KG)
- `2665` | `25 -> 7` | Printrbot PLA Filament for 3D Printers, 1.75mm Diameter, Metallic Charcoal Gray, 1Kg Spool
- `3629` | `25 -> 7` | [STAR] Alchement - PLA Series, 3D Filament, 1.75mm, 1kg (Transparent)
- `3675` | `25 -> 7` | Dremel PLA 3D Printer Filament, 1.75 mm Diameter, 0.5 kg Spool Weight, White Translucent
- `3554` | `17 -> 1` | Gizmo Dorks 1.75mm ABS Filament 1kg / 2.2lb for 3D Printers, White
