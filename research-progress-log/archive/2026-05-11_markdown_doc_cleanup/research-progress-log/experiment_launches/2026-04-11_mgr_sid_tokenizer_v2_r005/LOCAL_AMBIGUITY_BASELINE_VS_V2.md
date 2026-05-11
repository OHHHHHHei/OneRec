# Baseline vs Hierarchy Final SID Local Ambiguity Analysis

## Conclusion

- `same_l2` 有改善：测试目标 item 的平均 `l2` 叶子数从 `4.7999` 降到 `4.3422`。
- 测试目标落在多叶 `same_l2` bucket 的比例从 `0.6894` 变到 `0.4873`。
- 测试目标落在深拥挤 `l2` bucket (`>=4` leaves) 的比例从 `0.3283` 变到 `0.2228`。
- 测试样本里，有 `0.2674` 的目标 item 从 `same_l2` 多叶 bucket 被移到单叶 bucket；只有 `0.0653` 被移入更拥挤的 `same_l2` bucket。

## Catalog-Level

| Metric | Baseline | Hierarchy | Delta |
|---|---:|---:|---:|
| Item-weighted mean l2 leaf count | 3.186923 | 2.394466 | -0.792458 |
| Fraction items in multi-leaf l2 | 0.600651 | 0.423494 | -0.177157 |
| Fraction items in l2 with >=4 leaves | 0.258003 | 0.130765 | -0.127238 |
| Weighted H(level3|level1,level2) | 1.111955 | 0.715048 | -0.396907 |

## Test-Weighted

| Metric | Baseline | Hierarchy | Delta |
|---|---:|---:|---:|
| Mean target l2 leaf count | 4.799912 | 4.342158 | -0.457754 |
| Fraction targets in multi-leaf l2 | 0.689389 | 0.487315 | -0.202074 |
| Fraction targets in l2 with >=4 leaves | 0.328259 | 0.222811 | -0.105449 |
| Mean target l3 entropy under l2 | 1.453294 | 1.100115 | -0.353179 |

## Movement Summary

- `SID` changed on `100.00\%` of catalog items.
- Test-weighted targets with reduced `l2` leaf count: `44.21\%`.
- Test-weighted targets with increased `l2` leaf count: `19.32\%`.
- Test-weighted targets moved out of multi-leaf `same_l2`: `26.74\%`.
- Test-weighted targets moved into multi-leaf `same_l2`: `6.53\%`.
- Test-weighted mean delta of `l2` leaf count: `-0.457754`.

## Top Improved Examples

- `2665` | `25 -> 1` | Printrbot PLA Filament for 3D Printers, 1.75mm Diameter, Metallic Charcoal Gray, 1Kg Spool
- `3096` | `25 -> 1` | HATCHBOX 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, Bronze
- `3675` | `25 -> 1` | Dremel PLA 3D Printer Filament, 1.75 mm Diameter, 0.5 kg Spool Weight, White Translucent
- `2701` | `25 -> 2` | HATCHBOX 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, Transparent Black
- `249` | `25 -> 3` | UP! ABS Plastic Filament, 1.75 mm Diameter, 1.54 lbs Spool, Black
- `3629` | `25 -> 3` | [STAR] Alchement - PLA Series, 3D Filament, 1.75mm, 1kg (Transparent)
- `2909` | `22 -> 1` | 3D Solutech Silver Metal 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - PLA175TCMS
- `2060` | `17 -> 1` | Antique Bronze Metallic PLA - 3D Printing Filament (1.75mm 0.5 kg) Made in the USA
- `3042` | `17 -> 1` | RioRand 1.75mm PLA Filament 1kg/2.2lb for 3D Printers Reprap, MakerBot Replicator 2, Afinia, Solidoodle etc.(Black)
- `3490` | `17 -> 1` | MeltInk3D PLA- 1K175BLK05 Black PLA 3D Printer Filament &Oslash; 1.75mm, 1Kg (2.2 Lb), MADE in U.S.A, Dimensional Accuracy: &plusmn; 0.05mm
