# Baseline vs Hierarchy Final SID Local Ambiguity Analysis

## Conclusion

- `same_l2` 没改善：测试目标 item 的平均 `l2` 叶子数从 `3.6148` 变到 `4.1266`。
- 测试目标落在多叶 `same_l2` bucket 的比例从 `0.4988` 变到 `0.5831`。
- 测试目标落在深拥挤 `l2` bucket (`>=4` leaves) 的比例从 `0.1994` 变到 `0.2585`。
- 测试样本里，有 `0.1037` 的目标 item 从 `same_l2` 多叶 bucket 被移到单叶 bucket；只有 `0.1880` 被移入更拥挤的 `same_l2` bucket。

## Catalog-Level

| Metric | Baseline | Hierarchy | Delta |
|---|---:|---:|---:|
| Item-weighted mean l2 leaf count | 2.340206 | 2.574878 | +0.234672 |
| Fraction items in multi-leaf l2 | 0.444655 | 0.519262 | +0.074607 |
| Fraction items in l2 with >=4 leaves | 0.125068 | 0.174986 | +0.049919 |
| Weighted H(level3|level1,level2) | 0.728211 | 0.869628 | +0.141416 |

## Test-Weighted

| Metric | Baseline | Hierarchy | Delta |
|---|---:|---:|---:|
| Mean target l2 leaf count | 3.614825 | 4.126627 | +0.511802 |
| Fraction targets in multi-leaf l2 | 0.498787 | 0.583058 | +0.084271 |
| Fraction targets in l2 with >=4 leaves | 0.199426 | 0.258548 | +0.059122 |
| Mean target l3 entropy under l2 | 1.030772 | 1.212795 | +0.182023 |

## Movement Summary

- `SID` changed on `100.00\%` of catalog items.
- Test-weighted targets with reduced `l2` leaf count: `21.53\%`.
- Test-weighted targets with increased `l2` leaf count: `36.29\%`.
- Test-weighted targets moved out of multi-leaf `same_l2`: `10.37\%`.
- Test-weighted targets moved into multi-leaf `same_l2`: `18.80\%`.
- Test-weighted mean delta of `l2` leaf count: `+0.511802`.

## Top Improved Examples

- `1849` | `23 -> 4` | HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, Purple
- `2697` | `16 -> 2` | 3D Solutech Real White 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG)
- `2894` | `11 -> 1` | Gizmo Dorks 1.75mm HIPS Filament 1kg / 2.2lb for 3D Printers, White
- `319` | `12 -> 2` | 6061 Aluminum Rectangular Bar, Unpolished (Mill) Finish, Extruded, T6511 Temper, ASTM B221, 1/8" Thickness, 1" Width, 36" Length
- `323` | `12 -> 2` | 6101 Aluminum Rectangular Bar, Unpolished (Mill) Finish, T61 Temper, ASTM B317, 1/4" Thickness, 2" Width, 12" Length
- `2868` | `12 -> 2` | 6061 Aluminum Rectangular Bar, Unpolished (Mill) Finish, Extruded, T6511 Temper, ASTM B211/ASTM B221, 1-1/2" Thickness, 1-1/2" Width, 12" Length
- `1734` | `8 -> 1` | SOLOOP 328Pcs 8 Sizes Assortment 2:1 Heat Shrink Tube Tubing Sleeve Wrap Wire Kit Set
- `3382` | `8 -> 1` | E-Projects 100EP514200R 200 Ohm Resistors, 1/4 W, 5% (Pack of 100)
- `767` | `7 -> 1` | J-B Weld 50101 MinuteWeld Instant-Setting Epoxy Syringe - Dries Clear - 25ml
- `2215` | `7 -> 1` | J-B Weld 50172 MarineWeld Marine Adhesive Epoxy Syringe - Dries White - 25ml
