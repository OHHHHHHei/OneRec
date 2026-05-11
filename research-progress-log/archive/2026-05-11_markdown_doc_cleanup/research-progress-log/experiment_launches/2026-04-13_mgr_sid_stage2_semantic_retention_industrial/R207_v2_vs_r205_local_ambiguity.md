# Baseline vs Hierarchy Final SID Local Ambiguity Analysis

## Conclusion

- `same_l2` 没改善：测试目标 item 的平均 `l2` 叶子数从 `4.3422` 变到 `4.9572`。
- 测试目标落在多叶 `same_l2` bucket 的比例从 `0.4873` 变到 `0.5449`。
- 测试目标落在深拥挤 `l2` bucket (`>=4` leaves) 的比例从 `0.2228` 变到 `0.2621`。
- 测试样本里，有 `0.1114` 的目标 item 从 `same_l2` 多叶 bucket 被移到单叶 bucket；只有 `0.1690` 被移入更拥挤的 `same_l2` bucket。

## Catalog-Level

| Metric | Baseline | Hierarchy | Delta |
|---|---:|---:|---:|
| Item-weighted mean l2 leaf count | 2.394466 | 2.814162 | +0.419696 |
| Fraction items in multi-leaf l2 | 0.423494 | 0.520890 | +0.097396 |
| Fraction items in l2 with >=4 leaves | 0.130765 | 0.190993 | +0.060228 |
| Weighted H(level3|level1,level2) | 0.715048 | 0.923560 | +0.208512 |

## Test-Weighted

| Metric | Baseline | Hierarchy | Delta |
|---|---:|---:|---:|
| Mean target l2 leaf count | 4.342158 | 4.957203 | +0.615045 |
| Fraction targets in multi-leaf l2 | 0.487315 | 0.544893 | +0.057578 |
| Fraction targets in l2 with >=4 leaves | 0.222811 | 0.262078 | +0.039268 |
| Mean target l3 entropy under l2 | 1.100115 | 1.262250 | +0.162136 |

## Movement Summary

- `SID` changed on `100.00\%` of catalog items.
- Test-weighted targets with reduced `l2` leaf count: `17.85\%`.
- Test-weighted targets with increased `l2` leaf count: `37.00\%`.
- Test-weighted targets moved out of multi-leaf `same_l2`: `11.14\%`.
- Test-weighted targets moved into multi-leaf `same_l2`: `16.90\%`.
- Test-weighted mean delta of `l2` leaf count: `+0.615045`.

## Top Improved Examples

- `1206` | `17 -> 1` | eSUN 1.75mm Black PLA PRO (PLA+) 3D Printer Filament 1KG Spool (2.2lbs), Black
- `2284` | `17 -> 1` | HICTOP 1.75mm Black PLA 3D Printer Filament - 1kg Spool (2.2 lbs) - Dimensional Accuracy +/- 0.02mm
- `2740` | `17 -> 1` | 1.75mm White PLA 3D Printer Filament - 1kg Spool (2.2 lbs) - Dimensional Accuracy +/- 0.03mm
- `3505` | `17 -> 2` | Taulman 3D 618 Natural Nylon Filament for 3D Printer 1.75mm 2 1lb Spool Bundle MADE IN USA
- `2719` | `17 -> 4` | MeltInk3D Silver PLA 3D Printer Filament &Oslash; 1.75mm, 1Kg (2.2 Lb), MADE in U.S.A, Dimensional Accuracy: &plusmn; 0.05mm
- `2739` | `17 -> 4` | 1.75mm Black PLA 3D Printer Filament - 1kg Spool (2.2 lbs) - Dimensional Accuracy +/- 0.03mm
- `2741` | `17 -> 4` | 1.75mm Yellow PLA 3D Printer Filament - 1kg Spool (2.2 lbs) - Dimensional Accuracy +/- 0.03mm
- `2742` | `17 -> 4` | 1.75mm Dark Blue PLA 3D Printer Filament - 1kg Spool (2.2 lbs) - Dimensional Accuracy +/- 0.03mm
- `3555` | `17 -> 4` | 3D PLA 1.75MM BLUE Plastic 3D Printer Printing Filament, Dimensional Accuracy +/- 0.04 mm, 1KG 2.2LBS
- `3580` | `17 -> 4` | MeltInk3D PLA-1K175PRP05 Purple PLA 3D Printer Filament &Oslash; 1.75mm, 1Kg (2.2 Lb), Made in USA, Dimensional Accuracy: &plusmn; 0.05mm
