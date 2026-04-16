# Prefix Collaborative Consistency Analysis

## Conclusion

- This diagnostic treats multi-leaf `l2` prefixes as potentially useful if they are collaboratively consistent, not automatically bad.
- `graph-weak threshold`（图弱连接阈值） is `0.001463`; crowded prefixes are split into `consistent`, `mixed`, and `inconsistent` by intra-prefix graph support.
- Test-weighted `consistent crowded`（协同一致拥挤前缀） fraction: `0.1147 -> 0.1374`.
- Test-weighted `inconsistent crowded`（协同不一致拥挤前缀） fraction: `0.3711 -> 0.5001`.
- Test-weighted mean prefix graph affinity（测试加权平均前缀图亲和度）: `0.023510 -> 0.025290`.
- Test-weighted moved to `consistent crowded`（移入协同一致拥挤前缀）: `0.0675`; moved to `inconsistent crowded`（移入协同不一致拥挤前缀）: `0.2394`.

## Test-Weighted Summary

| Metric | Baseline | Compare | Delta |
|---|---:|---:|---:|
| Mean target leaf count | 4.342158 | 4.128171 | -0.213986 |
| Mean target prefix graph affinity | 0.023510 | 0.025290 | +0.001780 |
| Mean target prefix semantic sim | 0.525752 | 0.660276 | +0.134524 |
| Fraction targets in consistent crowded prefixes | 0.114714 | 0.137437 | +0.022722 |
| Fraction targets in mixed crowded prefixes | 0.001544 | 0.001324 | -0.000221 |
| Fraction targets in inconsistent crowded prefixes | 0.371057 | 0.500110 | +0.129054 |
| Fraction targets in singleton prefixes | 0.512685 | 0.361129 | -0.151555 |

## Movement Summary

- Test targets moved to `consistent crowded`（协同一致拥挤前缀）: `6.75%`.
- Test targets moved to `inconsistent crowded`（协同不一致拥挤前缀）: `23.94%`.
- Test targets moved from `inconsistent` to `consistent`（从协同不一致转为协同一致）: `4.24%`.
- Test-weighted mean delta of leaf count（测试加权叶子数变化）: `-0.213986`.
- Test-weighted mean delta of prefix graph affinity（测试加权前缀图亲和度变化）: `+0.001780`.

## Compare Top Consistent Crowded Prefixes

- `<a_91><b_63>` | items `11` | leaves `11` | test_weight `98` | strong `0.764` | positive `0.800` | graph `0.062222` | semantic `0.9909` | Inland 1.75mm Red PLA 3D Printer Filament - 1kg Spool (2.2 lbs) / Inland 1.75mm Gold PLA 3D Printer Filament - 1kg Spool (2.2 lbs) / Inland 1.75mm White PLA 3D Printer Filament - 1kg Spool (2.2 lbs)
- `<a_106><b_120>` | items `9` | leaves `9` | test_weight `48` | strong `0.528` | positive `0.528` | graph `0.020778` | semantic `0.9773` | eSUN 3D 1.75mm PETG Natural Filament 1kg (2.2lb), PETG 3D Printer Filament, Semi-Transparent 1.75mm Natural / eSUN 3D 1.75mm PETG Blue Filament 1kg (2.2lb), PETG 3D Printer Filament, 1.75mm Semi-Transparent Blue / eSUN 3D 1.75mm PETG Green Filament 1kg (2.2lb), PETG 3D Printer Filament, Semi-Transparent 1.75mm Green
- `<a_83><b_104>` | items `5` | leaves `5` | test_weight `28` | strong `0.500` | positive `0.500` | graph `0.009229` | semantic `0.9826` | Gorilla Tape, Black Duct Tape, 1.88" x 35 yd, Black, (Pack of 1) / Gorilla Tape, Black Tough & Wide Duct Tape, 2.88" x 30 yd, Black, (Pack of 1) / Gorilla Tape, White Duct Tape, 1.88" x 30 yd, White, (Pack of 1)
- `<a_106><b_73>` | items `20` | leaves `20` | test_weight `25` | strong `0.568` | positive `0.574` | graph `0.023878` | semantic `0.9995` | HATCHBOX ABS 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, White / HATCHBOX ABS 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, Silver / HATCHBOX ABS 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, Black
- `<a_227><b_116>` | items `11` | leaves `11` | test_weight `22` | strong `0.564` | positive `0.564` | graph `0.078192` | semantic `0.9575` | Acetal Copolymer Round Rod, Opaque Black, Standard Tolerance, ASTM D6778, 3" Diameter, 6" Length / 6061 Aluminum Round Rod, Unpolished (Mill) Finish, Extruded, T6511 Temper, ASTM B221, 2-1/4" Diameter, 12" Length / 6061 Aluminum Round Rod, Unpolished (Mill) Finish, Extruded, T6511 Temper, ASTM B221, 1/4" Diameter, 60" Length
- `<a_247><b_180>` | items `3` | leaves `2` | test_weight `21` | strong `1.000` | positive `1.000` | graph `0.382691` | semantic `0.9981` | T-fal C51407 Excite Nonstick Thermo-Spot Dishwasher Safe Oven Safe PFOA Free Fry Pan Cookware, 12-Inch, Red / T-fal C51402 Excite Nonstick Thermo-Spot Dishwasher Safe Oven Safe PFOA Free Fry Pan Cookware, 8-Inch, Red / T-fal C51402 Excite Nonstick Thermo-Spot Dishwasher Safe Oven Safe PFOA Free Fry Pan Cookware, 8-Inch, Red
- `<a_53><b_139>` | items `2` | leaves `2` | test_weight `21` | strong `1.000` | positive `1.000` | graph `0.008487` | semantic `0.9363` | HATCHBOX ABS 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, Red / HATCHBOX 3D Printer Filament, Dimensional Accuracy +/- 0.03mm, 1.75 mm, 1 kg Spool, Wood
- `<a_113><b_64>` | items `10` | leaves `10` | test_weight `21` | strong `0.733` | positive `0.733` | graph `0.057599` | semantic `0.9969` | X-Treme Tape TPE-X36ZLB Silicone Rubber Self Fusing Tape, 1" x 36', Triangular, Black / X-Treme Tape TPE-XZLCLR Silicone Rubber Self Fusing Tape, 1" x 10', Triangular, Clear / X-Treme Tape TPE-XZLB Silicone Rubber Self Fusing Tape, 1" x 10', Triangular, Black
- `<a_250><b_86>` | items `2` | leaves `2` | test_weight `13` | strong `1.000` | positive `1.000` | graph `0.097524` | semantic `0.9996` | PTFE Teflon Bowden Tube for 1.75 Filament (2.0mm ID/4.0mm OD) &ndash; White Connector Tubing for 3D Printer - 2.0 Meters / PTFE Teflon Bowden Tube for 1.75 Filament (2.0mm ID/4.0mm OD) &ndash; WhiteConnector Tubing for 3D Printer - 1.5 Meters
- `<a_53><b_247>` | items `2` | leaves `2` | test_weight `12` | strong `1.000` | positive `1.000` | graph `0.007951` | semantic `0.8875` | HATCHBOX 1 Spool 3D Printer Filament Tabletop Wall Mount Rack / BAMtack! 1.75mm Black PLA 3D Printer Filament - 1kg (2.2 lbs) - Dimensional Accuracy +/- 0.03mm

## Compare Top Inconsistent Crowded Prefixes

- `<a_53><b_237>` | items `26` | leaves `26` | test_weight `134` | strong `0.345` | positive `0.357` | graph `0.011932` | semantic `0.9875` | HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, Blue / HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, Black / HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, Glow in the Dark
- `<a_106><b_51>` | items `15` | leaves `15` | test_weight `121` | strong `0.124` | positive `0.124` | graph `0.015495` | semantic `0.9877` | 3D Solutech Teal Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) / 3D Solutech Hot Pink 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) / 3D Solutech Natural Clear 1.75mm 3D Printer PLA Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG)
- `<a_106><b_190>` | items `13` | leaves `13` | test_weight `56` | strong `0.090` | positive `0.090` | graph `0.006523` | semantic `0.9700` | eSUN 1.75mm Black PLA 3D Printer filament 1kg Spool (2.2lbs), Black / eSUN 1.75mm Glass Watermelon Red PLA 3D Printer filament 1kg Spool (2.2lbs), Glass Red / eSUN 1.75mm Black PLA 3D Printer filament 1kg Spool (2.2lbs), Black
- `<a_53><b_190>` | items `10` | leaves `10` | test_weight `47` | strong `0.222` | positive `0.222` | graph `0.022809` | semantic `0.9645` | Verbatim 3D Printer Filament - PLA High-Grade 1.75mm 1kg Reel - Widely Compatible with 3D Printers - Red / Filament Outlet Green PLA 1.75mm 3D Printer Filament 1kg (2.2lbs) spool USA / Filament Outlet Black PLA 1.75mm 3D Printer Filament 1kg (2.2lbs) spool USA
- `<a_91><b_134>` | items `10` | leaves `10` | test_weight `39` | strong `0.022` | positive `0.067` | graph `0.006232` | semantic `0.9820` | eSUN 1.75mm Black PLA PRO (PLA+) 3D Printer Filament 1KG Spool (2.2lbs), Black / HICTOP 1.75mm Black PLA 3D Printer Filament - 1kg Spool (2.2 lbs) - Dimensional Accuracy +/- 0.02mm / 1.75mm Black PLA 3D Printer Filament - 1kg Spool (2.2 lbs) - Dimensional Accuracy +/- 0.03mm
- `<a_53><b_146>` | items `2` | leaves `2` | test_weight `37` | strong `0.000` | positive `0.000` | graph `0.000000` | semantic `0.9998` | HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, Green / HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, White
- `<a_106><b_104>` | items `2` | leaves `2` | test_weight `36` | strong `0.000` | positive `0.000` | graph `0.000000` | semantic `0.9613` | 3D Solutech Silver Metal 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - PLA175TCMS / 3DDPLUS 1.75mm PLA 3D Printer Filament Black- 1kg Spool (2.2 lbs) - Dimensional Accuracy +/- 0.03mm (Plain Black)
- `<a_91><b_39>` | items `7` | leaves `7` | test_weight `32` | strong `0.095` | positive `0.095` | graph `0.016408` | semantic `0.9872` | Inland 1.75mm Peak Green PLA 3D Printer Filament - 1kg Spool (2.2 lbs) / Inland 1.75mm Pink PLA 3D Printer Filament - 1kg Spool (2.2 lbs) / Inland 1.75mm Green PLA 3D Printer Filament - 1kg Spool (2.2 lbs)
- `<a_29><b_110>` | items `5` | leaves `5` | test_weight `28` | strong `0.200` | positive `0.200` | graph `0.025945` | semantic `0.9555` | Gorilla 2 Part Epoxy, 5 Minute Set, .85 ounce Syringe, Clear / Loctite Heavy Duty Epoxy Quick Set 8-Fluid Ounce Bottle (1365736) / Loctite Epoxy Quick Set 0.85-Fluid Ounce Syringe (1395391)
- `<a_106><b_229>` | items `2` | leaves `2` | test_weight `23` | strong `0.000` | positive `0.000` | graph `0.000000` | semantic `0.9755` | 3D Solutech Navy Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) / 3D Solutech Real Green 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG)
