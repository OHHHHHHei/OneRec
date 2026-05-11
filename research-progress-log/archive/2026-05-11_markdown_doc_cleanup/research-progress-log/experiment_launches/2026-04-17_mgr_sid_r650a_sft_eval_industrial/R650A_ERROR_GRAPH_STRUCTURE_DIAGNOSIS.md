# R650a Error Graph Structure Diagnosis

## Graph Setup
- coarse_view: coarse_seq2g_rel_masked
- mid_view: fagsp_mid_seq2g_rel_masked
- seq2g_mix_alpha: 0.35
- seq2g_direct_tau: 0.5
- graph_topk: 32
- note: coarse_seq2g_rel_masked = 0.65 * coarse_purified + 0.35 * reliability-weighted shared-predecessor rescue edges masked to direct_support < 0.5, row-normalized. mid_view applies FAGSP-style mid-band transform on this coarse graph.

## Counts
- examples: 4533
- losses: 206
- gains: 143
- stable_hits: 457
- stable_misses: 3727

## Pair-Level Summary
### Top10 losses
- count: 206
- same_family_mean: 0.5680
- pred1_is_more_popular_mean: 0.4660
- target_pop_mean: 286.9515
- pred1_pop_mean: 192.2039
- coarse_seq2g_target_to_pred1_mean: 0.0068
- mid_seq2g_target_to_pred1_mean: 0.0011
- base_mid_target_to_pred1_mean: 0.0002
- rescue_target_to_pred1_mean: 0.0040
- context_target_to_pred1_mean: 0.0996
- reliability_target_to_pred1_mean: 0.0304
- direct_support_target_to_pred1_mean: 11.0680
- local_raw_target_to_pred1_mean: 0.5303
- coarse_raw_target_to_pred1_mean: 4.7934
- in_selective_separation_pairs_mean: 0.0437
- target_mid_seq2g_in_strength_pct_mean: 0.7573
- pred1_mid_seq2g_in_strength_pct_mean: 0.7690
- target_coarse_seq2g_in_strength_pct_mean: 0.7990
- pred1_coarse_seq2g_in_strength_pct_mean: 0.7484
- coarse_seq2g_edge_positive_rate: 0.2184
- coarse_seq2g_edge_top10_rate: 0.1117
- coarse_seq2g_edge_top32_rate: 0.2184
- mid_seq2g_edge_positive_rate: 0.1990
- mid_seq2g_edge_top10_rate: 0.1019
- mid_seq2g_edge_top32_rate: 0.1990
- base_mid_edge_positive_rate: 0.0437
- base_mid_edge_top10_rate: 0.0194
- base_mid_edge_top32_rate: 0.0437
- rescue_edge_positive_rate: 0.0485
- rescue_edge_top10_rate: 0.0243
- rescue_edge_top32_rate: 0.0485
- context_edge_positive_rate: 0.2573
- context_edge_top10_rate: 0.1408
- context_edge_top32_rate: 0.2573
- direct_support_edge_positive_rate: 0.5825
- direct_support_edge_top10_rate: 0.1990
- direct_support_edge_top32_rate: 0.3252

### Top10 gains
- count: 143
- same_family_mean: 0.7692
- pred1_is_more_popular_mean: 0.4266
- target_pop_mean: 166.5594
- pred1_pop_mean: 174.1399
- coarse_seq2g_target_to_pred1_mean: 0.0131
- mid_seq2g_target_to_pred1_mean: 0.0015
- base_mid_target_to_pred1_mean: 0.0004
- rescue_target_to_pred1_mean: 0.0032
- context_target_to_pred1_mean: 0.2318
- reliability_target_to_pred1_mean: 0.1702
- direct_support_target_to_pred1_mean: 9.7478
- local_raw_target_to_pred1_mean: 0.6834
- coarse_raw_target_to_pred1_mean: 4.2349
- in_selective_separation_pairs_mean: 0.0070
- target_mid_seq2g_in_strength_pct_mean: 0.7934
- pred1_mid_seq2g_in_strength_pct_mean: 0.7865
- target_coarse_seq2g_in_strength_pct_mean: 0.8127
- pred1_coarse_seq2g_in_strength_pct_mean: 0.7806
- coarse_seq2g_edge_positive_rate: 0.3217
- coarse_seq2g_edge_top10_rate: 0.2308
- coarse_seq2g_edge_top32_rate: 0.3217
- mid_seq2g_edge_positive_rate: 0.2308
- mid_seq2g_edge_top10_rate: 0.0979
- mid_seq2g_edge_top32_rate: 0.2308
- base_mid_edge_positive_rate: 0.1049
- base_mid_edge_top10_rate: 0.0140
- base_mid_edge_top32_rate: 0.1049
- rescue_edge_positive_rate: 0.0559
- rescue_edge_top10_rate: 0.0420
- rescue_edge_top32_rate: 0.0559
- context_edge_positive_rate: 0.4545
- context_edge_top10_rate: 0.3427
- context_edge_top32_rate: 0.4545
- direct_support_edge_positive_rate: 0.5524
- direct_support_edge_top10_rate: 0.2797
- direct_support_edge_top32_rate: 0.3916

## Item Segment Summary
### loss_heavy
- item_count: 33
- eval_mean: 17.4242
- loss_mean: 4.0000
- gain_mean: 0.9091
- loss_rate_mean: 0.3449
- gain_rate_mean: 0.0696
- mid_seq2g_in_strength_mean: 0.0690
- mid_seq2g_in_strength_pct_mean: 0.7503
- mid_seq2g_in_degree_mean: 19.3939
- coarse_seq2g_in_strength_mean: 1.5610
- coarse_seq2g_in_strength_pct_mean: 0.8342
- rescue_in_strength_mean: 0.9787
- rescue_in_degree_mean: 37.7879
- context_in_strength_mean: 18.4967
- direct_support_in_strength_mean: 1494.4651
- mid_seq2g_top10_same_family_rate_mean: 0.5606
- coarse_seq2g_top10_same_family_rate_mean: 0.5788
- base_mid_top10_same_family_rate_mean: 0.3818
- rescue_top10_same_family_rate_mean: 0.5574

### gain_heavy
- item_count: 24
- eval_mean: 20.5417
- loss_mean: 1.4583
- gain_mean: 3.3750
- loss_rate_mean: 0.0440
- gain_rate_mean: 0.3644
- mid_seq2g_in_strength_mean: 0.1879
- mid_seq2g_in_strength_pct_mean: 0.7889
- mid_seq2g_in_degree_mean: 38.7917
- coarse_seq2g_in_strength_mean: 1.5336
- coarse_seq2g_in_strength_pct_mean: 0.8469
- rescue_in_strength_mean: 0.8485
- rescue_in_degree_mean: 30.2917
- context_in_strength_mean: 13.4810
- direct_support_in_strength_mean: 1395.6745
- mid_seq2g_top10_same_family_rate_mean: 0.5500
- coarse_seq2g_top10_same_family_rate_mean: 0.4917
- base_mid_top10_same_family_rate_mean: 0.3750
- rescue_top10_same_family_rate_mean: 0.4958

### neutral_eval_ge3
- item_count: 354
- eval_mean: 4.7373
- loss_mean: 0.0000
- gain_mean: 0.0000
- loss_rate_mean: 0.0000
- gain_rate_mean: 0.0000
- mid_seq2g_in_strength_mean: 0.1917
- mid_seq2g_in_strength_pct_mean: 0.7536
- mid_seq2g_in_degree_mean: 37.2938
- coarse_seq2g_in_strength_mean: 0.7986
- coarse_seq2g_in_strength_pct_mean: 0.4359
- rescue_in_strength_mean: 0.7829
- rescue_in_degree_mean: 21.3333
- context_in_strength_mean: 6.2006
- direct_support_in_strength_mean: 311.0538
- mid_seq2g_top10_same_family_rate_mean: 0.4003
- coarse_seq2g_top10_same_family_rate_mean: 0.4679
- base_mid_top10_same_family_rate_mean: 0.4107
- rescue_top10_same_family_rate_mean: 0.4402

## Top Loss Items With Graph Neighbors
- item 182 | eval=200 loss=19 gain=3 net_loss=16 | family=gauge_meter | mid_in_pct=0.710 | AcuRite 00613 Humidity Monitor with Indoor Thermometer, Digital Hygrometer and Humidity Gauge Indic…
  mid_top: 2957:0.00601:DBPOWER 2 MP 15M Tube USB Waterproof Hd 6-led Borescop… || 2956:0.00497:DBPOWER 2 MP 15M Tube USB Waterproof Hd 6-led Borescop… || 1322:0.00428:Pro Gaff Gaffers Tape 1 and 2 inch widths, 17 colors a… || 1321:0.00415:Pro Gaff Gaffers Tape 1 and 2 inch widths, 17 colors a… || 1323:0.0039:Pro Gaff Gaffers Tape 1 and 2 inch widths, 17 colors a… || 2889:0.0039:GROW1 Panda Film, 10' x 25', Poly 5.5 mil, Black/White
  coarse_top: 1459:0.35:Forney 72730 Wire Cup Brush, Fine Crimped with 1/4-Inc… || 181:0.0217:AcuRite 00613 Humidity Monitor with Indoor Thermometer… || 2938:0.00645:VenTech VT DUCT-6 VTD625 Aluminum Duct for Ventilation… || 3018:0.00622:VenTech VT DUCT-4 VTD425 Aluminum Duct for Ventilation… || 3128:0.00583:VenTech VT IF6+CF6 IF6CF620 Inline Duct Fan with Virgi… || 2289:0.00507:60 Tube - 16x150mm Clear Plastic Test Tube Set with Ca…
  rescue_top: 1459:1:Forney 72730 Wire Cup Brush, Fine Crimped with 1/4-Inc…
- item 2909 | eval=31 loss=16 gain=0 net_loss=16 | family=3d_filament | mid_in_pct=0.771 | 3D Solutech Silver Metal 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm,…
  mid_top: 3620:0.0153:6 Pcs 100K ohm NTC Thermistors/Temp Sensor for Reprap … || 3474:0.0144:3D Solutech See Through Red 1.75mm PETG 3D Printer Fil… || 3619:0.0115:eSUN 1.75mm Red PLA PRO (PLA+) 3D Printer Filament 1KG… || 2724:0.0108:3D Solutech Natural Clear 1.75mm PETG 3D Printer Filam… || 3683:0.0056:3D Solutech Purple 3D Printer PLA Filament 1.75MM Fila… || 3469:0.00532:3D Solutech Real Pink 3D Printer PLA Filament 1.75MM F…
  coarse_top: 51:0.0758:HATCHBOX 1 Spool 3D Printer Filament Tabletop Wall Mou… || 1574:0.0514:HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy… || 3558:0.0424:Professional Black 1.75mm PLA 3D Printer Filament, 1kg… || 2420:0.0354:HATCHBOX ABS 3D Printer Filament, Dimensional Accuracy… || 2470:0.0338:HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy… || 2660:0.0305:3D Solutech Real Grey 3D Printer PLA Filament 1.75MM F…
  rescue_top: 51:0.217:HATCHBOX 1 Spool 3D Printer Filament Tabletop Wall Mou… || 1574:0.147:HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy… || 3558:0.121:Professional Black 1.75mm PLA 3D Printer Filament, 1kg… || 2420:0.101:HATCHBOX ABS 3D Printer Filament, Dimensional Accuracy… || 2470:0.0965:HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy… || 2660:0.0871:3D Solutech Real Grey 3D Printer PLA Filament 1.75MM F…
- item 3475 | eval=11 loss=8 gain=0 net_loss=8 | family=3d_filament | mid_in_pct=0.710 | 3D Solutech Real Orange 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, …
  mid_top: 3620:0.0131:6 Pcs 100K ohm NTC Thermistors/Temp Sensor for Reprap … || 3474:0.0121:3D Solutech See Through Red 1.75mm PETG 3D Printer Fil… || 3619:0.00988:eSUN 1.75mm Red PLA PRO (PLA+) 3D Printer Filament 1KG… || 2724:0.00908:3D Solutech Natural Clear 1.75mm PETG 3D Printer Filam… || 1729:0.00677:Smith-Cooper International CV30L Series Brass Check Va… || 437:0.00618:Millrose 70660 Monster Roll PTFE Thread Seal Tape, 1/2…
  coarse_top: 3022:0.131:Philips Sonicare HX9381/05 Diamond Clean Rechargeable … || 2188:0.0725:Dremel Digilab 3D20 3D Printer, Idea Builder for Brand… || 3657:0.0661:Taulman 3D 618 Natural Nylon Filament for 3D Printer 1… || 3466:0.0523:3D Solutech Real Red 3D Printer PLA Filament 1.75MM, D… || 3557:0.0453:3D Solutech Chocolate Brown 3D Printer PLA Filament 1.… || 3469:0.0387:3D Solutech Real Pink 3D Printer PLA Filament 1.75MM F…
  rescue_top: 3469:0.111:3D Solutech Real Pink 3D Printer PLA Filament 1.75MM F… || 3113:0.0771:Inland 1.75mm Silver PLA 3D Printer Filament - 1kg Spo… || 2927:0.0619:Barbariol 3 PCS NTC3950 Thermistors for RepRap 3D Prin… || 1202:0.0521:MakerGear M2 Desktop 3D Printer || 1315:0.0496:Alvord Polk 127-0 High-Speed Steel Chucking Reamer, St… || 1164:0.0464:Fowler Full Warranty 52-104-025-0 Machinist Jack Set, …
- item 3522 | eval=14 loss=8 gain=1 net_loss=7 | family=3d_filament | mid_in_pct=0.710 | 3D Solutech Real Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.…
  mid_top: 3620:0.0148:6 Pcs 100K ohm NTC Thermistors/Temp Sensor for Reprap … || 3474:0.0134:3D Solutech See Through Red 1.75mm PETG 3D Printer Fil… || 3619:0.011:eSUN 1.75mm Red PLA PRO (PLA+) 3D Printer Filament 1KG… || 2724:0.00985:3D Solutech Natural Clear 1.75mm PETG 3D Printer Filam… || 3599:0.00583:3D Solutech Aqua Blue 3D Printer PLA Filament 1.75MM F… || 3469:0.00532:3D Solutech Real Pink 3D Printer PLA Filament 1.75MM F…
  coarse_top: 3494:0.076:3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM… || 2659:0.0716:3D Solutech Apple Green 3D Printer PLA Filament 1.75MM… || 2728:0.0627:BIQU Upgrade Wear Resistant MK10 Nozzles M7 0.4mm Thre… || 2446:0.054:100x 1N4007 Diode 1A 1000V Rectifier Diodes Arduino Mo… || 3311:0.0492:Uxcell a10081700ux0008 20pcs 1mm Twisted Drilling Bit … || 2145:0.0484:ROBO 3D R1 Plus 10x9x8-Inch ABS/PLA 3D Printer, White …
  rescue_top: 3201:0.0946:SainSmart Wood-DarkBrown-1KG1.75 1.75 mm Wood 3D Print… || 3456:0.0804:HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy… || 3096:0.0687:HATCHBOX 3D Printer Filament, Dimensional Accuracy +/-… || 1203:0.0682:eSUN 3D 1.75mm PETG Natural Filament 1kg (2.2lb), PETG… || 3454:0.0635:Wangdd22 3D Printer J-head Hotend with Fan for 1.75mm … || 3429:0.0586:3D Printer Filament 1.75mm Black ABS - 1kg (2.2 lbs) 1…
- item 3016 | eval=17 loss=7 gain=1 net_loss=6 | family=3d_filament | mid_in_pct=0.710 | HATCHBOX 3D Printer Filament, Dimensional Accuracy +/- 0.03mm, 1.75 mm, 1 kg Spool, Wood
  mid_top: 3620:0.011:6 Pcs 100K ohm NTC Thermistors/Temp Sensor for Reprap … || 3474:0.011:3D Solutech See Through Red 1.75mm PETG 3D Printer Fil… || 2724:0.00851:3D Solutech Natural Clear 1.75mm PETG 3D Printer Filam… || 3619:0.00836:eSUN 1.75mm Red PLA PRO (PLA+) 3D Printer Filament 1KG… || 3683:0.00431:3D Solutech Purple 3D Printer PLA Filament 1.75MM Fila… || 1845:0.00428:Filament Outlet Blue PLA 1.75mm 3D Printer Filament 1k…
  coarse_top: 2704:0.0599:HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy… || 3136:0.0595:HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy… || 2677:0.0583:eSUN 1.75mm Black ABS 3D Printer filament 1kg Spool (2… || 3684:0.0437:HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy… || 3429:0.0425:3D Printer Filament 1.75mm Black ABS - 1kg (2.2 lbs) 1… || 3520:0.0302:SainSmart 1.75mm PVA Dissolvable 3D Printers Filament …
  rescue_top: 2704:0.171:HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy… || 3136:0.17:HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy… || 2677:0.167:eSUN 1.75mm Black ABS 3D Printer filament 1kg Spool (2… || 3684:0.125:HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy… || 3429:0.121:3D Printer Filament 1.75mm Black ABS - 1kg (2.2 lbs) 1… || 3520:0.0862:SainSmart 1.75mm PVA Dissolvable 3D Printers Filament …
- item 2703 | eval=14 loss=5 gain=0 net_loss=5 | family=3d_filament | mid_in_pct=0.710 | eSUN 3D 1.75mm PETG Black Filament 1kg (2.2lb), PETG 3D Printer Filament, 1.75mm Solid Opaque Black
  mid_top: 3620:0.00867:6 Pcs 100K ohm NTC Thermistors/Temp Sensor for Reprap … || 3474:0.00813:3D Solutech See Through Red 1.75mm PETG 3D Printer Fil… || 3619:0.0064:eSUN 1.75mm Red PLA PRO (PLA+) 3D Printer Filament 1KG… || 2724:0.00597:3D Solutech Natural Clear 1.75mm PETG 3D Printer Filam… || 1802:0.00326:Loos Cableware Division Stainless Steel 302/304 Wire R… || 3599:0.00314:3D Solutech Aqua Blue 3D Printer PLA Filament 1.75MM F…
  coarse_top: 3098:0.0775:SainSmart PETG-1KG1.75 PETG 3D Printers Filament, 1 kg… || 1793:0.0497:Matter and Form MFS1V1 3D Scanner || 3520:0.0452:SainSmart 1.75mm PVA Dissolvable 3D Printers Filament … || 2734:0.0443:Inland 1.75mm White PLA 3D Printer Filament - 1kg Spoo… || 3619:0.0441:eSUN 1.75mm Red PLA PRO (PLA+) 3D Printer Filament 1KG… || 3635:0.0369:Octave 4 Color Red-Green-Blue-Yellow ABS Filament for …
  rescue_top: 3098:0.221:SainSmart PETG-1KG1.75 PETG 3D Printers Filament, 1 kg… || 1793:0.142:Matter and Form MFS1V1 3D Scanner || 3520:0.129:SainSmart 1.75mm PVA Dissolvable 3D Printers Filament … || 2734:0.127:Inland 1.75mm White PLA 3D Printer Filament - 1kg Spoo… || 3619:0.126:eSUN 1.75mm Red PLA PRO (PLA+) 3D Printer Filament 1KG… || 3635:0.105:Octave 4 Color Red-Green-Blue-Yellow ABS Filament for …
- item 3442 | eval=9 loss=4 gain=0 net_loss=4 | family=3d_filament | mid_in_pct=0.891 | 3D Solutech Real Purple 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, …
  mid_top: 3620:0.0127:6 Pcs 100K ohm NTC Thermistors/Temp Sensor for Reprap … || 3474:0.0109:3D Solutech See Through Red 1.75mm PETG 3D Printer Fil… || 3619:0.0094:eSUN 1.75mm Red PLA PRO (PLA+) 3D Printer Filament 1KG… || 3626:0.00785:MUYI 10 Meters = 32.8 Feet PTFE Teflon Tube OD 4mm ID … || 2724:0.00785:3D Solutech Natural Clear 1.75mm PETG 3D Printer Filam… || 2485:0.00768:Inland 1.75mm Gold PLA 3D Printer Filament - 1kg Spool…
  coarse_top: 3477:0.125:MeltInk3D PLA Filament For 3D Printers - 1.75mm PLA Fi… || 3324:0.099:Aketek 3x Solderless BreadBoard, 400 tie-points, 4 pow… || 2088:0.0916:ELEGOO Stepstick Stepper Motor Driver Module A4988 + H… || 3099:0.0638:[POWER] Alchement - Flexible(TPU) Series, 3D Filament,… || 3494:0.0525:3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM… || 3546:0.0492:Inland 1.75mm Orange PLA 3D Printer Filament - 1kg Spo…
  rescue_top: 3556:0.0747:MeltInk3D PLA-1K175GLD05 Gold PLA 3D Printer Filament … || 2674:0.0687:6" x 36 yds - 1 Mil Kapton Tape for 3D Printer Platfor… || 3096:0.0533:HATCHBOX 3D Printer Filament, Dimensional Accuracy +/-… || 3532:0.0527:Mercurry 10 Meters GT2 timing belt width 6mm Fit for R… || 3136:0.0517:HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy… || 3619:0.0448:eSUN 1.75mm Red PLA PRO (PLA+) 3D Printer Filament 1KG…
- item 1847 | eval=25 loss=4 gain=1 net_loss=3 | family=3d_filament | mid_in_pct=0.923 | HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, White
  mid_top: 3480:0.00711:HATCHBOX ABS 3D Printer Filament, Dimensional Accuracy… || 3487:0.00691:HATCHBOX ABS 3D Printer Filament, Dimensional Accuracy… || 2476:0.00641:HATCHBOX ABS 3D Printer Filament, Dimensional Accuracy… || 3202:0.00614:XiKe 30012X (300 Qty) 1/2" Steel Balls, Slingshot Ammo… || 3488:0.00571:HATCHBOX ABS 3D Printer Filament, Dimensional Accuracy… || 3485:0.00562:SainSmart 1.75 mm ABS Filament 1 kg/2.2 lb. for 3D Pri…
  coarse_top: 3480:0.331:HATCHBOX ABS 3D Printer Filament, Dimensional Accuracy… || 164:0.0193:HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy… || 1851:0.0193:HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy… || 50:0.0152:HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy… || 1107:0.0128:HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy… || 444:0.0127:HATCHBOX ABS 3D Printer Filament, Dimensional Accuracy…
  rescue_top: 3480:0.945:HATCHBOX ABS 3D Printer Filament, Dimensional Accuracy… || 164:0.0552:HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy…
- item 201 | eval=8 loss=4 gain=1 net_loss=3 | family=connector_fitting | mid_in_pct=0.710 | ColorConnex Coupler & Plug Kit (7 Piece), Industrial Type D, 1/4 in. NPT, Red - A73457D
  mid_top: 2957:0.00661:DBPOWER 2 MP 15M Tube USB Waterproof Hd 6-led Borescop… || 2956:0.00533:DBPOWER 2 MP 15M Tube USB Waterproof Hd 6-led Borescop… || 2451:0.00468:Round Spacer, Nylon, Off-White, 0.194" ID, 3/4" Length… || 2450:0.00438:Round Spacer, Nylon, Off-White, 3/8" Screw Size, 0.675… || 2288:0.00428:Dawson Tools DZA50 AC Line Splitter || 2745:0.00426:HM Digital 1000ppm TDS Calibration Solution
  coarse_top: 218:0.0326:Phresh Duct Silencer 8 in x 24 in || 2938:0.0304:VenTech VT DUCT-6 VTD625 Aluminum Duct for Ventilation… || 2807:0.0218:VIVOSUN 6 Inch 440 CFM Inline Duct Ventilation Fan wit… || 3387:0.0198:iPower 4 Inch 190 CFM Duct Inline Fan with 4" Carbon F… || 2745:0.0198:HM Digital 1000ppm TDS Calibration Solution || 428:0.0193:Arrow Fastener 508IP Genuine T50 1/2-Inch Staples, 5,0…
  rescue_top: 218:0.0931:Phresh Duct Silencer 8 in x 24 in || 2938:0.0869:VenTech VT DUCT-6 VTD625 Aluminum Duct for Ventilation… || 2807:0.0622:VIVOSUN 6 Inch 440 CFM Inline Duct Ventilation Fan wit… || 3387:0.0566:iPower 4 Inch 190 CFM Duct Inline Fan with 4" Carbon F… || 2745:0.0566:HM Digital 1000ppm TDS Calibration Solution || 192:0.054:3M Vetbond Tissue Adhesive, 3ml Bottles w/MSDS
- item 1850 | eval=15 loss=3 gain=0 net_loss=3 | family=3d_filament | mid_in_pct=0.710 | HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, Red
  mid_top: 3474:0.0118:3D Solutech See Through Red 1.75mm PETG 3D Printer Fil… || 3620:0.0115:6 Pcs 100K ohm NTC Thermistors/Temp Sensor for Reprap … || 2724:0.00911:3D Solutech Natural Clear 1.75mm PETG 3D Printer Filam… || 3619:0.0087:eSUN 1.75mm Red PLA PRO (PLA+) 3D Printer Filament 1KG… || 1845:0.00486:Filament Outlet Blue PLA 1.75mm 3D Printer Filament 1k… || 3451:0.0048:Filament Outlet Orange PLA 1.75mm 3D Printer Filament …
  coarse_top: 3458:0.159:Wisamic 4 Variety Pack 3d Printer 0.2mm 0.3mm 0.4mm 0.… || 3619:0.0703:eSUN 1.75mm Red PLA PRO (PLA+) 3D Printer Filament 1KG… || 1790:0.0635:Gizmo Dorks 1.75mm PLA Filament 1kg / 2.2lb for 3D Pri… || 3245:0.0573:MakerBot Smart Extruder+ (For Z18) MP07376 || 1851:0.0298:HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy… || 50:0.0268:HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy…
  rescue_top: 3458:0.454:Wisamic 4 Variety Pack 3d Printer 0.2mm 0.3mm 0.4mm 0.… || 3619:0.201:eSUN 1.75mm Red PLA PRO (PLA+) 3D Printer Filament 1KG… || 1790:0.181:Gizmo Dorks 1.75mm PLA Filament 1kg / 2.2lb for 3D Pri… || 3245:0.164:MakerBot Smart Extruder+ (For Z18) MP07376
- item 2807 | eval=8 loss=3 gain=0 net_loss=3 | family=ventilation_fan | mid_in_pct=0.934 | VIVOSUN 6 Inch 440 CFM Inline Duct Ventilation Fan with Variable Speed Controller
  mid_top: 2957:0.00694:DBPOWER 2 MP 15M Tube USB Waterproof Hd 6-led Borescop… || 2956:0.00566:DBPOWER 2 MP 15M Tube USB Waterproof Hd 6-led Borescop… || 2451:0.0047:Round Spacer, Nylon, Off-White, 0.194" ID, 3/4" Length… || 2745:0.00442:HM Digital 1000ppm TDS Calibration Solution || 2450:0.00441:Round Spacer, Nylon, Off-White, 3/8" Screw Size, 0.675… || 2288:0.0044:Dawson Tools DZA50 AC Line Splitter
  coarse_top: 565:0.149:5lb co2 Tank- New Aluminum Cylinder with CGA320 Valve || 2938:0.142:VenTech VT DUCT-6 VTD625 Aluminum Duct for Ventilation… || 1215:0.1:Clear Polycarbonate Tubing, 1-1/8" ID, 1-1/4" OD, 1/16… || 2775:0.0915:Milwaukee Instruments MA9015 Storage Solution for pH/O… || 2321:0.0747:HM Digital C342 TDS and EC Calibration Solution, 342 p… || 1602:0.0358:STMicroelectronics L7805CV 5V 1.5A Positive Voltage Re…
  rescue_top: 3018:0.0977:VenTech VT DUCT-4 VTD425 Aluminum Duct for Ventilation… || 3128:0.0839:VenTech VT IF6+CF6 IF6CF620 Inline Duct Fan with Virgi… || 2745:0.0529:HM Digital 1000ppm TDS Calibration Solution || 446:0.0519:DEWALT DCS16150 1-1/2-Inch by 16 Gauge Finish Nail (2,… || 2889:0.0509:GROW1 Panda Film, 10' x 25', Poly 5.5 mil, Black/White || 1351:0.0447:GRIDMANN NSF Stainless Steel Commercial Kitchen Prep &…
- item 560 | eval=6 loss=3 gain=0 net_loss=3 | family=test_strip | mid_in_pct=0.710 | Litmus pH Test Strips, Universal Application (pH 1-14), 2 Packs of 100 Strips
  mid_top: 2957:0.00649:DBPOWER 2 MP 15M Tube USB Waterproof Hd 6-led Borescop… || 2956:0.00537:DBPOWER 2 MP 15M Tube USB Waterproof Hd 6-led Borescop… || 2451:0.0043:Round Spacer, Nylon, Off-White, 0.194" ID, 3/4" Length… || 2745:0.00425:HM Digital 1000ppm TDS Calibration Solution || 2889:0.00422:GROW1 Panda Film, 10' x 25', Poly 5.5 mil, Black/White || 2288:0.00411:Dawson Tools DZA50 AC Line Splitter
  coarse_top: 3471:0.0355:Signswise 200x200mm 12v 150w Silicone Rubber Heating H… || 2139:0.0346:Glowz Glow in the Dark Photoluminescent Green Luminous… || 2556:0.0305:Pure Soft Lead Ingot || 2864:0.0282:Etekcity pH-009 Digital Pocket-Sized Pen Type pH Meter… || 2049:0.0247:500 qty 3/8" Inch Steel Shot Slingshot Ammo Balls || 220:0.0231:Tach-It B-1 Single Edge Industrial Razor Blade (Pack o…
  rescue_top: 2864:0.0806:Etekcity pH-009 Digital Pocket-Sized Pen Type pH Meter… || 2049:0.0706:500 qty 3/8" Inch Steel Shot Slingshot Ammo Balls || 385:0.0477:Eclectic Supply B36-24 Cobalt Glass Bottles with Glass… || 1837:0.0473:Premium Vials B4702-12 Glass Vial with Screw Cap, 1 Dr… || 3602:0.0445:American Educational 7-771000 Clear Borosilicate Glass… || 176:0.0438:URBEST 530 Pcs 2:1 Heat Shrink Tubing Tube Sleeving Wr…
- item 98 | eval=5 loss=3 gain=0 net_loss=3 | family=metadata_placeholder | mid_in_pct=0.710 | Industrial & Scientific" />
  mid_top: 1729:0.0159:Smith-Cooper International CV30L Series Brass Check Va… || 437:0.0153:Millrose 70660 Monster Roll PTFE Thread Seal Tape, 1/2… || 2897:0.00898:Ginsco 110pcs Female Red 22-18 Gauge Nylon Fully-Insul… || 2835:0.00895:Ginsco 108pcs Insulated Heat Shrink Waterproof Butt Co… || 2895:0.00876:Ginsco 110pcs Female Blue 16-14 Gauge Nylon Fully-Insu… || 597:0.00822:Ginsco 110pcs Female Yellow 12-10 Gauge Nylon Fully-In…
  coarse_top: 2521:0.043:Kreg SML-C250B-50 Blue-Kote WR Pocket Screws 2-1/2-Inc… || 1267:0.0321:Wago 222-412 LEVER-NUTS 2 Conductor Compact Connectors… || 3470:0.0309:SCStyle 5 Meters GT2 2mm Pitch 6mm Wide Timing Belt fo… || 1917:0.0259:Jancy Slugger 10208W 1 Gallon Water Soluable Cutting F… || 1202:0.023:MakerGear M2 Desktop 3D Printer || 1345:0.0217:POWERTEC 17000 Workbench Caster Kit (Pack of 4)
  rescue_top: 2521:0.123:Kreg SML-C250B-50 Blue-Kote WR Pocket Screws 2-1/2-Inc… || 1267:0.0916:Wago 222-412 LEVER-NUTS 2 Conductor Compact Connectors… || 3470:0.0882:SCStyle 5 Meters GT2 2mm Pitch 6mm Wide Timing Belt fo… || 1917:0.0741:Jancy Slugger 10208W 1 Gallon Water Soluable Cutting F… || 1202:0.0656:MakerGear M2 Desktop 3D Printer || 1345:0.062:POWERTEC 17000 Workbench Caster Kit (Pack of 4)
- item 273 | eval=3 loss=3 gain=0 net_loss=3 | family=adhesive_epoxy | mid_in_pct=0.710 | Gorilla Original Gorilla Glue, Waterproof Polyurethane Glue, 2 ounce Bottle, Brown, (Pack of 4)
  mid_top: 3202:0.00619:XiKe 30012X (300 Qty) 1/2" Steel Balls, Slingshot Ammo… || 977:0.00606:Uriah Products UV001970 17-1/2", Vinyl Battery Carrier… || 1873:0.00594:Gorilla Crystal Clear Duct Tape, 1.88&rdquo; x 5 yd, C… || 1249:0.00519:T-fal C51402 Excite Nonstick Thermo-Spot Dishwasher Sa… || 2594:0.00493:EVINIS 5Pcs&nbsp;Stainless Steel Spatula Palette Knife… || 1250:0.00461:T-fal C51402 Excite Nonstick Thermo-Spot Dishwasher Sa…
  coarse_top: 272:0.23:Gorilla Original Gorilla Glue, Waterproof Polyurethane… || 205:0.0754:Gorilla Glue Adhesive, 2-Ounces #50001 || 311:0.0511:Krazy Glue KG517 Purpose Super Glue, Precision Tip, 2 … || 154:0.0445:3M Multi-Use Duct Tape, 2930-C, 1.88 Inches by 30 Yards || 279:0.0363:WD-40 100324 Multi-Use Product Spray with Smart Straw,… || 1843:0.036:Duco Cement Multi-Purpose Household Glue - 1 fl oz
  rescue_top: 154:0.127:3M Multi-Use Duct Tape, 2930-C, 1.88 Inches by 30 Yards || 1843:0.103:Duco Cement Multi-Purpose Household Glue - 1 fl oz || 1506:0.0884:Loctite Vinyl, Fabric and Plastic Repair Adhesive 1-Ou… || 209:0.0805:Duck Brand 442055 Wrap-Fix Repair Tape, 1-Inch by 10 F… || 945:0.0747:Science Purchase INSTANT Single Panel Drug Test Kit - … || 2322:0.072:16x150mm Glass Test Tube Set with Cork Stoppers, 3.3 B…
- item 175 | eval=8 loss=3 gain=1 net_loss=2 | family=tape | mid_in_pct=0.710 | Gorilla Crystal Clear Duct Tape, 1.88&rdquo; x 9 yd, Clear, (Pack of 1)
  mid_top: 3090:0.0122:WoodPro Fasteners AP9X212-1 #9 by 2-1/2-Inch All Purpo… || 3289:0.00939:WallPeg Pegboard Hooks - 100 pk. Flex-Lock J Style for… || 1249:0.00764:T-fal C51402 Excite Nonstick Thermo-Spot Dishwasher Sa… || 1250:0.00678:T-fal C51402 Excite Nonstick Thermo-Spot Dishwasher Sa… || 1248:0.00631:T-fal C51407 Excite Nonstick Thermo-Spot Dishwasher Sa… || 977:0.00356:Uriah Products UV001970 17-1/2", Vinyl Battery Carrier…
  coarse_top: 203:0.0628:Gorilla 7500101 07221000673 Glue Brush & Nozzle, 1-Pac… || 202:0.0485:Loctite Liquid Professional Super Glue  20-Gram Bottle… || 256:0.0261:Gorilla White Glue, Waterproof, 2 ounce Bottle, White,… || 2050:0.024:E6000 230022 Medium Viscosity Auto/Industrial Adhesive… || 2810:0.0228:TEMO 5 Micron 5gram Diamond Polish Lapping Paste Compo… || 172:0.0187:Gorilla Super Glue Gel, 15 Gram, Clear
  rescue_top: 203:0.179:Gorilla 7500101 07221000673 Glue Brush & Nozzle, 1-Pac… || 202:0.138:Loctite Liquid Professional Super Glue  20-Gram Bottle… || 2050:0.0686:E6000 230022 Medium Viscosity Auto/Industrial Adhesive… || 2810:0.0651:TEMO 5 Micron 5gram Diamond Polish Lapping Paste Compo… || 555:0.0511:3M 3920-BK Duct Tape Black, 1.88 Inches by 20 Yards || 1872:0.0475:ATP Vinyl-Flex PVC Food Grade Plastic Tubing, Clear, 1…

## Top Replacement Pairs
- n=7 | target 2909 3D Solutech Silver Metal 3D Printer PLA Filament 1.75MM Filament… -> pred1 3466 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional… | same_family=1 | pop 155->53 | mid_edge=0.004837 rank=9.0 | rescue=0 ctx=0.2637 direct=20.08
- n=6 | target 3522 3D Solutech Real Blue 3D Printer PLA Filament 1.75MM Filament, D… -> pred1 3466 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional… | same_family=1 | pop 36->53 | mid_edge=0.004579 rank=9.0 | rescue=0 ctx=0 direct=2.333
- n=4 | target 3475 3D Solutech Real Orange 3D Printer PLA Filament 1.75MM Filament,… -> pred1 3466 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional… | same_family=1 | pop 30->53 | mid_edge=0.00424 rank=17.0 | rescue=0 ctx=0 direct=13
- n=4 | target 3442 3D Solutech Real Purple 3D Printer PLA Filament 1.75MM Filament,… -> pred1 3466 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional… | same_family=1 | pop 21->53 | mid_edge=0.003737 rank=24.0 | rescue=0 ctx=0 direct=2.25
- n=4 | target 2909 3D Solutech Silver Metal 3D Printer PLA Filament 1.75MM Filament… -> pred1 3494 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM Filament,… | same_family=1 | pop 155->27 | mid_edge=0.003341 rank=24.0 | rescue=0 ctx=0.6578 direct=1.929
- n=3 | target 2807 VIVOSUN 6 Inch 440 CFM Inline Duct Ventilation Fan with Variable… -> pred1 176 URBEST 530 Pcs 2:1 Heat Shrink Tubing Tube Sleeving Wrap Cable W… | same_family=0 | pop 27->180 | mid_edge=0 rank=999.0 | rescue=0.01479 ctx=0.4439 direct=0
- n=3 | target 3016 HATCHBOX 3D Printer Filament, Dimensional Accuracy +/- 0.03mm, 1… -> pred1 2507 HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 … | same_family=1 | pop 199->266 | mid_edge=0 rank=999.0 | rescue=0 ctx=0.5325 direct=46.83
- n=2 | target 3016 HATCHBOX 3D Printer Filament, Dimensional Accuracy +/- 0.03mm, 1… -> pred1 3466 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional… | same_family=1 | pop 199->53 | mid_edge=0.003741 rank=9.0 | rescue=0 ctx=0 direct=1.667
- n=2 | target 2909 3D Solutech Silver Metal 3D Printer PLA Filament 1.75MM Filament… -> pred1 2697 3D Solutech Real White 3D Printer PLA Filament 1.75MM Filament, … | same_family=1 | pop 155->73 | mid_edge=0.003086 rank=28.0 | rescue=0 ctx=0.4121 direct=37
- n=2 | target 182 AcuRite 00613 Humidity Monitor with Indoor Thermometer, Digital … -> pred1 119 Rubbermaid Commercial BRUTE Heavy-Duty Round Waste/Utility Conta… | same_family=0 | pop 1518->1776 | mid_edge=0 rank=999.0 | rescue=0 ctx=0 direct=44.01
- n=2 | target 1847 HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 … -> pred1 50 HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 … | same_family=1 | pop 483->498 | mid_edge=0 rank=999.0 | rescue=0 ctx=0.6312 direct=151.2
- n=2 | target 2507 HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 … -> pred1 3466 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional… | same_family=1 | pop 266->53 | mid_edge=0 rank=999.0 | rescue=0 ctx=0 direct=10.67
- n=2 | target 3522 3D Solutech Real Blue 3D Printer PLA Filament 1.75MM Filament, D… -> pred1 2697 3D Solutech Real White 3D Printer PLA Filament 1.75MM Filament, … | same_family=1 | pop 36->73 | mid_edge=0 rank=999.0 | rescue=0 ctx=0 direct=7.833
- n=1 | target 2932 Inland 1.75mm Blue PLA 3D Printer Filament - 1kg Spool (2.2 lbs) -> pred1 2485 Inland 1.75mm Gold PLA 3D Printer Filament - 1kg Spool (2.2 lbs) | same_family=1 | pop 66->25 | mid_edge=0.02294 rank=1.0 | rescue=0 ctx=0.4168 direct=30.5
- n=1 | target 2934 Inland 1.75mm Yellow PLA 3D Printer Filament - 1kg Spool (2.2 lb… -> pred1 2485 Inland 1.75mm Gold PLA 3D Printer Filament - 1kg Spool (2.2 lbs) | same_family=1 | pop 26->25 | mid_edge=0.02191 rank=1.0 | rescue=0.3967 ctx=0.5825 direct=0
- n=1 | target 2576 First Aid Only 13-040 First Aid Burn Spray, 4oz Pump Bottle -> pred1 2793 First Aid Only Splinter Out, 10 Per Box | same_family=1 | pop 14->72 | mid_edge=0.01536 rank=7.0 | rescue=0 ctx=0.3658 direct=2.5
- n=1 | target 2059 MG Chemicals Wood 3D Printer Filament, 1.75mm, 0.5 Kg (1.1 lbs.)… -> pred1 2979 Inland 1.75mm Gray PLA 3D Printer Filament - 1kg Spool (2.2 lbs) | same_family=1 | pop 58->50 | mid_edge=0.007782 rank=11.0 | rescue=0 ctx=0.2666 direct=16.25
- n=1 | target 417 American Terminal E-FFB250N-100 16/14-Gauge Economy Nylon Fully-… -> pred1 593 Gardner Bender 10-106 50PK Ring Terminal, Yellow | same_family=0 | pop 1228->98 | mid_edge=0.006096 rank=21.0 | rescue=0 ctx=0 direct=37.37
- n=1 | target 1772 Dixon Valve HHP2M Brass Fitting, Hex Head Plug, 1/4" NPT Male -> pred1 1769 Dixon 179-0606 Brass Hose Splicer Fitting, Tee, 3/8" Hose ID Bar… | same_family=1 | pop 34->26 | mid_edge=0.005437 rank=15.0 | rescue=0 ctx=0 direct=0
- n=1 | target 2697 3D Solutech Real White 3D Printer PLA Filament 1.75MM Filament, … -> pred1 3466 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional… | same_family=1 | pop 73->53 | mid_edge=0.004809 rank=8.0 | rescue=0 ctx=0.2837 direct=5.214
- n=1 | target 1552 3D Solutech Teal Blue 3D Printer PLA Filament 1.75MM Filament, D… -> pred1 3466 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional… | same_family=1 | pop 41->53 | mid_edge=0.004624 rank=8.0 | rescue=0 ctx=0.2523 direct=10
- n=1 | target 1851 HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 … -> pred1 3466 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional… | same_family=1 | pop 409->53 | mid_edge=0.004257 rank=9.0 | rescue=0 ctx=0 direct=3.5
- n=1 | target 2063 3Doodler Create 3D Pen with 50 Plastic Strands, No Mess, Non-Tox… -> pred1 2143 3Doodler Start Emoji & Symbol DoodleBlock Kit with 2 Plastic Pac… | same_family=1 | pop 32->18 | mid_edge=0.003387 rank=15.0 | rescue=0 ctx=0.1813 direct=10.5
- n=1 | target 2152 HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 … -> pred1 1849 HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 … | same_family=1 | pop 152->121 | mid_edge=0.003192 rank=26.0 | rescue=0 ctx=0.6594 direct=28.5
- n=1 | target 3475 3D Solutech Real Orange 3D Printer PLA Filament 1.75MM Filament,… -> pred1 3494 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM Filament,… | same_family=1 | pop 30->27 | mid_edge=0.00295 rank=30.0 | rescue=0 ctx=0.3502 direct=7