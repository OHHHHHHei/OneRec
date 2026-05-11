# R650a vs MiniOneRec Original Tokenizer Error Overlap

## Comparison Scope
- Primary original baseline: strongest original MiniOneRec SFT, `title_history2sid_off + desc_align_p05`.
- Secondary reference: recipe-aligned original SFT, `title_history2sid_on + desc_align_p05`.
- Current tokenizer: R650a Seq2Graph-lite high-order carrier + mid-only pull-push, `title_history2sid_on + desc_align_p05`.
- Target item ids are read from the test CSV to avoid SID-collision reverse-map ambiguity.

## SID Space Metrics
| label | unique_sid | collision_items | unique_l1 | unique_l2_pairs | l1_bucket_mean | l1_bucket_max | l2_bucket_mean | l2_bucket_max | mean_l2_per_l1 | mean_l3_per_l2 | h_l2_given_l1 | h_l3_given_l12 | l1_family_purity_mean | l2_family_purity_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original_semantic | 3670 | 31 | 48 | 2295 | 76.7917 | 247 | 1.6061 | 47 | 47.8125 | 1.5991 | 5.4659 | 1.0254 | 0.7323 | 0.9782 |
| r650a_seq2graph_mid_pull_push | 3675 | 22 | 199 | 2782 | 18.5226 | 74 | 1.3249 | 22 | 13.9799 | 1.3210 | 3.7236 | 0.6280 | 0.7821 | 0.9876 |

## Strong original vs R650a
| k | original hit | R650a hit | both hit | original only hit | R650a only hit | both miss |
|---|---:|---:|---:|---:|---:|---:|
| @1 | 304 | 296 | 239 | 65 | 57 | 4172 |
| @3 | 446 | 424 | 318 | 128 | 106 | 3981 |
| @5 | 536 | 495 | 374 | 162 | 121 | 3876 |
| @10 | 684 | 600 | 487 | 197 | 113 | 3736 |
| @50 | 1112 | 1042 | 824 | 288 | 218 | 3203 |

## Recipe-aligned original vs R650a
| k | original hit | R650a hit | both hit | original only hit | R650a only hit | both miss |
|---|---:|---:|---:|---:|---:|---:|
| @1 | 288 | 296 | 240 | 48 | 56 | 4189 |
| @3 | 439 | 424 | 311 | 128 | 113 | 3981 |
| @5 | 513 | 495 | 356 | 157 | 139 | 3881 |
| @10 | 644 | 600 | 457 | 187 | 143 | 3746 |
| @50 | 1106 | 1042 | 815 | 291 | 227 | 3200 |

## Family Distribution For Strong Original Comparison
### both_miss10
- other: 1704 (45.6%)
- 3d_filament: 820 (21.9%)
- gauge_meter: 280 (7.5%)
- tape: 265 (7.1%)
- connector_fitting: 235 (6.3%)
- adhesive_epoxy: 200 (5.4%)
- fastener: 138 (3.7%)
- ventilation_fan: 55 (1.5%)
- metadata_placeholder: 29 (0.8%)
- test_strip: 10 (0.3%)

### orig_only_hit10_r650_miss
- 3d_filament: 92 (46.7%)
- other: 36 (18.3%)
- gauge_meter: 29 (14.7%)
- connector_fitting: 20 (10.2%)
- tape: 11 (5.6%)
- adhesive_epoxy: 5 (2.5%)
- ventilation_fan: 2 (1.0%)
- test_strip: 1 (0.5%)
- fastener: 1 (0.5%)

### r650_only_hit10_orig_miss
- 3d_filament: 47 (41.6%)
- other: 28 (24.8%)
- gauge_meter: 15 (13.3%)
- adhesive_epoxy: 11 (9.7%)
- tape: 6 (5.3%)
- connector_fitting: 5 (4.4%)
- metadata_placeholder: 1 (0.9%)

## Top Common Miss Items
| item_id | eval_count | family | orig_hit10_rate | r650_hit10_rate | both_miss10_count | orig_only_hit10_count | r650_only_hit10_count | orig_l2_bucket_size | r650_l2_bucket_size | orig_l2_family_purity | r650_l2_family_purity | coarse_seq2g_in_strength_pct | mid_seq2g_in_strength_pct | title |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 562 | 29 | 3d_filament | 0.000 | 0.000 | 29 | 0 | 0 | 47 | 22 | 1.000 | 1.000 | 0.100 | 0.710 | HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool,… |
| 2912 | 31 | 3d_filament | 0.129 | 0.161 | 25 | 1 | 2 | 19 | 16 | 1.000 | 1.000 | 0.955 | 0.890 | Inland 1.75mm Black PLA 3D Printer Filament - 1kg Spool (2.2 lbs) |
| 2058 | 25 | 3d_filament | 0.000 | 0.040 | 24 | 0 | 1 | 47 | 22 | 1.000 | 1.000 | 0.626 | 0.710 | HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool,… |
| 1959 | 22 | adhesive_epoxy | 0.000 | 0.000 | 22 | 0 | 0 | 2 | 1 | 1.000 | 1.000 | 0.330 | 0.710 | Gorilla Super Glue Gel, 20 Gram, Clear |
| 181 | 132 | gauge_meter | 0.826 | 0.720 | 19 | 18 | 4 | 2 | 2 | 1.000 | 1.000 | nan | nan | AcuRite 00613 Humidity Monitor with Indoor Thermometer, Digital Hygrometer and … |
| 84 | 18 | gauge_meter | 0.000 | 0.111 | 16 | 0 | 2 | 2 | 1 | 1.000 | 1.000 | 0.996 | 0.710 | Neiko 01409A Electronic Digital Caliper with Extra Large LCD Screen / 0 - 12 In… |
| 3112 | 32 | 3d_filament | 0.531 | 0.250 | 15 | 9 | 0 | 24 | 3 | 1.000 | 1.000 | 0.914 | 0.857 | 3D Solutech Real Black 3D Printer PLA Filament 1.75MM Filament, Dimensional Acc… |
| 2161 | 19 | 3d_filament | 0.158 | 0.211 | 15 | 0 | 1 | 24 | 18 | 1.000 | 1.000 | 0.877 | 0.710 | 3D Solutech Natural Clear 1.75mm 3D Printer PLA Filament, Dimensional Accuracy … |
| 338 | 19 | gauge_meter | 0.211 | 0.105 | 14 | 3 | 1 | 2 | 1 | 1.000 | 1.000 | 0.977 | 0.710 | Etekcity Lasergrip 1080 Non-Contact Digital Laser Infrared Thermometer Temperat… |
| 570 | 15 | other | 0.067 | 0.000 | 14 | 1 | 0 | 1 | 1 | 1.000 | 1.000 | 0.922 | 0.710 | Bissell 9595A CleanView Bagless Vacuum with OnePass |
| 3016 | 17 | 3d_filament | 0.118 | 0.176 | 13 | 1 | 2 | 3 | 4 | 0.667 | 1.000 | 0.979 | 0.710 | HATCHBOX 3D Printer Filament, Dimensional Accuracy +/- 0.03mm, 1.75 mm, 1 kg Sp… |
| 159 | 14 | gauge_meter | 0.000 | 0.071 | 13 | 0 | 1 | 2 | 1 | 1.000 | 1.000 | 0.976 | 0.710 | Etekcity Lasergrip 774 Non-contact Digital Laser Infrared Thermometer Temperatu… |
| 178 | 14 | other | 0.071 | 0.071 | 13 | 0 | 0 | 3 | 1 | 1.000 | 1.000 | 0.982 | 0.710 | Neiko 10194A Titanium Step Drill Bit, High Speed Steel / 1/4" to 1-3/8" / Total… |
| 2703 | 14 | 3d_filament | 0.071 | 0.071 | 13 | 0 | 0 | 10 | 8 | 1.000 | 1.000 | 0.991 | 0.710 | eSUN 3D 1.75mm PETG Black Filament 1kg (2.2lb), PETG 3D Printer Filament, 1.75m… |
| 2734 | 21 | 3d_filament | 0.333 | 0.381 | 12 | 1 | 2 | 19 | 16 | 1.000 | 1.000 | 0.859 | 0.956 | Inland 1.75mm White PLA 3D Printer Filament - 1kg Spool (2.2 lbs) |

## Top R650a Regression Items: original hit@10 but R650a miss@10
| item_id | eval_count | family | orig_hit10_rate | r650_hit10_rate | both_miss10_count | orig_only_hit10_count | r650_only_hit10_count | orig_l2_bucket_size | r650_l2_bucket_size | orig_l2_family_purity | r650_l2_family_purity | coarse_seq2g_in_strength_pct | mid_seq2g_in_strength_pct | title |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2909 | 31 | 3d_filament | 0.710 | 0.065 | 9 | 20 | 0 | 24 | 3 | 1.000 | 1.000 | 0.918 | 0.771 | 3D Solutech Silver Metal 3D Printer PLA Filament 1.75MM Filament, Dimensional A… |
| 181 | 132 | gauge_meter | 0.826 | 0.720 | 19 | 18 | 4 | 2 | 2 | 1.000 | 1.000 | nan | nan | AcuRite 00613 Humidity Monitor with Indoor Thermometer, Digital Hygrometer and … |
| 3112 | 32 | 3d_filament | 0.531 | 0.250 | 15 | 9 | 0 | 24 | 3 | 1.000 | 1.000 | 0.914 | 0.857 | 3D Solutech Real Black 3D Printer PLA Filament 1.75MM Filament, Dimensional Acc… |
| 2660 | 10 | 3d_filament | 0.700 | 0.000 | 3 | 7 | 0 | 24 | 18 | 1.000 | 1.000 | 0.630 | 0.710 | 3D Solutech Real Grey 3D Printer PLA Filament 1.75MM Filament, Dimensional Accu… |
| 444 | 8 | 3d_filament | 0.625 | 0.125 | 3 | 4 | 0 | 47 | 21 | 1.000 | 1.000 | 0.803 | 0.803 | HATCHBOX ABS 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool,… |
| 1847 | 25 | 3d_filament | 0.680 | 0.560 | 7 | 4 | 1 | 5 | 4 | 1.000 | 1.000 | 0.985 | 0.923 | HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool,… |
| 130 | 15 | tape | 0.333 | 0.267 | 7 | 4 | 3 | 8 | 4 | 1.000 | 1.000 | 0.987 | 0.710 | Gorilla Tape, Black Duct Tape, 1.88" x 35 yd, Black, (Pack of 1) |
| 273 | 3 | adhesive_epoxy | 1.000 | 0.000 | 0 | 3 | 0 | 5 | 4 | 1.000 | 1.000 | 0.180 | 0.710 | Gorilla Original Gorilla Glue, Waterproof Polyurethane Glue, 2 ounce Bottle, Br… |
| 2718 | 6 | 3d_filament | 0.500 | 0.000 | 3 | 3 | 0 | 19 | 16 | 1.000 | 1.000 | 0.755 | 0.710 | Inland 1.75mm Pink PLA 3D Printer Filament - 1kg Spool (2.2 lbs) |
| 3631 | 6 | 3d_filament | 0.500 | 0.000 | 3 | 3 | 0 | 24 | 6 | 1.000 | 1.000 | 0.290 | 0.953 | 3D Solutech Skin 3D Printer PLA Filament 1.75MM Filament 2.2 LBS (1.0KG) |
| 3557 | 8 | 3d_filament | 0.500 | 0.125 | 4 | 3 | 0 | 24 | 6 | 1.000 | 1.000 | 0.899 | 0.894 | 3D Solutech Chocolate Brown 3D Printer PLA Filament 1.75MM Filament 2.2 LBS (1.… |
| 338 | 19 | gauge_meter | 0.211 | 0.105 | 14 | 3 | 1 | 2 | 1 | 1.000 | 1.000 | 0.977 | 0.710 | Etekcity Lasergrip 1080 Non-Contact Digital Laser Infrared Thermometer Temperat… |
| 176 | 11 | other | 0.273 | 0.182 | 6 | 3 | 2 | 1 | 4 | 1.000 | 1.000 | 0.985 | 0.710 | URBEST 530 Pcs 2:1 Heat Shrink Tubing Tube Sleeving Wrap Cable Wire 5 Color 8 S… |
| 635 | 2 | other | 1.000 | 0.000 | 0 | 2 | 0 | 3 | 2 | 1.000 | 1.000 | 0.974 | 0.710 | Texas Instruments NE555N NE555 NE555P General Purpose Single Bipolar Timer DIP8… |
| 2869 | 3 | other | 0.667 | 0.000 | 1 | 2 | 0 | 2 | 2 | 1.000 | 1.000 | 0.277 | 0.816 | 13cm Dia. (5.1") Filter Funnel, Buchner Style - 2 Parts, Polypropylene, Designe… |

## Top R650a Rescue Items: R650a hit@10 but original miss@10
| item_id | eval_count | family | orig_hit10_rate | r650_hit10_rate | both_miss10_count | orig_only_hit10_count | r650_only_hit10_count | orig_l2_bucket_size | r650_l2_bucket_size | orig_l2_family_purity | r650_l2_family_purity | coarse_seq2g_in_strength_pct | mid_seq2g_in_strength_pct | title |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3466 | 11 | 3d_filament | 0.364 | 0.909 | 1 | 0 | 6 | 24 | 18 | 1.000 | 1.000 | 0.784 | 0.916 | 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional Accuracy +/- 0… |
| 40 | 17 | other | 0.118 | 0.353 | 9 | 2 | 6 | 2 | 1 | 1.000 | 1.000 | 0.967 | 0.710 | Forney 20859 Cutting Fluid, Industrial Pro Tap Magic, 1-Gallon |
| 2697 | 30 | 3d_filament | 0.500 | 0.667 | 10 | 0 | 5 | 24 | 18 | 1.000 | 1.000 | 0.841 | 0.828 | 3D Solutech Real White 3D Printer PLA Filament 1.75MM Filament, Dimensional Acc… |
| 57 | 16 | adhesive_epoxy | 0.000 | 0.250 | 12 | 0 | 4 | 1 | 1 | 1.000 | 1.000 | 0.916 | 0.710 | Gorilla 2 Part Epoxy, 5 Minute Set, .85 ounce Syringe, Clear |
| 181 | 132 | gauge_meter | 0.826 | 0.720 | 19 | 18 | 4 | 2 | 2 | 1.000 | 1.000 | nan | nan | AcuRite 00613 Humidity Monitor with Indoor Thermometer, Digital Hygrometer and … |
| 130 | 15 | tape | 0.333 | 0.267 | 7 | 4 | 3 | 8 | 4 | 1.000 | 1.000 | 0.987 | 0.710 | Gorilla Tape, Black Duct Tape, 1.88" x 35 yd, Black, (Pack of 1) |
| 1848 | 2 | 3d_filament | 0.000 | 1.000 | 0 | 0 | 2 | 2 | 1 | 1.000 | 1.000 | 0.941 | 0.710 | Kamo 5PCS 3D Printer 0.4mm Extruder Brass Nozzle Print Head for MK8 1.75mm ABS … |
| 2311 | 3 | adhesive_epoxy | 0.000 | 0.667 | 1 | 0 | 2 | 6 | 7 | 1.000 | 1.000 | 0.012 | 0.710 | TEMCo 3/4" Marine Heat Shrink Tube 3:1 Adhesive Glue Lined 4 ft RED |
| 970 | 5 | gauge_meter | 0.000 | 0.400 | 3 | 0 | 2 | 1 | 3 | 1.000 | 1.000 | 0.658 | 0.710 | Wixey WR25 Mini Digital Height Gauge |
| 675 | 5 | adhesive_epoxy | 0.200 | 0.600 | 2 | 0 | 2 | 4 | 3 | 1.000 | 1.000 | 0.976 | 0.710 | J-B Weld 8265S Original Cold-Weld Steel Reinforced Epoxy - 2 oz. |
| 2507 | 7 | 3d_filament | 0.286 | 0.571 | 3 | 0 | 2 | 47 | 22 | 1.000 | 1.000 | 0.970 | 0.710 | HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool,… |
| 1554 | 8 | 3d_filament | 0.000 | 0.250 | 6 | 0 | 2 | 19 | 16 | 1.000 | 1.000 | 0.500 | 0.907 | Inland 1.75mm Peak Green PLA 3D Printer Filament - 1kg Spool (2.2 lbs) |
| 3522 | 14 | 3d_filament | 0.000 | 0.143 | 12 | 0 | 2 | 24 | 3 | 1.000 | 1.000 | 0.191 | 0.710 | 3D Solutech Real Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accu… |
| 84 | 18 | gauge_meter | 0.000 | 0.111 | 16 | 0 | 2 | 2 | 1 | 1.000 | 1.000 | 0.996 | 0.710 | Neiko 01409A Electronic Digital Caliper with Extra Large LCD Screen / 0 - 12 In… |
| 3016 | 17 | 3d_filament | 0.118 | 0.176 | 13 | 1 | 2 | 3 | 4 | 0.667 | 1.000 | 0.979 | 0.710 | HATCHBOX 3D Printer Filament, Dimensional Accuracy +/- 0.03mm, 1.75 mm, 1 kg Sp… |

## Both Miss Example Pred1 Pairs
- n=5 | target 164 HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1… | original pred1 50: HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy… | R650a pred1 50: HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy…
- n=4 | target 562 HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1… | original pred1 50: HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy… | R650a pred1 1850: HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy…
- n=4 | target 570 Bissell 9595A CleanView Bagless Vacuum with OnePass | original pred1 181: AcuRite 00613 Humidity Monitor with Indoor Thermometer… | R650a pred1 181: AcuRite 00613 Humidity Monitor with Indoor Thermometer…
- n=3 | target 2807 VIVOSUN 6 Inch 440 CFM Inline Duct Ventilation Fan with Variable Spee… | original pred1 181: AcuRite 00613 Humidity Monitor with Indoor Thermometer… | R650a pred1 176: URBEST 530 Pcs 2:1 Heat Shrink Tubing Tube Sleeving Wr…
- n=3 | target 2058 HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1… | original pred1 50: HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy… | R650a pred1 50: HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy…
- n=3 | target 2721 3D Solutech Navy Blue 3D Printer PLA Filament 1.75MM Filament, Dimens… | original pred1 2659: 3D Solutech Apple Green 3D Printer PLA Filament 1.75MM… | R650a pred1 3466: 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, D…
- n=3 | target 3522 3D Solutech Real Blue 3D Printer PLA Filament 1.75MM Filament, Dimens… | original pred1 2909: 3D Solutech Silver Metal 3D Printer PLA Filament 1.75M… | R650a pred1 3466: 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, D…
- n=3 | target 159 Etekcity Lasergrip 774 Non-contact Digital Laser Infrared Thermometer… | original pred1 181: AcuRite 00613 Humidity Monitor with Indoor Thermometer… | R650a pred1 181: AcuRite 00613 Humidity Monitor with Indoor Thermometer…
- n=3 | target 3468 3D Solutech Real Green 3D Printer PLA Filament 1.75MM Filament, Dimen… | original pred1 2909: 3D Solutech Silver Metal 3D Printer PLA Filament 1.75M… | R650a pred1 3466: 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, D…
- n=2 | target 1522 Yosoo GM328 Lcd Display Transistor Tester ESR Meter Cymometer Square … | original pred1 635: Texas Instruments NE555N NE555 NE555P General Purpose … | R650a pred1 2624: Uxcell a13060500ux0042 3 Pins Split Shaft Rotary Linea…
- n=2 | target 3522 3D Solutech Real Blue 3D Printer PLA Filament 1.75MM Filament, Dimens… | original pred1 3494: 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM… | R650a pred1 3466: 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, D…
- n=2 | target 3446 Anycubic 1.75mm Skin Color PLA 3D Printer Filament - 1kg Spool (2.2 l… | original pred1 2979: Inland 1.75mm Gray PLA 3D Printer Filament - 1kg Spool… | R650a pred1 2979: Inland 1.75mm Gray PLA 3D Printer Filament - 1kg Spool…
- n=2 | target 3468 3D Solutech Real Green 3D Printer PLA Filament 1.75MM Filament, Dimen… | original pred1 3494: 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM… | R650a pred1 3466: 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, D…
- n=2 | target 726 5 Gallon Plastic Hedpack with cap | original pred1 94: White SiliconeTubing, 3/16"ID, 1/4"OD, 1/32" Wall, 10'… | R650a pred1 94: White SiliconeTubing, 3/16"ID, 1/4"OD, 1/32" Wall, 10'…
- n=2 | target 1959 Gorilla Super Glue Gel, 20 Gram, Clear | original pred1 1107: HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy… | R650a pred1 50: HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy…
- n=2 | target 3113 Inland 1.75mm Silver PLA 3D Printer Filament - 1kg Spool (2.2 lbs) | original pred1 2979: Inland 1.75mm Gray PLA 3D Printer Filament - 1kg Spool… | R650a pred1 2979: Inland 1.75mm Gray PLA 3D Printer Filament - 1kg Spool…
- n=2 | target 3113 Inland 1.75mm Silver PLA 3D Printer Filament - 1kg Spool (2.2 lbs) | original pred1 2979: Inland 1.75mm Gray PLA 3D Printer Filament - 1kg Spool… | R650a pred1 2485: Inland 1.75mm Gold PLA 3D Printer Filament - 1kg Spool…
- n=2 | target 1346 Aluminum Tape/Aluminum Foil Tape - 1.9 inch x 150 feet (3.4 mil) - Go… | original pred1 119: Rubbermaid Commercial BRUTE Heavy-Duty Round Waste/Uti… | R650a pred1 119: Rubbermaid Commercial BRUTE Heavy-Duty Round Waste/Uti…
- n=2 | target 2044 Gorilla 100 Percent Silicone Sealant Caulk, 2.8 ounce Squeeze Tube, C… | original pred1 140: DuPont Teflon Silicone Lubricant | R650a pred1 140: DuPont Teflon Silicone Lubricant
- n=2 | target 2744 BIQU Heat Bed Power Module Expansion Hot Bed MOS Tube for 3D Printer | original pred1 2979: Inland 1.75mm Gray PLA 3D Printer Filament - 1kg Spool… | R650a pred1 2912: Inland 1.75mm Black PLA 3D Printer Filament - 1kg Spoo…

## R650 Regression Pred1 Pairs
- n=3 | target 2660 3D Solutech Real Grey 3D Printer PLA Filament 1.75MM Filament, Dimens… | original pred1 2659: 3D Solutech Apple Green 3D Printer PLA Filament 1.75MM… | R650a pred1 3466: 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, D…
- n=3 | target 2909 3D Solutech Silver Metal 3D Printer PLA Filament 1.75MM Filament, Dim… | original pred1 2909: 3D Solutech Silver Metal 3D Printer PLA Filament 1.75M… | R650a pred1 3466: 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, D…
- n=3 | target 2909 3D Solutech Silver Metal 3D Printer PLA Filament 1.75MM Filament, Dim… | original pred1 2659: 3D Solutech Apple Green 3D Printer PLA Filament 1.75MM… | R650a pred1 3466: 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, D…
- n=2 | target 2932 Inland 1.75mm Blue PLA 3D Printer Filament - 1kg Spool (2.2 lbs) | original pred1 2979: Inland 1.75mm Gray PLA 3D Printer Filament - 1kg Spool… | R650a pred1 2485: Inland 1.75mm Gold PLA 3D Printer Filament - 1kg Spool…
- n=2 | target 3112 3D Solutech Real Black 3D Printer PLA Filament 1.75MM Filament, Dimen… | original pred1 2477: BAMtack! 1.75mm Black PLA 3D Printer Filament - 1kg (2… | R650a pred1 2697: 3D Solutech Real White 3D Printer PLA Filament 1.75MM …
- n=2 | target 3112 3D Solutech Real Black 3D Printer PLA Filament 1.75MM Filament, Dimen… | original pred1 2659: 3D Solutech Apple Green 3D Printer PLA Filament 1.75MM… | R650a pred1 3466: 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, D…
- n=2 | target 2909 3D Solutech Silver Metal 3D Printer PLA Filament 1.75MM Filament, Dim… | original pred1 3494: 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM… | R650a pred1 3494: 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM…
- n=2 | target 2909 3D Solutech Silver Metal 3D Printer PLA Filament 1.75MM Filament, Dim… | original pred1 3494: 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM… | R650a pred1 3466: 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, D…
- n=2 | target 1847 HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1… | original pred1 1107: HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy… | R650a pred1 50: HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy…
- n=2 | target 2909 3D Solutech Silver Metal 3D Printer PLA Filament 1.75MM Filament, Dim… | original pred1 2659: 3D Solutech Apple Green 3D Printer PLA Filament 1.75MM… | R650a pred1 3494: 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM…
- n=1 | target 2734 Inland 1.75mm White PLA 3D Printer Filament - 1kg Spool (2.2 lbs) | original pred1 2979: Inland 1.75mm Gray PLA 3D Printer Filament - 1kg Spool… | R650a pred1 1848: Kamo 5PCS 3D Printer 0.4mm Extruder Brass Nozzle Print…
- n=1 | target 2564 20 Pieces LM386N-1 LM386N LM386 Low Voltage Audio Power Amplifier | original pred1 1881: Phantom YoYo 170 Points Mini Breadboard for Arduino Pr… | R650a pred1 3541: 5PCS Nema 17 Stepper Motor Bipolar 2A 84oz.in 48mm 4-l…
- n=1 | target 2732 PTFE Teflon Bowden Tube for 1.75 Filament (2.0mm ID/4.0mm OD) &ndash;… | original pred1 2477: BAMtack! 1.75mm Black PLA 3D Printer Filament - 1kg (2… | R650a pred1 50: HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy…
- n=1 | target 2586 10 Assorted Kelly Locking Hemostat Forceps 5.5 | original pred1 2586: 10 Assorted Kelly Locking Hemostat Forceps 5.5 | R650a pred1 739: Tensive Parker Labs Conductive Adhesive Gel, 50 Gram
- n=1 | target 2721 3D Solutech Navy Blue 3D Printer PLA Filament 1.75MM Filament, Dimens… | original pred1 3468: 3D Solutech Real Green 3D Printer PLA Filament 1.75MM … | R650a pred1 3466: 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, D…
- n=1 | target 2718 Inland 1.75mm Pink PLA 3D Printer Filament - 1kg Spool (2.2 lbs) | original pred1 2979: Inland 1.75mm Gray PLA 3D Printer Filament - 1kg Spool… | R650a pred1 3633: [STAR] Alchement - TPU Series, 3D Filament, 1.75mm, 1k…
- n=1 | target 2718 Inland 1.75mm Pink PLA 3D Printer Filament - 1kg Spool (2.2 lbs) | original pred1 2979: Inland 1.75mm Gray PLA 3D Printer Filament - 1kg Spool… | R650a pred1 2912: Inland 1.75mm Black PLA 3D Printer Filament - 1kg Spoo…
- n=1 | target 2617 Yueton&reg; 100pcs Yellow 12/10-Gauge Economy Nylon Male Fully-Insula… | original pred1 32: 4-1/2" Cut-off Wheels for Metal, for Cutting All Ferro… | R650a pred1 793: Yost MU360 Universal Jaw Cover, 6
- n=1 | target 2659 3D Solutech Apple Green 3D Printer PLA Filament 1.75MM Filament, Dime… | original pred1 2659: 3D Solutech Apple Green 3D Printer PLA Filament 1.75MM… | R650a pred1 3494: 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM…
- n=1 | target 2718 Inland 1.75mm Pink PLA 3D Printer Filament - 1kg Spool (2.2 lbs) | original pred1 2979: Inland 1.75mm Gray PLA 3D Printer Filament - 1kg Spool… | R650a pred1 2485: Inland 1.75mm Gold PLA 3D Printer Filament - 1kg Spool…

## R650 Rescue Pred1 Pairs
- n=2 | target 2697 3D Solutech Real White 3D Printer PLA Filament 1.75MM Filament, Dimen… | original pred1 3112: 3D Solutech Real Black 3D Printer PLA Filament 1.75MM … | R650a pred1 3466: 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, D…
- n=1 | target 2605 Acid Blend - 1 lb. | original pred1 116: Rescue Tape RT1000201202USCO | R650a pred1 24: Wine Yeast Red Star Premier Classique Formerly Montrac…
- n=1 | target 2507 HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1… | original pred1 3151: Anycubic Pulley Version Unassemble Delta Rostock 3D Pr… | R650a pred1 50: HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy…
- n=1 | target 2507 HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1… | original pred1 51: HATCHBOX 1 Spool 3D Printer Filament Tabletop Wall Mou… | R650a pred1 50: HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy…
- n=1 | target 2477 BAMtack! 1.75mm Black PLA 3D Printer Filament - 1kg (2.2 lbs) - Dimen… | original pred1 2979: Inland 1.75mm Gray PLA 3D Printer Filament - 1kg Spool… | R650a pred1 3101: PolyMax PLA True Black - 1.75mm (0.75kg)
- n=1 | target 2472 HATCHBOX ABS 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1… | original pred1 2678: eSUN 1.75mm White ABS 3D Printer filament 1kg Spool (2… | R650a pred1 3414: eSUN 1.75mm White ABS+ 3D Printer filament 1kg Spool (…
- n=1 | target 2463 Adventure Medical SOL Duct Tape - Two 50in Rolls | original pred1 2827: Pac-Kit by First Aid Only 21-770 Disposable Thermomete… | R650a pred1 1350: Bluecell Pack of 9 PCS AA / AAA Battery Storage Hard C…
- n=1 | target 2446 100x 1N4007 Diode 1A 1000V Rectifier Diodes Arduino Motor Snubber Fly… | original pred1 2229: Uxcell a11093000ux0385 10x 8 Pin DIP IC Sockets Adapto… | R650a pred1 633: 100 x 2N2222 NPN TO-92 Plastic-Encapsulate Power Trans…
- n=1 | target 2383 Inkbird All-Purpose Digital Temperature Controller Fahrenheit &Centig… | original pred1 2321: HM Digital C342 TDS and EC Calibration Solution, 342 p… | R650a pred1 459: 50 Pack - 4.5"x.040"x7/8" Quality Thin Cut Off Wheels …
- n=1 | target 2333 eSUN 3D Printer CLEANING Filament 1.75mm Natural 0.1kg for all 1.75mm… | original pred1 2472: HATCHBOX ABS 3D Printer Filament, Dimensional Accuracy… | R650a pred1 2507: HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy…
- n=1 | target 2311 TEMCo 3/4" Marine Heat Shrink Tube 3:1 Adhesive Glue Lined 4 ft RED | original pred1 3232: TEMCo 3/8" Marine Heat Shrink Tube 3:1 Adhesive Glue L… | R650a pred1 2103: TEMCo 1/2" Marine Heat Shrink Tube 3:1 Adhesive Glue L…
- n=1 | target 2311 TEMCo 3/4" Marine Heat Shrink Tube 3:1 Adhesive Glue Lined 4 ft RED | original pred1 216: Ancor Tinned Copper Lugs 8 AWG - 4/0 AWG | R650a pred1 2103: TEMCo 1/2" Marine Heat Shrink Tube 3:1 Adhesive Glue L…
- n=1 | target 2161 3D Solutech Natural Clear 1.75mm 3D Printer PLA Filament, Dimensional… | original pred1 3112: 3D Solutech Real Black 3D Printer PLA Filament 1.75MM … | R650a pred1 2507: HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy…
- n=1 | target 2697 3D Solutech Real White 3D Printer PLA Filament 1.75MM Filament, Dimen… | original pred1 2544: SainSmart Clear Flexible TPU 3D Printing Filament, 1.7… | R650a pred1 2697: 3D Solutech Real White 3D Printer PLA Filament 1.75MM …
- n=1 | target 2156 Monoprice ABS 3D Printer Filament - White - 1kg Spool, 1.75mm Thick |… | original pred1 2716: 3D printer Filament 1.75 ABS Red 1kg 2.2lb 100% USA | R650a pred1 3462: Gizmo Dorks 3mm (2.85mm) ABS Filament 1kg / 2.2lb for …
- n=1 | target 2106 Noga NG8150 Heavy Duty Deburr Tool, with 10 S10 blades | original pred1 1042: uxcell Mandrel Mounted White Conical Felt Point Polish… | R650a pred1 1075: Industrial & Scientific" />
- n=1 | target 2102 TEMCo 3/8" Marine Heat Shrink Tube 3:1 Adhesive Glue Lined 4 ft RED | original pred1 3232: TEMCo 3/8" Marine Heat Shrink Tube 3:1 Adhesive Glue L… | R650a pred1 2103: TEMCo 1/2" Marine Heat Shrink Tube 3:1 Adhesive Glue L…
- n=1 | target 2058 HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1… | original pred1 2477: BAMtack! 1.75mm Black PLA 3D Printer Filament - 1kg (2… | R650a pred1 3456: HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy…
- n=1 | target 1982 Plymor 6" x 9", 4 Mil (Pack of 100) Heavy Duty Plastic Reclosable Zip… | original pred1 257: Gorilla 5000408  Original Gorilla Glue, Waterproof Pol… | R650a pred1 2813: Plymor 5" x 8", 4 Mil (Pack of 100) Heavy Duty Plastic…
- n=1 | target 1850 HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1… | original pred1 1847: HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy… | R650a pred1 50: HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy…
