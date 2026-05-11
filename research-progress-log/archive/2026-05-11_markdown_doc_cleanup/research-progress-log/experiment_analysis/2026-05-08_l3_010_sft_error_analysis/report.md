# L3=0.010 SFT Error Analysis（监督微调错误分析）

## Summary（摘要）

This compares the same 4533 test rows for the current mainline L3=0.020（当前主线第三层权重 0.020） and L3=0.010（第三层权重 0.010）.

| variant               |     hr@1 |     hr@3 |     hr@5 |    hr@10 |   ndcg@10 |   l1_prefix_hit@1 |   l12_prefix_hit@10 |   top1_history_copy |   target_has_history_l12 |
|:----------------------|---------:|---------:|---------:|---------:|----------:|------------------:|--------------------:|--------------------:|-------------------------:|
| Current main L3=0.020 | 0.070593 | 0.100816 | 0.117362 | 0.146923 |  0.104383 |          0.230973 |            0.203618 |            0.389367 |                 0.165012 |
| L3=0.010              | 0.062210 | 0.092213 | 0.110523 | 0.142731 |  0.097456 |          0.239576 |            0.211780 |            0.358262 |                 0.164130 |

## Hit Transition（命中迁移）

| main_hit10->l3_hit10   |   count |   pct |
|:-----------------------|--------:|------:|
| 0->0                   |    3724 | 82.15 |
| 1->1                   |     504 | 11.12 |
| 1->0                   |     162 |  3.57 |
| 0->1                   |     143 |  3.15 |

## Conditional Exact Accuracy（条件精确命中）

This table checks whether prefix routing（前缀路由） actually converts into exact SID（精确语义标识） ranking.

| variant               | condition              |   count |   exact_hr@10 |   ndcg@10 |   median_exact_rank |   top1_history_copy |
|:----------------------|:-----------------------|--------:|--------------:|----------:|--------------------:|--------------------:|
| Current main L3=0.020 | all                    |    4533 |      0.146923 |  0.104383 |          999.000000 |            0.389367 |
| Current main L3=0.020 | L1 prefix @1           |    1047 |      0.527221 |  0.403725 |            8.000000 |            0.376313 |
| Current main L3=0.020 | L12 prefix @10         |     923 |      0.721560 |  0.512644 |            4.000000 |            0.412784 |
| Current main L3=0.020 | target has history L12 |     748 |      0.645722 |  0.491720 |            4.000000 |            0.445187 |
| Current main L3=0.020 | target no history L12  |    3785 |      0.048349 |  0.027837 |          999.000000 |            0.378336 |
| L3=0.010              | all                    |    4533 |      0.142731 |  0.097456 |          999.000000 |            0.358262 |
| L3=0.010              | L1 prefix @1           |    1086 |      0.468692 |  0.350674 |           13.000000 |            0.372928 |
| L3=0.010              | L12 prefix @10         |     960 |      0.673958 |  0.460176 |            5.000000 |            0.417708 |
| L3=0.010              | target has history L12 |     744 |      0.604839 |  0.455398 |            5.000000 |            0.459677 |
| L3=0.010              | target no history L12  |    3789 |      0.051993 |  0.027172 |          999.000000 |            0.338348 |

## Prefix-Correct But Exact-Wrong（前缀对但精确错）

| variant               |   l12_prefix_but_exact_miss_count |   fraction_of_test |   mean_l12_prefix_rank |   top1_history_copy |
|:----------------------|----------------------------------:|-------------------:|-----------------------:|--------------------:|
| Current main L3=0.020 |                               257 |           0.056695 |               2.922179 |            0.315175 |
| L3=0.010              |                               313 |           0.069049 |               3.105431 |            0.242812 |

## Item-Level Lost Targets（物品级退化目标）

|   item_id |   test_count |   train_item_events |   main_hit10 |   l3_hit10 |   hit10_delta_l3_minus_main |   main_mean_rank |   l3_mean_rank |   main_same_l12_hist_mean |   l3_same_l12_hist_mean |   main_l12_bucket_size |   l3_l12_bucket_size | item_title                                                                                                    |
|----------:|-------------:|--------------------:|-------------:|-----------:|----------------------------:|-----------------:|---------------:|--------------------------:|------------------------:|-----------------------:|---------------------:|:--------------------------------------------------------------------------------------------------------------|
|       357 |            1 |                  34 |            1 |          0 |                          -1 |            1.000 |         23.000 |                     1.000 |                   0.000 |                      4 |                    1 | PORTER-CABLE PNS18050 1/2-Inch, 18 Gauge Narrow Crown (1/4-Inch) Staple (5000-Pack),PORTER-CABLE,PNS18050" /> |
|       421 |            1 |                  47 |            1 |          0 |                          -1 |            1.000 |         21.000 |                     1.000 |                   1.000 |                      2 |                    2 | Anytime Tools 7/32" Diamond Chainsaw Sharpener Burr 1/8" Shank, 4 Pack                                        |
|       814 |            1 |                  70 |            1 |          0 |                          -1 |            1.000 |         33.000 |                     0.000 |                   0.000 |                      1 |                    1 | DEWALT DW4902 1-Inch by 1/4-Inch High Performance Carbon Knot Wire End Brush, 0.020-Inch Wire                 |
|      1886 |            1 |                  66 |            1 |          0 |                          -1 |            1.000 |         11.000 |                     1.000 |                   0.000 |                      3 |                    1 | Arrow Fastener 256 Genuine T25 3/8-Inch Staples, 1,000-Pack                                                   |
|      2302 |            1 |                  46 |            1 |          0 |                          -1 |            1.000 |        999.000 |                     0.000 |                   0.000 |                      1 |                    1 | HBD Thermoid NBR/PVC SAE30R6 Fuel Line Hose, 5/16" x 25' Length, 0.3125" ID, Black                            |
|      3649 |            1 |                  22 |            1 |          0 |                          -1 |            1.000 |        999.000 |                     0.000 |                   0.000 |                      5 |                    1 | [3D CAM] 5 PCS DRV8825 StepStick Stepper Motor Drivers for 3D Printer Electronics, CNC Machine or Robotics    |
|       226 |            4 |                 145 |            4 |          1 |                          -3 |            1.250 |        507.750 |                     1.000 |                   1.000 |                      2 |                    4 | Anderson Metals Brass Hose Fitting, Connector, 3/4" Barb x 3/4" Male Pipe                                     |
|       273 |            3 |                  57 |            3 |          0 |                          -3 |            3.000 |        671.333 |                     1.000 |                   1.000 |                      4 |                    3 | Gorilla Original Gorilla Glue, Waterproof Polyurethane Glue, 2 ounce Bottle, Brown, (Pack of 4)               |
|      2104 |            2 |                  40 |            1 |          0 |                          -1 |          500.000 |        999.000 |                     0.000 |                   0.000 |                      2 |                    4 | Anderson Metals Brass Garden Hose Fitting, Connector, 3/4" Barb x 3/4" Male Hose                              |
|       265 |            1 |                 114 |            1 |          0 |                          -1 |            3.000 |        999.000 |                     0.000 |                   0.000 |                      4 |                    3 | ZJchao 10 pieces Tungsten Carbide Rotary Burr SET 1/8" shank [Misc.]                                          |
|      2446 |            1 |                 124 |            1 |          0 |                          -1 |            3.000 |        999.000 |                     0.000 |                   0.000 |                      2 |                    2 | 100x 1N4007 Diode 1A 1000V Rectifier Diodes Arduino Motor Snubber Flyback                                     |
|       526 |            1 |                 206 |            1 |          0 |                          -1 |            4.000 |        999.000 |                     0.000 |                   0.000 |                      1 |                    1 | Rubbermaid Commercial Products FG263100GRAY Rubbermaid Commercial Round Brute Container Lid, Gray, 32G        |

## Item-Level Gained Targets（物品级改善目标）

|   item_id |   test_count |   train_item_events |   main_hit10 |   l3_hit10 |   hit10_delta_l3_minus_main |   main_mean_rank |   l3_mean_rank |   main_same_l12_hist_mean |   l3_same_l12_hist_mean |   main_l12_bucket_size |   l3_l12_bucket_size | item_title                                                                                                         |
|----------:|-------------:|--------------------:|-------------:|-----------:|----------------------------:|-----------------:|---------------:|--------------------------:|------------------------:|-----------------------:|---------------------:|:-------------------------------------------------------------------------------------------------------------------|
|      2734 |           21 |                  48 |            0 |          9 |                           9 |          534.714 |        526.143 |                     0.762 |                   0.762 |                     21 |                   18 | Inland 1.75mm White PLA 3D Printer Filament - 1kg Spool (2.2 lbs)                                                  |
|       130 |           15 |                 397 |            2 |          9 |                           7 |          540.800 |        143.733 |                     0.267 |                   0.267 |                      5 |                    7 | Gorilla Tape, Black Duct Tape, 1.88" x 35 yd, Black, (Pack of 1)                                                   |
|      3016 |           17 |                 199 |            2 |          8 |                           6 |          252.529 |        188.588 |                     0.000 |                   0.588 |                      1 |                   27 | HATCHBOX 3D Printer Filament, Dimensional Accuracy +/- 0.03mm, 1.75 mm, 1 kg Spool, Wood                           |
|        59 |           12 |                 233 |            0 |          6 |                           6 |          999.000 |        343.833 |                     0.083 |                   0.083 |                      2 |                    3 | Loctite Ultra Gel Control Super Glue 4-Gram (1363589)                                                              |
|      3475 |           11 |                  30 |            0 |          5 |                           5 |          468.091 |        369.545 |                     1.727 |                   1.455 |                     23 |                   24 | 3D Solutech Real Orange 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) |
|      3442 |            9 |                  21 |            0 |          4 |                           4 |          786.222 |        447.556 |                     0.000 |                   2.333 |                      4 |                   24 | 3D Solutech Real Purple 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) |
|      1206 |            8 |                  51 |            0 |          4 |                           4 |          999.000 |        258.250 |                     0.000 |                   0.000 |                      7 |                    9 | eSUN 1.75mm Black PLA PRO (PLA+) 3D Printer Filament 1KG Spool (2.2lbs), Black                                     |
|      1847 |           25 |                 483 |           16 |         19 |                           3 |          205.600 |        126.240 |                     0.120 |                   0.120 |                      3 |                    4 | HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, White                     |
|        40 |           17 |                 266 |            0 |          3 |                           3 |          770.588 |        538.824 |                     0.000 |                   0.000 |                      1 |                    1 | Forney 20859 Cutting Fluid, Industrial Pro Tap Magic, 1-Gallon                                                     |
|      2703 |           14 |                  89 |            2 |          5 |                           3 |          719.357 |        367.929 |                     0.214 |                   0.214 |                      7 |                    7 | eSUN 3D 1.75mm PETG Black Filament 1kg (2.2lb), PETG 3D Printer Filament, 1.75mm Solid Opaque Black                |
|      2807 |            8 |                  27 |            4 |          7 |                           3 |          131.750 |        127.625 |                     0.625 |                   0.625 |                      3 |                    3 | VIVOSUN 6 Inch 440 CFM Inline Duct Ventilation Fan with Variable Speed Controller                                  |
|      2544 |            8 |                 281 |            0 |          3 |                           3 |          273.250 |        265.625 |                     0.000 |                   0.000 |                      1 |                    3 | SainSmart Clear Flexible TPU 3D Printing Filament, 1.75 mm, 0.8 kg, Dimensional Accuracy +/- 0.05 mm               |

## Weakest L1 Routes（最弱第一层路由）

### Current main L3=0.020（当前主线）
| main_l3_020_l1   |   test_count |   hit10 |   ndcg10 |   same_l12_hist_mean |   l1_bucket_size |   l1_train_events |
|:-----------------|-------------:|--------:|---------:|---------------------:|-----------------:|------------------:|
| <a_12>           |           23 |  0.0000 |   0.0000 |               0.0000 |               35 |              1210 |
| <a_172>          |           21 |  0.0000 |   0.0000 |               0.0476 |               47 |              2507 |
| <a_235>          |           50 |  0.0000 |   0.0000 |               0.0000 |               43 |              2494 |
| <a_247>          |           47 |  0.0000 |   0.0000 |               0.0638 |               54 |              1863 |
| <a_42>           |           48 |  0.0000 |   0.0000 |               0.0000 |               65 |              2826 |
| <a_48>           |           91 |  0.0000 |   0.0000 |               0.0220 |               44 |              1133 |
| <a_36>           |           36 |  0.0000 |   0.0000 |               0.0000 |               46 |              1786 |
| <a_34>           |           59 |  0.0000 |   0.0000 |               0.0339 |               74 |              2902 |
| <a_83>           |           66 |  0.0000 |   0.0000 |               0.0000 |               81 |              2862 |
| <a_145>          |           84 |  0.0119 |   0.0060 |               0.0000 |              109 |              3826 |
| <a_6>            |           67 |  0.0299 |   0.0093 |               0.0000 |               55 |              2965 |
| <a_89>           |           98 |  0.0204 |   0.0129 |               0.0408 |               75 |              2986 |

### L3=0.010（第三层权重 0.010）
| l3_010_l1   |   test_count |   hit10 |   ndcg10 |   same_l12_hist_mean |   l1_bucket_size |   l1_train_events |
|:------------|-------------:|--------:|---------:|---------------------:|-----------------:|------------------:|
| <a_0>       |           56 |  0.0000 |   0.0000 |               0.0714 |               66 |              2275 |
| <a_125>     |           41 |  0.0000 |   0.0000 |               0.0000 |               62 |              2319 |
| <a_167>     |           44 |  0.0000 |   0.0000 |               0.0000 |               50 |              2672 |
| <a_149>     |           22 |  0.0000 |   0.0000 |               0.0000 |               42 |              1634 |
| <a_229>     |           30 |  0.0000 |   0.0000 |               0.0000 |               37 |              1569 |
| <a_200>     |           24 |  0.0000 |   0.0000 |               0.0000 |               43 |              2360 |
| <a_29>      |           79 |  0.0000 |   0.0000 |               0.0000 |               42 |              1125 |
| <a_36>      |           37 |  0.0000 |   0.0000 |               0.0000 |               62 |              2451 |
| <a_62>      |           22 |  0.0000 |   0.0000 |               0.0000 |               24 |               864 |
| <a_77>      |           54 |  0.0185 |   0.0080 |               0.0000 |               62 |              2192 |
| <a_86>      |           47 |  0.0213 |   0.0106 |               0.0426 |               51 |              2011 |
| <a_16>      |           53 |  0.0189 |   0.0119 |               0.1887 |               59 |              1617 |

## Main Hit / L3 Miss Examples（主线命中但 L3=0.010 失败样例）

|   row_id |   item_id | item_title                                                                                 |   train_item_events |   history_len | main_l3_020_sid       | l3_010_sid          |   main_l3_020_rank |   l3_010_rank |   main_l3_020_same_l12_hist_count |   l3_010_same_l12_hist_count | main_l3_020_top1      | main_l3_020_top1_title                                                                     | l3_010_top1          | l3_010_top1_title                                                                          |
|---------:|----------:|:-------------------------------------------------------------------------------------------|--------------------:|--------------:|:----------------------|:--------------------|-------------------:|--------------:|----------------------------------:|-----------------------------:|:----------------------|:-------------------------------------------------------------------------------------------|:---------------------|:-------------------------------------------------------------------------------------------|
|      170 |       181 | AcuRite 00613 Humidity Monitor with Indoor Thermometer, Digital Hygrometer and Humidity Ga |                2351 |             3 | <a_31><b_221><c_0>    | <a_59><b_52><c_0>   |                  1 |           999 |                                 0 |                            0 | <a_31><b_221><c_0>    | AcuRite 00613 Humidity Monitor with Indoor Thermometer, Digital Hygrometer and Humidity Ga | <a_140><b_78><c_3>   | Philips Sonicare HX9381/05 Diamond Clean Rechargeable Electric Toothbrush                  |
|     1199 |      3649 | [3D CAM] 5 PCS DRV8825 StepStick Stepper Motor Drivers for 3D Printer Electronics, CNC Mac |                  22 |             6 | <a_186><b_227><c_180> | <a_15><b_74><c_32>  |                  1 |           999 |                                 0 |                            0 | <a_186><b_227><c_180> | [3D CAM] 5 PCS DRV8825 StepStick Stepper Motor Drivers for 3D Printer Electronics, CNC Mac | <a_29><b_239><c_50>  | Kamo 5PCS 3D Printer 0.4mm Extruder Brass Nozzle Print Head for MK8 1.75mm ABS PLA Printer |
|     1759 |       416 | PRO 1 Fuel Line Hose 1/4 Inch Inside Diameter X 25 Feet Length NRB/PVCC SAE30R6            |                  36 |             8 | <a_223><b_126><c_85>  | <a_56><b_228><c_9>  |                  1 |           999 |                                 1 |                            1 | <a_223><b_126><c_85>  | PRO 1 Fuel Line Hose 1/4 Inch Inside Diameter X 25 Feet Length NRB/PVCC SAE30R6            | <a_59><b_178><c_7>   | Nubee Temperature Gun Non-contact Digital Laser Infrared IR Thermometer                    |
|     2560 |       226 | Anderson Metals Brass Hose Fitting, Connector, 3/4" Barb x 3/4" Male Pipe                  |                 145 |             6 | <a_78><b_243><c_2>    | <a_31><b_98><c_13>  |                  1 |           999 |                                 1 |                            1 | <a_78><b_243><c_2>    | Anderson Metals Brass Hose Fitting, Connector, 3/4" Barb x 3/4" Male Pipe                  | <a_31><b_80><c_193>  | Anderson Metals Brass Pipe Fitting, 90 Degree Barstock Street Elbow, 3/8" Male Pipe x 3/8" |
|     3228 |      2104 | Anderson Metals Brass Garden Hose Fitting, Connector, 3/4" Barb x 3/4" Male Hose           |                  40 |            10 | <a_78><b_32><c_158>   | <a_31><b_98><c_200> |                  1 |           999 |                                 0 |                            0 | <a_78><b_32><c_158>   | Anderson Metals Brass Garden Hose Fitting, Connector, 3/4" Barb x 3/4" Male Hose           | <a_56><b_136><c_148> | Lifegard Aquatics 3/4-Inch Double Threaded Bulkhead                                        |
|     4144 |      2302 | HBD Thermoid NBR/PVC SAE30R6 Fuel Line Hose, 5/16" x 25' Length, 0.3125" ID, Black         |                  46 |             4 | <a_225><b_126><c_244> | <a_56><b_208><c_65> |                  1 |           999 |                                 0 |                            0 | <a_225><b_126><c_244> | HBD Thermoid NBR/PVC SAE30R6 Fuel Line Hose, 5/16" x 25' Length, 0.3125" ID, Black         | <a_115><b_160><c_15> | HBD Thermoid NBR/PVC SAE30R6 Fuel Line Hose, 3/8" x 25' Length, 0.375" ID, Black           |
|     2238 |       814 | DEWALT DW4902 1-Inch by 1/4-Inch High Performance Carbon Knot Wire End Brush, 0.020-Inch W |                  70 |             6 | <a_122><b_58><c_252>  | <a_71><b_88><c_6>   |                  1 |            33 |                                 0 |                            0 | <a_122><b_58><c_252>  | DEWALT DW4902 1-Inch by 1/4-Inch High Performance Carbon Knot Wire End Brush, 0.020-Inch W | <a_254><b_75><c_225> | iGaging ABSOLUTE ORIGIN 0-6" Digital Electronic Caliper - IP54 Protection/Extreme Accuracy |
|      343 |       226 | Anderson Metals Brass Hose Fitting, Connector, 3/4" Barb x 3/4" Male Pipe                  |                 145 |             5 | <a_78><b_243><c_2>    | <a_31><b_98><c_13>  |                  1 |            24 |                                 1 |                            1 | <a_78><b_243><c_2>    | Anderson Metals Brass Hose Fitting, Connector, 3/4" Barb x 3/4" Male Pipe                  | <a_59><b_52><c_0>    | AcuRite 00613 Humidity Monitor with Indoor Thermometer, Digital Hygrometer and Humidity Ga |

## L3 Hit / Main Miss Examples（L3=0.010 命中但主线失败样例）

|   row_id |   item_id | item_title                                                                                 |   train_item_events |   history_len | main_l3_020_sid       | l3_010_sid            |   main_l3_020_rank |   l3_010_rank |   main_l3_020_same_l12_hist_count |   l3_010_same_l12_hist_count | main_l3_020_top1      | main_l3_020_top1_title                                                                     | l3_010_top1           | l3_010_top1_title                                                                          |
|---------:|----------:|:-------------------------------------------------------------------------------------------|--------------------:|--------------:|:----------------------|:----------------------|-------------------:|--------------:|----------------------------------:|-----------------------------:|:----------------------|:-------------------------------------------------------------------------------------------|:----------------------|:-------------------------------------------------------------------------------------------|
|      158 |      1911 | uxcell 1 Meter 65 Flat width 40mm Dia Ratio 2:1 Heat Shrinkable Shrinking Tube Black       |                  23 |             6 | <a_202><b_236><c_188> | <a_115><b_22><c_156>  |                999 |             1 |                                 0 |                            1 | <a_202><b_30><c_0>    | Lucksender 100 Feet /30 Meter 1/4inch / 6mm I.D Polyolefin 2:1 Heat Shrink Tubing          | <a_115><b_22><c_156>  | uxcell 1 Meter 65 Flat width 40mm Dia Ratio 2:1 Heat Shrinkable Shrinking Tube Black       |
|     2265 |      2513 | 2 Meters PTFE Teflon Bowden Tube 1.75 Filament 3D printer RepRap Rostock Kossel            |                 121 |             6 | <a_186><b_184><c_137> | <a_251><b_236><c_192> |                 44 |             1 |                                 0 |                            0 | <a_85><b_236><c_157>  | eSUN 3D Printer CLEANING Filament 1.75mm Natural 0.1kg for all 1.75mm FDM 3D Printers, 1.7 | <a_251><b_236><c_192> | 2 Meters PTFE Teflon Bowden Tube 1.75 Filament 3D printer RepRap Rostock Kossel            |
|      895 |      1809 | UHMW (Ultra High Molecular Weight Polyethylene) Sheet, Opaque White, Standard Tolerance, A |                  44 |             3 | <a_225><b_238><c_150> | <a_253><b_211><c_172> |                 35 |             1 |                                 1 |                            0 | <a_225><b_238><c_136> | HDPE (High Density Polyethylene) Sheet, Opaque Off-White, Standard Tolerance, ASTM D4976-2 | <a_253><b_211><c_172> | UHMW (Ultra High Molecular Weight Polyethylene) Sheet, Opaque White, Standard Tolerance, A |
|     1358 |        40 | Forney 20859 Cutting Fluid, Industrial Pro Tap Magic, 1-Gallon                             |                 266 |             5 | <a_6><b_89><c_63>     | <a_173><b_59><c_42>   |                 35 |             1 |                                 0 |                            0 | <a_247><b_140><c_198> | Osborn International 75116SP Steel File Card, 3-3/4" Brush Area Length                     | <a_173><b_59><c_42>   | Forney 20859 Cutting Fluid, Industrial Pro Tap Magic, 1-Gallon                             |
|     2750 |       734 | microtivity IL188 5mm Assorted Clear LED w/Resistors (8 Colors, Pack of 80)                |                 446 |             9 | <a_104><b_9><c_3>     | <a_175><b_255><c_1>   |                 29 |             1 |                                 0 |                            0 | <a_145><b_175><c_74>  | 100x 1N4007 Diode 1A 1000V Rectifier Diodes Arduino Motor Snubber Flyback                  | <a_175><b_255><c_1>   | microtivity IL188 5mm Assorted Clear LED w/Resistors (8 Colors, Pack of 80)                |
|      106 |      1953 | Funnel, Regular stem, glass, 50 OD x 50mm stem                                             |                 244 |             4 | <a_172><b_19><c_73>   | <a_223><b_50><c_3>    |                 19 |             1 |                                 0 |                            0 | <a_36><b_49><c_134>   | SEOH Petri Dish 100 x 15Mm Sterile, Vented, 25/Pk                                          | <a_223><b_50><c_3>    | Funnel, Regular stem, glass, 50 OD x 50mm stem                                             |
|     2978 |      1850 | HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, R |                 501 |             4 | <a_222><b_157><c_250> | <a_184><b_67><c_38>   |                 19 |             1 |                                 1 |                            1 | <a_165><b_229><c_0>   | AmScope 25pc Assorted Specimen Collection of Prepared Microscope Slides Glass Slide with S | <a_184><b_67><c_38>   | HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, R |
|     2050 |      2063 | 3Doodler Create 3D Pen with 50 Plastic Strands, No Mess, Non-Toxic -                       |                  32 |             2 | <a_249><b_250><c_230> | <a_246><b_40><c_12>   |                 14 |             1 |                                 2 |                            2 | <a_222><b_109><c_177> | HATCHBOX ABS 3D Printer Filament, Dimensional Accuracy +/- 0.03 mm, 1 kg Spool, 1.75 mm, B | <a_246><b_40><c_12>   | 3Doodler Create 3D Pen with 50 Plastic Strands, No Mess, Non-Toxic -                       |

## Output Files（输出文件）

- per_sample（逐样本）: `/home/leejt/OneRec/research-progress-log/experiment_analysis/2026-05-08_l3_010_sft_error_analysis/per_sample.csv`
- per_item（逐物品）: `/home/leejt/OneRec/research-progress-log/experiment_analysis/2026-05-08_l3_010_sft_error_analysis/per_item.csv`
- report（报告）: `/home/leejt/OneRec/research-progress-log/experiment_analysis/2026-05-08_l3_010_sft_error_analysis/report.md`
