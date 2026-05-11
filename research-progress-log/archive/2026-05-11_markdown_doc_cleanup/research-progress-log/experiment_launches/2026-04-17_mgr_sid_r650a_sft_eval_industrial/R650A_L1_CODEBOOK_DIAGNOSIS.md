# R650a L1 Codebook Diagnosis

## Scope
- This report inspects active L1 code assignments, not raw continuous codebook vectors.
- R650a is compared against the original MiniOneRec semantic tokenizer on Industrial.
- Downstream comparison uses strongest original SFT and R650a SFT results already evaluated.

## Active L1 Overview
| tokenizer | active L1 | inactive L1 | mean items/L1 | median items/L1 | p90 items/L1 | max items/L1 |
|---|---:|---:|---:|---:|---:|---:|
| original | 48 | 208 | 76.79 | 67.5 | 119.0 | 247 |
| R650a | 199 | 57 | 18.52 | 17.0 | 26.2 | 74 |

## Routing Quality
| tokenizer | hit@10 | pred1 same L1 | top10 has same L1 | top10 has same L2 | top50 has same L1 | top50 has same L2 |
|---|---:|---:|---:|---:|---:|---:|
| original | 0.1509 | 0.2672 | 0.4306 | 0.2109 | 0.6340 | 0.3148 |
| R650a | 0.1324 | 0.1518 | 0.2804 | 0.1796 | 0.4694 | 0.2795 |

## R650a L1 Bucket Size Bins
| bucket_bin | l1_code_count | catalog_item_count | eval_count | orig_hit10_count | r650_hit10_count | delta_hit10_count | orig_hit10_rate | r650_hit10_rate | top10_has_same_l1_rate_weighted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 2-3 | 3 | 7 | 4 | 0 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 6-10 | 20 | 179 | 155 | 9 | 3 | -6 | 0.0581 | 0.0194 | 0.0387 |
| 11-20 | 107 | 1649 | 1782 | 265 | 250 | -15 | 0.1487 | 0.1403 | 0.2054 |
| 21-50 | 66 | 1716 | 2047 | 250 | 224 | -26 | 0.1221 | 0.1094 | 0.2487 |
| >50 | 2 | 134 | 544 | 160 | 123 | -37 | 0.2941 | 0.2261 | 0.7169 |

## Family Spread Across L1
| family | catalog_items | orig_l1_count | r650_l1_count | orig_items_per_l1 | r650_items_per_l1 | eval_count | orig_hit10_count | r650_hit10_count | delta_hit10_count | orig_only_hit10_count | r650_only_hit10_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3d_filament | 386 | 21 | 39 | 18.38 | 9.90 | 1101 | 234.00 | 189.00 | -45.00 | 92 | 47 |
| connector_fitting | 274 | 19 | 49 | 14.42 | 5.59 | 273 | 33.00 | 18.00 | -15.00 | 20 | 5 |
| gauge_meter | 338 | 29 | 85 | 11.66 | 3.98 | 513 | 218.00 | 204.00 | -14.00 | 29 | 15 |
| other | 2065 | 45 | 185 | 45.89 | 11.16 | 1873 | 141.00 | 133.00 | -8.00 | 36 | 28 |
| tape | 217 | 22 | 47 | 9.86 | 4.62 | 301 | 30.00 | 25.00 | -5.00 | 11 | 6 |
| ventilation_fan | 58 | 24 | 35 | 2.42 | 1.66 | 64 | 9.00 | 7.00 | -2.00 | 2 | 0 |
| test_strip | 9 | 3 | 4 | 3.00 | 2.25 | 12 | 2.00 | 1.00 | -1.00 | 1 | 0 |
| fastener | 187 | 27 | 44 | 6.93 | 4.25 | 142 | 4.00 | 3.00 | -1.00 | 1 | 0 |
| metadata_placeholder | 27 | 3 | 4 | 9.00 | 6.75 | 30 | 0.00 | 1.00 | 1.00 | 0 | 1 |
| adhesive_epoxy | 125 | 14 | 32 | 8.93 | 3.91 | 224 | 13.00 | 19.00 | 6.00 | 5 | 11 |

## Largest R650a L1 Buckets
| l1_code | catalog_item_count | dominant_family | dominant_family_rate | eval_count | orig_hit10_count | r650_hit10_count | delta_hit10_count | orig_only_hit10_count | r650_only_hit10_count | pred1_l1_count | pred1_over_target_ratio | top10_has_same_l1_rate | top_items |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <a_107> | 74 | 3d_filament | 1.000 | 374 | 115 | 76 | -39 | 60 | 21 | 370 | 0.989 | 0.733 | 3112:eval32/train23:3D Solutech Real Black 3D Printer PLA Filament 1.75MM Filament,… // 2909:eval31/train41:3D Solutech Silver Me… |
| <a_18> | 60 | 3d_filament | 1.000 | 170 | 45 | 47 | 2 | 5 | 7 | 317 | 1.865 | 0.682 | 562:eval29/train5:HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03… // 2058:eval25/train10:HATCHBOX PLA 3D Printer… |
| <a_215> | 50 | 3d_filament | 0.980 | 87 | 18 | 15 | -3 | 5 | 2 | 54 | 0.621 | 0.483 | 1847:eval25/train97:HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03… // 164:eval12/train1:HATCHBOX PLA 3D Printer… |
| <a_243> | 47 | 3d_filament | 0.766 | 223 | 46 | 41 | -5 | 17 | 12 | 267 | 1.197 | 0.507 | 2912:eval31/train15:Inland 1.75mm Black PLA 3D Printer Filament - 1kg Spool (2.2 lb… // 2734:eval21/train12:Inland 1.75mm White P… |
| <a_89> | 39 | tape | 0.615 | 84 | 21 | 15 | -6 | 9 | 3 | 101 | 1.202 | 0.452 | 130:eval15/train76:Gorilla Tape, Black Duct Tape, 1.88" x 35 yd, Black, (Pack of 1) // 175:eval8/train75:Gorilla Crystal Clear Du… |
| <a_63> | 39 | other | 0.436 | 26 | 0 | 0 | 0 | 0 | 0 | 27 | 1.038 | 0.000 | 1826:eval3/train24:Stainless Steel 316L Seamless Round Tubing, 1/8" OD, 0.027" ID,… // 87:eval3/train12:Wixey WR510 Digital Plane… |
| <a_199> | 38 | 3d_filament | 0.895 | 53 | 2 | 1 | -1 | 2 | 1 | 37 | 0.698 | 0.245 | 2145:eval6/train21:ROBO 3D R1 Plus 10x9x8-Inch ABS/PLA 3D Printer, White (A1-0002-… // 3171:eval6/train9:Signstek 10 PCS Reprap 1… |
| <a_51> | 38 | other | 0.737 | 48 | 0 | 0 | 0 | 0 | 0 | 11 | 0.229 | 0.021 | 469:eval8/train4:CO-Z 5pcs Hss Cobalt Multiple Hole 50 Sizes Step Drill Bit Set … // 1012:eval7/train7:4.5" x 7/8" Premium High D… |
| <a_21> | 36 | other | 0.583 | 40 | 7 | 7 | 0 | 0 | 0 | 60 | 1.500 | 0.300 | 2807:eval8/train12:VIVOSUN 6 Inch 440 CFM Inline Duct Ventilation Fan with Variabl… // 1744:eval5/train6:270 PCS Heat Shrink Wire… |
| <a_173> | 35 | gauge_meter | 0.714 | 46 | 23 | 22 | -1 | 1 | 0 | 39 | 0.848 | 0.500 | 2047:eval6/train8:Ginsco 160pcs Nylon Fully Insulated Male / Female Spade Wire Cr… // 1155:eval4/train75:6061 Aluminum Round Rod,… |
| <a_17> | 32 | connector_fitting | 0.938 | 29 | 3 | 0 | -3 | 3 | 0 | 31 | 1.069 | 0.276 | 157:eval8/train28:Yueton 100pcs Female Fully Insulated Wire Crimp Terminal Nylon … // 1481:eval2/train40:Install Bay CCL1614 Crim… |
| <a_246> | 32 | other | 0.906 | 28 | 1 | 0 | -1 | 1 | 0 | 21 | 0.750 | 0.179 | 570:eval15/train52:Bissell 9595A CleanView Bagless Vacuum with OnePass // 26:eval2/train17:Shark Navigator Lift-Away Professional… |

## Most Negative R650a L1 Buckets
| l1_code | catalog_item_count | dominant_family | dominant_family_rate | eval_count | orig_hit10_count | r650_hit10_count | delta_hit10_count | orig_only_hit10_count | r650_only_hit10_count | pred1_l1_count | pred1_over_target_ratio | top10_has_same_l1_rate | top_items |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <a_107> | 74 | 3d_filament | 1.000 | 374 | 115 | 76 | -39 | 60 | 21 | 370 | 0.989 | 0.733 | 3112:eval32/train23:3D Solutech Real Black 3D Printer PLA Filament 1.75MM Filament,… // 2909:eval31/train41:3D Solutech Silver Me… |
| <a_79> | 17 | other | 0.706 | 219 | 178 | 162 | -16 | 20 | 4 | 350 | 1.598 | 0.758 | 181:eval132/train557:AcuRite 00613 Humidity Monitor with Indoor Thermometer, Digital… // 182:eval68/train491:AcuRite 00613 Humidi… |
| <a_89> | 39 | tape | 0.615 | 84 | 21 | 15 | -6 | 9 | 3 | 101 | 1.202 | 0.452 | 130:eval15/train76:Gorilla Tape, Black Duct Tape, 1.88" x 35 yd, Black, (Pack of 1) // 175:eval8/train75:Gorilla Crystal Clear Du… |
| <a_243> | 47 | 3d_filament | 0.766 | 223 | 46 | 41 | -5 | 17 | 12 | 267 | 1.197 | 0.507 | 2912:eval31/train15:Inland 1.75mm Black PLA 3D Printer Filament - 1kg Spool (2.2 lb… // 2734:eval21/train12:Inland 1.75mm White P… |
| <a_215> | 50 | 3d_filament | 0.980 | 87 | 18 | 15 | -3 | 5 | 2 | 54 | 0.621 | 0.483 | 1847:eval25/train97:HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03… // 164:eval12/train1:HATCHBOX PLA 3D Printer… |
| <a_128> | 22 | connector_fitting | 1.000 | 44 | 10 | 7 | -3 | 5 | 2 | 17 | 0.386 | 0.386 | 226:eval4/train28:Anderson Metals Brass Hose Fitting, Connector, 3/4" Barb x 3/4"… // 225:eval4/train19:Anderson Metals Brass Hos… |
| <a_17> | 32 | connector_fitting | 0.938 | 29 | 3 | 0 | -3 | 3 | 0 | 31 | 1.069 | 0.276 | 157:eval8/train28:Yueton 100pcs Female Fully Insulated Wire Crimp Terminal Nylon … // 1481:eval2/train40:Install Bay CCL1614 Crim… |
| <a_191> | 29 | connector_fitting | 0.828 | 25 | 3 | 1 | -2 | 3 | 1 | 21 | 0.840 | 0.120 | 201:eval8/train64:ColorConnex Coupler & Plug Kit (7 Piece), Industrial Type D, 1/… // 2496:eval3/train0:Apera Instruments EC20 Va… |
| <a_145> | 17 | 3d_filament | 0.471 | 22 | 5 | 3 | -2 | 3 | 1 | 50 | 2.273 | 0.409 | 563:eval6/train20:Foreasy 3D Printer Tool 3D Print Removal Tool Enhanced Version … // 2063:eval5/train21:3Doodler Create 3D Pen w… |
| <a_56> | 23 | other | 0.913 | 20 | 3 | 1 | -2 | 3 | 1 | 30 | 1.500 | 0.150 | 1824:eval6/train39:DTOL 10 X Mini Laser Dot Diode Module Head WL Red 650nm 6mm 5V … // 2036:eval4/train23:3mm and 5mm LED Lights … |
| <a_122> | 22 | other | 0.773 | 32 | 3 | 1 | -2 | 2 | 0 | 13 | 0.406 | 0.125 | 410:eval4/train5:Century Drill and Tool 95107 Coarse Plug Hand Tap, 3/8-16 // 2870:eval4/train2:Hanson 1903ZR Tap 1/4"-18Npt Tape… |
| <a_156> | 18 | other | 0.444 | 29 | 3 | 1 | -2 | 2 | 0 | 5 | 0.172 | 0.172 | 416:eval7/train9:PRO 1 Fuel Line Hose 1/4 Inch Inside Diameter X 25 Feet Length … // 798:eval5/train2:PRO 1 Fuel Line Hose 5/16 I… |
| <a_84> | 23 | other | 0.435 | 25 | 4 | 2 | -2 | 2 | 0 | 3 | 0.120 | 0.080 | 2835:eval4/train2:Ginsco 108pcs Insulated Heat Shrink Waterproof Butt Connectors … // 3210:eval3/train6:Ancor Heat Shrink Ring Te… |
| <a_130> | 12 | gauge_meter | 0.583 | 12 | 2 | 0 | -2 | 2 | 0 | 9 | 0.750 | 0.250 | 331:eval4/train31:Stanley TRA700BN Heavy-Duty Staple & Brad Assortment, 2500-Pack // 20:eval2/train6:PORTER-CABLE (PBN18075-1) 18… |
| <a_3> | 9 | other | 0.778 | 5 | 2 | 0 | -2 | 2 | 0 | 2 | 0.400 | 0.000 | 2869:eval3/train6:13cm Dia. (5.1") Filter Funnel, Buchner Style - 2 Parts, Polypr… // 1407:eval1/train11:PowerSmith PAVC101 10 Am… |

## Most Positive R650a L1 Buckets
| l1_code | catalog_item_count | dominant_family | dominant_family_rate | eval_count | orig_hit10_count | r650_hit10_count | delta_hit10_count | orig_only_hit10_count | r650_only_hit10_count | pred1_l1_count | pred1_over_target_ratio | top10_has_same_l1_rate | top_items |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <a_195> | 25 | adhesive_epoxy | 0.480 | 60 | 1 | 5 | 4 | 0 | 4 | 48 | 0.800 | 0.167 | 57:eval16/train52:Gorilla 2 Part Epoxy, 5 Minute Set, .85 ounce Syringe, Clear // 1733:eval6/train5:Bob Smith Industries BSI-151H… |
| <a_193> | 12 | other | 0.917 | 27 | 3 | 6 | 3 | 3 | 6 | 66 | 2.444 | 0.296 | 40:eval17/train64:Forney 20859 Cutting Fluid, Industrial Pro Tap Magic, 1-Gallon // 105:eval3/train3:Mist Coolant Lubrication Spr… |
| <a_47> | 21 | adhesive_epoxy | 0.381 | 25 | 6 | 9 | 3 | 0 | 3 | 34 | 1.360 | 0.480 | 3312:eval4/train2:TEMCo 3/4" Marine Heat Shrink Tube 3:1 Adhesive Glue Lined 4 ft… // 133:eval3/train10:3M High Temperature Flue … |
| <a_18> | 60 | 3d_filament | 1.000 | 170 | 45 | 47 | 2 | 5 | 7 | 317 | 1.865 | 0.682 | 562:eval29/train5:HATCHBOX PLA 3D Printer Filament, Dimensional Accuracy +/- 0.03… // 2058:eval25/train10:HATCHBOX PLA 3D Printer… |
| <a_125> | 29 | adhesive_epoxy | 0.793 | 61 | 5 | 7 | 2 | 2 | 4 | 96 | 1.574 | 0.328 | 59:eval12/train54:Loctite Ultra Gel Control Super Glue 4-Gram (1363589) // 202:eval7/train66:Loctite Liquid Professional Super Gl… |
| <a_93> | 20 | gauge_meter | 0.750 | 42 | 1 | 3 | 2 | 1 | 3 | 33 | 0.786 | 0.214 | 84:eval18/train133:Neiko 01409A Electronic Digital Caliper with Extra Large LCD Sc… // 266:eval11/train48:iGaging ABSOLUTE ORIGIN… |
| <a_76> | 17 | 3d_filament | 0.941 | 32 | 0 | 2 | 2 | 0 | 2 | 52 | 1.625 | 0.312 | 3440:eval6/train2:HICTOP 5 Pieces 3D Printer Endstops/ Limit Switch/ Mechanical S… // 2726:eval5/train10:ANYCUBIC 3D Printer Heat… |
| <a_170> | 17 | gauge_meter | 0.588 | 17 | 1 | 2 | 1 | 1 | 2 | 22 | 1.294 | 0.118 | 970:eval5/train28:Wixey WR25 Mini Digital Height Gauge // 1499:eval2/train10:Smart Weigh CW-500G Carbon Steel 500g OIML Class M1:… |
| <a_38> | 28 | other | 0.750 | 32 | 7 | 8 | 1 | 0 | 1 | 59 | 1.844 | 0.312 | 94:eval6/train62:White SiliconeTubing, 3/16"ID, 1/4"OD, 1/32" Wall, 10' Length // 93:eval4/train43:White SiliconeTubing, 1/2"ID, … |
| <a_177> | 24 | metadata_placeholder | 0.958 | 28 | 0 | 1 | 1 | 0 | 1 | 25 | 0.893 | 0.143 | 54:eval7/train61:Industrial & Scientific" /> // 98:eval5/train74:Industrial & Scientific" /> // 523:eval2/train11:Industrial & Sc… |

## Over-Predicted R650a L1 Buckets
| l1_code | catalog_item_count | dominant_family | dominant_family_rate | eval_count | orig_hit10_count | r650_hit10_count | delta_hit10_count | orig_only_hit10_count | r650_only_hit10_count | pred1_l1_count | pred1_over_target_ratio | top10_has_same_l1_rate | top_items |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <a_95> | 14 | other | 0.857 | 13 | 2 | 2 | 0 | 0 | 0 | 50 | 3.846 | 0.308 | 793:eval5/train11:Yost MU360 Universal Jaw Cover, 6 // 33:eval2/train51:Bessey BV-NVJ Multi-Purpose Vise Jaws (Jaws Only) // 471:… |
| <a_166> | 20 | other | 0.800 | 8 | 0 | 0 | 0 | 0 | 0 | 28 | 3.500 | 0.000 | 2111:eval2/train2:Shop-Vac 5986100 8-Gallon 5.5 Peak HP Stainless Steel Wet Dry V… // 2168:eval1/train8:PORTER-CABLE Wet/Dry Vacu… |
| <a_54> | 21 | connector_fitting | 0.667 | 13 | 2 | 1 | -1 | 1 | 0 | 45 | 3.462 | 0.231 | 1187:eval3/train5:Parts Express Cable TV In-Line Coaxial Surge Protector // 1764:eval2/train3:SIEMENS WN2060U Non-Fused AC Discon… |
| <a_209> | 19 | other | 1.000 | 9 | 2 | 1 | -1 | 1 | 0 | 31 | 3.444 | 0.222 | 1953:eval4/train32:Funnel, Regular stem, glass, 50 OD x 50mm stem // 148:eval1/train121:Plastic Transfer Pipettes 3ml, Graduated,… |
| <a_211> | 12 | other | 0.833 | 12 | 1 | 2 | 1 | 0 | 1 | 40 | 3.333 | 0.250 | 809:eval3/train30:Loos Cableware AN100-C4 Stainless Steel Thimble for 3/32" and 1… // 2671:eval3/train5:Gardner Bender GHG-1538 5… |
| <a_233> | 16 | other | 0.500 | 13 | 0 | 0 | 0 | 0 | 0 | 40 | 3.077 | 0.154 | 2924:eval4/train14:Hilitchi 635 Pcs 40 Pin 2.54mm Pitch Single Row Pin Headers,Dup… // 2344:eval3/train21:Uxcell a12013100ux0116 … |
| <a_75> | 9 | other | 1.000 | 6 | 1 | 0 | -1 | 1 | 0 | 17 | 2.833 | 0.000 | 332:eval2/train24:DEWALT DW4523 4-1/2-Inch by 1/4-Inch by 5/8-Inch General Purpos… // 402:eval2/train2:Metabo Slicer Cut Off Whee… |
| <a_150> | 14 | other | 1.000 | 17 | 11 | 11 | 0 | 0 | 0 | 48 | 2.824 | 0.647 | 140:eval12/train26:DuPont Teflon Silicone Lubricant // 1588:eval2/train3:Super Lube 41160/UV Synthetic UV Grease (NLGI 2), 14.1 o… |
| <a_66> | 31 | tape | 1.000 | 26 | 4 | 3 | -1 | 1 | 0 | 65 | 2.500 | 0.269 | 77:eval5/train37:Scotch Heavy Duty Shipping Packaging Tape, 3" Core, 1.88" x 54.… // 467:eval3/train31:3M Utility Duct Tape 2929 … |
| <a_230> | 24 | connector_fitting | 0.583 | 20 | 2 | 1 | -1 | 2 | 1 | 50 | 2.500 | 0.150 | 803:eval7/train1:Koehler Enterprises KE20BX 10 Piece Hose Clamp Box (Size SAE 20) // 42:eval4/train82:Precision Brand M6S Micro S… |

## Under-Predicted R650a L1 Buckets
| l1_code | catalog_item_count | dominant_family | dominant_family_rate | eval_count | orig_hit10_count | r650_hit10_count | delta_hit10_count | orig_only_hit10_count | r650_only_hit10_count | pred1_l1_count | pred1_over_target_ratio | top10_has_same_l1_rate | top_items |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <a_223> | 9 | other | 0.556 | 19 | 0 | 0 | 0 | 0 | 0 | 0 | 0.000 | 0.000 | 2210:eval11/train0:HHIP 8000-0001 2 Flute High Speed Steel End Mill Set, 6 Piece, … // 3205:eval3/train3:Stens 125-508 Oil Drain … |
| <a_88> | 19 | other | 0.789 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0.000 | 0.062 | 849:eval3/train6:Neiko 53100A 4-Inch Pegboard Hooks and Organizer Assortment / 5… // 833:eval3/train6:WallPeg 43 Pc. Peg Board St… |
| <a_82> | 18 | other | 0.778 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0.000 | 0.000 | 2234:eval2/train4:Sterile Specimen Cups, Set of 3, Screw Cap, Tamper Evident, 4 o… // 231:eval1/train11:Polar Ice PI125200CT 125 … |
| <a_168> | 18 | other | 0.389 | 23 | 0 | 0 | 0 | 0 | 0 | 1 | 0.043 | 0.000 | 3137:eval4/train7:3M Dual Lock Reclosable Fastener SJ3560 250 Clear, 1 in x 6 Ft // 2689:eval4/train6:Loctite 22221 Purple 222MS … |
| <a_217> | 21 | other | 0.857 | 16 | 1 | 1 | 0 | 0 | 0 | 1 | 0.062 | 0.062 | 2933:eval5/train10:FlashForge Finder 3D Printers with Cloud, Wi-Fi, USB cable and … // 725:eval3/train15:Bissell Zing Bagged Cani… |
| <a_31> | 19 | other | 0.737 | 15 | 0 | 0 | 0 | 0 | 0 | 1 | 0.067 | 0.000 | 1324:eval5/train13:RiteBrew Rubber Stopper - Size 10 - Drilled // 2738:eval5/train4:XTC-3D High Performance 3D Print Coating, 6.4… |
| <a_134> | 17 | other | 0.882 | 24 | 0 | 0 | 0 | 0 | 0 | 2 | 0.083 | 0.083 | 1163:eval9/train3:AUTOTOOLHOME 1/8" 3/16" 1/4" 5/16" 3/8" 1/2"high Speed Steel HS… // 56:eval6/train29:RhinoGear 11909ABMI RhinoR… |
| <a_8> | 13 | fastener | 0.923 | 11 | 0 | 0 | 0 | 0 | 0 | 1 | 0.091 | 0.000 | 917:eval2/train12:Reliable Hardware Company RH-5112BO-A 1/2-Inch Wood Screw with … // 21:eval2/train5:Swordfish 32051 Brass Plate… |
| <a_244> | 17 | other | 0.882 | 11 | 0 | 0 | 0 | 0 | 0 | 1 | 0.091 | 0.000 | 1742:eval2/train4:Ajax Scientific Battery Holder with Lead Wire, 1x AA Cell (Pack… // 1899:eval2/train2:55pcs 6 Pin DPDT Self-loc… |
| <a_222> | 17 | other | 1.000 | 9 | 0 | 0 | 0 | 0 | 0 | 1 | 0.111 | 0.111 | 1419:eval2/train8:Eagle Brewing BE510 Siphon Spray Wort Aerator // 1750:eval2/train6:1 X Drilled Rubber Stopper #6.5 (Set of 3) /… |

## Output Files
- /home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-17_mgr_sid_r650a_sft_eval_industrial/R650A_L1_CODEBOOK_SUMMARY.csv
- /home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-17_mgr_sid_r650a_sft_eval_industrial/R650A_L1_CODEBOOK_ITEM_DETAIL.csv
- /home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-17_mgr_sid_r650a_sft_eval_industrial/ORIGINAL_L1_CODEBOOK_SUMMARY.csv
- /home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-17_mgr_sid_r650a_sft_eval_industrial/R650A_L1_BUCKET_BIN_SUMMARY.csv
- /home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-17_mgr_sid_r650a_sft_eval_industrial/R650A_VS_ORIGINAL_FAMILY_L1_SPREAD.csv
- /home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-17_mgr_sid_r650a_sft_eval_industrial/R650A_L1_CODEBOOK_DIAGNOSIS.json