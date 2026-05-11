# Loss Item Graph/Semantic Analysis（损失物品图与语义分析）

## Scope（范围）

This note analyzes the `top10` loss items（`top10` 损失物品） where `v2_on_p05` hits but `R630c` misses.

## Overview（概览）

- total `top10` loss examples（总 `top10` 损失样本）: `193`
- total `top10` gain examples（总 `top10` 增益样本）: `118`
- unique loss items（唯一损失物品数）: `112`
- unique gain items（唯一增益物品数）: `70`
- weak pair threshold（弱连接阈值）: `0.001611`

## Aggregate Comparison（聚合对比）

| metric | all_eval_items（全部评测物品） | top10_loss_items（`top10` 损失物品） | top10_gain_items（`top10` 增益物品） |
|---|---:|---:|---:|
| semantic_density（语义密度） | 0.9067 | 0.9208 | 0.9309 |
| semantic_collab_disagreement（语义-协同失配） | 0.9782 | 0.9615 | 0.9541 |
| graph_competition（图竞争度） | 0.9593 | 0.9717 | 0.9689 |
| offline_combined（离线合成分数） | 0.8309 | 0.8456 | 0.8498 |
| weak_pair_endpoint_count（弱连接对端点数） | 0.8012 | 1.1140 | 1.3136 |
| weak_pair_reliability_sum（弱连接可靠性和） | 0.0060 | 0.0086 | 0.0073 |
| semantic_topk_mean_sim（语义近邻平均相似度） | 0.9283 | 0.9413 | 0.9510 |
| semantic_topk_mean_mid_affinity（语义近邻中图平均亲和） | 0.0005 | 0.0004 | 0.0006 |
| semantic_topk_zero_mid_fraction（语义近邻零中图占比） | 0.9280 | 0.9161 | 0.8339 |
| semantic_topk_weak_mid_fraction（语义近邻弱中图占比） | 0.0160 | 0.0197 | 0.0415 |
| semantic_topk_graph_overlap_fraction（语义/中图邻居重叠占比） | 0.0306 | 0.0373 | 0.0949 |

## Common Traits（共同特点）

### Loss Families（损失家族）

| family | fraction | weight |
|---|---:|---:|
| 3d_filament | 0.425 | 82.0 |
| other | 0.373 | 72.0 |
| hose_fitting | 0.078 | 15.0 |
| monitor_gauge | 0.067 | 13.0 |
| tape | 0.036 | 7.0 |
| test_strip | 0.021 | 4.0 |

### Loss Brands（损失品牌）

| brand | fraction | weight |
|---|---:|---:|
| 3D Solutech | 0.187 | 36.0 |
| HATCHBOX | 0.057 | 11.0 |
| eSUN | 0.057 | 11.0 |
| Inland | 0.047 | 9.0 |
| 3D | 0.047 | 9.0 |
| AcuRite | 0.041 | 8.0 |
| Small Parts | 0.021 | 4.0 |
| Legacy Manufacturing | 0.021 | 4.0 |
| LabRat Supplies | 0.021 | 4.0 |
| First Aid Only | 0.021 | 4.0 |

### Gain Families（增益家族）

| family | fraction | weight |
|---|---:|---:|
| 3d_filament | 0.441 | 52.0 |
| other | 0.263 | 31.0 |
| monitor_gauge | 0.136 | 16.0 |
| tape | 0.093 | 11.0 |
| hose_fitting | 0.051 | 6.0 |
| staple_fastener | 0.017 | 2.0 |

## Case Studies（病例分析）

### 3475: 3D Solutech Real Orange 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG)

- brand（品牌）: `3D Solutech`
- family（家族）: `3d_filament`
- loss/gain counts（损失/增益次数）: `top10 loss = 9`, `top10 gain = 0`
- proxy scores（代理分数）: `semantic_density = 0.9770`, `semantic_collab_disagreement = 1.0000`, `graph_competition = 0.9973`
- weak-pair exposure（弱连接对暴露）: `endpoint_count = 0`, `reliability_sum = 0.0000`
- graph stats（图统计）: `mid_degree = 32`, `mid_strength = 0.0733`, `semantic_topk_mean_mid_affinity = 0.0000`, `semantic_topk_weak_mid_fraction = 0.0000`

Top semantic neighbors（顶部语义近邻）:
| neighbor | brand | semantic_sim | coarse | mid | local | weak_pair | reliability | neighbor_top10_loss | neighbor_top10_gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3469: 3D Solutech Real Pink 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3D Solutech | 0.9935 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 0 | 0 |
| 3442: 3D Solutech Real Purple 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3D Solutech | 0.9922 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 5 | 0 |
| 3631: 3D Solutech Skin 3D Printer PLA Filament 1.75MM Filament 2.2 LBS (1.0KG) | 3D Solutech | 0.9878 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 0 | 0 |
| 1888: 3D Solutech Hot Pink 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3D Solutech | 0.9877 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 0 | 0 |
| 2993: 3D Solutech Blue 3D Printer Ultra PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3D Solutech | 0.9866 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 3 | 0 |
| 3599: 3D Solutech Aqua Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3D Solutech | 0.9865 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 0 | 0 |

Top mid-graph neighbors（顶部中图近邻）:
| neighbor | brand | mid | semantic_sim | coarse | local | weak_pair | reliability |
|---|---|---:|---:|---:|---:|---:|---:|
| 2064: Argos Technologies B3125-50 Basins, 25 mL, White (Pack of 50) | Argos Technologies | 0.0031 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |
| 2155: H-B DURAC 4-Channel Electronic Timer with White Board and Certificate of Calibration (B61700-3700) | SP Scienceware | 0.0031 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |
| 1402: Heat High Temperature Resistant Adhesive Gold Tape for Electric Task 30m 12mm | MS WGO | 0.0030 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |
| 2143: 3Doodler Start Emoji & Symbol DoodleBlock Kit with 2 Plastic Packs, (3D Pen not Included) | 3Doodler | 0.0029 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |
| 2154: Rubbermaid Commercial AutoFlush Toilet System, FG401805A | Rubbermaid Commercial Products | 0.0027 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |
| 983: 3M(TM) Cubitron(TM) II Flap Disc 967A, T29 Giant (Multiple Sizes and Grit Types) |  | 0.0025 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |

### 3522: 3D Solutech Real Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - ST176BLPLA

- brand（品牌）: `3D`
- family（家族）: `3d_filament`
- loss/gain counts（损失/增益次数）: `top10 loss = 9`, `top10 gain = 0`
- proxy scores（代理分数）: `semantic_density = 0.9780`, `semantic_collab_disagreement = 0.9508`, `graph_competition = 0.9867`
- weak-pair exposure（弱连接对暴露）: `endpoint_count = 0`, `reliability_sum = 0.0000`
- graph stats（图统计）: `mid_degree = 32`, `mid_strength = 0.0737`, `semantic_topk_mean_mid_affinity = 0.0000`, `semantic_topk_weak_mid_fraction = 0.0000`

Top semantic neighbors（顶部语义近邻）:
| neighbor | brand | semantic_sim | coarse | mid | local | weak_pair | reliability | neighbor_top10_loss | neighbor_top10_gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3112: 3D Solutech Real Black 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - PLA175RBLK | 3D Solutech | 0.9937 | 0.0271 | 0.0000 | 0.0000 | 0 | 0.0000 | 0 | 12 |
| 2909: 3D Solutech Silver Metal 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - PLA175TCMS | 3D Solutech | 0.9920 | 0.0102 | 0.0000 | 0.0363 | 0 | 0.0000 | 0 | 1 |
| 2993: 3D Solutech Blue 3D Printer Ultra PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3D Solutech | 0.9898 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 3 | 0 |
| 3599: 3D Solutech Aqua Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3D Solutech | 0.9897 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 0 | 0 |
| 1552: 3D Solutech Teal Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3D Solutech | 0.9896 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 2 | 0 |
| 3493: 3D Solutech Printer Filament, Real Blue PLA, 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 1.1 LBS (0.5KG) | 3D Solutech | 0.9878 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 0 | 0 |

Top mid-graph neighbors（顶部中图近邻）:
| neighbor | brand | mid | semantic_sim | coarse | local | weak_pair | reliability |
|---|---|---:|---:|---:|---:|---:|---:|
| 3631: 3D Solutech Skin 3D Printer PLA Filament 1.75MM Filament 2.2 LBS (1.0KG) | 3D Solutech | 0.0040 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |
| 3474: 3D Solutech See Through Red 1.75mm PETG 3D Printer Filament 2.2 LBS (1.0KG) | 3D Solutech | 0.0038 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |
| 3592: 3D Solutech Printer Filament, Real Black PLA, 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 1.1 LBS (0.5KG) | 3D Solutech | 0.0037 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |
| 1249: T-fal C51402 Excite Nonstick Thermo-Spot Dishwasher Safe Oven Safe PFOA Free Fry Pan Cookware, 8-Inch, Red | T-fal | 0.0032 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |
| 3276: Thick Glass Graduated Measuring Cylinder Set 5ml 10ml 50ml 100ml Glass with Two Brushes | Ronyes Lifescience | 0.0031 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |
| 3275: Pyrex Erlenmeyer Flask Starter Pack | Corning | 0.0031 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |

### 181: AcuRite 00613 Humidity Monitor with Indoor Thermometer, Digital Hygrometer and Humidity Gauge Indicator

- brand（品牌）: `AcuRite`
- family（家族）: `monitor_gauge`
- loss/gain counts（损失/增益次数）: `top10 loss = 8`, `top10 gain = 9`
- proxy scores（代理分数）: `semantic_density = 0.8494`, `semantic_collab_disagreement = 1.0000`, `graph_competition = 0.9615`
- weak-pair exposure（弱连接对暴露）: `endpoint_count = 0`, `reliability_sum = 0.0000`
- graph stats（图统计）: `mid_degree = 32`, `mid_strength = 0.0681`, `semantic_topk_mean_mid_affinity = 0.0000`, `semantic_topk_weak_mid_fraction = 0.0000`

Top semantic neighbors（顶部语义近邻）:
| neighbor | brand | semantic_sim | coarse | mid | local | weak_pair | reliability | neighbor_top10_loss | neighbor_top10_gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 182: AcuRite 00613 Humidity Monitor with Indoor Thermometer, Digital Hygrometer and Humidity Gauge Indicator | AcuRite | 1.0000 | 0.0495 | 0.0000 | 0.1301 | 0 | 0.0000 | 0 | 0 |
| 2644: Vktech Home Appliance DHT22/AM2302 Digital Temperature And Humidity Measurement Sensor | Vktech | 0.8748 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 0 | 0 |
| 2645: SMAKN&reg; DHT22 AM2302 Digital Temperature And Humidity Measurement Sensor | SMAKN | 0.8672 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 0 | 0 |
| 1423: Uxcell a12080200ux0383 LCD Display Resettable Refrigerator Freezer Digital Thermometer | uxcell | 0.8647 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 0 | 0 |
| 1740: Chunshop WH5001 Celsius/Fahrenheit Digital Thermometer Temperature Meter Gauge C/F | Chunshop | 0.8633 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 0 | 0 |
| 2971: Taylor Precision Products Digital Panel Mount Thermometer (-40- to 300-Degrees Fahrenheit, -40- to 150-Degrees Celsius) | Taylor Precision Products | 0.8576 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 0 | 0 |

Top mid-graph neighbors（顶部中图近邻）:
| neighbor | brand | mid | semantic_sim | coarse | local | weak_pair | reliability |
|---|---|---:|---:|---:|---:|---:|---:|
| 2406: 5 PIECES SCISSORS FORCEPS HEMOSTATS NEEDLE HOLDERS DDP INSTRUMENTS | DDP | 0.0060 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |
| 1252: 3M Steri-Strip Reinforced Sterile Skin Closures, 10 Pack Variety Pack | Steri-Strip | 0.0050 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |
| 695: MABIS Stainless Steel Tweezers, Thumb Dressing Forceps, Serrated Forceps, Silver | Mabis | 0.0044 | 0.0000 | 0.0442 | 0.0323 | 0 | 0.0000 |
| 2231: 5 Pack - 10ml Sterile Syringe with Blunt Tip Needle and Storage Cap for Refilling Cartridges, E-Liquids, E-cigs, E-juice, Vape | C-U Innovations | 0.0044 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |
| 556: Waterjel 2120 Bacitracin Zinc Triple Antibiotic Ointment, 0.5gm Packet (Pack of 144) | Water Jel | 0.0035 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |
| 2447: Pac-Kit by First Aid Only 3-201 Gauze Pad, 3" Length x 3" Width (Box of 25) | First Aid Only | 0.0030 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |

### 3494: 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG)

- brand（品牌）: `3D Solutech`
- family（家族）: `3d_filament`
- loss/gain counts（损失/增益次数）: `top10 loss = 6`, `top10 gain = 0`
- proxy scores（代理分数）: `semantic_density = 0.9809`, `semantic_collab_disagreement = 0.9333`, `graph_competition = 0.9881`
- weak-pair exposure（弱连接对暴露）: `endpoint_count = 0`, `reliability_sum = 0.0000`
- graph stats（图统计）: `mid_degree = 32`, `mid_strength = 0.0552`, `semantic_topk_mean_mid_affinity = 0.0000`, `semantic_topk_weak_mid_fraction = 0.0000`

Top semantic neighbors（顶部语义近邻）:
| neighbor | brand | semantic_sim | coarse | mid | local | weak_pair | reliability | neighbor_top10_loss | neighbor_top10_gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3453: 3D Solutech Real Gold 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3D Solutech | 0.9923 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 0 | 0 |
| 3466: 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - RA-I00I-DQUY | 3D Solutech | 0.9918 | 0.0000 | 0.0000 | 0.0678 | 0 | 0.0000 | 1 | 0 |
| 3599: 3D Solutech Aqua Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3D Solutech | 0.9910 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 0 | 0 |
| 2993: 3D Solutech Blue 3D Printer Ultra PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3D Solutech | 0.9909 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 3 | 0 |
| 2660: 3D Solutech Real Grey 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3D Solutech | 0.9908 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 0 | 0 |
| 1552: 3D Solutech Teal Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3D Solutech | 0.9906 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 2 | 0 |

Top mid-graph neighbors（顶部中图近邻）:
| neighbor | brand | mid | semantic_sim | coarse | local | weak_pair | reliability |
|---|---|---:|---:|---:|---:|---:|---:|
| 3343: Corn Sugar 4lb | LD Carlson | 0.0030 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |
| 3631: 3D Solutech Skin 3D Printer PLA Filament 1.75MM Filament 2.2 LBS (1.0KG) | 3D Solutech | 0.0029 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |
| 3428: Alcotec 24-hour Turbo Yeast, 205 grams (Pack of 3) | Home Brew Ohio | 0.0029 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |
| 3592: 3D Solutech Printer Filament, Real Black PLA, 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 1.1 LBS (0.5KG) | 3D Solutech | 0.0028 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |
| 3275: Pyrex Erlenmeyer Flask Starter Pack | Corning | 0.0024 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |
| 3627: Dinglab,500ml Chemistry Lab Glassware Kit,glass Distilling,distillation Apparatus,24/40 | dinglab | 0.0022 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |

### 2697: 3D Solutech Real White 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG)

- brand（品牌）: `3D Solutech`
- family（家族）: `3d_filament`
- loss/gain counts（损失/增益次数）: `top10 loss = 6`, `top10 gain = 0`
- proxy scores（代理分数）: `semantic_density = 0.9812`, `semantic_collab_disagreement = 0.9333`, `graph_competition = 0.9809`
- weak-pair exposure（弱连接对暴露）: `endpoint_count = 2`, `reliability_sum = 0.0087`
- graph stats（图统计）: `mid_degree = 32`, `mid_strength = 0.0481`, `semantic_topk_mean_mid_affinity = 0.0000`, `semantic_topk_weak_mid_fraction = 0.0000`

Top semantic neighbors（顶部语义近邻）:
| neighbor | brand | semantic_sim | coarse | mid | local | weak_pair | reliability | neighbor_top10_loss | neighbor_top10_gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2660: 3D Solutech Real Grey 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3D Solutech | 0.9943 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 0 | 0 |
| 2993: 3D Solutech Blue 3D Printer Ultra PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3D Solutech | 0.9915 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 3 | 0 |
| 3599: 3D Solutech Aqua Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3D Solutech | 0.9912 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 0 | 0 |
| 3466: 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - RA-I00I-DQUY | 3D Solutech | 0.9912 | 0.0177 | 0.0000 | 0.0000 | 0 | 0.0000 | 1 | 0 |
| 1552: 3D Solutech Teal Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3D Solutech | 0.9911 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 2 | 0 |
| 3494: 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3D Solutech | 0.9902 | 0.0564 | 0.0000 | 0.0465 | 0 | 0.0000 | 6 | 0 |

Top mid-graph neighbors（顶部中图近邻）:
| neighbor | brand | mid | semantic_sim | coarse | local | weak_pair | reliability |
|---|---|---:|---:|---:|---:|---:|---:|
| 3665: American 3D Supply PLA 3D Printer Filament, 1 kg Spool, 1.75 mm, Red | American 3D Supply | 0.0032 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |
| 3432: 3D Solutech PETG175YLW 3D Printer Filament, Yellow, 1.75mm, 2.2 LBS (1.0KG) | 3D Solutech | 0.0031 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |
| 1755: BIQU GT2 20Teeth 5mm Bore Aluminum Timing Belt Idler Pulley for 3D Printer 6mm Width Timing Belt (Pack of 5pcs) | BIQU | 0.0029 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |
| 3532: Mercurry 10 Meters GT2 timing belt width 6mm Fit for RepRap Mendel Rostock Prusa GT2-6mm Belt | Mercurry | 0.0020 | 0.0000 | 0.0393 | 0.0408 | 0 | 0.0000 |
| 2675: ELEGOO 5 Sets 28BYJ-48 ULN2003 5V Stepper Motor + ULN2003 Driver Board for Arduino | ELEGOO | 0.0020 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |
| 2505: 100 3/8" Inch Chrome Steel Bearing Balls G25 | BC Precision | 0.0019 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |

## Reading（解读）

- If loss items have higher `weak_pair_endpoint_count`（弱连接对端点数） and higher `semantic_collab_disagreement`（语义-协同失配）, then the push term is concentrating on exactly those semantically dense but collaboratively weak neighborhoods.
- If their `semantic_topk_mean_mid_affinity`（语义近邻中图平均亲和） is low while semantic similarity stays high, then the method is facing semantic-near / graph-weak tension rather than a simple sparse-item problem.
- The case tables show whether the lost items are surrounded by same-family variants（同家族变体） that are semantically near but weakly supported by `G_mid`（中尺度图）.
