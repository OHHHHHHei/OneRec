# L2/L3 Codebook Reasonableness（码本合理性）分析

## 结论

- 当前主线 `Main L2=0.010,L3=0.020` 是目前最强的 validated tokenizer（已验证分词器）：它比 `L2=0.003` 更能把 semantic-near collaborative-far（语义近但协同远）物品保留在同一 L1（第一层）粗类下，并在 L2/L3（第二/三层）拆开；对应 SFT（监督微调）NDCG@10 也更高。
- `L3=0.010` 的结构指标和 pair-level（物品对级）指标最像一个强候选：same L1（同第一层）更高、same L12（同前两层）不升反降，说明它保留语义粗类同时进一步增强后层分辨。
- `S-far C-near`（语义远但协同近）上三组 tokenizer（分词器）的后层 token overlap（token 重合）都很弱；当前方法主要解决了“语义近但协同远要拆开”，还没有很好解决“语义远但协同近要拉近”。
- 因为 `L3=0.010` 的 SFT（监督微调）结果当前仍是 pending（待完成），现在不能声称“结构越合理下游一定越好”；只能说 `L2=0.003 -> 当前主线` 这组已完成对比支持该趋势。

## 数据定义

- semantic similarity（语义相似度）：`Industrial_and_Scientific.emb-qwen-td.npy` 的 cosine similarity（余弦相似度）。
- collaborative similarity（协同相似度）：只用 train interaction（训练交互），把每条样本的 `history_item_id -> item_id` 作为 direct edge（直接边），累计 co-occurrence（共现）并计算 PPMI（正点互信息）。
- valid/test interaction（验证/测试交互）没有参与构图，避免 leakage（泄露）。

| stat（统计） | value（数值） |
| --- | --- |
| train rows（训练样本数） | 36259 |
| direct edges（直接协同边） | 100920 |
| edge events（边事件数） | 124597 |
| cooc p95（共现次数 p95） | 2.0 |
| PPMI p95（正点互信息 p95） | 5.9157 |

## 结构指标

| tokenizer（分词器） | active L1（活跃第一层） | unique L12（唯一前两层） | unique SID（唯一语义标识） | collision（冲突） | max conflict（最大冲突簇） | top5 L1 cover（前五个第一层覆盖） | max L1 bucket（最大第一层桶） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L2=0.003 | 56 | 2182 | 3673 | 13 | 2 | 683 | 194 |
| Main L2=0.010,L3=0.020 | 60 | 2330 | 3670 | 16 | 2 | 707 | 167 |
| L3=0.010 | 60 | 2394 | 3673 | 13 | 2 | 738 | 210 |

## Pair-Level（物品对级）指标

### S-near C-far

pair count（物品对数量）=5677, semantic mean（语义均值）=0.9610, PPMI mean（正点互信息均值）=0.0000
| tokenizer（分词器） | same L1（同第一层） | same L12（同前两层） | same SID（同语义标识） | same L2 token（同第二层 token） | same L3 token（同第三层 token） | avg overlap（平均 token 重合） | avg LCP（平均最长前缀） | split after L1（同 L1 后拆开） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L2=0.003 | 89.71% | 31.65% | 0.04% | 31.72% | 0.88% | 1.223 | 1.214 | 58.06% |
| Main L2=0.010,L3=0.020 | 91.14% | 29.72% | 0.07% | 30.88% | 0.97% | 1.230 | 1.209 | 61.42% |
| L3=0.010 | 95.42% | 29.56% | 0.05% | 29.70% | 0.58% | 1.257 | 1.250 | 65.86% |

### S-near C-near

pair count（物品对数量）=3853, semantic mean（语义均值）=0.9510, PPMI mean（正点互信息均值）=4.3294
| tokenizer（分词器） | same L1（同第一层） | same L12（同前两层） | same SID（同语义标识） | same L2 token（同第二层 token） | same L3 token（同第三层 token） | avg overlap（平均 token 重合） | avg LCP（平均最长前缀） | split after L1（同 L1 后拆开） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L2=0.003 | 89.54% | 34.80% | 0.26% | 34.96% | 1.06% | 1.256 | 1.246 | 54.74% |
| Main L2=0.010,L3=0.020 | 89.46% | 34.26% | 0.29% | 35.32% | 0.96% | 1.257 | 1.240 | 55.20% |
| L3=0.010 | 90.86% | 34.00% | 0.26% | 34.13% | 0.96% | 1.260 | 1.251 | 56.86% |

### S-far C-near

pair count（物品对数量）=10000, semantic mean（语义均值）=0.7195, PPMI mean（正点互信息均值）=4.5175
| tokenizer（分词器） | same L1（同第一层） | same L12（同前两层） | same SID（同语义标识） | same L2 token（同第二层 token） | same L3 token（同第三层 token） | avg overlap（平均 token 重合） | avg LCP（平均最长前缀） | split after L1（同 L1 后拆开） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L2=0.003 | 0.54% | 0.00% | 0.00% | 0.28% | 0.78% | 0.016 | 0.005 | 0.54% |
| Main L2=0.010,L3=0.020 | 0.54% | 0.00% | 0.00% | 0.35% | 0.70% | 0.016 | 0.005 | 0.54% |
| L3=0.010 | 0.73% | 0.00% | 0.00% | 0.29% | 0.75% | 0.018 | 0.007 | 0.73% |

### S-far C-far

pair count（物品对数量）=10000, semantic mean（语义均值）=0.7170, PPMI mean（正点互信息均值）=0.0000
| tokenizer（分词器） | same L1（同第一层） | same L12（同前两层） | same SID（同语义标识） | same L2 token（同第二层 token） | same L3 token（同第三层 token） | avg overlap（平均 token 重合） | avg LCP（平均最长前缀） | split after L1（同 L1 后拆开） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L2=0.003 | 0.20% | 0.00% | 0.00% | 0.33% | 0.64% | 0.012 | 0.002 | 0.20% |
| Main L2=0.010,L3=0.020 | 0.09% | 0.00% | 0.00% | 0.29% | 0.41% | 0.008 | 0.001 | 0.09% |
| L3=0.010 | 0.14% | 0.00% | 0.00% | 0.35% | 0.52% | 0.010 | 0.001 | 0.14% |

## Downstream SFT（下游监督微调）

| tokenizer（分词器） | status（状态） | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | HR@1 | HR@3 | HR@5 | HR@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L2=0.003 | available | 0.066181 | 0.077773 | 0.085235 | 0.095737 | 0.066181 | 0.086477 | 0.104567 | 0.137216 |
| Main L2=0.010,L3=0.020 | available | 0.070593 | 0.088131 | 0.094889 | 0.104383 | 0.070593 | 0.100816 | 0.117362 | 0.146923 |
| L3=0.010 | pending | - | - | - | - | - | - | - | - |

## 具体物品例子

### S-near C-far

1. pair（物品对） `1543` - `1738`; sim（语义相似度）=0.9990; cooc（共现）=0; PPMI（正点互信息）=0.0000
   - A: Aluminum 6061-T6 Seamless Round Tubing, WW-T 700/6, 1-1/8" OD, 1.009" ID, 0.058" Wall, 12
   - B: Aluminum 6061-T6 Seamless Round Tubing, WW-T 700/6, 1-1/4" OD, 1.18" ID, 0.035" Wall, 24" Length
   - L2=0.003: `['<a_255>', '<b_112>', '<c_217>']` vs `['<a_255>', '<b_112>', '<c_186>']`; LCP=2, overlap=2
   - Main L2=0.010,L3=0.020: `['<a_225>', '<b_250>', '<c_237>']` vs `['<a_225>', '<b_11>', '<c_19>']`; LCP=1, overlap=1
   - L3=0.010: `['<a_115>', '<b_14>', '<c_70>']` vs `['<a_115>', '<b_14>', '<c_131>']`; LCP=2, overlap=2
2. pair（物品对） `2014` - `2233`; sim（语义相似度）=0.9986; cooc（共现）=0; PPMI（正点互信息）=0.0000
   - A: Red Brass Pipe Fitting, Nipple, Schedule 40 Seamless, 1/4" NPT Male X 1-1/2" Length
   - B: Red Brass Pipe Fitting, Nipple, Schedule 40 Seamless, 1/4" NPT Male X 4" Length
   - L2=0.003: `['<a_59>', '<b_78>', '<c_0>']` vs `['<a_59>', '<b_78>', '<c_129>']`; LCP=2, overlap=2
   - Main L2=0.010,L3=0.020: `['<a_78>', '<b_250>', '<c_71>']` vs `['<a_78>', '<b_250>', '<c_177>']`; LCP=2, overlap=2
   - L3=0.010: `['<a_31>', '<b_19>', '<c_4>']` vs `['<a_31>', '<b_30>', '<c_115>']`; LCP=1, overlap=1
3. pair（物品对） `753` - `2459`; sim（语义相似度）=0.9972; cooc（共现）=0; PPMI（正点互信息）=0.0000
   - A: PVC (Polyvinyl Chloride) Sheet, Opaque White, Standard Tolerance, UL 94/ASTM D1784, 1/4" Thickness, 12" Width, 12" Lengt
   - B: PVC (Polyvinyl Chloride) Sheet, Opaque Gray, Standard Tolerance, UL 94/ASTM D1784, 0.25" Thickness, 12" Width, 12" Lengt
   - L2=0.003: `['<a_255>', '<b_123>', '<c_209>']` vs `['<a_255>', '<b_123>', '<c_162>']`; LCP=2, overlap=2
   - Main L2=0.010,L3=0.020: `['<a_225>', '<b_96>', '<c_32>']` vs `['<a_225>', '<b_96>', '<c_222>']`; LCP=2, overlap=2
   - L3=0.010: `['<a_253>', '<b_44>', '<c_132>']` vs `['<a_253>', '<b_3>', '<c_105>']`; LCP=1, overlap=1
4. pair（物品对） `1543` - `1739`; sim（语义相似度）=0.9965; cooc（共现）=0; PPMI（正点互信息）=0.0000
   - A: Aluminum 6061-T6 Seamless Round Tubing, WW-T 700/6, 1-1/8" OD, 1.009" ID, 0.058" Wall, 12
   - B: Aluminum 6061-T6 Seamless Round Tubing, WW-T 700/6, 1/2" OD, 0.26" ID, 0.125" Wall, 24" Length
   - L2=0.003: `['<a_255>', '<b_112>', '<c_217>']` vs `['<a_255>', '<b_112>', '<c_209>']`; LCP=2, overlap=2
   - Main L2=0.010,L3=0.020: `['<a_225>', '<b_250>', '<c_237>']` vs `['<a_225>', '<b_11>', '<c_48>']`; LCP=1, overlap=1
   - L3=0.010: `['<a_115>', '<b_14>', '<c_70>']` vs `['<a_115>', '<b_14>', '<c_27>']`; LCP=2, overlap=2
5. pair（物品对） `3442` - `3469`; sim（语义相似度）=0.9963; cooc（共现）=0; PPMI（正点互信息）=0.0000
   - A: 3D Solutech Real Purple 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG)
   - B: 3D Solutech Real Pink 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG)
   - L2=0.003: `['<a_236>', '<b_69>', '<c_68>']` vs `['<a_236>', '<b_69>', '<c_10>']`; LCP=2, overlap=2
   - Main L2=0.010,L3=0.020: `['<a_222>', '<b_229>', '<c_23>']` vs `['<a_222>', '<b_68>', '<c_10>']`; LCP=1, overlap=1
   - L3=0.010: `['<a_184>', '<b_137>', '<c_244>']` vs `['<a_184>', '<b_137>', '<c_155>']`; LCP=2, overlap=2

### S-near C-near

1. pair（物品对） `181` - `182`; sim（语义相似度）=1.0000; cooc（共现）=327; PPMI（正点互信息）=2.4351
   - A: AcuRite 00613 Humidity Monitor with Indoor Thermometer, Digital Hygrometer and Humidity Gauge Indicator
   - B: AcuRite 00613 Humidity Monitor with Indoor Thermometer, Digital Hygrometer and Humidity Gauge Indicator
   - L2=0.003: `['<a_217>', '<b_82>', '<c_0>']` vs `['<a_217>', '<b_82>', '<c_0>']`; LCP=3, overlap=3
   - Main L2=0.010,L3=0.020: `['<a_31>', '<b_221>', '<c_0>']` vs `['<a_31>', '<b_221>', '<c_0>']`; LCP=3, overlap=3
   - L3=0.010: `['<a_59>', '<b_52>', '<c_0>']` vs `['<a_59>', '<b_52>', '<c_0>']`; LCP=3, overlap=3
2. pair（物品对） `417` - `418`; sim（语义相似度）=0.9918; cooc（共现）=204; PPMI（正点互信息）=3.1059
   - A: American Terminal E-FFB250N-100 16/14-Gauge Economy Nylon Fully-Insulated Female Quick Disconnects
   - B: American Terminal E-FMB250N-100 16/14-Gauge Economy Nylon Fully-Insulated Male Quick Disconnects
   - L2=0.003: `['<a_18>', '<b_197>', '<c_5>']` vs `['<a_18>', '<b_197>', '<c_35>']`; LCP=2, overlap=2
   - Main L2=0.010,L3=0.020: `['<a_202>', '<b_97>', '<c_123>']` vs `['<a_202>', '<b_97>', '<c_0>']`; LCP=2, overlap=2
   - L3=0.010: `['<a_191>', '<b_218>', '<c_173>']` vs `['<a_191>', '<b_218>', '<c_26>']`; LCP=2, overlap=2
3. pair（物品对） `119` - `526`; sim（语义相似度）=0.9473; cooc（共现）=157; PPMI（正点互信息）=3.9791
   - A: Rubbermaid Commercial BRUTE Heavy-Duty Round Waste/Utility Container with Venting Channels, 20-gallon, Gray (FG262000GRA
   - B: Rubbermaid Commercial Products FG263100GRAY Rubbermaid Commercial Round Brute Container Lid, Gray, 32G
   - L2=0.003: `['<a_104>', '<b_25>', '<c_77>']` vs `['<a_104>', '<b_25>', '<c_160>']`; LCP=2, overlap=2
   - Main L2=0.010,L3=0.020: `['<a_241>', '<b_248>', '<c_233>']` vs `['<a_241>', '<b_127>', '<c_58>']`; LCP=1, overlap=1
   - L3=0.010: `['<a_20>', '<b_239>', '<c_209>']` vs `['<a_20>', '<b_211>', '<c_60>']`; LCP=1, overlap=1
4. pair（物品对） `418` - `1452`; sim（语义相似度）=0.9879; cooc（共现）=111; PPMI（正点互信息）=3.3005
   - A: American Terminal E-FMB250N-100 16/14-Gauge Economy Nylon Fully-Insulated Male Quick Disconnects
   - B: American Terminal E-FMR250N-100 22/18-Gauge Economy Nylon Fully-Insulated Male Quick Disconnects
   - L2=0.003: `['<a_18>', '<b_197>', '<c_35>']` vs `['<a_18>', '<b_197>', '<c_100>']`; LCP=2, overlap=2
   - Main L2=0.010,L3=0.020: `['<a_202>', '<b_97>', '<c_0>']` vs `['<a_202>', '<b_97>', '<c_2>']`; LCP=2, overlap=2
   - L3=0.010: `['<a_191>', '<b_218>', '<c_26>']` vs `['<a_191>', '<b_218>', '<c_19>']`; LCP=2, overlap=2
5. pair（物品对） `1153` - `1156`; sim（语义相似度）=0.9990; cooc（共现）=75; PPMI（正点互信息）=4.1698
   - A: 6061 Aluminum Round Rod, Unpolished (Mill) Finish, Extruded, T6511 Temper, ASTM B221, 2-1/4" Diameter, 12" Length
   - B: 6061 Aluminum Round Rod, Unpolished (Mill) Finish, Extruded, T6511 Temper, ASTM B221, 1-5/8" Diameter, 24" Length
   - L2=0.003: `['<a_76>', '<b_108>', '<c_39>']` vs `['<a_76>', '<b_108>', '<c_44>']`; LCP=2, overlap=2
   - Main L2=0.010,L3=0.020: `['<a_225>', '<b_205>', '<c_169>']` vs `['<a_225>', '<b_205>', '<c_185>']`; LCP=2, overlap=2
   - L3=0.010: `['<a_253>', '<b_135>', '<c_77>']` vs `['<a_253>', '<b_135>', '<c_168>']`; LCP=2, overlap=2

### S-far C-near

1. pair（物品对） `1205` - `2871`; sim（语义相似度）=0.7544; cooc（共现）=23; PPMI（正点互信息）=4.0595
   - A: PEI (Polyetherimide) Sheet, Opaque Off-White, Standard Tolerance, ASTM D5205 PEI0113, 0.03" Thickness, 12" Width, 24" Le
   - B: 3M 468MP Adhesive Transfer Tape, 4" width x 5yd length (1 roll)
   - L2=0.003: `['<a_147>', '<b_233>', '<c_0>']` vs `['<a_101>', '<b_154>', '<c_96>']`; LCP=0, overlap=0
   - Main L2=0.010,L3=0.020: `['<a_51>', '<b_17>', '<c_1>']` vs `['<a_192>', '<b_87>', '<c_134>']`; LCP=0, overlap=0
   - L3=0.010: `['<a_251>', '<b_155>', '<c_96>']` vs `['<a_226>', '<b_111>', '<c_144>']`; LCP=0, overlap=0
2. pair（物品对） `182` - `2938`; sim（语义相似度）=0.7431; cooc（共现）=17; PPMI（正点互信息）=3.1805
   - A: AcuRite 00613 Humidity Monitor with Indoor Thermometer, Digital Hygrometer and Humidity Gauge Indicator
   - B: VenTech VT DUCT-6 VTD625 Aluminum Duct for Ventilation Ducting, 6''
   - L2=0.003: `['<a_217>', '<b_82>', '<c_0>']` vs `['<a_241>', '<b_38>', '<c_224>']`; LCP=0, overlap=0
   - Main L2=0.010,L3=0.020: `['<a_31>', '<b_221>', '<c_0>']` vs `['<a_223>', '<b_78>', '<c_17>']`; LCP=0, overlap=0
   - L3=0.010: `['<a_59>', '<b_52>', '<c_0>']` vs `['<a_31>', '<b_151>', '<c_200>']`; LCP=0, overlap=0
3. pair（物品对） `182` - `218`; sim（语义相似度）=0.7337; cooc（共现）=14; PPMI（正点互信息）=2.1268
   - A: AcuRite 00613 Humidity Monitor with Indoor Thermometer, Digital Hygrometer and Humidity Gauge Indicator
   - B: Phresh Duct Silencer 8 in x 24 in
   - L2=0.003: `['<a_217>', '<b_82>', '<c_0>']` vs `['<a_159>', '<b_173>', '<c_23>']`; LCP=0, overlap=0
   - Main L2=0.010,L3=0.020: `['<a_31>', '<b_221>', '<c_0>']` vs `['<a_223>', '<b_82>', '<c_200>']`; LCP=0, overlap=0
   - L3=0.010: `['<a_59>', '<b_52>', '<c_0>']` vs `['<a_57>', '<b_139>', '<c_131>']`; LCP=0, overlap=0
4. pair（物品对） `181` - `944`; sim（语义相似度）=0.7494; cooc（共现）=14; PPMI（正点互信息）=1.7890
   - A: AcuRite 00613 Humidity Monitor with Indoor Thermometer, Digital Hygrometer and Humidity Gauge Indicator
   - B: General Hydroponics PH Test Kit, 1-Ounce
   - L2=0.003: `['<a_217>', '<b_82>', '<c_0>']` vs `['<a_37>', '<b_7>', '<c_187>']`; LCP=0, overlap=0
   - Main L2=0.010,L3=0.020: `['<a_31>', '<b_221>', '<c_0>']` vs `['<a_62>', '<b_179>', '<c_33>']`; LCP=0, overlap=0
   - L3=0.010: `['<a_59>', '<b_52>', '<c_0>']` vs `['<a_59>', '<b_12>', '<c_121>']`; LCP=1, overlap=1
5. pair（物品对） `181` - `218`; sim（语义相似度）=0.7337; cooc（共现）=14; PPMI（正点互信息）=1.6893
   - A: AcuRite 00613 Humidity Monitor with Indoor Thermometer, Digital Hygrometer and Humidity Gauge Indicator
   - B: Phresh Duct Silencer 8 in x 24 in
   - L2=0.003: `['<a_217>', '<b_82>', '<c_0>']` vs `['<a_159>', '<b_173>', '<c_23>']`; LCP=0, overlap=0
   - Main L2=0.010,L3=0.020: `['<a_31>', '<b_221>', '<c_0>']` vs `['<a_223>', '<b_82>', '<c_200>']`; LCP=0, overlap=0
   - L3=0.010: `['<a_59>', '<b_52>', '<c_0>']` vs `['<a_57>', '<b_139>', '<c_131>']`; LCP=0, overlap=0

### S-far C-far

1. pair（物品对） `586` - `2883`; sim（语义相似度）=0.5450; cooc（共现）=0; PPMI（正点互信息）=0.0000
   - A: Neato XV-21 Pet & Allergy Automatic Vacuum Cleaner
   - B: Anderson Metals 56122 Brass Pipe Fitting, Hex Nipple, 1/2" x 1/2" NPT Male Pipe
   - L2=0.003: `['<a_42>', '<b_208>', '<c_62>']` vs `['<a_59>', '<b_162>', '<c_183>']`; LCP=0, overlap=0
   - Main L2=0.010,L3=0.020: `['<a_165>', '<b_183>', '<c_225>']` vs `['<a_78>', '<b_3>', '<c_91>']`; LCP=0, overlap=0
   - L3=0.010: `['<a_196>', '<b_82>', '<c_239>']` vs `['<a_31>', '<b_89>', '<c_136>']`; LCP=0, overlap=0
2. pair（物品对） `101` - `586`; sim（语义相似度）=0.5577; cooc（共现）=0; PPMI（正点互信息）=0.0000
   - A: Stainless Steel 304/304L Pipe Fitting, Nipple, Schedule 40 Welded, 4" X 6" NPT Male by Merit Brass
   - B: Neato XV-21 Pet & Allergy Automatic Vacuum Cleaner
   - L2=0.003: `['<a_59>', '<b_143>', '<c_53>']` vs `['<a_42>', '<b_208>', '<c_62>']`; LCP=0, overlap=0
   - Main L2=0.010,L3=0.020: `['<a_78>', '<b_250>', '<c_7>']` vs `['<a_165>', '<b_183>', '<c_225>']`; LCP=0, overlap=0
   - L3=0.010: `['<a_31>', '<b_167>', '<c_9>']` vs `['<a_196>', '<b_82>', '<c_239>']`; LCP=0, overlap=0
3. pair（物品对） `586` - `2916`; sim（语义相似度）=0.5612; cooc（共现）=0; PPMI（正点互信息）=0.0000
   - A: Neato XV-21 Pet & Allergy Automatic Vacuum Cleaner
   - B: Taulman BRIDGE Filament, 1.75 mm, BLACK
   - L2=0.003: `['<a_42>', '<b_208>', '<c_62>']` vs `['<a_149>', '<b_220>', '<c_180>']`; LCP=0, overlap=0
   - Main L2=0.010,L3=0.020: `['<a_165>', '<b_183>', '<c_225>']` vs `['<a_222>', '<b_158>', '<c_36>']`; LCP=0, overlap=0
   - L3=0.010: `['<a_196>', '<b_82>', '<c_239>']` vs `['<a_184>', '<b_183>', '<c_0>']`; LCP=0, overlap=0
4. pair（物品对） `204` - `2602`; sim（语义相似度）=0.5613; cooc（共现）=0; PPMI（正点互信息）=0.0000
   - A: Gorilla Super Glue, Two 3 Gram Tubes, Clear
   - B: Tektronix TBS1052B Digital Storage Oscilloscope, 2 Channel, 50 MHz Bandwidth, 5 Year Warranty
   - L2=0.003: `['<a_2>', '<b_220>', '<c_9>']` vs `['<a_42>', '<b_92>', '<c_100>']`; LCP=0, overlap=0
   - Main L2=0.010,L3=0.020: `['<a_150>', '<b_176>', '<c_184>']` vs `['<a_165>', '<b_200>', '<c_59>']`; LCP=0, overlap=0
   - L3=0.010: `['<a_19>', '<b_0>', '<c_2>']` vs `['<a_196>', '<b_171>', '<c_85>']`; LCP=0, overlap=0
5. pair（物品对） `2602` - `3059`; sim（语义相似度）=0.5615; cooc（共现）=0; PPMI（正点互信息）=0.0000
   - A: Tektronix TBS1052B Digital Storage Oscilloscope, 2 Channel, 50 MHz Bandwidth, 5 Year Warranty
   - B: Filabot ABS 3D Printing Smoothing Pen
   - L2=0.003: `['<a_42>', '<b_92>', '<c_100>']` vs `['<a_149>', '<b_95>', '<c_46>']`; LCP=0, overlap=0
   - Main L2=0.010,L3=0.020: `['<a_165>', '<b_200>', '<c_59>']` vs `['<a_249>', '<b_108>', '<c_124>']`; LCP=0, overlap=0
   - L3=0.010: `['<a_196>', '<b_171>', '<c_85>']` vs `['<a_246>', '<b_106>', '<c_214>']`; LCP=0, overlap=0

## 判断

1. `L2=0.003` 的 L1（第一层）并没有坏，但 L2（第二层）协同干预偏弱；在 `S-near C-far`（语义近但协同远）上，同 L12（同前两层）比例最高，说明后层拆分不够。
2. 当前主线的结构合理性和下游 SFT（监督微调）是一致的：它在 `S-near C-far` 上更会“同粗类、后层拆”，同时 NDCG@10 从 `0.095737` 提到 `0.104383`。
3. `L3=0.010` 在结构上目前最漂亮：same L1（同第一层）最高，same L12（同前两层）与当前主线相近甚至略低，split after L1（同第一层后拆开）最高。若它的 SFT（监督微调）结果也提升，就能更强地支持“码本合理性 -> 下游收益”的叙事。
4. 如果 `L3=0.010` 下游没有提升，优先怀疑 learnability（可学习性）或 route distribution（路由分布）问题，而不是单纯否定 pair-level reasonableness（物品对级合理性）。

## 输出文件

- `metrics.json`: `/home/leejt/OneRec/research-progress-log/experiment_analysis/2026-05-08_codebook_reasonableness/metrics.json`
- `structure_metrics.csv`: `/home/leejt/OneRec/research-progress-log/experiment_analysis/2026-05-08_codebook_reasonableness/structure_metrics.csv`
- `pair_metrics.csv`: `/home/leejt/OneRec/research-progress-log/experiment_analysis/2026-05-08_codebook_reasonableness/pair_metrics.csv`
- `pair_examples.json`: `/home/leejt/OneRec/research-progress-log/experiment_analysis/2026-05-08_codebook_reasonableness/pair_examples.json`
