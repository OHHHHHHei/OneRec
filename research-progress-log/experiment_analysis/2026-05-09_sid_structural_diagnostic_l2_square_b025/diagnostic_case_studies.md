# Diagnostic Case Studies（诊断案例分析）

### R690b L2=0.010 main

- split（划分）: `calibration`
- profile（画像）: `balanced-positive`; predicted SFT band（预测监督微调档位）: `high`; actual（实际）: `high`
- SFT（监督微调）: NDCG@10=0.104383, HR@10=0.146923
- L1 stability（第一层稳定性）: `pass`; L1 routing remains semantically stable.
- selective separation（选择性拆分）: `pass`; Semantic-near collaborative-far pairs are selectively separated after L1.
- collaborative preservation（协同保持）: `pass`; Semantic-near collaborative-near pairs are preserved.
- learnability（可学习性）: `pass`; Prefix distribution is compact enough for SFT.
- interpretation（解释）: All four axes pass.


### R690b L2=0.003 weak

- split（划分）: `calibration`
- profile（画像）: `under-separated`; predicted SFT band（预测监督微调档位）: `medium`; actual（实际）: `low`
- SFT（监督微调）: NDCG@10=0.095737, HR@10=0.137216
- L1 stability（第一层稳定性）: `pass`; L1 routing remains semantically stable.
- selective separation（选择性拆分）: `warn`; split_after_l1=58.06% < 60%; S-near C-far same_l12=31.65% > 30.5%
- collaborative preservation（协同保持）: `pass`; Semantic-near collaborative-near pairs are preserved.
- learnability（可学习性）: `pass`; Prefix distribution is compact enough for SFT.
- interpretation（解释）: Stable and learnable, but collaborative separation is weak.


### R690b L2=0.015 fragmented

- split（划分）: `calibration`
- profile（画像）: `separating-but-risky`; predicted SFT band（预测监督微调档位）: `low`; actual（实际）: `low`
- SFT（监督微调）: NDCG@10=0.094080, HR@10=0.135010
- L1 stability（第一层稳定性）: `warn`; active_l1=108 > 100; top5_l1_cover=443 < 500; S-near C-near same_l1=84.56% < 86%
- selective separation（选择性拆分）: `pass`; Semantic-near collaborative-far pairs are selectively separated after L1.
- collaborative preservation（协同保持）: `warn`; S-near C-near same_l1=84.56% < 86%; S-near C-near same_l12=29.38% < 31%
- learnability（可学习性）: `warn`; active_l1=108 > 100; unique_l12=2619 > 2550; top5_l1_cover=443 < 500; l12_singletons=2009 > 2000
- interpretation（解释）: Can split target pairs, but preservation or routing stability is risky.


### R690b no L1 semantic

- split（划分）: `calibration`
- profile（画像）: `over-separated-unstable`; predicted SFT band（预测监督微调档位）: `low`; actual（实际）: `low`
- SFT（监督微调）: NDCG@10=0.093812, HR@10=0.131921
- L1 stability（第一层稳定性）: `warn`; active_l1=115 > 100; S-near C-near same_l1=83.52% < 86%
- selective separation（选择性拆分）: `pass`; Semantic-near collaborative-far pairs are selectively separated after L1.
- collaborative preservation（协同保持）: `fail`; S-near C-near same_l12=24.53% < 27%
- learnability（可学习性）: `warn`; active_l1=115 > 100; unique_l12=2649 > 2550; l12_singletons=2072 > 2000
- interpretation（解释）: Can split semantic-near collaborative-far pairs, but also damages near/near preservation.


### QCR L2 conflict ranking

- split（划分）: `validation`
- profile（画像）: `separating-but-risky`; predicted SFT band（预测监督微调档位）: `low`; actual（实际）: `medium`
- SFT（监督微调）: NDCG@10=0.099810, HR@10=0.138760
- L1 stability（第一层稳定性）: `warn`; active_l1=117 > 100; top5_l1_cover=496 < 500; S-near C-near same_l1=84.87% < 86%
- selective separation（选择性拆分）: `pass`; Semantic-near collaborative-far pairs are selectively separated after L1.
- collaborative preservation（协同保持）: `warn`; S-near C-near same_l1=84.87% < 86%; S-near C-near same_l12=28.19% < 31%
- learnability（可学习性）: `warn`; active_l1=117 > 100; unique_l12=2632 > 2550; top5_l1_cover=496 < 500; l12_singletons=2057 > 2000
- interpretation（解释）: Can split target pairs, but preservation or routing stability is risky.


### V2 offline

- split（划分）: `validation`
- profile（画像）: `out-of-family-flat-routing`; predicted SFT band（预测监督微调档位）: `unknown`; actual（实际）: `high`
- SFT（监督微调）: NDCG@10=0.102708, HR@10=0.146261
- L1 stability（第一层稳定性）: `fail`; active_l1=203 > 150; top5_l1_cover=282 < 300; S-near C-near same_l1=67.79% < 80%
- selective separation（选择性拆分）: `fail`; S-near C-far same_l1=67.24% < 80%; split_after_l1=44.37% < 55%
- collaborative preservation（协同保持）: `fail`; S-near C-near same_l1=67.79% < 80%
- learnability（可学习性）: `fail`; active_l1=203 > 180; top5_l1_cover=282 < 300
- interpretation（解释）: Strong historical tokenizer with non-semantic-flat L1; diagnostic should not be used as a pure rank predictor here.


### R690b L3=0.010 pending

- split（划分）: `prospective`
- profile（画像）: `balanced-positive`; predicted SFT band（预测监督微调档位）: `high`; actual（实际）: `medium`
- SFT（监督微调）: NDCG@10=0.097456, HR@10=0.142731
- L1 stability（第一层稳定性）: `pass`; L1 routing remains semantically stable.
- selective separation（选择性拆分）: `pass`; Semantic-near collaborative-far pairs are selectively separated after L1.
- collaborative preservation（协同保持）: `pass`; Semantic-near collaborative-near pairs are preserved.
- learnability（可学习性）: `pass`; Prefix distribution is compact enough for SFT.
- interpretation（解释）: All four axes pass.


### R690b L2 square dominant b025

- split（划分）: `prospective`
- profile（画像）: `separating-but-risky`; predicted SFT band（预测监督微调档位）: `low`; actual（实际）: `pending`
- SFT（监督微调）: pending（待完成）
- L1 stability（第一层稳定性）: `pass`; L1 routing remains semantically stable.
- selective separation（选择性拆分）: `pass`; Semantic-near collaborative-far pairs are selectively separated after L1.
- collaborative preservation（协同保持）: `warn`; S-near C-near same_l12=29.85% < 31%
- learnability（可学习性）: `pass`; Prefix distribution is compact enough for SFT.
- interpretation（解释）: Can split target pairs, but preservation or routing stability is risky.

