# Representative Codebook-SFT Correlation（代表性码本-监督微调相关性）

## 结论

- 这批代表 tokenizer（分词器）显示，codebook reasonableness（码本合理性）确实能解释一部分 SFT（监督微调）趋势，尤其是当前 `R690b L2=0.010 main`、`L2=0.003 weak`、`L2=0.015 fragmented`、`no L1 semantic` 这一组。
- 但它不是单变量充分条件：original semantic（原版语义）和 v2 offline（离线 v2）仍然受 recipe（训练配方）和 SID learnability（语义标识可学习性）影响，不能只看某一个结构指标。
- 最有解释力的模式是：L1 routing（第一层路由）不能过碎，`S-near C-far`（语义近但协同远）需要在保持较高 same L1（同第一层）的同时降低 same L12（同前两层）；过强约束会破坏 `S-near C-near`（语义近且协同近）。

## 代表样本

| label                        |   ndcg_at_10 |   hr_at_10 |   active_l1 |   unique_l12 |   top5_l1_cover |   snear_cfar_same_l1 |   snear_cfar_same_l12 |   snear_cfar_split_after_l1 |   snear_cnear_same_l1 |   sfar_cnear_avg_overlap |
|:-----------------------------|-------------:|-----------:|------------:|-------------:|----------------:|---------------------:|----------------------:|----------------------------:|----------------------:|-------------------------:|
| R690b L2=0.010 main          |       0.1044 |     0.1469 |          60 |         2330 |             707 |              91.1397 |               29.7164 |                     61.4233 |               89.4628 |                   0.0159 |
| Original semantic            |       0.1037 |     0.1509 |          48 |         2295 |             833 |              96.7765 |               30.2448 |                     66.5316 |               93.1742 |                   0.0242 |
| V2 offline                   |       0.1027 |     0.1463 |         203 |         2680 |             282 |              67.2362 |               22.8642 |                     44.3720 |               67.7913 |                   0.0128 |
| Original L2 multihop ranking |       0.1017 |     0.1474 |          88 |         2449 |             516 |              84.8512 |               26.0877 |                     58.7634 |               84.5315 |                   0.0161 |
| QCR L2 conflict ranking      |       0.0998 |     0.1388 |         117 |         2632 |             496 |              92.8307 |               22.2124 |                     70.6183 |               84.8689 |                   0.0114 |
| Stage3 prefix retained       |       0.0991 |     0.1385 |         256 |         2991 |             203 |              65.6333 |               12.7180 |                     52.9153 |               67.9990 |                   0.0133 |
| TAGCF attr mid               |       0.0976 |     0.1315 |         125 |         2570 |             295 |              76.4488 |               21.7016 |                     54.7472 |               76.2523 |                   0.0151 |
| R690b L2=0.003 weak          |       0.0957 |     0.1372 |          56 |         2182 |             683 |              89.7129 |               31.6540 |                     58.0588 |               89.5406 |                   0.0160 |
| V2 LMH mid=0.010             |       0.0952 |     0.1377 |         190 |         2722 |             293 |              78.1927 |               20.6271 |                     57.5656 |               75.2920 |                   0.0126 |
| R690b L2=0.005 weak          |       0.0944 |     0.1315 |         131 |         2322 |             488 |              90.4879 |               26.5457 |                     63.9422 |               79.0812 |                   0.0134 |
| R690b L2=0.015 fragmented    |       0.0941 |     0.1350 |         108 |         2619 |             443 |              88.4446 |               24.0444 |                     64.4002 |               84.5575 |                   0.0151 |
| R690b no L1 semantic         |       0.0938 |     0.1319 |         115 |         2649 |             616 |              87.1411 |               20.5390 |                     66.6021 |               83.5193 |                   0.0116 |

## Correlation（相关性）

### All Representative Tokenizers（全部代表分词器）

| metric                    |   pearson_with_ndcg10 |   spearman_with_ndcg10 |   pearson_with_hr10 |
|:--------------------------|----------------------:|-----------------------:|--------------------:|
| snear_cnear_same_l12      |                0.3656 |                 0.4196 |              0.5084 |
| sfar_cnear_avg_overlap    |                0.4697 |                 0.3993 |              0.5843 |
| snear_cfar_same_l12       |                0.2528 |                 0.3636 |              0.3761 |
| sfar_cnear_same_l1        |                0.4918 |                 0.3187 |              0.6005 |
| top5_l1_cover             |                0.2423 |                 0.2448 |              0.3643 |
| snear_cnear_same_l1       |                0.1328 |                 0.2238 |              0.2594 |
| snear_cfar_same_l1        |               -0.0194 |                 0.1818 |              0.0873 |
| l1_entropy_norm           |               -0.0141 |                 0.1748 |             -0.1538 |
| snear_cfar_split_after_l1 |               -0.2109 |                -0.1888 |             -0.1520 |
| unique_l12                |               -0.1468 |                -0.1888 |             -0.2236 |
| l1_gini                   |               -0.0215 |                -0.1958 |              0.1477 |
| active_l1                 |               -0.1608 |                -0.2308 |             -0.2322 |

### Same SFT Recipe Subset（同监督微调配方子集）

| metric                    |   pearson_with_ndcg10 |   spearman_with_ndcg10 |   pearson_with_hr10 |
|:--------------------------|----------------------:|-----------------------:|--------------------:|
| l1_entropy_norm           |                0.0929 |                 0.2636 |             -0.0386 |
| snear_cnear_same_l12      |                0.2201 |                 0.2636 |              0.3583 |
| sfar_cnear_avg_overlap    |                0.2368 |                 0.2460 |              0.2958 |
| snear_cfar_same_l12       |                0.1158 |                 0.2455 |              0.2282 |
| sfar_cnear_same_l1        |                0.2863 |                 0.1367 |              0.3418 |
| top5_l1_cover             |                0.0141 |                 0.0273 |              0.1004 |
| snear_cnear_same_l1       |               -0.0706 |                 0.0091 |              0.0249 |
| active_l1                 |               -0.0018 |                -0.0182 |             -0.0388 |
| unique_l12                |               -0.0084 |                -0.0273 |             -0.0585 |
| snear_cfar_same_l1        |               -0.2234 |                -0.0455 |             -0.1593 |
| l1_gini                   |               -0.0785 |                -0.2091 |              0.1013 |
| snear_cfar_split_after_l1 |               -0.3798 |                -0.3818 |             -0.3735 |

## 输出文件

- `summary.csv`: `/home/leejt/OneRec/research-progress-log/experiment_analysis/2026-05-08_representative_codebook_correlation/summary.csv`
- `correlation.csv`: `/home/leejt/OneRec/research-progress-log/experiment_analysis/2026-05-08_representative_codebook_correlation/correlation.csv`
- `correlation_same_recipe.csv`: `/home/leejt/OneRec/research-progress-log/experiment_analysis/2026-05-08_representative_codebook_correlation/correlation_same_recipe.csv`
- `metrics.json`: `/home/leejt/OneRec/research-progress-log/experiment_analysis/2026-05-08_representative_codebook_correlation/metrics.json`
