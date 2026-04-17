# 2026-04-16 `R630c` SFT Evaluate Industrial

Status（状态）: `snapshot（快照）`

## Goal（目标）

把 `R630c` 作为当前 selective separation（选择性分离）方向唯一值得继续的 tokenizer candidate（分词器候选），推进到最小完整下游：

- `title_history2sid_on + desc_align_p05`
- `SFT -> evaluate`（监督微调到评测）

这轮的目标不是宣布 `R630c` 已经赢下项目主线，而是回答一个更严格也更关键的问题：

> 在 current `v2`（当前 `v2`）最强的 graph-aware recipe（图感知配方）上，`R630c` 的 `pull + push`（拉近加推远）码本空间能不能转化成真实的下游排名收益？

## Source Tokenizer（来源分词器）

- source run（来源运行）:
  - `R630c`
- generated index（生成索引）:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_mid_pull_push_20260416/generated_indices/Industrial_and_Scientific.r630c_mid_pull_push.index.json`
- tokenizer stage summary（分词器阶段摘要）:
  - [2026-04-16_mgr_sid_r630_mid_pull_push_industrial/README.md](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_r630_mid_pull_push_industrial/README.md)

## Data Root（数据根目录）

- variant root（变体根目录）:
  - `/home/leejt/OneRec/data_experiment/Amazon/r630c_mid_pull_push`

## Configs（配置）

- SFT：
  - [sft_industrial_mgr_r630c_title_on_desc_p05.yaml](/home/leejt/OneRec/config/experiments/sft_industrial_mgr_r630c_title_on_desc_p05.yaml)
- Evaluate：
  - [evaluate_industrial_mgr_r630c_title_on_desc_p05.yaml](/home/leejt/OneRec/config/experiments/evaluate_industrial_mgr_r630c_title_on_desc_p05.yaml)

## Launcher（启动脚本）

- [experiment_mgr_sid_r630c_sft_eval_chain.sh](/home/leejt/OneRec/scripts/experiment_mgr_sid_r630c_sft_eval_chain.sh)

## Runtime（运行时）

- tmux（终端复用）:
  - `mgr_r630c_sft_eval`
- GPUs：
  - `2,3,4,5`
- SFT log（日志）:
  - `/home/leejt/OneRec/logs/experiment_mgr_sid_r630c_sft_20260416.log`
- Evaluate log（日志）:
  - `/home/leejt/OneRec/logs/experiment_mgr_sid_r630c_eval_20260416.log`
- SFT output（输出）:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r630c_sft_eval_20260416/title_on_desc_p05/sft`
- Evaluate result（结果）:
  - `/home/leejt/OneRec/results/experiments/mgr_sid_r630c_sft_eval_20260416/final_result_sft_mgr_r630c_title_on_desc_p05_Industrial_and_Scientific.json`

## Status（状态）

- launch date（启动日期）:
  - `2026-04-16`
- current status（当前状态）:
  - `COMPLETED`

## Decision Rule（裁决规则）

这轮只回答一个问题：

- `R630c` 相对 current `v2_on_p05`（当前 `v2_on_p05`）到底是正、负，还是不成立

因此最主要的比较对象是：

- current strongest validated line（当前最强已验证主线）:
  - `v2_on_p05 -> SFT`
  - `v2_on_p05 -> RL`

如果 `R630c` 连 `SFT -> evaluate`（监督微调到评测）都站不住，这条 selective separation（选择性分离）简化线就不能进入下一阶段。

## Final Metrics（最终指标）

- `NDCG@1/3/5/10/20/50`
  - `0.06199 / 0.07876 / 0.08503 / 0.09261 / 0.10278 / 0.11358`
- `HR@1/3/5/10/20/50`
  - `0.06199 / 0.09067 / 0.10589 / 0.12972 / 0.17009 / 0.22458`
- `constraint_invalid_total`（约束失配总数）
  - `0`
- SFT stop epoch（监督微调停止轮次）
  - `5.5`
- SFT final eval loss（监督微调最终验证损失）
  - `1.59752`
- SFT final train loss（监督微调最终训练损失）
  - `0.46236`

## Comparison（对比）

### Against Current `v2_on_p05`（对比当前 `v2_on_p05`）

- current `v2_on_p05`
  - `NDCG@10 = 0.10271`
  - `HR@10 = 0.14626`
- `R630c`
  - `NDCG@10 = 0.09261`
  - `HR@10 = 0.12972`

delta（差值）:

- `NDCG@10`: `-0.01010`
- `HR@10`: `-0.01654`
- `NDCG@1`: `-0.00860`
- `HR@1`: `-0.00860`

### Against Strongest Original MiniOneRec SFT（对比原版最强 `SFT`）

- strongest original `SFT`
  - `NDCG@10 = 0.10372`
  - `HR@10 = 0.15089`
- delta（差值）:
  - `NDCG@10`: `-0.01111`
  - `HR@10`: `-0.02118`

### Against `R510`（对比 `R510`）

- `R510`
  - `NDCG@10 = 0.09758`
  - `HR@10 = 0.13148`
- delta（差值）:
  - `NDCG@10`: `-0.00497`
  - `HR@10`: `-0.00176`

## Reading（解读）

这次结果是一个**明确负结论**：

- `R630c` 没有超过 current `v2_on_p05`（当前 `v2_on_p05`）
- `R630c` 也没有超过 strongest original MiniOneRec `SFT`（原版最强 `SFT`）
- 更关键的是：
  - 它甚至没有超过 `R510`
  - 而 `R510` 已经被证明是下游负结果

因此这次可以确认的不是“pull + push（拉近加推远）有下游收益”，而是：

> **当前这版 `mid-only pull + push`（仅中层拉近加推远）虽然能改善 tokenizer-side（分词器侧）冲突，但这种改进没有转化成更强的下游 `SFT` 排名效果。**

这也进一步强化了一个已有负结论：

> **generated collision（生成后冲突率）即使继续变好，也不能被当作 downstream（下游）正结果的可靠前验指标。**

## Verdict（裁决）

- `R630c`:
  - `SFT -> evaluate`（监督微调到评测）`NEGATIVE`
- decision（决策）:
  - **不要**把 `R630c` 推进到 `RL`
  - 当前这版 simplified selective separation（简化选择性分离）主张，在现有实现下不能进入 strongest line（最强主线）

## Posterior Output Diagnosis（后验输出诊断）

本轮额外补做了 output-side diagnosis（输出侧诊断），重点不是再看 tokenizer-side proxy（分词器侧代理指标），而是直接看模型 `evaluate`（评测）输出到底错在什么地方。

### Artifacts（产物）

- single-run posterior diagnostics（单模型后验诊断）:
  - [R630c_output_sid_diagnostics.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_r630c_sft_eval_industrial/R630c_output_sid_diagnostics.json)
  - [v2_on_p05_output_sid_diagnostics.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_r630c_sft_eval_industrial/v2_on_p05_output_sid_diagnostics.json)
  - [strongest_orig_sft_output_sid_diagnostics.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_r630c_sft_eval_industrial/strongest_orig_sft_output_sid_diagnostics.json)
- aligned top-k comparison（对齐 `top-k` 对比）:
  - [TOPK_V2_ON_P05_VS_R630C.md](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_r630c_sft_eval_industrial/TOPK_V2_ON_P05_VS_R630C.md)
  - [TOPK_STRONGEST_ORIG_SFT_VS_R630C.md](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_r630c_sft_eval_industrial/TOPK_STRONGEST_ORIG_SFT_VS_R630C.md)
- detailed output pair diagnosis（细粒度输出成对诊断）:
  - [OUTPUT_PAIR_DIAGNOSIS_V2_ON_P05_VS_R630C.md](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_r630c_sft_eval_industrial/OUTPUT_PAIR_DIAGNOSIS_V2_ON_P05_VS_R630C.md)
  - [OUTPUT_PAIR_DIAGNOSIS_STRONGEST_ORIG_SFT_VS_R630C.md](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_r630c_sft_eval_industrial/OUTPUT_PAIR_DIAGNOSIS_STRONGEST_ORIG_SFT_VS_R630C.md)
- loss-item graph/semantic analysis（损失物品图/语义分析）:
  - [LOSS_ITEM_GRAPH_SEMANTIC_V2_ON_P05_VS_R630C.md](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_r630c_sft_eval_industrial/LOSS_ITEM_GRAPH_SEMANTIC_V2_ON_P05_VS_R630C.md)

### Main Findings（主要发现）

- `R630c` 的 catalog cleanliness（码本目录清洁度）确实更强：
  - generated collision rate（生成碰撞率）=`0.00298`
  - weighted `H(level3 | level1, level2)`（加权三级条件熵）=`0.68791`
  - 都优于 `v2_on_p05` 的 `0.00353 / 0.71505`
- 但 output-side behavior（输出侧行为）更差：
  - `R630c` `top10 hit`（`top10` 命中）=`0.12972`
  - `v2_on_p05` `top10 hit`（`top10` 命中）=`0.14626`
  - strongest original `SFT` `top10 hit`（`top10` 命中）=`0.15089`
- `R630c` 相对 `v2_on_p05` 的唯一局部正信号，只出现在 crowded targets（拥挤目标）的 head ranking（头部排序）：
  - baseline `l2>=4` bucket（`l2>=4` 桶）上，`top1` delta（差值）=`+0.00990`
  - 同一 bucket（桶）上，`top3` delta（差值）=`+0.00990`
  - 但到 `top10` 就变成 `-0.02871`
- 这说明 `R630c` 不是完全没有改善局部歧义（local ambiguity，局部歧义），而是：
  - 它对 hardest crowded cases（最难的拥挤样本）有一点 head disambiguation（头部消歧）收益
  - 但 beam retention（候选束保留）更差，收益存活不到 `top10`
- 一个很直接的证据是：
  - 当 `v2_on_p05` 已经 `top10 hit`（`top10` 命中）但 `R630c` 丢失时，共有 `193` 个样本
  - 其中 `90` 个样本在 `R630c` 里直接掉到 `>50`
  - 另外 `59` 个掉到 `11-20`
- 更细的 retention diagnosis（保留诊断）说明：
  - `top1` 丢失样本里，有 `71.4%` 的 target（目标）其实仍然留在 `R630c top10`
  - `top10` 丢失样本里，有 `53.4%` 的 target（目标）其实仍然留在 `R630c top50`
  - 这说明很多失败不是“完全找不到”，而是“排不到前面去”
- 同时，tokenizer-side structure（分词器侧结构）并不能直接解释这些输出失败：
  - 在 `v2_on_p05 -> R630c` 的 `top10` 丢失样本里，有 `79.3%` 的样本其 hierarchy-side `l2 fanout`（层级侧 `l2` 扇出）并没有变大
  - 也就是说，哪怕局部结构更干净，输出依然可能更差
- 更细的 loss-item graph/semantic analysis（损失物品图/语义分析）进一步说明：
  - `top10` loss（`top10` 损失）里，`3d_filament`（`3D` 打印耗材）家族占 `42.5%`
  - 最典型的 loss hotspot（损失热点）是 `3D Solutech` 多颜色 `PLA` 变体
  - 这些物品的语义近邻非常密：
    - 例如 `3475 / 3522 / 3494 / 2697` 的 `semantic_density`（语义密度）都接近 `0.98`
  - 但这些语义近邻在当前图里几乎不可见：
    - 对 `3475` 来说，前 `6` 个语义近邻在 `coarse / mid / local`（粗图 / 中图 / 局部图）里的亲和几乎全是 `0`
    - 它自己的 `semantic_topk_zero_mid_fraction`（语义近邻零中图占比）就是 `1.0`
  - 更关键的是：
    - 这类最差 hotspot（热点）很多甚至**不在** `weak pair`（弱连接物品对）里
    - 说明它们并不是因为被 `push`（推远）过度打散才失败
    - 而是因为语义邻域本来就没有被当前 graph carrier（图载体）有效看到
- 与此相对，gain items（增益物品）虽然同样经常处在高语义密度区域，但：
  - 它们平均有更高的 `semantic_topk_mean_mid_affinity`（语义近邻中图平均亲和）
  - 更高的 `semantic_topk_graph_overlap_fraction`（语义/中图近邻重叠占比）
  - 以及更高的 `weak_pair_endpoint_count`（弱连接对端点数）
  - 说明当前方法只有在“语义近邻至少有一点图可见性”的时候，才更可能把局部收益传到输出侧
- 所以当前最重要的经验不是“继续看 tokenizer cleaner 不 cleaner（更不更干净）”，而是：
  - **输出侧真正失败的是 neighborhood retention（邻域保留）/ beam retention（候选束保留），而不是单纯 same-prefix confusion（同前缀混淆）。**
  - **对最差热点来说，当前更具体的问题甚至不是 `push`（推远）本身，而是 dense semantic variant neighborhoods（稠密语义变体邻域）在 graph carrier（图载体）里基本不可见。**
