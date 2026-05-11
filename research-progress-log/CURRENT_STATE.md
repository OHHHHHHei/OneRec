# Current State（当前状态）

Status（状态）: `canonical（权威）`

Last updated（更新日期）: `2026-05-11`

## One-Line State（一句话状态）

The current evidence line（当前证据线） is still:

> `r690b_lmh_l2_contrastive_pull_weight001`

It is a meaningful tokenizer-side（分词器侧） positive signal on Industrial SFT（Industrial 监督微调）, but not yet a clean overall winner（整体胜者）:

- Industrial SFT（监督微调） improves NDCG（归一化折损累计增益） at @1/@3/@5/@10 over the strongest original MiniOneRec SFT baseline（原版最强监督微调基线）.
- Industrial HR（命中率） is mixed, especially HR@10（命中率@10） is lower than the strongest original SFT baseline.
- Industrial RL（强化学习） does not beat the strongest original MiniOneRec RL baseline（原版最强强化学习基线）.
- Office SFT（Office 监督微调） gives small NDCG gains but mixed HR.
- Toys SFT（Toys 监督微调） is negative.

So the current claim should be conservative（保守）:

> Local multihop collaborative shaping（局部多跳协同塑形） can improve SID tokenizer（语义标识分词器） learnability（可学习性） in SFT on Industrial, but the effect is not yet robust enough to claim broad or RL-level superiority.

## Core Research Question（核心研究问题）

Current phrasing（当前表述）:

> Can hierarchical collaborative information（层级协同信息） be injected into SID tokenizer construction（语义标识分词器构建） so that downstream recommendation models（下游推荐模型） learn item routing（物品路由） more easily than with the original semantic-only SID（原版纯语义标识）?

Working answer（阶段性答案）:

- Heavy RQ-VAE graph propagation（重残差量化变分自编码器图传播） remains unsupported.
- Broad mid-graph carriers（宽中层图载体） and `fagsp_mid_base`-style variants remain unreliable.
- The strongest useful signal so far is low-disturbance local multihop shaping（低扰动局部多跳塑形） around the L2/L3 SID hierarchy（第二/三层语义标识层级）.
- Tokenizer proxy metrics（分词器代理指标） are useful for diagnosis, but downstream SFT/RL（监督微调/强化学习） remains the final judge（最终裁决）.

## Current Evidence Line（当前证据线）

Tokenizer（分词器）:

- method name（方法名）: `LMH-HCSID`（局部多跳层级协同语义标识）
- `r690b_lmh_l2_contrastive_pull_weight001`

Method interpretation（方法解释）:

- L1（第一层） keeps coarse semantic routing（粗语义路由）.
- L2（第二层） uses weak local multihop collaborative signal（弱局部多跳协同信号）.
- L3（第三层） keeps local fine-grained refinement（局部细粒度修正）.
- The method is best described as hierarchical collaborative SID shaping（层级协同语义标识塑形）, not as a new downstream model architecture（下游模型架构）.

Main branch document（主线分支文档）:

- [MAINLINE.md](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/MAINLINE.md)

Current tokenizer code（当前分词器代码）:

- [hcsid/trainer.py](/home/leejt/OneRec/src/onerec/experiments/hcsid/trainer.py)
- [hcsid/train_entry.py](/home/leejt/OneRec/src/onerec/experiments/hcsid/train_entry.py)
- [sid_train_industrial_lmh_hcsid.yaml](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/configs/hcsid/sid_train_industrial_lmh_hcsid.yaml)

Current report draft（当前汇报草稿）:

- [main.tex](/home/leejt/OneRec/research-progress-log/advisor_reports/2026-05-11_mainline_vs_baseline_multidataset_results/main.tex)

## Industrial Results（Industrial 结果）

Single-run SFT comparison（单次监督微调对比）:

| Method（方法） | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | HR@1 | HR@3 | HR@5 | HR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original SFT baseline（原版监督微调基线） | 0.067064 | 0.085008 | 0.093153 | 0.103720 | 0.067064 | 0.098390 | 0.118244 | 0.150893 |
| Current tokenizer SFT（当前分词器监督微调） | 0.070593 | 0.088131 | 0.094889 | 0.104383 | 0.070593 | 0.100816 | 0.117362 | 0.146923 |

Repeated-run SFT mean（多次重复监督微调均值）:

| Method（方法） | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | HR@1 | HR@3 | HR@5 | HR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original SFT mean（原版监督微调均值） | 0.06574 | 0.08337 | 0.09089 | 0.10201 | 0.06574 | 0.09655 | 0.11486 | 0.14935 |
| Current tokenizer SFT mean（当前分词器监督微调均值） | 0.07015 | 0.08590 | 0.09317 | 0.10291 | 0.07015 | 0.09736 | 0.11508 | 0.14538 |

Industrial RL comparison（Industrial 强化学习对比）:

| Method（方法） | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | HR@1 | HR@3 | HR@5 | HR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original RL baseline（原版强化学习基线） | 0.073241 | 0.089032 | 0.097045 | 0.107263 | 0.073241 | 0.100375 | 0.119788 | 0.151335 |
| Current tokenizer RL（当前分词器强化学习） | 0.073020 | 0.087362 | 0.094663 | 0.105132 | 0.073020 | 0.097948 | 0.115597 | 0.148026 |

Interpretation（解释）:

- SFT（监督微调） gain is real but modest.
- Repeated runs（重复实验） show NDCG gains are more stable than HR gains.
- RL（强化学习） does not currently promote this method into a strongest-result claim（最强结果主张）.

## Transfer Evidence（迁移证据）

Office SFT（Office 监督微调）:

- NDCG@1/@3/@5/@10 all improve slightly.
- HR@1/@3/@5 improve, while HR@10 drops slightly.
- This supports possible transfer（迁移） of the tokenizer idea, but only weakly.

Toys SFT（Toys 监督微调）:

- Current tokenizer is below baseline on NDCG/HR（归一化折损累计增益/命中率） at @1/@3/@5/@10.
- This is a clear negative transfer（负迁移） warning.

## Documentation Map（文档地图）

Live canonical docs（仍需维护的权威文档）:

- [CURRENT_STATE.md](/home/leejt/OneRec/research-progress-log/CURRENT_STATE.md)
- [experiment_registry/README.md](/home/leejt/OneRec/research-progress-log/experiment_registry/README.md)
- [MAINLINE.md](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/MAINLINE.md)

Snapshot docs（快照文档）:

- `research-progress-log/advisor_reports/`
- older dated reports（旧日期报告）
- branch-level trackers（分支级追踪表）

Maintenance rule（维护规则）:

- Finalized results（定稿结果） go to split registries（分表总账） first.
- Current interpretation（当前解释） goes here.
- Branch details（分支细节） go to `MAINLINE.md`.
- Advisor-facing narratives（给导师看的叙事） stay under `advisor_reports/` and should not be treated as live state（实时状态）.

## Closed Or Deprioritized Lines（关闭或降优先级路线）

Still closed or low priority（仍关闭或低优先级）:

- Heavy RQ-VAE graph propagation（重残差量化变分自编码器图传播）.
- QCR-L2 conflict ranking（量化冲突感知第二层排序） as previously tested.
- `fagsp_mid_base` and broad mid graph（宽中图） carriers.
- Hard L1 capacity reduction（硬第一层容量压缩）.
- L3 ranking loss（第三层排序损失） under the tested setting.
- Removing L1 semantic pull（移除第一层语义拉近）.
- Recipe setting `title_history2sid_off + desc_align_p05` for the current tokenizer（当前分词器）.

## Next Checkpoint（下一检查点）

The next documentation update should happen only when one of these changes（仅在这些变化发生时更新）:

1. A new finalized SFT/RL result（定稿监督微调/强化学习结果） changes the current best evidence.
2. A method formula（方法公式） or code-aligned implementation（代码对齐实现） changes.
3. The active research question（活跃研究问题） or next-step decision（下一步决策） changes.
4. A report for the advisor（导师汇报） is finalized and should be linked as a snapshot（快照）.
