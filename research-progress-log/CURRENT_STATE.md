# Current State（当前状态）

Status（状态）: `canonical（权威）`

Last updated（更新日期）: `2026-05-07`

## One-Line State（一句话状态）

The active mainline（活跃主线） has shifted to:

> `R690b local_multihop L2 rescue + weak contrastive pull=0.01`（R690b 局部多跳第二层救援 + 弱对比拉近 0.01） for SID tokenizer（语义标识分词器） construction, followed by standard OneRec SFT / RL（监督微调 / 强化学习）.

Reason（原因）:

- It is the first tokenizer-side（分词器侧） line in this stage that beats the strongest original SFT baseline（原版最强监督微调基线） on `NDCG@1/@3/@5/@10`（归一化折损累计增益 @1/@3/@5/@10）.
- It also beats `v2_on_p05` SFT（监督微调） on all primary `NDCG/HR@1/@3/@5/@10`（主要排序/命中指标） checkpoints.
- Error analysis（错误分析） shows the gain is structurally meaningful: it improves rank quality（排序质量） especially when the target shares L2 prefix（第二层前缀） with historical items（历史物品）.

Current caveat（当前限制）:

- `HR@10`（命中率@10） is still below strongest original SFT（原版最强监督微调）.
- The method improves ranking depth（排序靠前程度） more than coverage（覆盖面）.
- RL/evaluate（强化学习/评测） is running and is the current decisive test（当前裁决实验）.

Branch pointer（分支指针）:

- [L2 Local Multihop Rescue Tokenizers](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/README.md)

## Current Mainline Experiment（当前主线实验）

Variant（变体）:

- `r690b_lmh_l2_contrastive_pull_weight001`

Tokenizer idea（分词器想法）:

- Start from the R690b hierarchy-cost guided tokenizer（层级代价引导分词器） family.
- Replace the stale/overweighted prior graph（旧先验图） setting with a refreshed local multihop graph（刷新后的局部多跳图）.
- Apply a weak L2 contrastive pull（弱第二层对比拉近） with weight `0.01`.
- Keep the downstream recipe（下游配方） aligned with the strongest comparable OneRec setup（OneRec 可比设置）.

SFT result（监督微调结果）:

| Metric（指标） | @1 | @3 | @5 | @10 |
| --- | ---: | ---: | ---: | ---: |
| NDCG（归一化折损累计增益） | 0.070593 | 0.088131 | 0.094889 | 0.104383 |
| HR（命中率） | 0.070593 | 0.100816 | 0.117362 | 0.146923 |

Primary comparisons（主要对比）:

- Versus strongest original SFT（原版最强监督微调）:
  - `NDCG@1/@3/@5/@10` all improve（全部提升）.
  - `HR@1/@3` improve（提升）.
  - `HR@5/@10` regress（下降）, with `HR@10 -0.003971`.
- Versus `v2_on_p05` SFT:
  - `NDCG/HR@1/@3/@5/@10` are all tied or improved（全部持平或提升）.

Interpretation（解释）:

- The model is not winning by simple hit-count expansion（命中数量扩张）.
- It wins because more correct items move into earlier ranks（正确物品被排到更靠前位置）.
- Pairwise against strongest original SFT（逐样本对比原版最强监督微调）:
  - New Hit@10（本次命中@10）: `666`
  - Original Hit@10（原版命中@10）: `684`
  - New rank-better examples（本次排名更优样本）: `268`
  - New rank-worse examples（本次排名更差样本）: `251`
  - Net `NDCG@10` improvement（净提升）: `+0.000663`

Error analysis（错误分析）:

- Strongest slice（最强切片） versus `v2_on_p05`:
  - `same_L2_seen`（历史中出现同第二层前缀）:
    - `NDCG@10 +0.044051`
    - `HR@10 +0.066845`
- Main weakness（主要弱点）:
  - `no_same_L1`（历史中没有同第一层前缀） has little gain.
  - Fine-grained sibling items（同族兄弟物品） such as filament color/material variants（耗材颜色/材料变体） are still confused.

Analysis artifacts（分析产物）:

- [report.md](/home/leejt/OneRec/results/analysis/r690b_lmh_pull001_error_analysis_20260507/report.md)
- [summary.json](/home/leejt/OneRec/results/analysis/r690b_lmh_pull001_error_analysis_20260507/summary.json)
- [slice_comparison.csv](/home/leejt/OneRec/results/analysis/r690b_lmh_pull001_error_analysis_20260507/slice_comparison.csv)

## Current Running Experiment（当前运行实验）

RL/evaluate（强化学习/评测） is running:

- tmux session（会话）: `mgr_r690b_lmh_pull001_rl_eval_0507`
- GPUs（显卡）: `2,3,4,5`
- W&B run（实验追踪）: `fug78gw7`
- Log（日志）: [mgr_r690b_lmh_pull001_rl_eval_0507.log](/home/leejt/OneRec/logs/l2_lmh_rl/mgr_r690b_lmh_pull001_rl_eval_0507.log)
- RL config（强化学习配置）: [rl_industrial_r690b_lmh_l2_contrastive_pull_weight001_title_on_desc_p05_4gpu.yaml](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/configs/rl_industrial_r690b_lmh_l2_contrastive_pull_weight001_title_on_desc_p05_4gpu.yaml)
- Eval config（评测配置）: [evaluate_industrial_r690b_lmh_l2_contrastive_pull_weight001_title_on_desc_p05_rl_4gpu.yaml](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/configs/evaluate_industrial_r690b_lmh_l2_contrastive_pull_weight001_title_on_desc_p05_rl_4gpu.yaml)
- Expected result（预期结果路径）: [final_result_rl_mgr_r690b_lmh_l2_contrastive_pull_weight001_title_on_desc_p05_4gpu_Industrial_and_Scientific.json](/home/leejt/OneRec/results/experiments/mgr_sid_l2_lmh_sweep_rl_eval_20260507/final_result_rl_mgr_r690b_lmh_l2_contrastive_pull_weight001_title_on_desc_p05_4gpu_Industrial_and_Scientific.json)

Do not write this RL run to `rl_registry.csv`（强化学习总账） until evaluate（评测） finishes and metrics are finalized（指标定稿）.

## Core Research Question（核心研究问题）

Current phrasing（当前表述）:

> Can collaborative hierarchy information（协同层级信息） be injected into SID tokenizer construction（语义标识分词器构建） so that the downstream model can learn item routing（物品路由） more easily than with the original OneRec SID（原版 OneRec 语义标识）?

Working answer（阶段性答案）:

- Heavy graph propagation（重图传播） and broad graph-carrier upgrades（宽图载体升级） remain unsupported（不支持）.
- Low-disturbance local hierarchy shaping（低扰动局部层级塑形） is now reopened as the most promising direction（最有希望方向）.
- The useful signal appears to be at L2 local multihop structure（第二层局部多跳结构）, not broad mid-graph spectral/carrier features（宽中图谱/载体特征）.

## Baseline Gate（基线门槛）

Primary baseline（主要基线）:

- Strongest original OneRec SFT（原版最强监督微调）:
  - `NDCG@10 = 0.10372025`
  - `HR@10 = 0.15089345`

Upper reference（上界参考）:

- Strongest original OneRec RL（原版最强强化学习）:
  - `NDCG@10 = 0.10726345`
  - `HR@10 = 0.15133466`

Current mainline SFT（当前主线监督微调）:

- `NDCG@10 = 0.10438342`
- `HR@10 = 0.14692257`

Decision rule（决策规则）:

- SFT（监督微调） has established a real positive signal（真实正向信号）, especially on `NDCG@1/@3/@5/@10`.
- RL（强化学习） decides whether this becomes a strong main result（强主结果） or remains a promising SFT-only lead（仅监督微调有希望线索）.
- A clean RL win should improve or at least preserve the SFT ranking gain（排序增益） while recovering `HR@10`（命中率@10）.

## Closed Or Deprioritized Lines（关闭或降优先级路线）

Still closed under current evidence（在当前证据下仍关闭）:

- Heavy RQ-VAE graph propagation（重残差量化变分自编码器图传播）.
- QCR-L2 conflict ranking（量化冲突感知第二层排序） as previously tested.
- FaGSP-mid-base style broad mid graph（宽中图） carriers.
- Hard L1 capacity reduction（硬第一层容量压缩）.
- AttnRQ reconstruction-only residual weighting（仅重构路径注意力残差加权） as a mainline, unless reused in a more directly downstream-aligned tokenizer design（更直接对齐下游的分词器设计）.

Reopened with constraints（有条件重启）:

- L2 local multihop（第二层局部多跳） structure.
- Weak pull strength（弱拉近强度） around `0.01`, not the earlier overweighted `0.15` setup.
- R690b-style hierarchy-aware codebook shaping（层级感知码本塑形）, if it preserves downstream learnability（下游可学习性）.

## Next Steps（下一步）

1. Monitor the running RL/evaluate（强化学习/评测） chain and finalize metrics（定稿指标） into `rl_registry.csv` and `downstream_scoreboard.csv`.
2. If RL improves over the strongest original RL（原版最强强化学习） or gives a clear `NDCG@10` gain without further HR collapse（命中率坍塌）, promote this branch to the strongest result line（最强结果线）.
3. If RL is mixed, analyze whether RL recovers coverage（覆盖面） or amplifies sibling confusion（兄弟物品混淆） before deciding between:
   - nearby pull-weight ablation（近邻拉近权重消融）: `0.005 / 0.02`
   - weak sibling separation（弱兄弟物品分离）
   - preserving this as SFT-only evidence（仅监督微调证据）
