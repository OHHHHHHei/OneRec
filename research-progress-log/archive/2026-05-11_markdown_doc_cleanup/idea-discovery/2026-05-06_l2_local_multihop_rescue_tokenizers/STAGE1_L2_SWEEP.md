# Stage 1 L2 Sweep Tracker

Status（状态）: `snapshot（快照）`

Last updated（更新日期）: `2026-05-11`

## Purpose（目的）

This tracker（追踪表） records the first-stage L2 hyperparameter sweep（第二层超参数扫描） around the current R690b local-multihop anchor（局部多跳锚点）.

This file is now a snapshot（快照）. Current interpretation（当前解释） lives in:

- [CURRENT_STATE.md](/home/leejt/OneRec/research-progress-log/CURRENT_STATE.md)
- [MAINLINE.md](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/MAINLINE.md)

Finalized results（定稿结果） should be checked in the split registry（分表总账） rather than inferred from this tracker（追踪表）.

## Fixed Setup（固定设置）

- Base method（基础方法）: `r690b_lmh_l2_contrastive_pull_weight001`
- `l1_contrastive_pull_weight = 0.03`
- `l3_contrastive_pull_weight = 0.02`
- `l2_contrastive_mode = graph_infonce`
- `l2_infonce_temperature = 0.1`
- `mid_view_name = local_multihop`
- `local_multihop_alpha = 0.35`
- `local_multihop_max_hop = 2`
- `hierarchy_stopgrad_previous_levels = true`
- RQVAE tokenizer hyperparameters（分词器超参数）: `10000 epochs`, `batch_size=20480`, `lr=0.001`, `num_emb_list=[256,256,256]`

## Tracker CSV（追踪表）

- [stage1_l2_sweep_tracker.csv](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/stage1_l2_sweep_tracker.csv)

## Final Snapshot（最终快照）

| L2 weight（第二层权重） | Status（状态） | Collision（碰撞） | Active L1（活跃第一层码） | Unique L12（唯一一二层前缀） | SFT NDCG@10 | Decision（决策） |
|---:|---|---:|---:|---:|---:|---|
| `0.003` | SFT-evaluated（已监督微调评测） | `13 / 3686` | `56` | `2182` | `0.09573735` | negative versus anchor（相对锚点负向） |
| `0.005` | SFT-evaluated（已监督微调评测） | `15 / 3686` | `131` | `2322` | `0.09435198` | negative versus anchor（相对锚点负向） |
| `0.010` | SFT-evaluated anchor（已监督微调评测锚点） | `16 / 3686` | `60` | `2330` | `0.10438342` | current anchor（当前锚点） |
| `0.015` | SFT-evaluated（已监督微调评测） | `13 / 3686` | `108` | `2619` | `0.09408004` | negative versus anchor（相对锚点负向） |
| `0.020` | generated-only / structure-reviewed（仅分词器生成 / 已结构审计） | `13 / 3686` | `45` | `2106` | - | not promoted（未晋级） |
| `0.030` | generated-only completed（仅分词器生成完成） | `1824 / 3686` | `2` | `21` | - | tokenizer no-go（分词器停止） |

## Session Note（会话备注）

The May 7 running sessions（运行会话） have finished or been superseded（已完成或被替代）. Logs remain available for traceability（可追溯性）:

- [mgr_r690b_l2_stage1_A_0507.log](/home/leejt/OneRec/logs/l2_lmh_stage1/mgr_r690b_l2_stage1_A_0507.log)
- [mgr_r690b_l2_stage1_B_0507.log](/home/leejt/OneRec/logs/l2_lmh_stage1/mgr_r690b_l2_stage1_B_0507.log)

## Stage Decision Rule（阶段决策规则）

Tokenizer-side（分词器侧） promotion screen（晋级筛选）:

- generated collision（生成碰撞） should remain near the healthy range, preferably `<= 20 / 3686`;
- max conflict（最大冲突簇） should remain `<= 2` when possible;
- active L1（活跃第一层码） should not collapse, and should be interpreted together with L1 bucket concentration（第一层桶集中度）;
- unique L12（唯一一二层前缀） should remain close to the original / anchor range;
- final SFT（监督微调） promotion should prioritize candidates that plausibly improve over the `0.010` anchor, not candidates that only improve collision proxies（碰撞代理指标）.
