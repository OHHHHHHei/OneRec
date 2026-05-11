# R720a L2 Ranking Contrastive SID

Status（状态）: `sft_evaluated（监督微调已评测）`

Date（日期）: `2026-04-19`

## Purpose（目的）

`R720a` 是当前收敛后的主线方法候选：不再继续横向堆新分支，而是把核心问题写成一个直接的 `L2`（第二层）排序约束：

在 semantic-close（语义相近）的候选物品里，collaborative-positive（协同正样本）应该比 collaborative-weak hard negative（协同弱困难负样本）在 `SID` 中层表示里更接近。

## Method（方法）

当前代码目标：

$$
\mathcal L_{\mathrm{R720a}}
=
\mathcal L_{\mathrm{rec}}
+
\mathcal L_{\mathrm{rq}}
+
0.05\,\mathcal L_{\mathrm{pull}}^{(1)}
+
0.03\,\mathcal L_{\mathrm{rank}}^{(2)}
+
0.03\,\mathcal L_{\mathrm{pull}}^{(3)}.
$$

层级分工：

- `L1`（第一层）: 使用 `coarse_purified`（净化粗图）做轻量 graph pull（图拉近），维护 coarse routing（粗粒度路由）。
- `L2`（第二层）: 使用 ranking contrastive loss（排序对比损失），让 `fagsp_mid_base`（基础中图）正样本排在 semantic-near mid-weak（语义近但中图弱连接）负样本前面。
- `L3`（第三层）: 使用 `local_purified`（净化局部图）做轻量 local refinement（局部细化）。

当前关闭项：

- `mid_weight = 0.0`，不再叠加 `L2` graph smoothness（第二层图平滑）。
- `semantic_coarse_weight = 0.0`，`semantic_mid_weight = 0.0`，不再额外叠加 semantic retention（语义保持）。
- `selective_separation_weight = 0.0`，不再使用旧的 selective separation（选择性分离）接口。

## Artifacts（产物）

- config（配置）: `/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_collab_ranking_mainline.yaml`
- train code（训练代码）: `/home/leejt/OneRec/src/onerec/experiments/mgr_sid/train_collab_ranking_sid.py`
- train runner（训练入口）: `/home/leejt/OneRec/scripts/experiment_mgr_sid_collab_ranking_train.py`
- pair source（物品对构造）: `/home/leejt/OneRec/scripts/experiment_mgr_sid_collab_ranking_pair_source.py`
- train/generate chain（训练生成链路）: `/home/leejt/OneRec/scripts/experiment_mgr_sid_collab_ranking_train_generate.sh`
- tmux launcher（终端复用器启动脚本）: `/home/leejt/OneRec/scripts/launch_mgr_sid_collab_ranking_tmux.sh`
- hard negatives（困难负样本）: `R720a_all_semantic_near_mid_weak_pairs.csv`

## Pair Source Summary（物品对摘要）

- `n_items = 3686`
- `mid_view_name = fagsp_mid_base`
- `semantic_topk = 64`
- `graph_topk = 32`
- `weak_threshold = 1e-8`
- `semantic_pair_count = 168036`
- `negative_pair_count = 159735`
- `negative_item_coverage_rate = 1.0`
- `reliability_mean = 0.0158494778`

## Smoke Check（冒烟检查）

Code checks（代码检查）已通过：

- `python -m py_compile src/onerec/experiments/mgr_sid/train_collab_ranking_sid.py scripts/experiment_mgr_sid_collab_ranking_train.py scripts/experiment_mgr_sid_collab_ranking_pair_source.py`
- `bash -n scripts/experiment_mgr_sid_collab_ranking_train_generate.sh scripts/launch_mgr_sid_collab_ranking_tmux.sh`

One-epoch smoke run（单轮冒烟运行）已通过：

- summary（摘要）: `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r720_l2_ranking_contrastive_20260418/smoke_r720a_l2_ranking/Apr-18-2026_23-21-04/summary.json`
- `l2_ranking_loss = 0.1577096283`
- `collision_rate = 0.0382528486`

## Final Tokenizer Result（最终分词器结果）

- run dir（运行目录）:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r720_l2_ranking_contrastive_20260418/industrial_r720a_l2_ranking_contrastive/Apr-18-2026_23-58-41`
- best train collision（训练最佳冲突）:
  - `0.1866521975`
- best epoch（最佳轮次）:
  - `9599`
- final eval collision（最终评估冲突）:
  - `0.1907216495`
- generated collision（生成后冲突）:
  - `14 / 3686 = 0.0037981552`
- max conflict（最大冲突簇）:
  - `2`
- active L1（活跃第一层码）:
  - `88`
- unique L2 pairs（唯一第二层前缀数）:
  - `1558`
- unique leaf codes（唯一叶子码）:
  - `3672`
- generated SID index（生成 SID 索引）:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r720_l2_ranking_contrastive_20260418/generated_indices/Industrial_and_Scientific.r720a_l2_ranking_contrastive.index.json`

## Reading（结果判读）

- 这次主线结果是 non-catastrophic（非灾难性）的。
  - 训练完整结束，`summary.json` 和 generated SID（生成 SID）都已落盘。
  - 没有出现彻底塌缩或大规模冲突爆炸。
- 但结构上出现了明显的 `L2 compression`（第二层压缩）。
  - original MiniOneRec（原版 MiniOneRec）: `active L1 = 48`, `unique L2 pairs = 2295`
  - current v2（当前 v2）: `active L1 = 203`, `unique L2 pairs = 2680`
  - `R693a`: `active L1 = 90`, `unique L2 pairs = 2274`
  - current mainline（当前主线）: `active L1 = 88`, `unique L2 pairs = 1558`
- 这说明主线方法没有把 `L1`（第一层）压得特别极端，但把 `L2`（第二层）压得过紧了。
  - 从层级分工角度看，这更像是“中层协同分叉不够展开”，而不是“粗层入口彻底坏掉”。
  - 因此它的风险点不是塌缩，而是 code space compression（码本空间压缩）导致的信息承载不足。

## Downstream SFT Result（下游监督微调结果）

- SFT output dir（监督微调输出目录）:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_collab_ranking_mainline_sft_eval_20260419/title_on_desc_p05/sft`
- best checkpoint（最佳检查点）:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_collab_ranking_mainline_sft_eval_20260419/title_on_desc_p05/sft/checkpoint-456`
- final model path（最终模型路径）:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_collab_ranking_mainline_sft_eval_20260419/title_on_desc_p05/sft/final_checkpoint`
- result JSON（结果文件）:
  - `/home/leejt/OneRec/results/experiments/mgr_sid_collab_ranking_mainline_sft_eval_20260419/final_result_sft_mgr_collab_ranking_mainline_title_on_desc_p05_Industrial_and_Scientific.json`
- W&B（实验追踪）:
  - `sft_mgr_collab_ranking_mainline_title_on_desc_p05_industrial`
  - `2ebzal1e`
- final eval loss（最终验证损失）:
  - `1.6338781118`
- final train loss（最终训练损失）:
  - `0.4765079771`
- stop epoch（停止轮次）:
  - `5.5`
- root branch count（根分支数）:
  - `88`
- constraint invalid total（约束失配总数）:
  - `0`
- `NDCG@1/3/5/10`:
  - `0.06243106 / 0.07755000 / 0.08420364 / 0.09234589`
- `HR@1/3/5/10`:
  - `0.06243106 / 0.08868299 / 0.10500772 / 0.13015663`

## Final Verdict（最终裁决）

- `R720a` 的第一次下游验证结果为 negative（负结果）。
- 它虽然没有出现 invalid constrained decoding（非法约束解码），但 `L2 compression`（第二层压缩）确实传导到了下游：
  - 相比当前 `v2_on_p05`，`NDCG@10` 低 `0.01036`
  - 相比严格 recipe-aligned original baseline（配方对齐原版基线），`NDCG@10` 低 `0.00948`
- 因此 `R720a` 不应继续作为默认推进版本，也不应推入 `RL`（强化学习）。
- 后续主线判断已经切换到 `R720b`：保持同一套 loss（损失）与 stop-gradient（停梯度）逻辑，只把 `mid graph`（中层图）从 `fagsp_mid_base` 换成 `local_multihop`。
