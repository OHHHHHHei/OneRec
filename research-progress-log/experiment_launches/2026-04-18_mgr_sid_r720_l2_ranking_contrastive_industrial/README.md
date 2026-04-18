# R720a L2 Ranking Contrastive SID

Status（状态）: `implemented_smoke_passed（已实现，冒烟通过）`

Date（日期）: `2026-04-18`

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

## Current Decision（当前决策）

`R720a` 已具备正式 tokenizer train -> generate（分词器训练到生成）的代码条件。当前活跃入口已经改为语义化命名；下一步如果推进，应直接用 `launch_mgr_sid_collab_ranking_tmux.sh` 启动完整训练；完成后再根据 generated SID（生成 SID）和 downstream SFT/evaluate（下游监督微调/评测）判断。
