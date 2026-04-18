# Current State（当前状态）

Status（状态）: `canonical（权威）`

Last updated（更新日期）: `2026-04-18`

## One-Line State（一句话状态）

当前 active mainline（活跃主线）已经收敛为：

`R720a: ambiguity-aware stop-gradient L2 ranking contrastive SID`（歧义感知停止梯度中层排序对比 SID）。

后续不要再横向发散新主线；只围绕 `R720a` 做小范围微调，例如 loss weight（损失权重）、margin（间隔）、positive/negative pair construction（正负样本构造）和 graph source（图来源）。

## Core Problem（核心问题）

我们的目标不是让新 SID space（SID 空间）靠近旧 baseline（基线），而是构造更好的 SID codebook space（SID 码本空间），让 fresh downstream SFT（全新下游监督微调）和后续 RL（强化学习）更容易学出推荐能力。

目前最清晰的问题表述是：

> 语义相近不等于协同相近。SID 的中层需要学会：在语义相近的候选里，协同正样本应该比协同弱样本更接近。

这对应 `R720a` 的核心约束：

$$
s_{ip}^{(2)}
\ge
s_{in}^{(2)}
+
m,
$$

其中 $p$ 是 collaborative-positive item（协同正样本），$n$ 是 semantic-near but collaborative-weak hard negative（语义近但协同弱困难负样本）。

## Main Method（主线方法）

当前主线训练目标是：

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

- `L1`（第一层）负责 coarse routing（粗粒度路由），使用 `coarse_purified`（净化粗图）做轻量 graph pull（图拉近）。
- `L2`（第二层）负责 collaborative branching（协同分叉），使用 ranking contrastive loss（排序对比损失）。
- `L3`（第三层）负责 local refinement（局部细化），使用 `local_purified`（净化局部图）做轻量 graph pull（图拉近）。

当前明确关闭：

- `mid_weight = 0.0`，不叠加 `L2` graph smoothness（第二层图平滑）。
- `semantic_coarse_weight = 0.0`，`semantic_mid_weight = 0.0`，不额外叠加 semantic retention（语义保持）。
- `selective_separation_weight = 0.0`，不再使用旧的 selective separation（选择性分离）接口。

## Code Entry Points（代码入口）

主线配置和脚本：

- `/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_collab_ranking_mainline.yaml`
- `/home/leejt/OneRec/scripts/launch_mgr_sid_collab_ranking_tmux.sh`
- `/home/leejt/OneRec/scripts/experiment_mgr_sid_collab_ranking_train_generate.sh`
- `/home/leejt/OneRec/scripts/experiment_mgr_sid_collab_ranking_pair_source.py`
- `/home/leejt/OneRec/scripts/experiment_mgr_sid_collab_ranking_train.py`
- `/home/leejt/OneRec/src/onerec/experiments/mgr_sid/train_collab_ranking_sid.py`

主线阶段文档：

- `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-18_mgr_sid_r720_l2_ranking_contrastive_industrial/README.md`

代码对齐公式：

- `/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/19_mgr_sid_current_method_code_aligned_formulas.md`

## Repository Cleanup（仓库整理状态）

旧实验入口已经归档：

- old configs（旧配置）: `/home/leejt/OneRec/config/archive/2026-04-18_pre_r720_legacy_experiments/`
- old scripts（旧脚本）: `/home/leejt/OneRec/scripts/archive/pre_r720_legacy_experiments_20260418/`
- previous current-state snapshot（旧当前状态快照）: `/home/leejt/OneRec/research-progress-log/archive/2026-04-18_pre_r720_state_cleanup/CURRENT_STATE_before_r720_cleanup.md`

注意：`R690b` 的 SFT/evaluate（监督微调/评测）tmux（终端复用器）仍在运行，因此以下 legacy（历史）入口暂时保留在明面，等运行结束并完成结果登记后再归档：

- `/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_r690b_hier_cost_guided.yaml`
- `/home/leejt/OneRec/config/experiments/sft_industrial_mgr_r690b_title_on_desc_p05.yaml`
- `/home/leejt/OneRec/config/experiments/evaluate_industrial_mgr_r690b_title_on_desc_p05.yaml`
- `/home/leejt/OneRec/scripts/experiment_mgr_sid_r690b_hier_cost_guided_train_generate.sh`
- `/home/leejt/OneRec/scripts/experiment_mgr_sid_r690b_sft_eval_chain.sh`
- `/home/leejt/OneRec/scripts/launch_mgr_sid_r690b_sft_eval_tmux.sh`

## Current Evidence（当前证据）

`R720a` 已完成 implementation smoke check（实现冒烟检查）：

- hard negative pairs（困难负样本对）: `159735`
- item coverage（物品覆盖率）: `1.0`
- one-epoch smoke run（单轮冒烟运行）通过
- smoke `l2_ranking_loss = 0.1577096283`
- smoke `collision_rate = 0.0382528486`

这只证明代码链路可运行，不代表方法已被验证。最终裁决仍然必须依赖 downstream evaluate（下游评测）。

## Strongest Validated Line（最强已验证线）

当前 strongest validated line（最强已验证线）仍然是：

`v2_on_p05 -> RL`

但它现在只是 baseline/reference（基线/参考），不是当前继续迭代的主线方法。

## Next Steps（下一步）

1. 等当前 `R690b` legacy SFT/evaluate（历史监督微调/评测）结束，记录结果，然后归档它的剩余入口。
2. 启动完整 `R720a tokenizer train -> generate`（分词器训练到生成）。
3. 若 `R720a` generated SID（生成 SID）非灾难性，再推进对齐 recipe（配方）的 `SFT -> evaluate`（监督微调到评测）。

## Reading Rule（阅读规则）

任何 `R720a` 之前的 dated notes（带日期笔记）、旧 stage README（阶段说明）、旧 scripts/configs（脚本/配置）默认都是 archived provenance（归档追溯材料），不应再作为新实验起点。
