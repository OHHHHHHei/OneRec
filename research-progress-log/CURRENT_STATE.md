# Current State（当前状态）

Status（状态）: `canonical（权威）`

Last updated（更新日期）: `2026-04-20`

## One-Line State（一句话状态）

当前 active mainline（活跃主线）已经收敛为：

`R720b: collab-ranking SID with local_multihop mid graph`（使用 local_multihop 中层图的协同排序 SID）。

后续不要再横向发散新主线；只围绕 `R720b/R720e` 做小范围微调，例如 loss weight（损失权重）、margin（间隔）、positive/negative pair construction（正负样本构造）和 graph source（图来源）。当前最新、也是 collab-ranking（协同排序）方法族内最强的 SFT（监督微调）候选是 `R720e`：在 `R720b` 基座上只把 `L1`（第一层）粗图加权改成 inverse ambiguity（逆歧义）。

## Core Problem（核心问题）

我们的目标不是让新 SID space（SID 空间）靠近旧 baseline（基线），而是构造更好的 SID codebook space（SID 码本空间），让 fresh downstream SFT（全新下游监督微调）和后续 RL（强化学习）更容易学出推荐能力。

目前最清晰的问题表述是：

> 语义相近不等于协同相近。SID 的中层需要学会：在语义相近的候选里，协同正样本应该比协同弱样本更接近。

这对应当前 collab-ranking（协同排序）主线的核心约束：

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
\mathcal L_{\mathrm{collab\_ranking}}
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
- `L2`（第二层）负责 collaborative branching（协同分叉），使用 ranking contrastive loss（排序对比损失）；当前活跃变体的 positive graph（正样本图）是 `local_multihop`（局部多跳图）。
- `L3`（第三层）负责 local refinement（局部细化），使用 `local_purified`（净化局部图）做轻量 graph pull（图拉近）。

当前明确关闭：

- `mid_weight = 0.0`，不叠加 `L2` graph smoothness（第二层图平滑）。
- `semantic_coarse_weight = 0.0`，`semantic_mid_weight = 0.0`，不额外叠加 semantic retention（语义保持）。
- `selective_separation_weight = 0.0`，不再使用旧的 selective separation（选择性分离）接口。

## Code Entry Points（代码入口）

当前活跃变体配置和脚本：

- `/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_collab_ranking_local_multihop_mid.yaml`
- `/home/leejt/OneRec/scripts/launch_mgr_sid_collab_ranking_local_multihop_mid_tmux.sh`
- `/home/leejt/OneRec/scripts/experiment_mgr_sid_collab_ranking_local_multihop_mid_train_generate.sh`
- `/home/leejt/OneRec/scripts/experiment_mgr_sid_collab_ranking_pair_source.py`
- `/home/leejt/OneRec/scripts/experiment_mgr_sid_collab_ranking_train.py`
- `/home/leejt/OneRec/src/onerec/experiments/mgr_sid/train_collab_ranking_sid.py`

主线阶段文档：

- `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-19_mgr_sid_collab_ranking_local_multihop_mid_industrial/README.md`

代码对齐公式：

- `/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/19_mgr_sid_current_method_code_aligned_formulas.md`

## Repository Cleanup（仓库整理状态）

旧实验入口已经归档：

- old configs（旧配置）: `/home/leejt/OneRec/config/archive/2026-04-18_pre_r720_legacy_experiments/`
- old scripts（旧脚本）: `/home/leejt/OneRec/scripts/archive/pre_r720_legacy_experiments_20260418/`
- previous current-state snapshot（旧当前状态快照）: `/home/leejt/OneRec/research-progress-log/archive/2026-04-18_pre_r720_state_cleanup/CURRENT_STATE_before_r720_cleanup.md`

注意：`R690b` 的 SFT/evaluate（监督微调/评测）已经结束并完成结果登记，但相关 legacy（历史）入口仍暂时保留，便于回溯与复核：

- `/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_r690b_hier_cost_guided.yaml`
- `/home/leejt/OneRec/config/experiments/sft_industrial_mgr_r690b_title_on_desc_p05.yaml`
- `/home/leejt/OneRec/config/experiments/evaluate_industrial_mgr_r690b_title_on_desc_p05.yaml`
- `/home/leejt/OneRec/scripts/experiment_mgr_sid_r690b_hier_cost_guided_train_generate.sh`
- `/home/leejt/OneRec/scripts/experiment_mgr_sid_r690b_sft_eval_chain.sh`
- `/home/leejt/OneRec/scripts/launch_mgr_sid_r690b_sft_eval_tmux.sh`

## Current Evidence（当前证据）

当前 collab-ranking（协同排序）主线已经有两轮关键证据：

- `R720a`（`fagsp_mid_base` 作为中层图）:
  - tokenizer（分词器）结果非灾难性，但有明显 `L2 compression`（第二层压缩）:
    - generated collision（生成冲突）: `14 / 3686 = 0.0037981552`
    - active L1（活跃第一层码）: `88`
    - unique L2 pairs（唯一第二层前缀数）: `1558`
  - downstream SFT/evaluate（下游监督微调/评测）为负结果:
    - `NDCG@10 = 0.09235`
    - `HR@10 = 0.13016`
- `R720b`（只把中层图换成 `local_multihop`）:
  - tokenizer（分词器）结构明显展开:
    - generated collision（生成冲突）: `11 / 3686 = 0.0029842648`
    - active L1（活跃第一层码）: `247`
    - unique L2 pairs（唯一第二层前缀数）: `2889`
  - downstream SFT/evaluate（下游监督微调/评测）明显优于 `R720a`:
    - `NDCG@10 = 0.09940`
    - `HR@10 = 0.13942`
    - 相比 `R720a`, `NDCG@10 +0.00706`, `HR@10 +0.00927`
  - strongest recipe（最强配方）补充验证为负:
    - `title_history2sid_off + desc_align_p05` 下 `NDCG@10 = 0.09807`
    - `HR@10 = 0.13920`
    - 相比 `R720b` 的 `title_history2sid_on + desc_align_p05`, `NDCG@10 -0.00133`, `HR@10 -0.00022`
    - 仍低于 strongest original SFT（原版最强监督微调）:
      - `NDCG@10 -0.00565`
      - `HR@10 -0.01169`

这说明当前最关键的正信号不是“collab-ranking（协同排序）主线已经赢了”，而是：

- `local_multihop`（局部多跳图）比 `fagsp_mid_base`（基础中层图）更适合作为 `L2`（第二层）载体
- `R720b` 已经是当前 collab-ranking（协同排序）主线内的 screening winner（筛选胜出版本）
- 但它仍未超过 `v2_on_p05` 或 strongest original SFT（原版最强 SFT），所以还不是 strongest validated line（最强已验证线）
- strongest recipe（最强配方）没有把 `R720b` 救回来，所以当前差距不能主要归因于 recipe mismatch（配方错配）
- `R720d`（把 `L1` 图载体从 `coarse_purified` 换成 `prism_anchor_coarse`）已经给出明确负结果：
  - generated collision（生成冲突）: `1755 / 3686 = 0.47613`
  - active L1（活跃第一层码）: `2`
  - unique L2 pairs（唯一第二层前缀数）: `20`
  - 一个超大根码 `<a_146>` 吸收了 `3270` 个 item（物品），说明当前 `prism_anchor_coarse` 作为 `L1`（第一层）载体会导致灾难性塌缩
- `R720e`（只把 `L1` coarse pull（粗图拉近）从正向歧义改成逆歧义）给出目前最干净的 `L1` 修复正信号：
  - generated collision（生成冲突）: `13 / 3686 = 0.0035268584`
  - active L1（活跃第一层码）: `190`
  - unique L2 pairs（唯一第二层前缀数）: `2695`
  - `L1 buckets <= 10`（第一层小桶数）从 `R720b / R720c` 的 `70 / 50` 降到 `17`
  - `L1 median bucket size`（第一层中位桶大小）从 `13 / 14` 提升到 `18`
  - 重点家族（focus families）上，`gauge_meter / connector_fitting / tape` 的同 `L1` 配对占比明显改善
  - downstream SFT/evaluate（下游监督微调/评测）也给出正收益：
    - `NDCG@1/3/5/10 = 0.06398 / 0.08397 / 0.09185 / 0.10094`
    - `HR@1/3/5/10 = 0.06398 / 0.09883 / 0.11802 / 0.14604`
    - 相比 `R720b`, `NDCG@10 +0.00154`, `HR@10 +0.00662`
    - 相比 `R720a`, `NDCG@10 +0.00860`, `HR@10 +0.01588`
    - 但仍略低于 `v2_on_p05`（`NDCG@10 -0.00176`, `HR@10 -0.00022`）和 strict recipe-aligned original baseline（严格配方对齐原版基线，`NDCG@10 -0.00088`, `HR@10 -0.00022`）
  - 这是当前 collab-ranking（协同排序）方法族内的 strongest SFT candidate（最强监督微调候选），但还不是 RL-promotable（可推进强化学习）的最终版本

## Strongest Validated Line（最强已验证线）

当前 strongest validated line（最强已验证线）仍然是：

`v2_on_p05 -> RL`

但它现在只是 baseline/reference（基线/参考），不是当前继续迭代的主线方法。

## Next Steps（下一步）

1. 后续所有 collab-ranking（协同排序）微调都应以 `R720b` 为基座，而不是回到 `R720a`。
2. 当前最核心的 tokenizer-side（分词器侧）候选已经从 `R720b` 的原始形态推进到 `R720e`：它在不引入新 loss（损失）、不换图的前提下，给出了最清晰的 `L1`（第一层）收缩正信号，并且已经转化为 SFT（监督微调）收益。
3. 当前不应直接推进任何 collab-ranking（协同排序）分支到 `RL`（强化学习）；`R720e` 虽然是 collab-ranking（协同排序）家族内最强 SFT（监督微调）候选，但仍略低于 `v2_on_p05` 和 strict recipe-aligned original baseline（严格配方对齐原版基线）。
4. 下一步优先做 `R720e` 的 layerwise/output error analysis（分层命中率与输出错误分析），确认 SFT 收益来自 `L1` hit（第一层命中）改善，还是来自候选束覆盖/后层细分变化。
5. 后续微调应优先围绕 `R720e` 的 existing loss weighting（现有损失加权）和 `L2 ranking`（第二层排序）细节做小改；不应重新开启横向方法发散。
6. `prism_anchor_coarse`（语义锚定粗图）这条 `L1`（第一层）载体路线在当前实现下不应继续推进；后续 `L1` 修复应继续坚持“小改现有损失/加权方式”，避免再次引入会导致根码塌缩的 coarse graph source（粗图来源）。
7. `R690b` 相关 legacy（历史）入口可以在确认不再复跑后归档。

## Reading Rule（阅读规则）

任何 `R720a` 之前的 dated notes（带日期笔记）、旧 stage README（阶段说明）、旧 scripts/configs（脚本/配置）默认都是 archived provenance（归档追溯材料），不应再作为新实验起点。
