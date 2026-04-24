# Current State（当前状态）

Status（状态）: `canonical（权威）`

Last updated（更新日期）: `2026-04-21`

## One-Line State（一句话状态）

当前 active mainline（活跃主线）已经从横向发散收敛为 diagnostic-driven minimal-edit collaborative injection（诊断驱动的最小编辑协同注入）验证线：

- `R720e`: collab-ranking SID（协同排序 SID）里当前最强的已入账 SFT（监督微调）候选。
- `original_l3_collab_local`: original RQ-VAE base（原版残差量化变分自编码器基座）上只加 `L3`（第三层）local collaborative pull（局部协同拉近）的 minimal-edit（最小编辑）正信号。
- `original_l2_multihop_ranking`: original RQ-VAE base（原版残差量化变分自编码器基座）上只加 `L2`（第二层）local_multihop ranking（局部多跳排序）的当前最强 minimal-edit（最小编辑）SFT（监督微调）结果。

后续不要再横向发散新主线；下一步先围绕 `original_l2_multihop_ranking`（原版第二层多跳排序）完成 beam / prefix diagnostics（束搜索 / 前缀诊断）和单因素 L2（第二层）扫描，而不是直接叠加 `L2 + L3`（第二层 + 第三层）组合。

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

上一轮 collab-ranking（协同排序）组合目标是：

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

Current diagnostic conclusion（当前诊断结论）: this full combination（完整组合） is no longer the immediate next action（立即下一步行动）. Since the project mainly optimizes `@1/@3/@5/@10`（主要评测截断）, the current priority（当前优先级） is not to chase `HR@50`（命中率@50） tail retention（尾部保留）, but to understand why the minimal `L2`（第二层） route-preserving ranking（路由保持排序） fails to improve top ranking / calibration（头部排序 / 校准）.

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
- `R720f`（`K1=128` 的 hard L1 capacity reduction（硬性第一层容量缩减））给出明确负结果：
  - generated collision（生成冲突）: `11 / 3686 = 0.0029842648`
  - downstream SFT/evaluate（下游监督微调/评测）:
    - `NDCG@10 = 0.08955`
    - `HR@10 = 0.13060`
  - 说明硬性压缩 `L1` capacity（第一层容量）即便改善 compactness proxy（紧凑性代理指标），也会严重伤害 downstream learnability（下游可学习性）
- `original_l3_collab_local`（原版基座 + 第三层局部协同）给出最新 minimal-edit（最小编辑）正信号：
  - tokenizer（分词器）:
    - generated collision（生成冲突）: `13 / 3686 = 0.0035268584`
    - max conflict（最大冲突簇）: `2`
  - downstream SFT/evaluate（下游监督微调/评测）:
    - `NDCG@1/3/5/10 = 0.06684 / 0.08445 / 0.09226 / 0.10159`
    - `HR@1/3/5/10 = 0.06684 / 0.09773 / 0.11692 / 0.14604`
    - 相比 strongest original SFT（原版最强监督微调）: `NDCG@10 -0.00213`, `HR@10 -0.00485`
    - 相比 `v2_on_p05` SFT（当前 v2_on_p05 监督微调）: `NDCG@10 -0.00112`, `HR@10 -0.00022`
    - 相比 `R720e` SFT（R720e 监督微调）: `NDCG@10 +0.00065`, `HR@10 +0.00000`
  - 这条线仍不应直接推进 RL（强化学习），但它说明 original SID routeability（原版 SID 可路由性）很重要，协同信息可能更适合以少扰动方式进入 `L2/L3`（第二/第三层）
- `original_l2_multihop_ranking`（原版基座 + 第二层局部多跳排序）给出当前最强 minimal-edit（最小编辑）SFT（监督微调）信号：
  - tokenizer（分词器）:
    - generated collision（生成冲突）: `15 / 3686 = 0.0040694520`
    - max conflict（最大冲突簇）: `2`
    - active L1（活跃第一层码）: `88`
    - unique L2 pairs（唯一第二层前缀数）: `2449`
  - downstream SFT/evaluate（下游监督微调/评测）:
    - `NDCG@1/3/5/10 = 0.06618 / 0.08283 / 0.09144 / 0.10165`
    - `HR@1/3/5/10 = 0.06618 / 0.09486 / 0.11582 / 0.14736`
    - 相比 strongest original SFT（原版最强监督微调）: `NDCG@10 -0.00207`, `HR@10 -0.00353`
    - 相比 `v2_on_p05` SFT（当前 v2_on_p05 监督微调）: `NDCG@10 -0.00106`, `HR@10 +0.00110`
    - 相比 `original_l3_collab_local` SFT（原版第三层局部协同监督微调）: `NDCG@10 +0.00006`, `HR@10 +0.00132`
    - 相比 `R720e` SFT（R720e 监督微调）: `NDCG@10 +0.00071`, `HR@10 +0.00132`
  - 这条线仍未达到 strongest original SFT（原版最强监督微调）或 `v2_on_p05` SFT（当前 v2_on_p05 监督微调）的 `NDCG@10`，所以不应直接推进 RL（强化学习）；但它支持“协同信息下沉到 `L2/L3`（第二/第三层）且少扰动原版可路由性”的判断。
  - 2026-04-21 bootstrap / proxy diagnostics（自助法 / 代理诊断）进一步收窄了这个判断：
    - paired bootstrap（配对自助法）显示相对 recipe-aligned original（配方对齐原版）只有 `HR@50 +0.01368` 是稳定信号，95% CI（置信区间）为 `[0.00397, 0.02338]`，approximate p（近似 p 值）为 `0.0048`；但 `HR@50` 只是 secondary diagnostic signal（次级诊断信号），不是主要成功标准
    - `NDCG@10 -0.00018` 和 `HR@10 +0.00110` 的 confidence interval（置信区间）都跨 0，不能作为稳定收益；因此按 `@1/@3/@5/@10`（主要评测截断）口径，这条线还没有给出 primary downstream win（主要下游胜利）
    - final Top50 proxy（最终前 50 代理诊断）显示 `GT L2 covered@50`（真实目标第二层前缀覆盖率@50）提高到 `0.32760`，但 `GT L2 covered@10`（真实目标第二层前缀覆盖率@10）没有提高，mean hit rank@50（前 50 命中平均排名）反而变差到 `13.73`
    - primary-cutoff rerun（主要截断重诊断）确认：相对 recipe-aligned original（配方对齐原版），`original_l2` 在 `NDCG/HR@1/@3/@5/@10`（主要指标）上没有任何稳定正收益；相对 strongest original SFT（原版最强监督微调），所有 primary raw metrics（主要原始指标）都更低
    - final Top10 proxy（最终前 10 代理诊断）显示 `GT L2 covered@1/3/5/10`（真实目标第二层前缀覆盖率@1/3/5/10）为 `0.11052 / 0.16016 / 0.17935 / 0.20671`，低于 recipe-aligned original（配方对齐原版）的 `0.13016 / 0.16391 / 0.18266 / 0.20869`，所以不能说它改善了前 10 区域的 target prefix survival（目标前缀存活）
    - tokenizer prefix diagnostics（分词器前缀诊断）没有显示 global graph-neighbor prefix sharing（全局图邻居前缀共享）提升；recipe-aligned original（配方对齐原版）的 graph-neighbor L2 overlap（图邻居第二层前缀重合率）仍更高
    - code audit（代码审查）确认该配置已经使用 `hierarchy_stopgrad_previous_levels = true`，`L2` ranking representation（第二层排序表示）已经是 `detach(q1) + q2`，所以单纯重做 `route_preserving_teacher_rank`（路由保持教师排序）不是有意义的新机制
- `original_l3_ambiguity_aware`（原版第三层歧义感知）给出明确 tokenizer-side no-go（分词器侧停止）结果：
  - intended change（意图变化）: compared with `original_l3_collab_local`（原版第三层局部协同）, only change `graph_scale_min/max`（图缩放范围） from `1.0 / 1.0` to `0.5 / 1.5`, so high-ambiguity items（高歧义物品） receive stronger `L3` local graph smoothing（第三层局部图平滑）
  - generated collision（生成冲突）: `657 / 3686 = 0.1782419967`
  - max conflict（最大冲突簇）: `72`
  - active L1（活跃第一层码）: `18`
  - unique L2 pairs（唯一第二层前缀数）: `256`
  - compared with `original_l3_collab_local`（原版第三层局部协同） at `13 / 3686` collision（冲突）, max conflict（最大冲突簇） `2`, active L1（活跃第一层码） `95`, and unique L2（唯一第二层前缀） `2632`, this is a structural collapse（结构塌缩）
  - ambiguity bucket diagnostic（歧义分桶诊断） shows collided item rate（冲突物品率） is high across all buckets (`21.8%` to `26.4%`), so the failure is global routing collapse（全局路由塌缩）, not only high-ambiguity overfitting（高歧义过拟合）
  - verdict（裁决）: do not prepare data_experiment（实验数据转换）, do not push to SFT（监督微调）, and do not treat this L3 ambiguity-aware scaling（第三层歧义感知缩放） interface as viable
- `original_l2_ambiguity_aware`（原版第二层歧义感知）已经完成 tokenizer/generate（分词器训练与生成）：
  - generated collision（生成冲突）: `13 / 3686 = 0.0035268584`
  - max conflict（最大冲突簇）: `2`
  - train best collision（训练最佳冲突率）: `0.1527400977`
  - active L1（活跃第一层码）: `50`
  - unique L2 pairs（唯一第二层前缀数）: `1693`
  - compared with `original_l2_multihop_ranking`（原版第二层多跳排序） at active L1（活跃第一层码） `88` and unique L2（唯一第二层前缀） `2449`, and with `original_l3_collab_local`（原版第三层局部协同） at active L1（活跃第一层码） `95` and unique L2（唯一第二层前缀） `2632`, this branch is structurally much more concentrated（结构上更集中）
  - top-5 `L1` bucket total（前 5 个第一层桶总覆盖） reaches `808`, versus `516` for `original_l2_multihop_ranking`（原版第二层多跳排序）, `594` for `original_l3_collab_local`（原版第三层局部协同）, and `282` for `v2`
  - unlike `original_l3_ambiguity_aware`（原版第三层歧义感知）, it is not a catastrophic collapse（灾难性塌缩）; instead, it is a non-catastrophic but over-compressed tokenizer（非灾难性但过度压缩的分词器）
  - verdict（裁决）: keep as a paired control（成对对照）, but do not prioritize this branch for SFT（监督微调） ahead of `original_l2_ranking_ambiguity_aware`（原版第二层排序歧义感知）
- `original_l2_ranking_ambiguity_aware`（原版第二层排序歧义感知）已经完成 tokenizer/generate（分词器训练与生成）：
  - generated collision（生成冲突）: `15 / 3686 = 0.0040694520`
  - max conflict（最大冲突簇）: `2`
  - train best collision（训练最佳冲突率）: `0.1855670103`
  - active L1（活跃第一层码）: `77`
  - unique L2 pairs（唯一第二层前缀数）: `1649`
  - compared with `original_l2_multihop_ranking`（原版第二层多跳排序） at the same generated collision（相同生成冲突） `15 / 3686`, this branch is more concentrated（更集中）: active L1（活跃第一层码） `88 -> 77`, unique L2（唯一第二层前缀） `2449 -> 1649`, top-5 L1 total（前 5 个第一层桶总覆盖） `516 -> 743`
  - compared with `original_l2_ambiguity_aware`（原版第二层歧义感知平滑）, the ranking variant（排序版本） is safer at `L1`（第一层） but still highly compressed（仍明显压缩） at `L2`（第二层）: active L1（活跃第一层码） `50 -> 77`, unique L2（唯一第二层前缀） `1693 -> 1649`
  - compared with strongest original tokenizer（最强原版分词器）, this branch spreads `L1`（第一层） more (`48 -> 77`) but sharply coarsens `L2`（第二层） (`2295 -> 1649`)
  - the largest `L1` bucket（第一层最大桶） is a heterogeneous mixed industrial bucket（异质工业混合桶） of size `236`; the second bucket（第二大桶） is a coherent 3D-printing cluster（3D 打印簇） of size `196`
  - ambiguity bucket diagnostic（歧义分桶诊断） shows no catastrophic collapse（灾难性塌缩）, but active L1（活跃第一层码） drops from `76` in low-ambiguity bucket（低歧义分桶） to `47` in high-ambiguity bucket（高歧义分桶）, consistent with stronger compression（更强压缩） on high-ambiguity items（高歧义物品）
  - verdict（裁决）: this branch is logically aligned with the push-pull motivation（推拉动机）, but the current ambiguity-aware scaling（歧义感知缩放） still over-compresses the mid-level structure（中层结构）; keep it as tokenizer-side control（分词器侧对照）, do not prioritize it for SFT（监督微调）
- `v2_l1cap128`（v2 第一层容量限制 128）已经完成 tokenizer/generate（分词器训练与生成）并给出明确 no-go（停止）结果：
  - intended change（意图变化）: keep original v2 ambiguity-aware graph supervision（歧义感知图监督） and semantic retention（语义保持） unchanged, only reduce `num_emb_list`（码本大小列表） from `[256,256,256]` to `[128,256,256]`
  - train best collision（训练最佳冲突率）: `0.5919696148`
  - generated collision（生成冲突）: `114 / 3686 = 0.0309278351`
  - max conflict（最大冲突簇）: `21`
  - active L1（活跃第一层码）: `15`
  - unique L2 pairs（唯一第二层前缀数）: `452`
  - compared with original v2（原始 v2） at generated collision（生成冲突） `13 / 3686`, max conflict（最大冲突簇） `2`, active L1（活跃第一层码） `203`, and unique L2（唯一第二层前缀） `2680`, this is severe global over-compression（严重全局过度压缩）
  - verdict（裁决）: do not push to SFT（监督微调）; hard `K1=128` capacity capping（硬性第一层容量限制） is not a safe fix for v2 L1 fragmentation（v2 第一层碎片化）
- `qcr_l2_conflict_ranking`（量化冲突感知第二层排序）已经完成 tokenizer/generate（分词器训练与生成）和 SFT/evaluate（监督微调/评测）；tokenizer-side（分词器侧）健康，但 downstream learnability（下游可学习性）没有兑现：
  - intended change（意图变化）: keep original RQ-VAE backbone（原版残差量化变分自编码器主干）, disable global graph propagation（全局图传播） and ordinary L2 ranking（普通第二层排序）, and activate L2 repulsion（第二层推开） only when semantic-near graph-weak negatives（语义近但图弱负样本） currently share the L2 prefix（第二层前缀）
  - generated collision（生成冲突）: `11 / 3686 = 0.0029842648`
  - max conflict（最大冲突簇）: `2`
  - active L1（活跃第一层码）: `117`
  - unique L2 pairs（唯一第二层前缀数）: `2632`
  - compared with `original_l2_multihop_ranking`（原版第二层多跳排序）, QCR（量化冲突感知排序） is structurally healthier: collision（冲突） `15 -> 11`, active L1（活跃第一层码） `88 -> 117`, unique L2（唯一第二层前缀） `2449 -> 2632`
  - on the QCR negative-pair set（QCR 负样本对集合）, same-L2 rate（同第二层率） drops from `0.01217` in `original_l2_multihop_ranking`（原版第二层多跳排序） to `0.01119`, but remains above v2/original_l3_collab_local（原始 v2 / 原版第三层局部协同）的 `0.01025`
  - downstream SFT/evaluate（下游监督微调/评测） under `title_history2sid_on + desc_align_p05`（标题历史转 SID 开启 + 描述对齐 0.05）: `NDCG@10 = 0.09980951`, `HR@10 = 0.13876020`, `constraint_invalid_total = 0`
  - compared with `original_l2_multihop_ranking`（原版第二层多跳排序）: `NDCG@10 -0.00184`, `HR@10 -0.00860`
  - compared with `R720e`（协同排序强候选）: `NDCG@10 -0.00114`, `HR@10 -0.00728`
  - compared with strongest original SFT（原版最强监督微调）: `NDCG@10 -0.00391`, `HR@10 -0.01213`
  - verdict（裁决）: no-go（停止） for RL promotion（强化学习晋级）; QCR（量化冲突感知排序） improved structural proxies（结构代理指标） but hurt the primary downstream objective（主要下游目标）

## Strongest Validated Line（最强已验证线）

当前 strongest validated line（最强已验证线）仍然是：

`v2_on_p05 -> RL`

但它现在只是 baseline/reference（基线/参考），不是当前继续迭代的主线方法。

## Next Steps（下一步）

1. 下一步诊断必须围绕 `@1/@3/@5/@10`（主要评测截断），尤其是 `NDCG@10` 和 `HR@10`，不要把 `HR@50`（命中率@50）作为晋级依据。
2. `original_l2_multihop_ranking`（原版第二层多跳排序）按当前 primary objective（主要目标）应视为 no-go（停止），不能作为方法扩展基座；如果继续训练，只能做 tightly controlled L2-only repair（严格受控的仅第二层小修复），例如只改 `lambda_2`（第二层损失权重）或 teacher pairs（教师样本对）质量，并以 `@1/@3/@5/@10`（主要评测截断）是否改善作为 go / no-go gate（推进 / 停止门槛）。
3. `original_l3_ambiguity_aware`（原版第三层歧义感知）和 `original_l2_ranking_ambiguity_aware`（原版第二层排序歧义感知）都已经完成 tokenizer-side（分词器侧）检查；前者是 clear no-go（明确停止），后者是 non-catastrophic but over-compressed（非灾难性但过度压缩）. 当前不应优先把这两条线推进到 SFT（监督微调）。
4. 当前不应直接推进 `R720e`、`original_l3_collab_local`、`original_l2_multihop_ranking` 或 `R720f` 到 `RL`（强化学习）；它们都没有超过 `v2_on_p05` SFT（当前 v2_on_p05 监督微调）和 strongest original SFT（原版最强监督微调）的 `NDCG@10`。
5. `R720f` 已经否定当前 hard L1 capacity reduction（硬性第一层容量缩减）方向；不要再把 L1 compactness proxy（第一层紧凑性代理指标）当作可靠晋级依据。
6. `v2_l1cap128`（v2 第一层容量限制 128）进一步说明：即使问题来自 v2 L1 fragmentation（第一层碎片化），也不能简单靠 hard codebook cap（硬码本限制）修复；后续若修 L1（第一层），应考虑 soft usage / entropy / routing regularization（软使用率 / 熵 / 路由正则），而不是直接缩小 `K1`（第一层码本大小）。
7. `qcr_l2_conflict_ranking`（量化冲突感知第二层排序）已经完成 SFT/evaluate（监督微调/评测）且主指标为负；不要推进 RL（强化学习），也不要把 tokenizer-side structural health（分词器侧结构健康）单独当作晋级依据。
8. `prism_anchor_coarse`（语义锚定粗图）这条 `L1`（第一层）载体路线在当前实现下不应继续推进；后续 `L1` 修复应继续坚持“小改现有损失/加权方式”，避免再次引入会导致根码塌缩的 coarse graph source（粗图来源）。

## Reading Rule（阅读规则）

任何 `R720a` 之前的 dated notes（带日期笔记）、旧 stage README（阶段说明）、旧 scripts/configs（脚本/配置）默认都是 archived provenance（归档追溯材料），不应再作为新实验起点。
