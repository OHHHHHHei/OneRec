# 2026-04-16 `R610` Selective Separation（`R610` 选择性分离） Industrial

Status（状态）: `snapshot（快照）`

## Scope（范围）

这是 selective separation（选择性分离）阶段的第一个正式 tokenizer screen（分词器筛选）实验入口。

当前只先推进最小版本：

- `R610a`: base `v2` + `L3`-only selective separation（仅 `L3` 选择性分离）

## Why This Run Exists（为什么做这个实验）

当前 `v2`（第二版）的方法骨架已经证明：

- graph-aware attraction（图感知吸引）有价值
- ambiguity-aware weighting（歧义感知加权）有价值

但当前图监督仍然主要是 attraction-only graph smoothness（仅吸引式图平滑），还不能显式分离 `semantic-close but collaboratively inconsistent`（语义接近但协同不一致）的物品。

`D600` 的结论进一步说明：

- `semantic-near + graph-non-neighbor`（语义接近 + 图上无邻接）太宽
- 第一批训练更合理的 pair rule（物品对规则）应当从 `semantic-near + graph-weak`（语义接近 + 图弱连接）开始

所以 `R610a` 的目的就是验证：

> 在不改当前 `v2` 主骨架的前提下，只在 `L3`（第 3 层）加一个很克制的 reliability-aware selective separation（可靠性感知选择性分离）项，能不能改善局部 `SID` 判别而不引发 tokenizer collapse（分词器塌缩）。

## Run Matrix（运行矩阵）

- `R610a`
  - backbone（骨干）: current base `v2`
  - pair source（物品对来源）: `D600_all_graph_weak_pairs.csv`
  - pair rule（物品对规则）: `semantic_near_graph_weak`
  - separation levels（分离层级）: `L3` only
  - pair weighting（物品对加权）: reliability-aware（可靠性感知）
  - ambiguity scaling（歧义缩放）: on
  - warmup（预热）: off

## Files（文件）

- config（配置）:
  - [sid_train_industrial_mgr_sid_r610a_selective_separation_l3.yaml](/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_r610a_selective_separation_l3.yaml)
- launcher（启动脚本）:
  - [experiment_mgr_sid_r610a_selective_separation_l3_train_generate.sh](/home/leejt/OneRec/scripts/experiment_mgr_sid_r610a_selective_separation_l3_train_generate.sh)
- pair diagnostics source（物品对诊断来源）:
  - [D600_all_graph_weak_pairs.csv](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_selective_separation_pair_diagnostics_industrial/D600_all_graph_weak_pairs.csv)

## Runtime（运行时）

- training log（训练日志）:
  - `/home/leejt/OneRec/logs/experiment_mgr_sid_r610a_selective_separation_l3_20260416.log`
- checkpoints（检查点）:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_selective_separation_20260416/industrial_r610a_l3_graph_weak`
- generated index（生成索引）:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_selective_separation_20260416/generated_indices/Industrial_and_Scientific.r610a_l3_graph_weak.index.json`
- generate summary（生成摘要）:
  - `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_r610_selective_separation_industrial/R610a_generate_summary.json`
- local ambiguity diagnostics（局部歧义诊断）:
  - [R611_v2_vs_r610a_local_ambiguity.md](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_r610_selective_separation_industrial/R611_v2_vs_r610a_local_ambiguity.md)
  - [R611_v2_vs_r610a_local_ambiguity.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_r610_selective_separation_industrial/R611_v2_vs_r610a_local_ambiguity.json)
- prefix collaborative consistency（前缀协同一致性）:
  - [R612_v2_vs_r610a_prefix_collaborative_consistency.md](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_r610_selective_separation_industrial/R612_v2_vs_r610a_prefix_collaborative_consistency.md)
  - [R612_v2_vs_r610a_prefix_collaborative_consistency.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_r610_selective_separation_industrial/R612_v2_vs_r610a_prefix_collaborative_consistency.json)

## Promotion Rule（推进规则）

- 原始推进规则是：如果 `R610a` 没有明显 collision explosion（冲突爆炸），并且 tokenizer-side（分词器侧）局部歧义指标优于 base `v2`，就继续推进 `R610b`
- 但 `R611` 现在已经补充证明：`R610a` 只有 generated collision（生成后冲突率）是正向，局部结构不是 clean win（干净胜利）
- 因此当前实际推进规则已经更新为：
  - **不要直接推进 `R610b`**
  - 优先回头收紧：
    - separation margin（分离间隔）
    - pair reliability（物品对可靠性）
    - pair coverage（物品对覆盖面）
    - `L3` only 是否仍然过强

## Result Snapshot（结果快照）

`R610a` 已完成，当前最关键结果是：

- train-stage best collision rate（训练阶段最佳冲突率）: `0.1489419425`
- generated collision rate（生成后冲突率）: `0.0032555616`
- generated collision count（生成后冲突数）: `12 / 3686`
- max conflict（最大冲突大小）: `2`
- collision rounds used（冲突修复轮数）: `20`

### Interpretation（解读）

- 这条线在 final generated SID（最终生成 SID）上是**明显正向**的：
  - 优于 current `v2`（当前 `v2`）的 `13 / 3686`
  - 明显优于 `R542a` 的 `42 / 3686`
  - 也远优于 `R530a` 的 `107 / 3686`
- 但它同时暴露出一个很重要的新风险：
  - train-stage collision（训练阶段冲突）仍然非常高
  - 说明当前分离项并没有直接改善 raw quantization uniqueness（原始量化唯一性）
  - final result 更像是 **generate-stage repair（生成阶段修复）+ selective separation（选择性分离）** 的组合效果

所以 `R610a` 当前最准确的判断不是“已经站稳”，而是：

> **这是 selective separation（选择性分离）方向的第一个强 tokenizer-side（分词器侧）正信号，但还不能直接等价成稳健的 downstream-ready（可直接下游推进）结论。**

## Structural Diagnostics（结构诊断）

`R611` 已完成，对比的是 current `v2`（当前 `v2`）和 `R610a` 的 final generated SID（最终生成 SID）局部结构：

- 诊断文件：
  - [R611_v2_vs_r610a_local_ambiguity.md](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_r610_selective_separation_industrial/R611_v2_vs_r610a_local_ambiguity.md)
- 关键数字：
  - test-weighted mean `l2` leaf count（测试加权平均 `l2` 叶子数）: `4.3422 -> 4.1282`
  - test-weighted multi-leaf `same_l2` fraction（测试加权多叶 `same_l2` 比例）: `0.4873 -> 0.6389`
  - test-weighted deep crowded `l2` fraction（测试加权深拥挤 `l2` 比例，`>=4`）: `0.2228 -> 0.2994`
  - test-weighted mean `l3` entropy（测试加权平均 `l3` 熵）: `1.1001 -> 1.3129`
  - catalog item-weighted mean `l2` leaf count（全表 item 加权平均 `l2` 叶子数）: `2.3945 -> 2.9425`
- movement summary（迁移摘要）:
  - test targets（测试目标）里 `22.52%` 的 item 叶子数下降
  - 但有 `39.58%` 的 item 叶子数上升
  - `9.27%` 从 multi-leaf `same_l2` 移出
  - `24.42%` 被移入 multi-leaf `same_l2`

### Diagnostic Reading（诊断解读）

- `R610a` 确实拆开了一批极端拥挤的 hard cases（困难样本），这和 `12 / 3686` 的 generated collision（生成后冲突率）改进是一致的。
- 但它**不是**一个“整体 local ambiguity cleanup（整体局部歧义清理）变好”的结果：
  - 少数 hardest prefixes（最难前缀）被明显拆散
  - 但更多 item 被重新挤进了新的 multi-leaf / high-entropy `l2` bucket（多叶 / 高熵 `l2` 桶）
- 所以当前最合理的判断是：

> `R610a` 更像是 **targeted rescue（定点抢救）**，而不是 **clean structural win（干净的结构性胜利）**。现阶段不能因为 `12 / 3686` 就直接推进 `R610b` 或下游链。

## Prefix Collaborative Consistency（前缀协同一致性）

`R612` 进一步补充了一个更贴近 downstream（下游）需求的判断：

- 多叶 `l2` prefix（`l2` 前缀）并不自动是坏事
- 真正要区分的是：
  - collaboratively consistent crowded prefix（协同一致拥挤前缀）
  - collaboratively inconsistent crowded prefix（协同不一致拥挤前缀）

这次诊断的关键数字是：

- test-weighted consistent crowded fraction（测试加权协同一致拥挤占比）: `0.1147 -> 0.1374`
- test-weighted inconsistent crowded fraction（测试加权协同不一致拥挤占比）: `0.3711 -> 0.5001`
- test-weighted mean prefix graph affinity（测试加权平均前缀图亲和度）: `0.023510 -> 0.025290`
- moved to consistent crowded（移入协同一致拥挤）: `6.75%`
- moved to inconsistent crowded（移入协同不一致拥挤）: `23.94%`

### Reading（解读）

- `R610a` 不是纯负：
  - 它确实增加了一部分 collaboratively consistent crowding（协同一致拥挤）
  - 也就是说，有一些多叶前缀更像是“合理的局部备选结构”
- 但当前版本的主要问题也很清楚：
  - collaboratively inconsistent crowding（协同不一致拥挤）增长得更快、更大
  - 所以 `R610a` 目前是在“同时增加好拥挤和坏拥挤”，而且坏拥挤占主导

因此，后续优化目标不该再表述成“继续压低多叶前缀”，而应该表述成：

> **提升 good crowding（良性拥挤）占比，压低 bad crowding（恶性拥挤）占比。**
