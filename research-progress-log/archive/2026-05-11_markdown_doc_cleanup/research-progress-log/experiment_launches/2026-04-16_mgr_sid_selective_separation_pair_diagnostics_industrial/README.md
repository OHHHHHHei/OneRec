# 2026-04-16 `D600` Selective-Separation Pair Diagnostics（`D600` 选择性分离物品对诊断）

Status（状态）: `archived（归档）`
Archived on（归档日期）: `2026-04-16`

This run is preserved as provenance（来源记录） only.
After `S000`, it is no longer an active diagnostics gate（活跃诊断门）.

## 目的

这是 selective separation（选择性分离）阶段的第一个正式 diagnostics gate（诊断门）。

它回答的问题是：

> 如果我们下一阶段想显式分离 `semantic-close but collaboratively inconsistent`（语义接近但协同不一致）的物品，
> 那么第一批最合理的 candidate pairs（候选物品对）应该怎么定义？

这轮不训练 tokenizer（分词器），只做 pair construction（物品对构造）诊断。

## 对应实现

- 诊断脚本：
  - [experiment_mgr_sid_selective_separation_pair_diagnostics.py](/home/leejt/OneRec/scripts/archive/retired_prior_diagnostics/experiment_mgr_sid_selective_separation_pair_diagnostics.py)
- 当前基础图组：
  - `coarse_purified`
  - `fagsp_mid_base`
  - `local_purified`

## 诊断规则

这轮主要检查两类 pair rule（物品对规则）：

- `semantic_near_graph_non_neighbor`
  - 语义接近，但在当前基础图组的联合邻接里完全无支持
- `semantic_near_graph_weak`
  - 语义接近，但在当前基础图组里只有非常弱的支持

其中：

- semantic-near（语义接近）来自 semantic `kNN`（近邻）图
- graph affinity（图亲和度）来自 `coarse / mid / local` 三张基础图的联合视图
- reliability（可靠性）是一个保守 first-pass score（第一版分数），用于排序而不是直接作为最终训练公式

## 主要输出

- [SUMMARY.md](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_selective_separation_pair_diagnostics_industrial/SUMMARY.md)
- [D600_pair_summary.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_selective_separation_pair_diagnostics_industrial/D600_pair_summary.json)
- [D600_top_non_neighbor_pairs.csv](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_selective_separation_pair_diagnostics_industrial/D600_top_non_neighbor_pairs.csv)
- [D600_top_graph_weak_pairs.csv](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_selective_separation_pair_diagnostics_industrial/D600_top_graph_weak_pairs.csv)

## 关键结果

- semantic-near（语义近邻）pair 总量：
  - `82,596`
- `semantic_near_graph_non_neighbor`
  - pair count（物品对数量）: `74,332`
  - pair ratio（物品对比例）: `0.8999`
  - item coverage rate（物品覆盖率）: `0.9995`
- `semantic_near_graph_weak`
  - pair count（物品对数量）: `2,066`
  - pair ratio（物品对比例）: `0.0250`
  - item coverage rate（物品覆盖率）: `0.3733`
- graph-weak threshold（图弱连接阈值）:
  - `0.002001`

## 历史判断（仅供回溯）

`D600` 最重要的结论不是“能不能找到候选负对”，而是：

- 朴素的 `semantic-near + graph-non-neighbor`（语义接近 + 图上无邻接）规则**太宽**
- 它几乎覆盖了整个语义邻域，不适合作为第一批训练里的直接 separation rule（分离规则）
- 更合理的 first-pass rule（第一版规则）是：
  - `semantic-near + graph-weak`

也就是说，下一步 selective separation（选择性分离）不应从“完全无边就推开”开始，而应从：

- 语义上接近
- 协同上只有弱支持
- 更接近当前 collaborative boundary（协同边界）

的 pair 先做

## 当前状态

- 该诊断结论已退役，不再直接影响后续实验推进。
- 如果未来继续 selective separation（选择性分离），必须由新的方法设计直接触发，并通过 downstream `SFT -> evaluate`（监督微调到评测）裁决。

## 历史上的直接影响

这轮之后，下一步最合理的首发训练定义不再是：

- `R610a = semantic-near + graph-non-neighbor`

而应改成：

- `R610a = semantic-near + graph-weak`

然后再把：

- reliability-aware weighting（可靠性感知加权）

作为第二步消融推进。
