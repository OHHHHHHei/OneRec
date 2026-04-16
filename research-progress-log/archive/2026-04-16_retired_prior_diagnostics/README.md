# Retired Prior Diagnostics（已退役前验诊断）

Status（状态）: `archived（归档）`
Archived on（归档日期）: `2026-04-16`

## Why This Archive Exists（为什么建立这个归档）

`S000` retrospective audit（回顾性审计） showed that the currently used prior
diagnostics（前验诊断） do not have enough pairwise consistency（成对一致率） against
historical downstream results（历史下游结果） to justify tokenizer promotion
decisions（分词器推进决策）.

As a result, these diagnostics were retired from the active workflow（活跃工作流）.

## What Was Retired（哪些内容被退役）

- generated collision（生成后冲突率） as a tokenizer ranking metric（分词器排序指标）
- local ambiguity（局部歧义） as a tokenizer promotion gate（分词器推进门槛）
- prefix collaborative consistency（前缀协同一致性） as a promotion gate
- stage-2 / stage-3 interface diagnostics（阶段 2 / 阶段 3 接口诊断） as promotion evidence
- coarse/local graph diagnostics（粗图 / 局部图诊断） as offline promotion gates
- selective-separation pair diagnostics（选择性分离物品对诊断） as gating evidence

## Current Rule（当前规则）

- downstream `SFT -> evaluate`（监督微调到评测） remains the only final verdict（最终裁决）
- prior diagnostics（前验诊断） may survive only as historical artifacts（历史产物）
- any future diagnostic must first pass retrospective audit（回顾性审计） before it can
  re-enter the active workflow（活跃工作流）

## Script Archive（脚本归档）

See:

- [scripts/archive/retired_prior_diagnostics/README.md](/home/leejt/OneRec/scripts/archive/retired_prior_diagnostics/README.md)
