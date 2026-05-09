# MGR-SID LMH Current Mainline（当前主线）

This package is the canonical code entrypoint（规范代码入口） for the active
SID tokenizer（语义标识分词器） research line:

- L1 semantic routing（第一层语义路由）
- L2 local-multihop collaborative graph InfoNCE（第二层局部多跳协同图对比损失）
- L3 local transition refinement（第三层局部转移精修）
- hierarchy stop-gradient（层级停梯度）

The original implementation was developed inside:

`research-progress-log/archive/2026-04-24_mgr_sid_negative_research_archive/archived_workspace`

That archived workspace remains the validated backend（已验证历史后端） for
currently running and already completed experiments. New launch scripts should
import through this package instead of referring to archived paths directly.

## Current Boundary（当前边界）

Included in the current mainline:

- `local_multihop_base_weight` for L2 graph radius ablations（第二层图半径消融）
- `local_multihop_alpha` for two-hop strength（两跳强度）
- L2 `graph_infonce` mode（图对比模式）
- L3 `pairwise_pull` local refinement（局部拉近精修）

Kept as legacy-only（仅历史兼容） unless explicitly re-promoted:

- QCR conflict ranking（冲突排序）
- TAGCF / FAGSP / Seq2Graph transplants（外部图方法迁移）
- attention residual tokenizer variants（注意力残差分词器变体）

