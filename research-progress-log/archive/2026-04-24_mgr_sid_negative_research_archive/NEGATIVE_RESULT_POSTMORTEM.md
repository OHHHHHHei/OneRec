# Negative Result Postmortem（负结果复盘）

Status（状态）: `archived（归档）`
Snapshot date（快照日期）: `2026-04-24`

## Summary（摘要）

The current MGR-SID line（当前 MGR-SID 线） did not produce a robust publishable improvement（可发表的稳定提升） over the strongest OneRec baseline（最强 OneRec 基线）.

The main failure is not implementation collapse（实现崩溃）. The experiments were often technically healthy:

- constrained decoding（约束解码） usually had `constraint_invalid_total = 0`;
- many tokenizer variants（分词器变体） had low final collision（低最终冲突）;
- some variants had healthier-looking hierarchy statistics（层级统计）.

The failure is conceptual:

> Better-looking SID structure（更好看的 SID 结构） did not consistently become better downstream learnability（下游可学习性）.

## Failed Assumptions（失败假设）

### 1. Structural Proxies Could Promote Tokenizers（结构代理指标可以推进分词器）

This failed.

Generated collision（生成冲突）, active L1 count（活跃第一层码数量）, unique L2 prefixes（唯一第二层前缀）, local ambiguity（局部歧义）, and graph-neighbor prefix sharing（图邻居前缀共享） were useful diagnostics（诊断）, but none became a reliable promotion gate（可靠推进门槛）.

### 2. More Collaborative Smoothness Would Help SID Learning（更多协同平滑会帮助 SID 学习）

This mostly failed.

Heavy graph propagation（重图传播） and broad graph-carrier changes（宽图载体改动） often over-compressed（过度压缩） the hierarchy or damaged routeability（可路由性）.

### 3. L2 Is the Right Place for Collaborative Branching（第二层适合协同分叉）

This remains only partially informative（部分有信息量）.

Local multihop（局部多跳） at L2 was better than earlier mid-graph carriers（中图载体）, but even the best L2 minimal-edit branch（最小编辑分支） did not beat the strongest baseline（最强基线） on primary `NDCG@10`（主要归一化折损累计增益@10）.

### 4. Conflict-Aware Repulsion Would Fix Shared Prefix Errors（冲突感知推开会修复共享前缀错误）

QCR（量化冲突感知排序） tested this more directly.

Result:

- tokenizer structure（分词器结构） looked healthy;
- downstream SFT/evaluate（下游监督微调/评测） got worse than the stronger minimal-edit branch（最小编辑分支） and strongest baseline（最强基线）.

So conflict-specific repulsion（冲突特异推开） alone is not enough.

### 5. Hard L1 Compression Would Improve Routing（硬性第一层压缩会改善路由）

This failed clearly.

Both R720f and v2 `K1=128` showed that hard capacity reduction（硬容量缩减） can make compactness proxies（紧凑性代理指标） look better while damaging downstream performance（下游性能）.

## What Remains Useful（仍然有用的知识）

- Original OneRec routeability（原始 OneRec 可路由性） is a strong baseline property（强基线性质）.
- Low-disturbance collaborative injection（低扰动协同注入） is safer than full graph-injection（完整图注入）.
- Local multihop（局部多跳） is the least bad collaborative carrier（协同载体） found in this stage.
- Downstream SFT/evaluate（下游监督微调/评测） must remain the promotion gate（推进门槛）.
- Any future method should explain learnability（可学习性） directly, not just codebook structure（码本结构）.

## What Not To Continue（不应继续的方向）

- Do not keep stacking graph losses（图损失） inside RQ-VAE（残差量化变分自编码器）.
- Do not promote a tokenizer（分词器） only because structural proxies（结构代理指标） look good.
- Do not treat `HR@50`（命中率@50） as success if `@1/@3/@5/@10`（主要截断） do not improve.
- Do not shrink `K1`（第一层码本大小） as a simple fix for L1 fragmentation（第一层碎片化）.
- Do not use posterior tokenizer tuning（后验分词器调参） that relies on already seeing target answers（目标答案）.
- Do not reopen QCR（量化冲突感知排序） without a new downstream-learnability mechanism（下游可学习性机制）.

## Clean Exit Rule（干净退出规则）

The archived line should be treated as closed unless a new proposal can answer:

1. What exactly becomes easier for SFT / RL（监督微调 / 强化学习） to learn?
2. Why does that not merely improve tokenizer statistics（分词器统计）?
3. What baseline（基线） does it need to beat?
4. Which primary metric（主要指标） decides go / no-go（推进 / 停止）?

