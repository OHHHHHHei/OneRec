# Current State（当前状态）

Status（状态）: `canonical（权威）`

Last updated（更新日期）: `2026-04-24`

## One-Line State（一句话状态）

The MGR-SID / ACLR / QCR research line（研究线） is now archived（已归档） as a negative-result stage（负结果阶段）.

Current decision（当前决策）:

> Do not continue the present collaborative-SID construction（协同 SID 构建） variants. No current branch gives a robust primary downstream win（主要下游稳定胜利） over the strongest OneRec baseline（最强 OneRec 基线） on `@1/@3/@5/@10`（主要评测截断）.

The repository should now treat this line as provenance（追溯材料）, not as an active optimization target（活跃优化目标）.

## Archive Pointer（归档指针）

Canonical archive（权威归档）:

- [MGR-SID Negative Research Archive](/home/leejt/OneRec/research-progress-log/archive/2026-04-24_mgr_sid_negative_research_archive/README.md)
- [Classified Stage Manifest](/home/leejt/OneRec/research-progress-log/archive/2026-04-24_mgr_sid_negative_research_archive/CLASSIFIED_STAGE_MANIFEST.md)
- [Negative Result Postmortem](/home/leejt/OneRec/research-progress-log/archive/2026-04-24_mgr_sid_negative_research_archive/NEGATIVE_RESULT_POSTMORTEM.md)

Snapshots（快照） from before this archive checkpoint（归档检查点）:

- [CURRENT_STATE_before_archive.md](/home/leejt/OneRec/research-progress-log/archive/2026-04-24_mgr_sid_negative_research_archive/CURRENT_STATE_before_archive.md)
- [EXPERIMENT_LAUNCH_INDEX_before_archive.md](/home/leejt/OneRec/research-progress-log/archive/2026-04-24_mgr_sid_negative_research_archive/EXPERIMENT_LAUNCH_INDEX_before_archive.md)

Registries（总账） remain the source for finalized metrics（定稿指标）:

- [experiment_registry/README.md](/home/leejt/OneRec/research-progress-log/experiment_registry/README.md)
- [tokenizer_registry.csv](/home/leejt/OneRec/research-progress-log/experiment_registry/tokenizer_registry.csv)
- [sft_registry.csv](/home/leejt/OneRec/research-progress-log/experiment_registry/sft_registry.csv)
- [rl_registry.csv](/home/leejt/OneRec/research-progress-log/experiment_registry/rl_registry.csv)
- [downstream_scoreboard.csv](/home/leejt/OneRec/research-progress-log/experiment_registry/downstream_scoreboard.csv)

## Core Verdict（核心裁决）

The tested hypothesis（被测试假设） was:

> Collaborative hierarchy information（协同层级信息） can be injected into SID construction（语义标识构建） to produce a SID codebook space（SID 码本空间） that is easier for downstream SFT / RL（下游监督微调 / 强化学习） than the standard OneRec baseline（标准 OneRec 基线）.

Current result（当前结果）:

- Not supported（不支持） as a paper-level claim（论文级主张）.
- Some variants improved tokenizer-side structure（分词器侧结构）.
- Some variants improved over weaker MGR-SID variants（较弱 MGR-SID 变体）.
- None established a robust win（稳定胜利） over the strongest baseline（最强基线） on the primary metrics（主要指标）.

Primary gate（主要门槛）:

- `NDCG@1/@3/@5/@10`（归一化折损累计增益）
- `HR@1/@3/@5/@10`（命中率）

Secondary diagnostics（次级诊断） such as `HR@50`（命中率@50）, tokenizer collision（分词器冲突）, active L1 count（活跃第一层码数量）, and L2 prefix spread（第二层前缀展开） are not promotion criteria（推进标准） by themselves.

## Closed Method Families（已关闭方法族）

The following families（方法族） are closed under current evidence（当前证据）:

- MGR-SID v1/v2 graph hierarchy（图层级） construction.
- Stage-2 / Stage-3 retention and codebook-space refinement（保持项与码本空间修复）.
- TAGCF / FaGSP / MGDCF / Seq2Graph-lite graph-carrier upgrades（图载体升级）.
- Selective separation（选择性分离） and mid-only pull-push（中层拉近推远）.
- CoST-inspired contrastive quantization（受 CoST 启发的对比量化）.
- Collab-ranking（协同排序） R720 variants.
- Minimal-edit original-RQVAE（最小编辑原版残差量化变分自编码器） L2/L3 variants.
- QCR-L2 conflict ranking（量化冲突感知第二层排序）.
- Hard L1 capacity reduction（硬性第一层容量缩减）.

## What Remains Useful（仍然有用的结论）

- Original OneRec SID routeability（原始 OneRec 语义标识可路由性） is a strong baseline property（强基线性质）.
- Low-disturbance collaborative injection（低扰动协同注入） is safer than heavy graph propagation（重图传播）.
- Local multihop（局部多跳） is more informative than broad mid-graph carriers（宽中图载体）, but still not enough for a primary downstream win（主要下游胜利）.
- Tokenizer-side proxies（分词器侧代理指标） cannot replace downstream SFT/evaluate（下游监督微调/评测）.
- Future work must explain downstream learnability（下游可学习性） directly, not only hierarchy structure（层级结构）.

## Repository State（仓库状态）

This archive is logical（逻辑归档） rather than physical relocation（物理搬移）:

- Experiment folders（实验文件夹） remain under `research-progress-log/experiment_launches/`.
- Research configs（研究配置） remain under `config/experiments/`.
- Research scripts（研究脚本） remain under `scripts/` and `scripts/archive/`.
- Method code（方法代码） remains under `src/onerec/experiments/mgr_sid/`.
- Research data variants（研究数据变体） remain under `data_experiment/`.
- Large checkpoints（大模型权重） remain under `/data/leejt/OneRec/output_weights/experiments/`.

This avoids breaking registry pointers（总账指针） and artifact paths（产物路径）.

## Next Steps（下一步）

1. If sending code to the advisor（导师）, export the clean OneRec baseline（干净 OneRec 基线） from the historical commit（历史提交）, not from the current research-polluted workspace（研究污染工作区）.
2. Do not launch more MGR-SID / QCR / R720-style experiments（实验） unless a genuinely new mechanism（新机制） is defined.
3. If a new research direction starts, create a new dated idea folder（带日期想法目录） and a fresh experiment plan（实验计划） with a clean baseline protocol（干净基线协议）.

## Reading Rule（阅读规则）

All dated MGR-SID notes（带日期 MGR-SID 笔记）, old stage README files（旧阶段说明）, old configs（旧配置）, and old scripts（旧脚本） are archived provenance（归档追溯材料） unless explicitly reactivated by a future canonical current-state update（未来权威当前状态更新）.

