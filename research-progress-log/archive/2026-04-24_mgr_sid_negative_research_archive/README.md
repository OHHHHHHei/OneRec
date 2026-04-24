# MGR-SID Negative Research Archive（MGR-SID 负结果研究归档）

Status（状态）: `archived（归档）`
Snapshot date（快照日期）: `2026-04-24`

## Archive Verdict（归档裁决）

This archive freezes（冻结） the current MGR-SID / ACLR / QCR research line（研究线） as a negative-result stage（负结果阶段）.

The core research question was:

> Can collaborative hierarchy information（协同层级信息） be injected into SID construction（语义标识构建） so that downstream SFT / RL（下游监督微调 / 强化学习） becomes easier than the standard OneRec baseline（标准 OneRec 基线）?

Current answer:

> No current method branch gives a robust primary downstream win（主要下游稳定胜利） over the strongest baseline（最强基线） on `@1/@3/@5/@10`（主要评测截断）. The line should be archived rather than further extended from the present variants.

Important nuance（重要口径）:

- Some internal structural proxies（结构代理指标） improved.
- Some variants improved over weaker MGR-SID variants.
- `v2_on_p05 -> RL` remains a useful reference（参考） run.
- None of these is enough for the paper-level claim（论文级主张） that collaborative SID construction（协同 SID 构建） beats the clean strongest OneRec baseline（干净最强 OneRec 基线）.

## What Was Archived（归档范围）

This archive now has two layers（两层）:

- Logical archive（逻辑归档） for experiment stages（实验阶段）:
  - Experiment stages（实验阶段） remain in `research-progress-log/experiment_launches/` so registry（总账） and artifact pointers（产物指针） do not break.
  - Research notes（研究笔记） remain under `idea-discovery/` and `research-progress-log/`.
- Physical archive（物理归档） for visible research workspace（显式研究工作区）:
  - Research configs（研究配置） moved to `archived_workspace/config/`.
  - Research scripts（研究脚本） moved to `archived_workspace/scripts/`.
  - MGR-SID method code（方法代码） moved to `archived_workspace/src/onerec/experiments/`.
  - ACLR-lite collaborative rerank code（协同重排代码） moved to `archived_workspace/src/onerec/evaluate/`.

This archive adds a classification layer（分类层）, closes the active decision loop（活跃决策循环）, and restores the visible baseline layout（显式基线布局）.

## Archive Files（归档文件）

- `CURRENT_STATE_before_archive.md`: canonical current state（权威当前状态） before this archive checkpoint（归档检查点）.
- `EXPERIMENT_LAUNCH_INDEX_before_archive.md`: stage index（阶段索引） before this archive checkpoint.
- `CLASSIFIED_STAGE_MANIFEST.md`: classified map（分类地图） of all current research stage folders and research assets（研究资产）.
- `NEGATIVE_RESULT_POSTMORTEM.md`: concise postmortem（复盘） of what failed, what remains informative, and what should not be continued.
- `archived_workspace/`: physically archived research configs（配置）, scripts（脚本）, and method code（方法代码）.
- `root_tmp_diagnostics/`: archived root-level `tmp_*` diagnostic artifacts（根目录临时诊断产物）.

## Closed Families（已关闭方法族）

The following research families（研究族） are archived as no-go（停止） or non-promotable（不可推进） under the current evidence:

- MGR-SID v1/v2 graph hierarchy（图层级） construction.
- Stage-2 / Stage-3 retention and codebook-space（码本空间） refinement.
- Graph-carrier upgrades（图载体升级） based on TAGCF / FaGSP / MGDCF / Seq2Graph-lite.
- Selective separation（选择性分离）, mid-only pull-push（中层拉近推远）, and push-pull restoration（推拉恢复） variants.
- CoST-inspired contrastive quantization（受 CoST 启发的对比量化） variants.
- Collab-ranking（协同排序） variants including R720a/b/e/f.
- Minimal-edit original-RQVAE（最小编辑原版残差量化变分自编码器） variants at L2/L3.
- QCR-L2 conflict ranking（量化冲突感知第二层排序）.
- Hard L1 capacity reduction（硬性第一层容量缩减） and v2 `K1=128` capacity cap（容量限制）.

## Main Lessons（主要教训）

- Tokenizer-side health（分词器侧健康） is not enough. Low collision（低冲突）, more active L1（更多活跃第一层码）, or better L2 spread（更好的第二层展开） did not reliably translate into downstream learnability（下游可学习性）.
- Routeability（可路由性） of the original SID space（原始 SID 空间） is stronger than expected.
- Local multihop（局部多跳） is a better L2 carrier（第二层载体） than broad mid-graph variants, but it still did not produce primary downstream gains（主要下游收益）.
- Global or heavy graph injection（重图注入） tends to damage hierarchy routing（层级路由） or over-compress（过度压缩） the code space（码空间）.
- `HR@50`（命中率@50） improvements are not enough when `NDCG/HR@1/@3/@5/@10`（主要推荐指标） do not improve.

## Reopen Criteria（重启条件）

Do not reopen this line by adding another nearby loss（相近损失） or graph carrier（图载体） to the existing pipeline（流水线）.

A future line should start only if it has:

1. A new mechanism（新机制） that explains downstream learnability（下游可学习性）, not only tokenizer structure（分词器结构）.
2. A clean baseline protocol（干净基线协议） anchored to the original OneRec baseline（原始 OneRec 基线）.
3. A primary metric gate（主要指标门槛） on `@1/@3/@5/@10`, especially `NDCG@10`（归一化折损累计增益@10） and `HR@10`（命中率@10）.
4. No posterior tokenizer modification（后验分词器修改） based on already seeing test answers（测试答案）.
