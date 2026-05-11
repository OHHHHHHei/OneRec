# L2 Local Multihop Rescue Tokenizers（第二层局部多跳救援分词器）

Status（状态）: `navigation + stage-snapshot（导航 + 阶段快照）`

Last updated（更新日期）: `2026-05-11`

## Role（角色）

This branch（分支） contains the May 2026 local multihop SID tokenizer experiments（局部多跳语义标识分词器实验）.

The current representative tokenizer（当前代表分词器） is:

- `r690b_lmh_l2_contrastive_pull_weight001`

This branch should be read as evidence and implementation history（证据与实现历史）, not as the sole live current-state document（唯一实时状态文档）.

Canonical current state（权威当前状态）:

- [CURRENT_STATE.md](/home/leejt/OneRec/research-progress-log/CURRENT_STATE.md)

Mainline detail（主线细节）:

- [MAINLINE.md](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/MAINLINE.md)

## Reading Order（阅读顺序）

1. [CURRENT_STATE.md](/home/leejt/OneRec/research-progress-log/CURRENT_STATE.md): current claim boundary（当前主张边界）.
2. [MAINLINE.md](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/MAINLINE.md): method, artifacts, and finalized mainline metrics（方法、产物、定稿主线指标）.
3. [mainline/](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/mainline): symlinked clean entry（软链接干净入口） for configs（配置）, scripts（脚本）, reports（报告）, and diagnostics（诊断）.
4. [active_candidates/](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/active_candidates): parking lot and evidence index（暂存区与证据索引） for follow-up ideas.
5. [ablations/](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/ablations): completed or lower-priority ablations（已完成或低优先级消融）.
6. [archive/](/home/leejt/OneRec/idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/archive): superseded helpers（已替代辅助脚本） and old queue wrappers（旧队列脚本）.

## Stage Summary（阶段总结）

Question（问题）:

> Can local multihop collaborative information（局部多跳协同信息） improve SID tokenizer hierarchy（语义标识分词器层级） without heavy graph propagation（重图传播）?

Current answer（当前答案）:

- Industrial SFT（Industrial 监督微调）: positive on NDCG（归一化折损累计增益）, mixed on HR（命中率）.
- Industrial RL（Industrial 强化学习）: below strongest original RL baseline（原版最强强化学习基线）.
- Office transfer（Office 迁移）: small NDCG improvement, mixed HR.
- Toys transfer（Toys 迁移）: negative.

Practical conclusion（实践结论）:

- Keep `r690b_lmh_l2_contrastive_pull_weight001` as a meaningful evidence line（有意义证据线）.
- Do not describe it as a settled SOTA（最优） result.
- Do not use tokenizer proxy metrics（分词器代理指标） alone to promote a variant.

## Maintenance Notes（维护备注）

- This directory keeps original paths（原始路径） for reproducibility（可复现性）.
- Organized subdirectories use symlinks（软链接） where possible.
- Large checkpoints（大检查点） should remain under `/data/leejt/OneRec/output_weights`.
- Finalized results（定稿结果） should be recorded in `research-progress-log/experiment_registry/` before they are summarized in prose（文字总结）.
