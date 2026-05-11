# Document Cleanup Candidates（文档清理候选）

Status（状态）: `cleanup-plan（清理计划）`

Last updated（更新日期）: `2026-05-09`

## Goal（目标）

Reduce stale documentation noise（降低过期文档噪声） while preserving canonical state（权威状态）, registries（总账）, current mainline（当前主线）, and reproducibility evidence（可复现实验证据）.

## Must Keep（必须保留）

Do not delete these without a separate explicit decision（没有单独明确决策不要删除）:

- `research-progress-log/CURRENT_STATE.md`
- `research-progress-log/experiment_registry/`
- `research-progress-log/experiment_launches/README.md`
- `research-progress-log/advisor_reports/`
- `idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/`
- `idea-discovery/2026-04-08_sid_collab_signal_injection/RESEARCH_DIRECTION.md`
- `idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/CURRENT_TASK_ALIGNMENT.md`
- `idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/18_mgr_sid_v2_ambiguity_aware_method.md`
- `idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/19_mgr_sid_current_method_code_aligned_formulas.md`
- `research-progress-log/archive/2026-04-24_mgr_sid_negative_research_archive/archived_workspace/`

## Safe Delete Batch A（可安全删除 A 批）

These are early automatic idea-discovery outputs（早期自动想法发现输出） superseded by the April 8 direction and current May 6 mainline（已被 4 月 8 日方向和 5 月 6 日主线替代）.

Candidate directory（候选目录）:

- `idea-discovery/2026-04-07-sid-collab/`

Reason（理由）:

- Duplicate / superseded exploratory material（重复 / 已被替代探索材料）.
- No active scripts or experiments depend on it（没有活跃脚本或实验依赖）.
- Current research direction is documented elsewhere（当前研究方向已在其他位置记录）.

Estimated removal（预计删除）:

- about 30+ stale docs（约 30 多个过期文档）.

## Safe Delete Batch B（可安全删除 B 批）

These are archived duplicate cleanup sources（归档重复清理来源） inside the old graph hierarchy workspace（旧图层级工作区）.

Candidate directories（候选目录）:

- `idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/archive/2026-04-11_doc_cleanup_duplicates/`
- `idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/archive/2026-04-11_doc_cleanup_v1_superseded/`
- `idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/archive/2026-04-12_doc_reorg_merged_sources/`

Reason（理由）:

- Their useful content was merged into live reference docs（有用内容已合并进参考文档）.
- They are already explicitly under archive（已经明确处于归档目录）.
- They create search noise（搜索噪声）.

## Safe Delete Batch C（可安全删除 C 批）

These are old post-stage2 review packets（旧 stage2 评审材料） that are superseded by the negative-result archive（负结果归档） and current mainline（当前主线）.

Candidate directory（候选目录）:

- `research-progress-log/archive/2026-04-14_post_stage2_review_materials/`

Reason（理由）:

- Historical discussion-only（历史讨论材料）.
- Not a registry（总账） or active method spec（活跃方法说明）.
- Superseded by later archive and current state（已被后续归档和当前状态替代）.

## Delete or Compress Batch D（删除或压缩 D 批）

These are repeated diagnostic report variants（重复诊断报告变体）.

Candidates（候选）:

- `research-progress-log/experiment_analysis/2026-05-08_sid_structural_diagnostic/`
- `research-progress-log/experiment_analysis/2026-05-08_codebook_reasonableness_with_l2_0015/`

Recommendation（建议）:

- Delete only if the enhanced versions（增强版本） are confirmed sufficient:
  - `research-progress-log/experiment_analysis/2026-05-08_sid_structural_diagnostic_enhanced/`
  - `research-progress-log/experiment_analysis/2026-05-09_sid_structural_diagnostic_l2_square_b025/`
  - `research-progress-log/experiment_analysis/2026-05-09_sid_structural_diagnostic_l3_ranking/`

## Archive-Only Batch E（只归档不删除 E 批）

These should not be deleted yet（暂不建议删除） because they are useful for method history（方法历史） or negative-result tracing（负结果追溯）:

- `research-progress-log/archive/2026-04-24_mgr_sid_negative_research_archive/`
- `research-progress-log/2026-04-21_mgr_sid_research_synthesis_report.md`
- `research-progress-log/INDEPENDENT_CRITICAL_REVIEW_20260414.md`
- `research-progress-log/DEEP_REVIEW_MGR_SID_PROJECT_STATE_20260414.md`
- `research-progress-log/research_progress_log.tex`
- `idea-discovery/2026-05-01_attentive_residual_rqvae_tokenizer/`
- `idea-discovery/2026-05-06_hierarchy_aware_attnres_sid_readout/`

Recommendation（建议）:

- Move these under a clearly named archive index（归档索引） later if they keep disturbing search.
- Do not delete them in the first cleanup pass（第一轮清理不要删除）.

## Proposed First Command（建议第一轮命令）

After confirmation（确认后）, delete only A/B/C first:

```bash
rm -rf \
  idea-discovery/2026-04-07-sid-collab \
  idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/archive/2026-04-11_doc_cleanup_duplicates \
  idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/archive/2026-04-11_doc_cleanup_v1_superseded \
  idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/archive/2026-04-12_doc_reorg_merged_sources \
  research-progress-log/archive/2026-04-14_post_stage2_review_materials
```

This keeps current mainline（当前主线）, registries（总账）, active candidates（活跃候选）, and negative-result archive（负结果归档） intact.
