# R700a Semantic-Collaborative Intersection

R700a tests a cleaner tokenizer（分词器） design after R693a underperformed the original MiniOneRec baseline（基线）.

Core design:

- L1 uses only high-confidence semantic-collaborative intersection positives（语义-协同交集正边）.
- L2 uses semantic-conditioned local-multihop InfoNCE（语义条件下的局部多跳对比损失）.
- L3 uses a very weak local pull（局部拉近）.
- Generic graph smoothness（通用图平滑）, semantic retention（语义保持）, and selective separation（选择性推远） are disabled.
- stop-gradient（停止梯度） from L2/L3 auxiliary losses to previous levels stays enabled.

Primary question:

Can a conservative, semantically clean L1 improve first-token routing（第一层路由） while leaving stronger collaborative refinement（协同细分） to L2?

Important files:

- Config: `config/experiments/sid_train_industrial_mgr_sid_r700a_semantic_collab_intersection.yaml`
- Graph builder: `scripts/experiment_mgr_sid_r700a_graph_sources.py`
- Launcher: `scripts/launch_mgr_sid_r700_semantic_collab_intersection_tmux.sh`

## Final Tokenizer Result

Status: tokenizer/generate（分词器训练与生成） completed, but the result is not promoted to SFT（监督微调） by default.

Core artifacts:

- Train summary: `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r700_semantic_collab_intersection_20260418/industrial_r700a_semantic_collab_intersection/Apr-18-2026_19-50-21/summary.json`
- Best checkpoint: `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r700_semantic_collab_intersection_20260418/industrial_r700a_semantic_collab_intersection/Apr-18-2026_19-50-21/best_collision_model.pth`
- Generated SID index（生成语义 ID 索引）: `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r700_semantic_collab_intersection_20260418/generated_indices/Industrial_and_Scientific.r700a_semantic_collab_intersection.index.json`
- Result snapshot: `R700A_TOKENIZER_RESULT_SNAPSHOT.md`

Key numbers:

- Best train collision（训练最佳冲突）: `0.08491589799240369` at epoch `9699`
- Generated collision（生成冲突）: `14 / 3686 = 0.0037981552`
- Max conflict（最大冲突簇）: `2`
- Active L1（活跃第一层码）: `125`
- Unique L2 pairs（唯一第二层前缀数）: `2631`
- Unique SID（唯一语义 ID）: `3672`

Main diagnosis:

- R700a did not collapse, and collision（冲突） is acceptable.
- However, it does not improve the target L1 routing（第一层路由） problem versus the original MiniOneRec baseline（原版基线）.
- `target_l1_in_history`（目标第一层出现在历史中） is `0.3018`, below original `0.4019` and below R693a `0.3446`.
- Focus families（重点物品族） such as `3d_filament`, `tape`, `connector_fitting`, and `fastener` become more fragmented（更碎片化） than in R693a/original, while `adhesive_epoxy` improves.

Interpretation:

The semantic-collaborative intersection（语义-协同交集） idea is conceptually clean, but the current L1 graph（第一层图） is too sparse/conservative: coverage is only `64.1%`, mean row degree is `1.55`, and median row degree is `1`. It creates many local high-purity fragments instead of a routeable coarse entrance（可路由的粗粒度入口）. This is likely harmful for downstream LLM（大语言模型） learning, so this tokenizer should not be the next default SFT candidate.
