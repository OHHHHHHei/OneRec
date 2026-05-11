# 2026-04-18 R710a v2 No Semantic Retention（去语义保持消融）

Status（状态）: `tokenizer_finished`

## Question（问题）

Does `semantic retention`（语义保持） in `MGR-SID v2` provide necessary downstream-relevant structure, or can it be removed because the `RQ-VAE`（残差量化变分自编码器） input and reconstruction objective already preserve semantic information?

## Design（设计）

`R710a` is a strict ablation（严格消融） of the current `v2` tokenizer（分词器）:

- keep `RQ-VAE` backbone（骨干） unchanged
- keep `offline_combined` ambiguity prior（离线组合歧义先验） unchanged
- keep graph assignment（图分配） unchanged:
  - `L1 <- coarse_purified`
  - `L2 <- fagsp_mid_base`
  - `L3 <- local_purified`
- keep graph weights（图权重） unchanged:
  - `coarse_weight = 0.05`
  - `mid_weight = 0.15`
  - `local_weight = 0.05`
- disable semantic retention（关闭语义保持）:
  - `semantic_coarse_weight = 0.0`
  - `semantic_mid_weight = 0.0`

## Files（文件）

- config（配置）: `/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_r710a_v2_no_semantic_retention.yaml`
- train/generate script（训练与生成脚本）: `/home/leejt/OneRec/scripts/experiment_mgr_sid_r710a_v2_no_semantic_retention_train_generate.sh`
- launch script（启动脚本）: `/home/leejt/OneRec/scripts/launch_mgr_sid_r710_v2_no_semantic_retention_tmux.sh`
- log（日志）: `/home/leejt/OneRec/logs/experiment_mgr_sid_r710a_v2_no_semantic_retention_20260418.log`
- checkpoints（权重）: `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r710_v2_no_semantic_retention_20260418/industrial_r710a_v2_no_semantic_retention`
- generated SID（生成 SID）: `/data/leejt/OneRec/output_weights/experiments/mgr_sid_r710_v2_no_semantic_retention_20260418/generated_indices/Industrial_and_Scientific.r710a_v2_no_semantic_retention.index.json`

## Interpretation（解释口径）

If `R710a` remains close to `v2`, then semantic retention（语义保持） is likely not essential and should be removed or downgraded from the main method. If `R710a` collapses or drops clearly, then semantic retention（语义保持） should be described as a stability anchor（稳定锚点）, not as an additional semantic information source（额外语义信息来源）.

## Result（结果）

Training and SID generation（训练与 SID 生成） finished at `2026-04-18 23:10:58 CST`.

| Metric（指标） | `R710a` | `v2 offline` |
|---|---:|---:|
| best train collision（训练最佳冲突率） | `0.1454150841` | `0.1226261530` |
| generated collision（生成冲突率） | `0.0029842648` | `0.0035268584` |
| generated collision count（生成冲突数） | `11 / 3686` | `13 / 3686` |
| max conflict（最大冲突簇） | `2` | `2` |
| active L1（活跃第一层码） | `123` | `203` |
| unique L2 pairs（唯一第二层前缀数） | `2234` | `2680` |

Reading（解读）:

- `R710a` is not a tokenizer collapse（分词器塌缩）.
- Removing semantic retention（语义保持） does not hurt generated collision（生成冲突）; it slightly improves this narrow artifact metric（窄口径产物指标）.
- However, prefix space（前缀空间） becomes more compressed: active L1（活跃第一层码） and unique L2 pairs（唯一第二层前缀数） both drop clearly versus `v2`.
- Therefore, this result supports a careful interpretation: semantic retention（语义保持） may not be necessary for collision repair（冲突修复）, but it may still help preserve routeable hierarchy（可路由层级）.
- Downstream SFT/evaluate（监督微调/评测） is required before claiming semantic retention（语义保持） is removable.
