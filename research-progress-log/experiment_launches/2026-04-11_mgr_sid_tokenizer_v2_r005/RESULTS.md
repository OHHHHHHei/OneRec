# 2026-04-11 MGR-SID Tokenizer v2 R005/R006 Results

## Variant

- run: `mgr_sid_tokenizer_v2_offline_combined`
- training config:
  `/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_tokenizer_v2.yaml`
- ambiguity prior:
  `offline_combined`

## Train-Stage Result

- run dir:
  `/home/leejt/OneRec/output/experiments/mgr_sid_tokenizer_v2/industrial_offline_combined/Apr-11-2026_01-36-05`
- summary:
  `/home/leejt/OneRec/output/experiments/mgr_sid_tokenizer_v2/industrial_offline_combined/Apr-11-2026_01-36-05/summary.json`
- best train-stage collision:
  `0.1226261530`
- best epoch:
  `9899`

## Generate-Stage Result

- generated index:
  `/home/leejt/OneRec/output/experiments/mgr_sid_tokenizer_v2/generated_indices/Industrial_and_Scientific.mgr_tokenizer_v2_offline.index.json`
- generate summary:
  `/home/leejt/OneRec/output/experiments/mgr_sid_tokenizer_v2/generated_indices/Industrial_and_Scientific.mgr_tokenizer_v2_offline.summary.json`
- final collision:
  `0.0035268584`
- max conflict:
  `2`
- collision rounds used:
  `20`

## Comparison Against Current References

### Train Stage

| setting | best sid-train collision |
|---|---:|
| MiniOneRec baseline | `0.1066196419` |
| `v1 hierarchy_reg` | `0.1318502442` |
| `v2 offline_combined` | `0.1226261530` |

### Final Generated SID

| setting | final collision | max conflict |
|---|---:|---:|
| MiniOneRec baseline | `0.0035268584` | `2` |
| `v1 hierarchy_reg` | `0.0032555616` | `2` |
| `v2 offline_combined` | `0.0035268584` | `2` |

## Current Reading

- `v2 offline_combined` improves over `v1 hierarchy_reg` at train-stage collision.
- After `sid-generate`, the gain does not survive: final collision matches the MiniOneRec baseline exactly.
- Therefore, this first `v2` run does **not** beat the MiniOneRec baseline on the final SID artifact, and it also does not surpass `v1 hierarchy_reg`.
- The most likely next diagnosis target is not whether ambiguity priors exist, but how the current ambiguity-aware weighting and semantic-retention terms interact with the final collision-resolution stage.
