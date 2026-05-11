# 2026-04-09 MGR-SID v1 Upstream-Aligned Industrial Results

## Train-Stage Best Collision

| Setting | Best sid-train collision | Best epoch |
|---|---:|---:|
| baseline | `0.1066196419` | `9899` |
| uniform_reg | `0.2457948996` | `9649` |
| hierarchy_reg | `0.1318502442` | `9149` |

Training summaries:

- [baseline summary.json](/home/leejt/OneRec/output/experiments/mgr_sid_v1_upstream/industrial_baseline/Apr-09-2026_23-09-22/summary.json)
- [uniform summary.json](/home/leejt/OneRec/output/experiments/mgr_sid_v1_upstream/industrial_uniform_reg/Apr-09-2026_23-09-22/summary.json)
- [hierarchy summary.json](/home/leejt/OneRec/output/experiments/mgr_sid_v1_upstream/industrial_hierarchy_reg/Apr-09-2026_23-09-22/summary.json)

## Generate-Stage Final Collision

| Setting | Final index collision | Max conflict |
|---|---:|---:|
| baseline | `0.0035268584` | `2` |
| uniform_reg | `0.0059685296` | `6` |
| hierarchy_reg | `0.0032555616` | `2` |

Generated summaries:

- [baseline generate summary](/home/leejt/OneRec/output/experiments/mgr_sid_v1_upstream/generated_indices/Industrial_and_Scientific.mgr_upstream_baseline.summary.json)
- [uniform generate summary](/home/leejt/OneRec/output/experiments/mgr_sid_v1_upstream/generated_indices/Industrial_and_Scientific.mgr_upstream_uniform.summary.json)
- [hierarchy generate summary](/home/leejt/OneRec/output/experiments/mgr_sid_v1_upstream/generated_indices/Industrial_and_Scientific.mgr_upstream_hierarchy.summary.json)

Generated indices:

- [baseline index.json](/home/leejt/OneRec/output/experiments/mgr_sid_v1_upstream/generated_indices/Industrial_and_Scientific.mgr_upstream_baseline.index.json)
- [uniform index.json](/home/leejt/OneRec/output/experiments/mgr_sid_v1_upstream/generated_indices/Industrial_and_Scientific.mgr_upstream_uniform.index.json)
- [hierarchy index.json](/home/leejt/OneRec/output/experiments/mgr_sid_v1_upstream/generated_indices/Industrial_and_Scientific.mgr_upstream_hierarchy.index.json)

## Key Takeaways

- If we only look at `sid-train`, the semantic `baseline` is still best, and `uniform_reg` is clearly harmful.
- After `sid-generate`, the ranking changes: `hierarchy_reg` becomes the best final index among the three.
- The final `hierarchy_reg` collision (`0.0032555616`) is lower than the final `baseline` collision (`0.0035268584`).
- This means the current `MGR-SID v1` integration is not improving raw train-stage collision, but it may still improve the final SID after collision resolution.
- `uniform_reg` is not promising in the current form and should likely be treated as a weak control rather than a serious method candidate.

## Current Interpretation

The most important fact from this run is not that the train-stage signal is perfect, but that the final generated SID does show a positive result for `hierarchy_reg`. This gives the current graph-hierarchy direction a real reason to continue. The next step should focus on validating whether this improvement persists under repeated runs and whether it translates into better downstream ambiguity behavior.
