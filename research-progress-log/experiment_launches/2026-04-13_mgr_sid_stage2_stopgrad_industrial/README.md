# 2026-04-13 MGR-SID Stage-2 Stop-Gradient Industrial

This stage launches the first retention-targeted tokenizer refinements from
`../../idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/refine-logs/archive/2026-04-14_stage3_scope_cleanup/EXPERIMENT_PLAN_STAGE2_RETENTION.md`.

## Runs

- `R202a`: stop-gradient hierarchy isolation
- `R202b`: stop-gradient hierarchy isolation + modest level-1 compensation

## Configs

- `/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_stage2_r202a.yaml`
- `/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_stage2_r202b.yaml`

## Output Roots

- `/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage2_retention_20260413/industrial_r202a_stopgrad`
- `/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage2_retention_20260413/industrial_r202b_stopgrad_comp`

## Notes

- graph bank fixed
- ambiguity prior fixed to offline combined proxy
- only stage-2 change is hierarchy stop-gradient, with and without a stronger `coarse_weight`

## Launch Status

- Date: `2026-04-13`
- `R202a`
  - tmux: `mgr_stage2_r202a`
  - GPU: `6`
  - log: `/home/leejt/OneRec/logs/experiment_mgr_sid_stage2_r202a_20260413.log`
- `R202b`
  - tmux: `mgr_stage2_r202b`
  - GPU: `5`
  - log: `/home/leejt/OneRec/logs/experiment_mgr_sid_stage2_r202b_20260413.log`

Both runs write checkpoints directly to the data disk under:

- `/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage2_retention_20260413/`

## Results Summary

Both tokenizer runs finished normally on `2026-04-13`.

- `R202a`: healthy stop-gradient branch
  - best collision: `0.1006511123`
  - best epoch: `9899`
  - summary:
    - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage2_retention_20260413/industrial_r202a_stopgrad/Apr-13-2026_00-11-11/summary.json`
- `R202b`: regressed level-1 compensation branch
  - best collision: `0.9584915898`
  - best epoch: `9299`
  - summary:
    - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage2_retention_20260413/industrial_r202b_stopgrad_comp/Apr-13-2026_00-11-11/summary.json`

## Interpretation

- `R202a` is the clear block-2 winner and should be carried forward to `R203/R204`.
- `R202b` differs from `R202a` only by the stronger `coarse_weight` compensation (`0.10` vs `0.05`), yet it regresses badly. This suggests the level-1 compensation is highly sensitive under stop-gradient isolation and should not be pushed downstream in its current form.
- `R202b` is **not discarded**. It is retained as a documented failed branch, and can be restarted later with a smaller compensation step such as a mild `coarse_weight` increase rather than a direct jump to `0.10`.

## Follow-up Status

- `R203` (`R202a -> sid-generate`) completed
  - output index:
    - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage2_retention_20260413/generated_indices/Industrial_and_Scientific.stage2_r202a_stopgrad.index.json`
  - summary:
    - `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_stopgrad_industrial/R203_r202a_generate_summary.json`
  - final generated collision: `0.0035268584`
  - max conflict: `2`

- `R204` (`current v2` vs `R202a`) completed
  - markdown:
    - `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_stopgrad_industrial/R204_v2_vs_r202a_local_ambiguity.md`
  - json:
    - `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_stopgrad_industrial/R204_v2_vs_r202a_local_ambiguity.json`
  - main readout:
    - final generated collision is unchanged (`13/3686`)
    - but test-weighted mean `l2` leaf count improves from `4.3422` to `3.6148`
    - target-weighted `H(level3 | level1, level2)` improves from `1.1001` to `1.0308`
    - fraction of targets in deep crowded `l2` buckets (`>=4` leaves) improves from `0.2228` to `0.1994`
  - caution:
    - fraction of targets in multi-leaf `same_l2` changes slightly in the wrong direction (`0.4873 -> 0.4988`), so this is a mixed-but-promising structural result rather than a clean across-the-board win

- `R202b` retry launched with a smaller compensation step
  - variant: `coarse_weight = 0.075`
  - config:
    - `/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_stage2_r202b_retry075.yaml`
  - tmux:
    - `mgr_stage2_r202b075`
  - log:
    - `/home/leejt/OneRec/logs/experiment_mgr_sid_stage2_r202b_retry075_20260413.log`
  - output root:
    - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage2_retention_20260413/industrial_r202b_retry075`

## Retry Result: `R202b-r075`

- train summary:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage2_retention_20260413/industrial_r202b_retry075/Apr-13-2026_01-48-27/summary.json`
- best train collision:
  - `0.1115029843`
- best epoch:
  - `9899`

- generated index:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage2_retention_20260413/generated_indices/Industrial_and_Scientific.stage2_r202b_retry075.index.json`
- generate summary:
  - `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_stopgrad_industrial/R203b_r202b_retry075_generate_summary.json`
- final generated collision:
  - `0.0032555616`
  - this is `12 / 3686`, better than both current `v2` and `R202a` on the final collision count

## Structural Comparison for `R202b-r075`

- `current v2` vs `R202b-r075`
  - markdown:
    - `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_stopgrad_industrial/R204b_v2_vs_r202b_retry075_local_ambiguity.md`
  - main readout:
    - final collision improves (`13 -> 12`)
    - but local ambiguity structure regresses:
      - mean target `l2` leaf count: `4.3422 -> 4.1266` (slight improvement)
      - fraction targets in multi-leaf `same_l2`: `0.4873 -> 0.5831` (worse)
      - fraction targets in deep crowded `l2>=4`: `0.2228 -> 0.2585` (worse)
      - target-weighted entropy: `1.1001 -> 1.2128` (worse)

- `R202a` vs `R202b-r075`
  - markdown:
    - `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-13_mgr_sid_stage2_stopgrad_industrial/R204c_r202a_vs_r202b_retry075_local_ambiguity.md`
  - main readout:
    - `R202b-r075` is clearly worse than `R202a` on retention-oriented structure:
      - mean target `l2` leaf count: `3.6148 -> 4.1266`
      - fraction targets in multi-leaf `same_l2`: `0.4988 -> 0.5831`
      - fraction targets in deep crowded `l2>=4`: `0.1994 -> 0.2585`
      - target-weighted entropy: `1.0308 -> 1.2128`

## Interpretation Update

- `R202b-r075` confirms that smaller level-1 compensation can recover training stability and can even improve final generated collision.
- But this gain appears to come at the cost of noticeably worse local ambiguity structure.
- So, for the retention-targeted stage-2 objective, `R202a` remains the cleaner and more promising branch to build on.
