# 2026-04-14 MGR-SID Stage-3 Prefix-Retained Industrial

This launch starts the first Stage-3 tokenizer-only mechanism-validation run
from `EXPERIMENT_PLAN_STAGE3_PREFIX_RETAINED_HIERARCHY.md`.

## Narrative Calibration

Stage-3 is **not** trying to make SID stay close to the current baseline by
default.

The actual project goal is:

> to find a better SID codebook space for downstream recommendation learning.

So `R401b`, `R401d`, and later branches should be read as **candidate codebook
spaces**.

In that framing:

- prefix stability is a diagnostic
- codebook drift is a diagnostic
- the hardest selection criterion is still full downstream `SFT -> evaluate`
- staying near `v2` is only one hypothesis, not the project objective

## Run

- `R401b-g0.05`: warm-start `v2` + `L1/L2` teacher-guided retention

## Goal

Test one conservative Stage-3 hypothesis first:

- keep the existing hierarchy-aware three-level graph supervision
- keep the existing semantic retention terms
- warm-start from the current best `v2` checkpoint
- add light `L1/L2` prefix-stabilizing retention as one candidate design
- do **not** add ambiguity-aware weighting yet
- do **not** push to SFT yet

## Config

- `/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_stage3_r401b_g005.yaml`

## Teacher / Warm-Start Checkpoint

- `/data/leejt/OneRec/output_weights/experiments/mgr_sid_tokenizer_v2/industrial_offline_combined/Apr-11-2026_01-36-05/best_collision_model.pth`

## Output Root

- `/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage3_prefix_retained_20260414/industrial_r401b_g005`

## Launch Status

- Date: `2026-04-14`
- tmux: `mgr_stage3_r401b_g005`
- GPU: `2`
- status: `COMPLETED`
- train pid:
  - `3306240`
- Log:
  - `/home/leejt/OneRec/logs/experiment_mgr_sid_stage3_r401b_g005_20260414.log`

## Sanity Status

- 1-epoch sanity: `PASSED`
- sanity output:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage3_prefix_retained_20260414/sanity_r401b_g005/Apr-14-2026_00-46-46/summary.json`
- sanity readout:
  - warm-start load: success
  - frozen teacher load: success
  - one training epoch + eval: success
  - logged retention losses:
    - `retain_l1 = 0.103063`
    - `retain_l2 = 0.098333`

## `R401b-g0.05` Final Status

- train summary:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage3_prefix_retained_20260414/industrial_r401b_g005/Apr-14-2026_00-47-31/summary.json`
- best train collision:
  - `0.0903418340`
- best epoch:
  - `9949`

- generated index:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage3_prefix_retained_20260414/generated_indices/Industrial_and_Scientific.stage3_r401b_g005.index.json`
- generate summary:
  - `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-14_mgr_sid_stage3_prefix_retained_industrial/R402_r401b_g005_generate_summary.json`
- final generated collision:
  - `0.0029842648`
  - this is `11 / 3686`

## Current Diagnostic Readout

- prefix stability:
  - `R401b` did **not** recover the planned upper-prefix stability gate
  - compared with `R202a`:
    - `l1` pair retention is slightly lower (`0.4106` vs `0.4137`)
    - `l2` pair retention is much lower (`0.4542` vs `0.6122`)
- local ambiguity:
  - `R401b` is much more aggressive than both `current v2` and `R202a`
  - vs `current v2`:
    - mean target `l2` leaf count: `4.3422 -> 2.6967`
    - target-weighted entropy: `1.1001 -> 0.7373`
  - vs `R202a`:
    - mean target `l2` leaf count: `3.6148 -> 2.6967`
    - target-weighted entropy: `1.0308 -> 0.7373`
- codebook drift:
  - large drift remains despite representation retention
  - quick readout vs current `v2` checkpoint:
    - `L1` relative RMS drift: `1.5461`
    - `L2` relative RMS drift: `1.4920`
    - `L3` relative RMS drift: `1.4634`

## Interpretation

- `R401b-g0.05` is a useful mechanism result, but it is **not** eliminated as a
  downstream candidate only because prefix stability is weak.
- What it shows is:
  - plain representation-level retention does **not** achieve the intended
    conservative-prefix hypothesis
  - yet it still creates a very different and structurally stronger SID space
- So the correct reading is:
  - `R401b` remains a valid candidate codebook space
  - `R401d` is launched as a parallel follow-up that tests whether adding a
    codebook anchor yields an even better candidate
- Final downstream selection should still be made by full `SFT -> evaluate`.

## Follow-up Launch: `R401d-g0.05-a0.05`

- variant:
  - `R401b` recipe + light `L1/L2` codebook anchor
- config:
  - `/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_stage3_r401d_g005_a005.yaml`
- tmux:
  - `mgr_stage3_r401d_g005_a005`
- GPU:
  - `3`
- status:
  - `COMPLETED`
- log:
  - `/home/leejt/OneRec/logs/experiment_mgr_sid_stage3_r401d_g005_a005_20260414.log`

## `R401d` Sanity Status

- 1-epoch sanity: `PASSED`
- sanity output:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage3_prefix_retained_20260414/sanity_r401d_g005_a005/Apr-14-2026_17-02-35/summary.json`
- sanity readout:
  - warm-start load: success
  - frozen teacher load: success
  - anchor loss active:
    - `anchor_l1 = 0.351512`
    - `anchor_l2 = 0.172048`

## `R401d-g0.05-a0.05` Final Status

- train summary:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage3_prefix_retained_20260414/industrial_r401d_g005_a005/Apr-14-2026_17-03-32/summary.json`
- best train collision:
  - `0.0887140532`
- best epoch:
  - `8649`
- generated index:
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage3_prefix_retained_20260414/generated_indices/Industrial_and_Scientific.stage3_r401d_g005_a005.index.json`
- generate summary:
  - `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-14_mgr_sid_stage3_prefix_retained_industrial/R402b_r401d_g005_a005_generate_summary.json`
- final generated collision:
  - `0.0029842648`
  - this is `11 / 3686`

## `R401d` Immediate Diagnostic Readout

- prefix stability vs `current_v2`:
  - changed `l1` rate drops from `0.9957` (`R401b`) to `0.9631`
  - changed `l2` rate drops from `1.0000` (`R401b`) to `0.9965`
  - but pair retention does **not** improve:
    - `l1` pair retention: `0.4065` vs `0.4106` in `R401b`
    - `l2` pair retention: `0.4381` vs `0.4542` in `R401b`
- ambiguity-bucket readout:
  - `l1` retention is better than `R401b` on `easy` and `medium`, but worse on `hard`
  - `l2` retention is still weak overall and remains below `R202a`
- local ambiguity:
  - `R401d` is even more aggressive than `R401b`
  - vs `current v2`:
    - mean target `l2` leaf count: `4.3422 -> 2.5711`
    - target-weighted entropy: `1.1001 -> 0.7156`
  - vs `R202a`:
    - mean target `l2` leaf count: `3.6148 -> 2.5711`
    - target-weighted entropy: `1.0308 -> 0.7156`
- codebook drift vs current `v2` checkpoint:
  - reduced on the anchored upper levels relative to `R401b`
    - `L1` relative RMS drift: `1.2355` vs `1.5461`
    - `L2` relative RMS drift: `0.8609` vs `1.4920`
  - `L3` remains large:
    - `L3` relative RMS drift: `1.4544` vs `1.4634`
- code polysemy:
  - `b/c` prefix drift is slightly cleaner than `R401b`
  - but the branch is still far from a conservative-prefix recovery

## Current Comparison Takeaway

- `R401d` does succeed at what it was explicitly designed to test:
  - it pulls `L1/L2` codebooks closer to the `v2` teacher than `R401b` does
- But that tighter codebook anchoring does **not** translate into better
  upper-prefix pair retention.
- Instead, the branch still produces a globally different SID space and pushes
  local ambiguity cleanup even harder than `R401b`.
- So the current tokenizer-side reading is:
  - `R401d` is **not** a clean conservative-prefix fix
  - `R401d` is still a valid candidate codebook space
  - the final choice between `R401b` and `R401d` should now be delegated to
    full downstream `SFT -> evaluate`

## Diagnostics Artifact Paths

- `R401d` stage-3 diagnostics dir:
  - `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-14_mgr_sid_stage3_prefix_retained_industrial/r401d_g005_a005_diagnostics`
- local ambiguity vs `current_v2`:
  - `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-14_mgr_sid_stage3_prefix_retained_industrial/R403c_v2_vs_r401d_local_ambiguity.md`
- local ambiguity vs `R202a`:
  - `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-14_mgr_sid_stage3_prefix_retained_industrial/R403d_r202a_vs_r401d_local_ambiguity.md`

## Notes

- This is the first Stage-3 run because it is the cleanest test of whether
  plain `L1/L2` retention can already improve SID-space learnability.
- `R304` multi-seed learnability probe for `R401d` is slower than the other
  tokenizer-side diagnostics and may finish later than the immediate readout
  above.
