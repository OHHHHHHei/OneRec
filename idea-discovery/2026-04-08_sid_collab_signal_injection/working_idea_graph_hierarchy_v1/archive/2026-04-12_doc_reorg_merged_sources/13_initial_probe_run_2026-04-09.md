# Initial Probe Run (2026-04-09)

## What was actually run

This round did **not** modify the existing MiniOneRec training/evaluation pipeline.  
Instead, we added an isolated experimental probe under:

- `src/onerec/experiments/mgr_sid/`
- `scripts/archive/retired_prior_diagnostics/experiment_mgr_sid_graph_bank_probe.py`

The goal was to test one small but important hypothesis for the current direction:

> Can graph-structured collaborative views provide useful signals on top-1 SID errors, especially inside local ambiguity buckets such as `same_l2`?

## Outputs

Industrial:

- `results/experiments/mgr_sid_graph_bank_probe_industrial_20260409_1/summary.md`
- `results/experiments/mgr_sid_graph_bank_probe_industrial_20260409_1/summary.json`
- `logs/experiment_mgr_sid_graph_bank_probe_industrial_20260409_1.log`

Office:

- `results/experiments/mgr_sid_graph_bank_probe_office_20260409_1/summary.md`
- `results/experiments/mgr_sid_graph_bank_probe_office_20260409_1/summary.json`
- `logs/experiment_mgr_sid_graph_bank_probe_office_20260409_1.log`

## Views tested

- `coarse_raw`, `coarse_purified`
- `local_raw`, `local_purified`
- `mid_diffusion_raw`, `mid_diffusion_purified`
- `mid_band_pass_raw`, `mid_band_pass_purified`
- `mid_community_raw`, `mid_community_purified`

Here:

- `coarse` = stable global collaborative graph
- `local` = transition-style short-range graph
- `mid` = candidate middle-resolution collaborative graph
- `purified` = a light denoising / pruning version of the corresponding view

## Main observations

### 1. Mid-resolution graph views are clearly useful

On both datasets, the strongest `same_l2` signals come from **purified mid views**, not from the simplest local graph:

- Industrial best `same_l2`: `mid_band_pass_purified = 0.2747`
- Office best `same_l2`: `mid_diffusion_purified = 0.2381`

This is encouraging for the current direction because it supports the idea that:

> the useful collaborative signal for deep leaf ambiguity is not purely global and not purely last-step local; a middle-resolution graph view may be the most informative.

### 2. Purification often helps the mid views

Examples:

- Industrial:
  - `mid_diffusion_raw = 0.2253` -> `mid_diffusion_purified = 0.2593` on `same_l2`
  - `mid_band_pass_raw = 0.2593` -> `mid_band_pass_purified = 0.2747`
- Office:
  - `mid_diffusion_raw = 0.1429` -> `mid_diffusion_purified = 0.2381`
  - `mid_band_pass_raw = 0.1667` -> `mid_band_pass_purified = 0.2143`

So the "graph as collaborative carrier" idea currently looks stronger when the graph view is lightly purified instead of used in raw form.

### 3. Community-style mid view is strong globally, but weak on `same_l2`

This pattern is stable enough to be meaningful:

- Industrial:
  - `mid_community_purified`: `all = 0.2766`, `same_l2 = 0.1512`
- Office:
  - `mid_community_purified`: `all = 0.2472`, `same_l2 = 0.1905`

Interpretation:

- community-style graphs may be good as a broad collaborative prior
- but they currently do **not** look like the best carrier for local leaf disambiguation

This matters because it suggests that different graph views may really serve different levels or roles.

### 4. Coarse and local views still matter, but they do different jobs

Examples:

- Industrial `coarse_purified` and `local_purified` both reach `0.3148` on `same_l2`, but with lower coverage
- Office `coarse_raw = 0.6071` on `same_l2`, while `local_raw = 0.4286`

These numbers should be read carefully:

- strong bucket performance does **not** mean the view should dominate the whole tokenizer
- coverage still matters
- local graphs remain useful, but they are not obviously the only answer

## What this run does and does not prove

### It supports

- graph-structured collaborative information is worth exploring for SID construction
- middle-resolution graph views deserve to be treated as a first-class design object
- light purification is likely necessary
- different graph views show different strengths, which is consistent with a hierarchy-aware direction

### It does not yet prove

- full `MGR-SID` beats published baselines
- graph regularization is better than feature fusion inside end-to-end training
- a specific level-to-view assignment is already fixed

This run is still a **probe**, not the final method.

## Immediate next step

The most natural next step is:

> freeze a small `v1` graph bank and move from post-hoc probing to training-time integration.

Current practical choice for `v1`:

- `G_coarse`: purified coarse collaborative graph
- `G_mid`: purified band-pass or purified diffusion residual
- `G_local`: purified transition graph

Then compare:

- semantic-only MiniOneRec
- naive graph feature fusion
- uniform graph regularization
- hierarchy-aware graph regularization (`MGR-SID v1`)
