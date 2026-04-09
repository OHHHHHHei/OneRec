# Paper-Transplant Probe Run (2026-04-09)

## What was added

This round kept the previous baseline code and the previous probe untouched.  
All changes were added as **isolated experimental modules**:

- `src/onerec/experiments/mgr_sid/paper_transplants.py`
- `src/onerec/experiments/mgr_sid/transplanted_graph_bank.py`
- `scripts/experiment_mgr_sid_paper_transplant_probe.py`

The goal was to test whether two families of ideas borrowed from related work are useful in our setting:

- `PRISM`-inspired semantic-anchor purification
- `GSPRec / FaGSP`-inspired spectral middle-resolution graph views

This is still a **probe** on top-1 SID errors, not a training-time tokenizer modification.

## What was run

Industrial:

- `results/experiments/mgr_sid_paper_transplant_probe_industrial_20260409_1/summary.md`
- `results/experiments/mgr_sid_paper_transplant_probe_industrial_20260409_1/summary.json`
- `logs/experiment_mgr_sid_paper_transplant_probe_industrial_20260409_1.log`

Office:

- `results/experiments/mgr_sid_paper_transplant_probe_office_20260409_1/summary.md`
- `results/experiments/mgr_sid_paper_transplant_probe_office_20260409_1/summary.json`
- `logs/experiment_mgr_sid_paper_transplant_probe_office_20260409_1.log`

## New transplanted views

- `prism_anchor_coarse`
- `prism_anchor_local`
- `fagsp_mid_base`
- `fagsp_mid_prism`
- `gsprec_mid_prism`

Here:

- `prism_anchor_*` means graph purification guided by semantic neighbors
- `fagsp_mid_*` means a spectral band reconstruction view
- `gsprec_mid_prism` means a temporal-mixed spectral middle-resolution view

## Main findings

### 1. The transplanted spectral mid views are much stronger than the previous mid-view proxies

Industrial:

- previous best mid view: `mid_band_pass_purified = 0.2747` on `same_l2`
- transplanted best mid view: `fagsp_mid_base = 0.5401`
- transplanted second best: `gsprec_mid_prism = 0.5247`

Office:

- previous best mid view: `mid_diffusion_purified = 0.2381` on `same_l2`
- transplanted best mid view: `fagsp_mid_base = 0.7619`
- transplanted second best: `gsprec_mid_prism = 0.6429`

This is the clearest signal so far that:

> the current `G_mid` should probably be designed as a proper graph-spectral middle-resolution collaborative view, instead of only using heuristic diffusion / band-pass proxies.

### 2. `PRISM`-style semantic anchoring alone is not the main win

The anchor-only purified graphs are not dominant:

- Industrial:
  - `prism_anchor_coarse = 0.3241` on `same_l2`
  - `prism_anchor_local = 0.3056`
- Office:
  - `prism_anchor_coarse = 0.5119`
  - `prism_anchor_local = 0.4167`

They are useful, but the real gain comes when the graph is further turned into a spectral middle-resolution view.

Interpretation:

- semantic anchoring may be a helpful purification step
- but the biggest gain currently comes from **how the middle-resolution collaborative structure is represented**

### 3. `FaGSP`-style mid view is currently the strongest single candidate

Across both datasets:

- `fagsp_mid_base` is the top-ranked transplanted mid view
- it has both high `same_l2` and high overall coverage

This makes it the strongest current candidate for `G_mid v1`.

### 4. `GSPRec`-style temporal mixing is also promising

`gsprec_mid_prism` is consistently near the top:

- Industrial: `0.5247`
- Office: `0.6429`

This is useful because it suggests:

- temporal transition information still matters
- but it may matter most when folded into a middle-resolution spectral view, instead of being used as a standalone local graph

## What this run supports

- paper-inspired module transplantation is worth doing
- `G_mid` should be the priority module
- the graph-frequency / graph-spectral route looks much stronger than the first heuristic mid-view design
- a pure "anchor-only denoising" story is probably not enough by itself

## What this run still does not prove

- training-time `MGR-SID` improvement on main recommendation metrics
- superiority over published baselines
- whether the probe gain will survive full SID integration

Also, the gains here are large enough that we should be cautious:

- spectral reconstruction may create smoother and more expressive candidate relations
- but it may also make the probe easier than the real training setup

So this result should be treated as:

> a strong green light for the next stage, not a final paper claim

## Immediate next step

The most justified next move is now:

1. Freeze `G_mid v1 = fagsp_mid_base`
2. Keep `gsprec_mid_prism` as the main alternative
3. Integrate `G_coarse + G_mid + G_local` into a training-time experimental `MGR-SID v1`
4. Compare:
   - semantic-only MiniOneRec
   - naive fusion
   - uniform graph regularization
   - hierarchy-aware graph regularization
