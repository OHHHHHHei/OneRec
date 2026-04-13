# 2026-04-13 Stage-2 Semantic Retention KL Results

## Run Identity

- primary run:
  `R205`
- summary:
  `/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage2_retention_20260413/industrial_r205_stopgrad_kl/Apr-13-2026_02-15-01/summary.json`
- generated index:
  `/data/leejt/OneRec/output_weights/experiments/mgr_sid_stage2_retention_20260413/generated_indices/Industrial_and_Scientific.stage2_r205_stopgrad_kl.index.json`

## Tokenizer Training Result

- best train collision:
  `0.1136733587`
- best epoch:
  `1149`
- final epoch collision:
  `0.1155724362`

Compared with `R202a`:

- `R202a best train collision = 0.1006511123`
- `R205 best train collision = 0.1136733587`

So the first `batch_local_kl` implementation underperforms the simpler
`stop-grad` branch during tokenizer training.

## Final Generate Result

- final generated collision:
  `0.0032555616`
- collision count:
  `12 / 3686`

This is numerically better than:

- current `v2`: `13 / 3686`
- `R202a`: `13 / 3686`

## Post-Generate Structural Diagnosis

### vs current `v2`

- mean target `l2` leaf count:
  `4.3422 -> 4.9572`
- fraction targets in multi-leaf `same_l2`:
  `0.4873 -> 0.5449`
- fraction targets in deep crowded `l2>=4`:
  `0.2228 -> 0.2621`
- target-weighted entropy:
  `1.1001 -> 1.2623`

### vs `R202a`

- mean target `l2` leaf count:
  `3.6148 -> 4.9572`
- fraction targets in multi-leaf `same_l2`:
  `0.4988 -> 0.5449`
- fraction targets in deep crowded `l2>=4`:
  `0.1994 -> 0.2621`
- target-weighted entropy:
  `1.0308 -> 1.2623`

## Interpretation

`R205` is not a collapse, but it is a clear tradeoff branch:

- it improves final collision from `13` to `12`
- but it does so by making the local ambiguity structure significantly worse

So the current stage-2 conclusion is:

> the first `batch_local_kl` semantic-retention implementation should **not**
> replace `R202a` as the main tokenizer candidate, and it should **not** be
> pushed downstream in its current form.
