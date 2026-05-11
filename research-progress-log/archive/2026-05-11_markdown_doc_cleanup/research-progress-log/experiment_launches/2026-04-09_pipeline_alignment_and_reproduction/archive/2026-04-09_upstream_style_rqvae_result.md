# Upstream-Style RQ-VAE Result (2026-04-09)

## Summary

Using upstream-style RQ-VAE hyperparameters from the official MiniOneRec repository:

- `epochs = 10000`
- `batch_size = 20480`
- `lr = 1e-3`

the current repo successfully reproduced the low-collision Industrial SID index after `sid-generate`.

## Training

Checkpoint root:

- `output/reproductions/2026-04-09_upstream_style_rqvae/industrial_sid_train/Apr-09-2026_21-50-02`

Training log:

- `logs/repro_upstream_style_sid_train_industrial_20260409.log`

Best training collision:

- `Best Collision Rate = 0.09929462832338579`

This is much better than the current repo default-config rerun (`~0.9989`), which confirms that the default `sid_train.yaml` hyperparameters are not aligned with the upstream MiniOneRec RQ-VAE recipe.

## Generation

Generated index:

- `output/reproductions/2026-04-09_upstream_style_rqvae/generated_indices/Industrial_and_Scientific.upstream_style.index.json`

Generation log:

- `logs/repro_upstream_style_sid_generate_industrial_20260409.log`

Generated index collision:

- `0.004340748779164406`

Max conflict:

- `3`

## Comparison with Existing Official Index

Existing official index:

- `data/Amazon/index/Industrial_and_Scientific.index.json`

Its collision is also:

- `0.004340748779164406`

and its max conflict is also:

- `3`

## Exact Match Check

Item-wise exact token sequence match between:

- reproduced upstream-style index
- current official repo index

is:

- `0 / 3686`

## Interpretation

This means:

1. The current repo **can** reproduce the official low-collision Industrial SID quality, if the training hyperparameters are aligned with upstream MiniOneRec.
2. The earlier reproduction failure was mainly caused by using the current repo default `sid_train.yaml` settings (`epochs=10`, `batch_size=256`), not by a broken `sid-train` or `sid-generate` code path.
3. The reproduced index and the existing official index are **not item-wise identical**, but they are collision-equivalent.

The most likely explanation is codebook permutation / alternative valid SID assignment:

- different code IDs
- same overall collision quality
- same 3-level SID structure

## Main Conclusion

The provenance gap is now substantially narrowed:

> The current official Industrial index is consistent with an upstream-style `RQ-VAE (10000 epochs, 20480 batch) -> sid-generate` pipeline.

So the primary issue was not a migration bug in the core logic, but a mismatch between:

- the current repo default training config
- and the upstream MiniOneRec RQ-VAE recipe that likely produced the official low-collision SID index.
