# 2026-04-09 Pipeline Alignment and Reproduction

This folder merges the early provenance and reproduction notes that were previously scattered as flat files under `experiment_launches/`.

## Why This Stage Matters

This stage established three foundational facts that every later MGR-SID experiment depends on:

1. the current repository default `sid-train` settings do not reproduce the official low-collision MiniOneRec tokenizer
2. upstream-style RQ-VAE training does reproduce the official-quality final SID after `sid-generate`
3. tokenizer quality should be judged on the final generated SID, not only on train-stage collision

## Merged Source Notes

The original split notes are now archived under:

- `archive/2026-04-09_mgr_sid_v1_aligned_training.md`
- `archive/2026-04-09_sid_pipeline_reproduction.md`
- `archive/2026-04-09_upstream_style_rqvae_launch.md`
- `archive/2026-04-09_upstream_style_rqvae_result.md`

## Consolidated Story

### 1. Default fresh rerun did not reproduce the official SID

Using the current repo default `sid-train -> sid-generate` path, a clean rerun produced:

- very high train-stage collision
- very high final generated collision
- zero item-wise exact SID match with the official index

This showed that the issue was real and that the repo had a provenance gap if we only looked at the default config.

### 2. Upstream-style RQ-VAE fixed the reproduction gap

After aligning to the upstream MiniOneRec RQ-VAE regime:

- `epochs = 10000`
- `batch_size = 20480`
- `lr = 1e-3`

the current repo successfully reproduced the official Industrial SID quality after `sid-generate`.

The important point is:

> the core pipeline was not broken;
> the mismatch was mainly in training regime, not in the existence of `sid-generate` itself.

### 3. Train-stage and final generated SID are not the same evaluation object

This stage also made another important distinction explicit:

- `sid-train` collision can still be relatively high
- but `sid-generate` may substantially repair the final index

From this point onward, the project stopped using train-stage collision as the final tokenizer verdict.

### 4. Aligned training sanity clarified what was and was not the problem

The aligned-training sanity run showed that once the training semantics were matched correctly, experimental baselines and the mainline baseline agreed.

That meant:

- old full-batch or semantically mismatched runs should not be treated as valid evidence
- later graph-method judgments had to be made against a properly aligned MiniOneRec reproduction

## What This Stage Concluded

The practical conclusion of this stage was:

> the trustworthy MiniOneRec tokenizer baseline is the upstream-style reproduced one, and all later tokenizer comparisons should be anchored to final generated SID quality rather than default-config train-stage collision.

## What It Enabled Next

This stage unlocked the next two stages:

- `2026-04-09_mgr_sid_v1_upstream/`
  - first upstream-aligned tokenizer comparison
- later `v2` tokenizer experiments
  - because the baseline provenance was finally stable enough to support real method comparisons
