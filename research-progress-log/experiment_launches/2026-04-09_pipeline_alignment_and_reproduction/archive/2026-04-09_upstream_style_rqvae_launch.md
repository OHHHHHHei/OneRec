# Upstream-Style RQ-VAE Launch (2026-04-09)

## Goal

Run a clean reproduction attempt using hyperparameters aligned to the upstream MiniOneRec repository, to test whether the current repo's poor `sid-train -> sid-generate` reproduction is caused by mismatched training hyperparameters.

Current decision:

- only launch `Industrial`
- do not launch `Office` in this round

## Upstream Reference

Upstream MiniOneRec repository:

- `AkaliKong/MiniOneRec`
- local mirror:
  - `/home/leejt/.cache/onerec_upstream_refs/MiniOneRec`

Relevant upstream training script:

- `/home/leejt/.cache/onerec_upstream_refs/MiniOneRec/rq/rqvae.sh`

The upstream script uses:

- `lr = 1e-3`
- `epochs = 10000`
- `batch_size = 20480`

## Launch Plan

We launch one long-running local job in an isolated `tmux` session:

- Industrial on GPU 2

Both jobs write to a dedicated reproduction-only output root:

- `output/reproductions/2026-04-09_upstream_style_rqvae/`

This avoids mixing with:

- current repo outputs
- previous reproduction attempts
- existing official `data/Amazon/index/*.index.json`

## Commands

Industrial:

```bash
CUDA_VISIBLE_DEVICES=2 python -m onerec.main sid-train \
  --config config/sid_train.yaml \
  data_path=./data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy \
  device=cuda:0 \
  epochs=10000 \
  batch_size=20480 \
  num_workers=4 \
  ckpt_dir=./output/reproductions/2026-04-09_upstream_style_rqvae/industrial_sid_train
```

## tmux Sessions

- `rqvae_upstream_ind`

## Logs

- `logs/repro_upstream_style_sid_train_industrial_20260409.log`

## Expected Follow-Up

If training completes successfully, the next step is:

1. locate `best_collision_model.pth`
2. run `sid-generate` in the same isolated reproduction tree
3. compare reproduced `index.json` collision with:
   - current repo official index
   - previous low-epoch rerun

## Runtime Estimate

These jobs are long but still feasible because the upstream batch size exceeds the dataset size, so each epoch is effectively one batch.

Rough estimate:

- around `15` to `40` minutes for this Industrial run, depending on k-means init and checkpoint overhead

This is only an estimate, not a guarantee.
