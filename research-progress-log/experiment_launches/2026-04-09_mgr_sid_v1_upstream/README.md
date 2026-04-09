# 2026-04-09 MGR-SID v1 Upstream-Aligned Industrial Run

## Goal

Use the upstream MiniOneRec `RQ-VAE` training regime on `Industrial` to test whether the training-time `MGR-SID v1` integration can improve over the semantic baseline under a fair `sid-train` setup.

## Scope

- Dataset: `Industrial_and_Scientific`
- Split: existing local train csv and precomputed semantic embeddings
- Training regime: align to upstream MiniOneRec `RQ-VAE`
  - `epochs = 10000`
  - `batch_size = 20480`
  - `lr = 1e-3`
- Modes:
  - `baseline`
  - `uniform_reg`
  - `hierarchy_reg`

## Configs

- [baseline](/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_v1_upstream_baseline.yaml)
- [uniform](/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_v1_upstream_uniform.yaml)
- [hierarchy](/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_v1_upstream_hierarchy.yaml)

## Launch

- Status: launched
- Launch time: `2026-04-09 23:09` Asia/Shanghai
- GPUs:
  - `baseline`: GPU `2`
  - `uniform_reg`: GPU `3`
  - `hierarchy_reg`: GPU `4`
- tmux sessions:
  - `mgr_up_base_ind`
  - `mgr_up_uniform_ind`
  - `mgr_up_hier_ind`

## Logs

- `/home/leejt/OneRec/logs/experiment_mgr_sid_v1_upstream_industrial_baseline_20260409.log`
- `/home/leejt/OneRec/logs/experiment_mgr_sid_v1_upstream_industrial_uniform_20260409.log`
- `/home/leejt/OneRec/logs/experiment_mgr_sid_v1_upstream_industrial_hierarchy_20260409.log`

## Commands

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/experiment_mgr_sid_v1_train.py \
  --config config/experiments/sid_train_industrial_mgr_sid_v1_upstream_baseline.yaml \
  --device cuda:0

CUDA_VISIBLE_DEVICES=3 python scripts/experiment_mgr_sid_v1_train.py \
  --config config/experiments/sid_train_industrial_mgr_sid_v1_upstream_uniform.yaml \
  --device cuda:0

CUDA_VISIBLE_DEVICES=4 python scripts/experiment_mgr_sid_v1_train.py \
  --config config/experiments/sid_train_industrial_mgr_sid_v1_upstream_hierarchy.yaml \
  --device cuda:0
```

## Monitoring

```bash
tmux ls
tmux attach -t mgr_up_base_ind
tmux attach -t mgr_up_uniform_ind
tmux attach -t mgr_up_hier_ind

tail -f /home/leejt/OneRec/logs/experiment_mgr_sid_v1_upstream_industrial_baseline_20260409.log
tail -f /home/leejt/OneRec/logs/experiment_mgr_sid_v1_upstream_industrial_uniform_20260409.log
tail -f /home/leejt/OneRec/logs/experiment_mgr_sid_v1_upstream_industrial_hierarchy_20260409.log
```

## Outputs

- `/home/leejt/OneRec/output/experiments/mgr_sid_v1_upstream/industrial_baseline`
- `/home/leejt/OneRec/output/experiments/mgr_sid_v1_upstream/industrial_uniform_reg`
- `/home/leejt/OneRec/output/experiments/mgr_sid_v1_upstream/industrial_hierarchy_reg`

## Notes

- This directory is the single place for this run's launch and follow-up notes.
- Existing baseline/default configs were not modified.
- This run only covers `Industrial`, as requested.
