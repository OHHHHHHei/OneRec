# 2026-04-10 MGR-SID SFT + Evaluate (Industrial)

## Goal

Run parallel `SFT -> evaluate` chains for the two Industrial SID variants prepared under `data_experiment/`:

- `mgr_upstream_baseline`
- `mgr_upstream_hierarchy`

The default `data/`, `config/sft.yaml`, and `config/evaluate.yaml` mainline paths remain untouched.

## Data Roots

- Baseline data root:
  `/home/leejt/OneRec/data_experiment/Amazon/mgr_upstream_baseline`
- Hierarchy data root:
  `/home/leejt/OneRec/data_experiment/Amazon/mgr_upstream_hierarchy`

## Configs

- SFT baseline:
  `/home/leejt/OneRec/config/experiments/sft_industrial_mgr_upstream_baseline.yaml`
- SFT hierarchy:
  `/home/leejt/OneRec/config/experiments/sft_industrial_mgr_upstream_hierarchy.yaml`
- Evaluate baseline:
  `/home/leejt/OneRec/config/experiments/evaluate_industrial_mgr_upstream_baseline.yaml`
- Evaluate hierarchy:
  `/home/leejt/OneRec/config/experiments/evaluate_industrial_mgr_upstream_hierarchy.yaml`

## Runtime Plan

- `baseline` SFT and evaluate use GPUs `2,3`
- `hierarchy` SFT and evaluate use GPUs `4,5`
- each variant runs in its own `tmux` session
- evaluate is chained after successful SFT completion

## Logs

- baseline SFT:
  `/home/leejt/OneRec/logs/experiment_mgr_sid_sft_industrial_baseline_20260410.log`
- baseline evaluate:
  `/home/leejt/OneRec/logs/experiment_mgr_sid_eval_industrial_baseline_20260410.log`
- hierarchy SFT:
  `/home/leejt/OneRec/logs/experiment_mgr_sid_sft_industrial_hierarchy_20260410.log`
- hierarchy evaluate:
  `/home/leejt/OneRec/logs/experiment_mgr_sid_eval_industrial_hierarchy_20260410.log`

## Outputs

- baseline SFT checkpoint root:
  `/home/leejt/OneRec/output/experiments/mgr_sid_sft_eval_industrial_20260410/mgr_upstream_baseline/sft`
- hierarchy SFT checkpoint root:
  `/home/leejt/OneRec/output/experiments/mgr_sid_sft_eval_industrial_20260410/mgr_upstream_hierarchy/sft`
- baseline evaluate result:
  `/home/leejt/OneRec/results/experiments/mgr_sid_sft_eval_industrial_20260410/final_result_sft_mgr_upstream_baseline_Industrial_and_Scientific.json`
- hierarchy evaluate result:
  `/home/leejt/OneRec/results/experiments/mgr_sid_sft_eval_industrial_20260410/final_result_sft_mgr_upstream_hierarchy_Industrial_and_Scientific.json`

## tmux Sessions

- `mgr_sft_eval_base_ind`
- `mgr_sft_eval_hier_ind`

## Monitoring

```bash
tmux ls
tmux attach -t mgr_sft_eval_base_ind
tmux attach -t mgr_sft_eval_hier_ind
```

or

```bash
tail -f /home/leejt/OneRec/logs/experiment_mgr_sid_sft_industrial_baseline_20260410.log
tail -f /home/leejt/OneRec/logs/experiment_mgr_sid_sft_industrial_hierarchy_20260410.log
```
