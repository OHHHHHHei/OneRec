# 2026-04-11 MGR-SID Tokenizer v2 R005

## Goal

Launch the first tokenizer-only `v2` run for Industrial using the validated
offline ambiguity prior from `R001`.

This run is intentionally isolated from:

- the reproduced MiniOneRec baseline
- the existing `v1 hierarchy_reg`
- any downstream `SFT / RL` chain

The purpose is only to answer the first tokenizer-stage question:

> Can `v2 = ambiguity-aware graph supervision + semantic-structure retention`
> beat the MiniOneRec baseline and improve tokenizer structure relative to `v1`?

## Variant

- variant name: `mgr_sid_tokenizer_v2_offline_combined`
- ambiguity prior: `offline_combined`
- online uncertainty: not used in this first run

## Core Design

- graph views:
  - `coarse_purified`
  - `fagsp_mid_base`
  - `local_purified`
- ambiguity prior:
  - semantic density
  - semantic-collaborative disagreement
  - graph competition
- semantic retention:
  - semantic kNN graph on the original semantic embedding

## Config

- config:
  `/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_tokenizer_v2.yaml`
- runner:
  `/home/leejt/OneRec/scripts/experiment_mgr_sid_v2_train.py`
- trainer:
  `/home/leejt/OneRec/src/onerec/experiments/mgr_sid/train_v2.py`

## Inputs

- item embedding:
  `/home/leejt/OneRec/data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy`
- train csv:
  `/home/leejt/OneRec/data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv`
- ambiguity prior csv:
  `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-11_mgr_sid_v2_proxy_sanity/proxy_item_scores.csv`

## Runtime Plan

- device: `cuda:0` inside process
- target physical GPU: `6`
- session name: `mgr_tok_v2_r005_ind`

## Logs

- train log:
  `/home/leejt/OneRec/logs/experiment_mgr_sid_tokenizer_v2_r005_industrial_20260411.log`

## Outputs

- checkpoint root:
  `/home/leejt/OneRec/output/experiments/mgr_sid_tokenizer_v2/industrial_offline_combined`
- launch record dir:
  `/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-11_mgr_sid_tokenizer_v2_r005`

## Next Steps After Training

1. Run `sid-generate` from the best collision checkpoint.
2. Compare final SID against:
   - MiniOneRec baseline
   - `v1 hierarchy_reg`
3. Re-run local ambiguity structural diagnostics.
