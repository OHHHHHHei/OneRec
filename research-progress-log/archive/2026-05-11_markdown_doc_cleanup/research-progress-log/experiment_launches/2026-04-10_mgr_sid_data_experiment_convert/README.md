# 2026-04-10 MGR-SID Experimental Convert

## Goal

Prepare isolated SFT-ready data under `data_experiment/` for the two Industrial SID variants:

- `mgr_upstream_baseline`
- `mgr_upstream_hierarchy`

The existing `data/` tree is left untouched.

## Source

- Source converted CSV root: `/home/leejt/OneRec/data/Amazon`
- Source item metadata: `/home/leejt/OneRec/data/Amazon/index/Industrial_and_Scientific.item.json`
- Experimental baseline index:
  `/home/leejt/OneRec/output/experiments/mgr_sid_v1_upstream/generated_indices/Industrial_and_Scientific.mgr_upstream_baseline.index.json`
- Experimental hierarchy index:
  `/home/leejt/OneRec/output/experiments/mgr_sid_v1_upstream/generated_indices/Industrial_and_Scientific.mgr_upstream_hierarchy.index.json`

## Output

- Variant root:
  `/home/leejt/OneRec/data_experiment/Amazon/mgr_upstream_baseline`
- Variant root:
  `/home/leejt/OneRec/data_experiment/Amazon/mgr_upstream_hierarchy`
- Manifest:
  `/home/leejt/OneRec/data_experiment/Amazon/Industrial_and_Scientific.manifest.json`

Each variant root mirrors the existing OneRec data structure:

- `index/Industrial_and_Scientific.index.json`
- `index/Industrial_and_Scientific.item.json`
- `info/Industrial_and_Scientific_5_2016-10-2018-11.txt`
- `train/Industrial_and_Scientific_5_2016-10-2018-11.csv`
- `valid/Industrial_and_Scientific_5_2016-10-2018-11.csv`
- `test/Industrial_and_Scientific_5_2016-10-2018-11.csv`

## Conversion Rule

This experimental convert reuses the existing CSV splits and only refreshes:

- `history_item_sid`
- `item_sid`

using the target experimental `index.json`.

Other fields such as:

- `user_id`
- `history_item_title`
- `item_title`
- `history_item_id`
- `item_id`

are preserved from the current `data/Amazon` CSVs.

## Verification

- Train rows: `36259`
- Valid rows: `4532`
- Test rows: `4533`
- Output columns exactly match the current OneRec CSV schema.
- Experimental baseline and hierarchy variants produce different SID strings from the current mainline data, as expected.

## Implementation

- Script:
  `/home/leejt/OneRec/scripts/experiment_mgr_sid_prepare_data.py`

## Next Step

Use the two variant roots above as parallel data sources for SFT:

- baseline SFT uses
  `/home/leejt/OneRec/data_experiment/Amazon/mgr_upstream_baseline`
- hierarchy SFT uses
  `/home/leejt/OneRec/data_experiment/Amazon/mgr_upstream_hierarchy`
