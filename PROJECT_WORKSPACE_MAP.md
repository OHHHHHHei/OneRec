# Project Workspace Map

This file is the shortest reliable entry point for the current repository layout.

If you need to sync to the project quickly, read in this order:

1. `research-progress-log/research_progress_log.tex`
2. `idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/00_ACTIVE_CONTEXT.md`
3. `idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/18_mgr_sid_v2_ambiguity_aware_method.md`
4. `research-progress-log/experiment_launches/README.md`
5. `experiment_results.csv`

## Folder Roles

### `idea-discovery/`

Research-direction and method-design workspace.

- current mainline:
  `idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/`
- archived earlier branches stay under sibling `archive/` folders

Use this directory for:

- direction notes
- method design
- plan / tracker documents
- literature-to-method mapping

### `research-progress-log/`

Canonical experiment-and-interpretation log.

- primary document:
  `research_progress_log.tex`
- experiment index:
  `experiment_launches/README.md`

Use this directory for:

- stage summaries
- per-run conclusions
- compact analysis reports

Do not treat this directory as raw artifact storage.

### `results/`

Lightweight result artifacts.

Use this directory for:

- final merged `json` outputs
- diagnostic summaries
- recovered legacy outputs

### `logs/`

Training and evaluation logs.

Keep logs here, not on the data disk.

### `output/`

Repository-side entry points only.

Large checkpoints are now expected to live on the data disk under:

- `/data/leejt/OneRec/output_weights`

Local `output/` may contain only lightweight links or compatibility paths.

### `paper-mgr-sid-draft/` and `paper-mgr-sid-draft-zh/`

Current English and Chinese paper drafts.

Keep only:

- source `.tex`
- bibliography
- final `.pdf`

Generated LaTeX auxiliaries should not be kept unless actively debugging compilation.

## Current Canonical Document Set

### Method and direction

- `idea-discovery/2026-04-08_sid_collab_signal_injection/README.md`
- `idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/00_ACTIVE_CONTEXT.md`
- `idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/18_mgr_sid_v2_ambiguity_aware_method.md`
- `idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/19_mgr_sid_current_method_code_aligned_formulas.md`

### Experiment status

- `research-progress-log/research_progress_log.tex`
- `research-progress-log/experiment_launches/README.md`
- `experiment_results.csv`

### Latest strongest run family

- `research-progress-log/experiment_launches/2026-04-11_mgr_sid_v2_recipe_isolation_industrial/RESULTS.md`
- `research-progress-log/experiment_launches/2026-04-12_mgr_sid_v2_rl_on_p05_industrial/RESULTS.md`
- `research-progress-log/experiment_launches/2026-04-12_mgr_sid_v2_rl_on_p05_industrial/EVAL_ANALYSIS.md`

## Cleanup Policy

- keep a small number of canonical summaries in active folders
- move superseded split notes into `archive/`
- keep raw `.json` / `.csv` analysis artifacts next to the run that generated them
- keep filenames stable for externally referenced core docs when possible
