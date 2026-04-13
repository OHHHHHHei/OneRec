# Probe and Early Evidence

This note merges the two early probe records that established why the current graph-hierarchy line was worth pursuing.

Merged source notes are archived under:

- `archive/2026-04-12_doc_reorg_merged_sources/13_initial_probe_run_2026-04-09.md`
- `archive/2026-04-12_doc_reorg_merged_sources/14_paper_transplant_probe_run_2026-04-09.md`

## Why These Probes Were Run

Before changing tokenizer training, we first asked a cheaper question:

> on real top-1 SID error cases, which graph-structured collaborative views actually help distinguish the target from its competitors?

This kept the design space under control before we paid for full tokenizer training.

## Probe Round 1: Basic Graph Bank

The first probe tested:

- coarse collaborative graph
- local transition graph
- heuristic middle-resolution views
  - diffusion residual
  - band-pass proxy
  - community-style proxy

Each view was evaluated in raw and lightly purified forms.

### Main takeaway

The strongest `same_l2` signals came from purified middle views, not from a naive single global or local view.

This established an important early thesis:

> the collaborative signal that matters for deep leaf ambiguity is neither purely global nor purely last-step local; a middle-resolution graph view is necessary.

## Probe Round 2: Paper-Transplant Views

The second probe added two imported idea families:

- `PRISM`-style semantic-anchor purification
- `GSPRec / FaGSP`-style spectral middle-resolution graph design

### Main takeaway

The strongest middle view jumped from heuristic proxies to a proper spectral view:

- current strongest `G_mid` candidate:
  `fagsp_mid_base`

The gain was large enough to change the design direction:

> `G_mid` should be treated as a first-class graph-spectral collaborative structure, not as a hand-made mid-hop heuristic.

## What These Probes Proved

Together, the two probe rounds supported five stable conclusions:

1. graph-structured collaborative information is worth injecting into SID construction
2. light graph purification is usually necessary
3. `G_mid` is the key view, not a minor auxiliary branch
4. semantic anchoring helps, but the largest gain comes from better middle-resolution structure design
5. the graph story should be framed as structural supervision, not as a graph-encoder benchmark

## What They Did Not Prove

These probes did not yet prove:

- final recommendation gains
- end-to-end tokenizer superiority
- the final exact downstream recipe

They were still probes, not final method validation.

## Why They Still Matter Today

Even after `v2` and RL, these probe notes remain important because they justify why the current graph bank was frozen the way it was:

- `G_coarse`:
  broad collaborative prior
- `G_mid`:
  ambiguity-relevant middle-resolution structure
- `G_local`:
  transition-sensitive short-range structure

That graph bank is still the backbone of the current `v2` line.
