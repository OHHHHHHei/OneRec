# Graph Pilot

## Purpose

This pilot is not the final method. Its role is much narrower:

- replace hand-crafted time windows with simple graph proxies
- test whether graph-derived collaborative signals also show non-uniform utility across ambiguity buckets
- see whether a graph view can support the hierarchy-aware motivation

## Setup

Train-only graphs were built from the current Amazon splits and existing prediction outputs:

- undirected collaborative graph:
  - item-item co-occurrence from train histories
- directed transition graph:
  - last-item to target transitions from train histories

Three simple graph views were probed:

- `coarse_graph`
  - 1-hop weighted score on the normalized undirected collaborative graph
- `mid_graph`
  - 2-hop diffusion score on the normalized undirected collaborative graph
- `local_trans`
  - directed last-item transition probability

For each failed recommendation case in the current result files, the probe compared:

- score of the ground-truth target item
- score of the model's chosen candidate within the predicted SID bucket

Reported number:

- `target-better rate`
  - fraction of cases where the probe prefers the true target over the chosen candidate

Buckets:

- `all`
- `same_l1`
- `same_l2`

Also reported:

- `coverage`
  - fraction of cases with a nonzero score signal

## Results

### Industrial

| View | all | same_l1 | same_l2 | coverage |
|------|-----|---------|---------|----------|
| `coarse_graph` | `0.145204` | `0.273230` | `0.398148` | `0.397048` |
| `mid_graph` | `0.079029` | `0.144912` | `0.206790` | `0.920495` |
| `local_trans` | `0.057605` | `0.132743` | `0.209877` | `0.082599` |

### Office

| View | all | same_l1 | same_l2 | coverage |
|------|-----|---------|---------|----------|
| `coarse_graph` | `0.154866` | `0.274123` | `0.416667` | `0.438076` |
| `mid_graph` | `0.099123` | `0.114035` | `0.154762` | `0.955720` |
| `local_trans` | `0.062486` | `0.177632` | `0.428571` | `0.097775` |

## What this pilot says

### 1. Graph signals are also non-uniform

This is the main positive result. Even crude graph proxies already behave differently:

- `coarse_graph` is the strongest robust prior
- `local_trans` is too sparse to use everywhere, but it can become very strong on deep ambiguity

### 2. Naive `mid_graph` is not enough

Plain 2-hop diffusion is not a good final answer for Level 2 collaborative structure.

This is actually useful:

- it tells us `mid-scale` cannot be treated as an arbitrary interpolation between global and local
- it likely needs a real graph operator such as community extraction, band-pass filtering, or diffusion residuals

### 3. The graph view story is stronger than the window story

The earlier window-based probe said that different collaborative granularities matter.  
This graph-based pilot sharpens that observation:

- the real distinction may be graph structure and graph frequency, not literal history-window length

## Caveats

- these are compatibility probes, not end-to-end model results
- the current `mid_graph` is deliberately simple and should not be treated as the final design
- low `coverage` for `local_trans` means it cannot stand alone as a global tokenizer signal

## Design implication

The paper-grade method should likely:

- construct multiple graph views
- denoise them differently
- let different SID levels use them differently
- avoid treating 2-hop diffusion as the final mid-scale answer
