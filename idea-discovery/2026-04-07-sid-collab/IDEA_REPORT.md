# Idea Discovery Report

Direction: MiniOneRec SID bottlenecks around missing collaboration, unstable prefix semantics, and collision

Date: 2026-04-07

Pipeline used:

- local repo diagnosis
- latest arXiv survey
- idea generation and ranking
- local novelty assessment
- reviewer-style critical check
- refined proposal and experiment plan

## Executive Summary

The user's three motivations are all real, but they do not point to the same paper equally strongly. The latest arXiv landscape in 2025-2026 already heavily covers global collaborative tokenizer redesign, prefix-aware training, and collision-aware SID learning. In contrast, the current repo's strongest evidence says the dominant practical bottleneck is local ambiguity inside semantically plausible subtrees, not raw collision.

The best direction for this repo is therefore `ACLR-Plus`: Ambiguity-Calibrated Local Collaborative Repair. The method keeps the current SID/tokenizer, detects when the current semantic prefix path is unreliable, and injects collaborative evidence only inside the risky local subtree. This is more repo-faithful, cheaper to validate, and more differentiated from the latest tokenizer papers.

## Literature Landscape

### Motivation 1: semantic-only SID misses collaborative information

Latest papers strongly support this:

- ReSID: https://arxiv.org/abs/2602.02338
- PIT: https://arxiv.org/abs/2602.08530
- PRISM: https://arxiv.org/abs/2601.16556
- UNGER: https://arxiv.org/abs/2502.06269

Conclusion:

- The motivation is real.
- But using it as the sole paper anchor is now crowded.

### Motivation 2: prefix semantics are unstable and not globally fixed

Latest papers also support this:

- TrieRec: https://arxiv.org/abs/2602.21677
- APAO: https://arxiv.org/abs/2603.02730
- ReSID: https://arxiv.org/abs/2602.02338

Conclusion:

- This is both real and central.
- In this repo, it appears as local same-prefix ambiguity.

### Motivation 3: collision exists

Supported by both local diagnostics and recent papers:

- QuaSID: https://arxiv.org/abs/2603.00632
- HiD-VAE: https://arxiv.org/abs/2508.04618

Conclusion:

- Collision is real.
- But in the current repo it looks more secondary than local subtree ambiguity.

## Local Evidence Summary

Representative repo evidence:

- collision is around `0.434%` on both Industrial and Office
- same-prefix errors are much larger than collision
- the beam often already contains same-prefix candidates
- collaborative reranking improves current predictions

Lightweight local pilot:

### Industrial hit@1

- baseline: `0.073241`
- global rerank: `0.083830`
- same_l1: `0.080079`
- same_l2: `0.081624`
- ambiguity_l2: `0.079197`

### Office hit@1

- baseline: `0.085697`
- global rerank: `0.088368`
- same_l1: `0.087752`
- same_l2: `0.087957`
- ambiguity_l2: `0.084875`

Interpretation:

- collaborative signal clearly helps
- local-prefix-constrained repair can help
- but ambiguity detection must be smarter than the current leaf-count heuristic

## Ranked Ideas

### 1. ACLR-Plus: Ambiguity-Calibrated Local Collaborative Repair

Status: recommended

Method thesis:

- semantics is already good at routing into a coarse subtree
- collaboration should only be injected when that subtree is locally ambiguous

Why it wins:

- strongest match to repo evidence
- strongest differentiation from recent tokenizer papers
- can be built on top of existing code

Novelty assessment:

- good, if framed as selective local repair rather than tokenizer redesign

Main risk:

- may look like a decoding heuristic unless the ambiguity trigger is principled

### 2. Selective Hot-Subtree Retokenization

Status: backup

Why not first:

- more complex
- more overlap with tokenizer papers

### 3. Collision-Qualified Alias Tokens

Status: backup

Why not first:

- collision is not the dominant local bottleneck

### 4. Full Collaborative Tokenizer Rebuild

Status: eliminated as top direction

Reason:

- too crowded relative to ReSID, PIT, PRISM, UNGER, QuaSID, and related work

## Reviewer-Style Bottom Line

External-style assessment:

- good internal research direction
- not yet a strong paper if implemented only as heuristic reranking

Minimum viable paper version:

- principled ambiguity calibration
- local candidate restriction
- clean comparison against global rerank and current heuristic local modes
- evidence that the method helps risky local cases without damaging globally easy ones

## Recommended Next Step

Proceed with:

- `refine-logs/FINAL_PROPOSAL.md`
- `refine-logs/EXPERIMENT_PLAN.md`

and treat the first implementation stage as a falsification exercise:

- if calibrated local repair cannot beat the simple local baselines, stop and reconsider
- if it does beat them cleanly, this becomes a strong repo-specific method story
