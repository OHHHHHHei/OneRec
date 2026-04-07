# Research Proposal: Ambiguity-Aware Collaborative Leaf Refinement for MiniOneRec

## Problem Anchor

- **Bottom-line problem**: MiniOneRec's remaining errors are concentrated in local SID ambiguity, where the model often reaches the right semantic prefix but misses the final leaf among behaviorally different yet textually similar items.
- **Must-solve bottleneck**: improve local leaf discrimination without replacing the whole tokenizer or destabilizing the current pipeline.
- **Non-goals**:
  - no full collaborative tokenizer rebuild
  - no end-to-end personalized tokenizer
  - no new RL method
  - no large graph model unless a simpler residual is insufficient
- **Constraints**:
  - must stay leakage-safe and train-only for collaborative signals
  - must fit current `item.json`, `index.json`, `convert`, SFT, and evaluate flow
  - should be implementable as a small extension to the current codebase
- **Success condition**:
  - overall HR/NDCG improves on Industrial and ideally Office
  - same-prefix error rates drop, especially at the `l2` level
  - gains are achieved without global retokenization

## Technical Gap

The current literature already covers global collaborative tokenization aggressively. `DiscRec`, `PRORec`, `HiD-VAE`, `PRISM`, `ReSID`, `PIT`, and related work all make the broad idea "inject collaboration into tokenization" far less novel than it was even a year ago. The repo's own diagnostics also argue against using that whole space as the paper anchor:

- collision is low
- prefix ambiguity is high
- local same-prefix confusions are common
- train-only collaborative evidence often still favors the target inside these local confusions

So the operational gap is not:

> how do we build an entirely new tokenizer?

It is:

> how do we repair the final leaf decision when the current tokenizer already places the sequence in the right neighborhood but text-only separation is still insufficient?

## Method Thesis

**One-sentence thesis**: Instead of rebuilding MiniOneRec's tokenizer, ACLR adds a train-only collaborative residual only to ambiguous `(a,b)` prefixes, so the model learns and applies behavior-aware leaf disambiguation exactly where current errors remain concentrated.

## Contribution Focus

- **Dominant contribution**: a prefix-local collaborative residual for third-token SID prediction
- **Optional supporting contribution**: a diagnostic-driven ambiguity profiler that decides when the residual should be active
- **Explicit non-contributions**:
  - no new global semantic ID construction method
  - no claim of solving collision in general
  - no claim of replacing recent collaborative-tokenizer methods

## Proposed Method

### Complexity Budget

- **Frozen / reused backbone**:
  - existing text-based SID construction
  - existing `index.json`
  - existing SFT/RL backbone and constrained decoding
  - existing data contracts and metrics pipeline
- **New trainable components**:
  - one small projection head `W` from decoder hidden state to a collaborative query space
- **Offline non-trainable artifacts**:
  - ambiguity scores per `(a,b)` prefix
  - collaborative residual vectors per item
  - prefix-local leaf prototypes
- **Tempting additions intentionally rejected**:
  - full retokenization
  - graph-heavy end-to-end encoder
  - extra RL reward redesign
  - global activation of collaborative bias on every prefix

### System Overview

```text
train interactions
  -> ambiguity profiler over prefixes
  -> collaborative residual vectors per item
  -> leaf prototypes e_(p,c) for ambiguous prefixes

SFT sample with target SID <a><b><c>
  -> normal MiniOneRec forward
  -> if prefix p=(a,b) is ambiguous:
       hidden state h_t at leaf step
       q_t = W h_t
       local leaf contrastive loss over valid c in prefix p

inference
  -> normal constrained decoding
  -> if current prefix p is ambiguous:
       add beta * alpha_p * (q_t . e_(p,c)) to valid leaf logits
```

### Core Mechanism

#### 1. Ambiguity Profiler

Use existing diagnostics to score each two-level prefix `p = (a,b)` with a normalized ambiguity score `alpha_p`. A practical score can combine:

- prefix entropy under the prefix
- same-prefix top-1 miss rate
- collaborative-gap rate inside that prefix

Only prefixes above a threshold are marked active.

#### 2. Collaborative Residual Bank

Build train-only item-level collaborative residual vectors `r_i` from simple transition/co-occurrence compression. Start with the smallest viable option:

- weighted item-item co-occurrence
- low-rank compression or normalized residual embedding

Do not require LightGCN unless the simple version fails.

#### 3. Prefix-Local Leaf Prototypes

For each ambiguous prefix `p` and each valid leaf `c` under that prefix, aggregate the residual vectors of items assigned to that leaf:

`e_(p,c) = mean { r_i | SID(i) has prefix p and leaf c }`

This turns global item collaboration into a compact local bias over the valid leaf choices.

#### 4. Leaf-Level Auxiliary Loss

At the third SID token step, obtain the decoder hidden state `h_t` and project it:

`q_t = W h_t`

Then apply a local contrastive loss over the valid leaf candidates under the target prefix:

`L_leaf = -log exp(q_t . e_(p,c*) / tau) / sum_{c in C(p)} exp(q_t . e_(p,c) / tau)`

where `c*` is the target leaf and `C(p)` is the valid leaf set under prefix `p`.

Use the final training objective:

`L_total = L_ce + lambda * alpha_p * L_leaf`

for ambiguous prefixes, and `L_total = L_ce` otherwise.

#### 5. Inference-Time Local Residual

During constrained decoding, once the model reaches the leaf step under prefix `p`, update logits only across the valid leaves:

`z'_c = z_c + beta * alpha_p * (q_t . e_(p,c))`

This is local, prefix-valid, and does not require wider beams or larger vocabularies.

## Why This Is The Smallest Adequate Intervention

- It directly targets the failure mode shown by the repo diagnostics.
- It avoids the crowded global tokenizer literature.
- It preserves nearly all existing infrastructure.
- It yields a clear ablation story:
  - no local residual
  - static heuristic residual
  - learned local residual
  - global activation

## Inference Path

- If a prefix is not ambiguous: use standard MiniOneRec decoding.
- If a prefix is ambiguous:
  - compute the collaborative query from the decoder hidden state
  - add the local residual only to the valid leaf logits
  - continue normal constrained decoding

## Failure Handling

- **Failure mode**: no target-like candidate exists in the beam or subtree
  - **Mitigation**: measure beam coverage and do not overclaim local repair
- **Failure mode**: collaborative residuals are noisy for long-tail items
  - **Mitigation**: shrink or smooth prototypes, and compare simple co-occurrence vs stronger residuals
- **Failure mode**: inference-only gains do not survive training
  - **Mitigation**: compare heuristic-only, training-only, and full ACLR

## Novelty and Elegance Argument

The contribution is not "yet another collaborative tokenizer." The claim is narrower and cleaner:

- recent work largely modifies the tokenizer globally
- this repo's remaining error is local
- a selective collaborative residual at the leaf step is enough to improve recommendation without retokenizing the catalog

That is both more grounded and more elegant for the current MiniOneRec setting.

## Must-Prove Claims

- **Claim 1**: local collaborative residualization improves recommendation by reducing ambiguous leaf mistakes inside already-useful semantic prefixes
- **Claim 2**: selective local activation is a better complexity-performance tradeoff than global collaborative injection

## Minimum Validation Needed

- overall HR/NDCG gain on Industrial and ideally Office
- reduction in same-prefix local error rates
- comparison against heuristic rerank-only and global activation baselines
- evidence that the gain comes from ambiguous-prefix repair, not broader beam effects

## Final Verdict

**READY**

This is the sharpest route that stays faithful to the repo's diagnostics, avoids the most crowded literature overlap, and remains realistic to implement in MiniOneRec.
