# MGR-SID v2: Ambiguity-Aware Graph Supervision with Semantic-Structure Retention

**Date**: 2026-04-10  
**Status**: current proposed next-step method after `MGR-SID v1`  
**Positioning**: this is not a post-hoc patch. It is a **training-time refinement** of hierarchy-aware graph regularization, motivated by the current tokenizer and SFT evidence.

---

## 1. Why We Need v2

Our current evidence is already strong enough to say that the main problem is no longer:

- whether graph-structured collaborative information is useful
- whether hierarchy-aware supervision is meaningful

Those two questions have been answered positively.

What the current results actually show is more specific:

### Confirmed by tokenizer-side results

- compared with the reproduced **MiniOneRec baseline**, `hierarchy_reg` produces a cleaner final SID
- final collision is lower
- local `same_l2` ambiguity is lower
- `H(level3 | level1, level2)` is lower

### Confirmed by downstream SFT + evaluate

- the gain is **real**, but not broad
- `top3` improves
- crowded local `same_l2` / high-`l2`-fanout cases improve
- but `@10+` does not consistently improve

### Confirmed by top-k structural analysis

- `hierarchy` helps exactly the difficult local disambiguation cases we care about
- but it also breaks some samples that the MiniOneRec baseline already keeps in the correct local neighborhood

So the current bottleneck is:

> `MGR-SID v1` applies graph supervision too uniformly and too rigidly.  
> It improves hard local ambiguity, but it also over-corrects some already-stable semantic structures.

This means the next step is **not** to throw away the current direction, and **not** to patch the system after evaluation.  
Instead, we should refine the **training-time integration mechanism**.

---

## 2. Core Idea of v2

`MGR-SID v2` keeps the main thesis of v1:

> different SID levels should receive different graph-structured collaborative supervision

But it changes *how* this supervision is applied:

- `v1`: fixed, globally uniform graph regularization
- `v2`: ambiguity-aware graph regularization + semantic-structure retention

In one sentence:

> graph collaborative supervision should be strong where the item is hard to distinguish and weak where the original semantic geometry already provides stable local structure.

This is still a **front-end tokenizer method**.  
It does not use downstream evaluation to patch tokenizer outputs.  
It only uses:

- train-only graph structure
- train-only semantic geometry
- train-time quantization uncertainty

---

## 3. What v2 Is Not

To avoid confusion, v2 is **not**:

- a post-hoc hybrid replacement of final SIDs after seeing evaluate errors
- a rule that says “if `same_l2` then switch to another SID”
- a downstream reranker
- a full free-form learned gate over all graph views and levels

At the current stage, v2 is intentionally more controlled:

- keep the level-to-graph role assignment interpretable
- make graph supervision adaptive through ambiguity scores
- explicitly protect already-good semantic structures

This is cleaner than immediately jumping to a fully free learned gate.

---

## 4. High-Level Method Summary

`MGR-SID v2` contains three ingredients:

1. **Graph Bank**  
   Keep the three-view graph bank used in the current line:
   - `G_coarse`
   - `G_mid`
   - `G_local`

2. **Ambiguity Proxies**  
   Estimate how ambiguous each item is before final SID generation:
   - offline structural ambiguity
   - online quantization uncertainty

3. **Ambiguity-Aware Loss Design**  
   Use ambiguity to control:
   - how strongly graph regularization acts
   - how strongly semantic-structure retention is enforced

The overall design is:

- hard / crowded / uncertain items:
  - stronger graph supervision
- easy / already-stable items:
  - stronger semantic-structure retention

---

## 5. The Graph Side: Keep the Current Roles, Do Not Reopen Everything

At the current stage, the graph story itself is already reasonably supported by evidence:

- `G_coarse` captures stable broad collaborative structure
- `G_mid` is the most valuable signal for local semantic-collaborative disambiguation
- `G_local` captures sharper short-range transition structure

So `v2` should not reopen the entire graph-design search.

### Recommended graph-role assignment for v2

- **Level 1**: `G_coarse`
  - role: preserve broad collaborative consistency
  - target: avoid globally unreasonable grouping

- **Level 2**: `G_mid`
  - role: resolve semantic-collaborative ambiguity in crowded middle-scale neighborhoods
  - target: improve local subtree organization

- **Level 3**: `G_local`
  - role: refine leaf-level competition with short-range transition information
  - target: improve final leaf separability

### Why keep fixed level roles for now

Although our earlier ideal method included a learned gate, the current evidence says:

- the graph-role hierarchy itself is not the main unresolved issue
- the main unresolved issue is **where and how strongly** graph supervision should act

So v2 should first solve the weighting problem before reopening the allocation problem.

---

## 6. Ambiguity Proxies: How to Estimate Hard vs Easy Before Final SID

The key difficulty is:

> we cannot use post-hoc `same_l2` labels as training inputs

So v2 must use **train-time ambiguity proxies** instead.

We currently recommend four families of proxies.

### 6.1 Semantic Density

Intuition:

- some items sit in very dense semantic neighborhoods
- many neighbors are semantically plausible alternatives
- these items are likely harder for pure semantic quantization

Possible definitions:

- mean cosine similarity to top-`k` semantic neighbors
- density estimated from local `k`-NN distances
- number of neighbors above a high semantic similarity threshold

What it captures:

- “how crowded the semantic neighborhood is”

What it does **not** capture:

- whether collaborative structure agrees with this neighborhood

So semantic density alone is useful, but not sufficient.

---

### 6.2 Semantic-Collaborative Disagreement

Intuition:

- some items are semantically close to one set of neighbors
- but collaboratively close to a different set
- these are exactly the items where semantic SID is likely to under-resolve local discrimination

Possible definitions:

- overlap / Jaccard distance between semantic `k`-NN and collaborative `k`-NN
- divergence between semantic neighbor weights and graph neighbor weights
- disagreement between semantic and graph-induced local communities

What it captures:

- “how much the semantic view and the collaborative view disagree on local structure”

Why this is especially attractive:

- it directly matches our current motivation
- it is still train-only
- it can be computed before final SID generation

This is currently the strongest candidate offline ambiguity proxy.

---

### 6.3 Graph Competition Strength

Intuition:

- some items have many graph-supported alternatives with similar support
- the problem is not just degree, but the **competition profile**

Possible definitions:

- entropy of normalized graph neighbor weights
- number of strong neighbors above threshold
- spectral mid-band energy around the item
- overlap among top graph neighbors
- competition between collaborative and transition neighborhoods

What it captures:

- “how many plausible graph-supported competitors exist”

Why it matters for us:

- this is the cleanest train-time approximation of what later becomes crowded local leaf ambiguity

---

### 6.4 Quantization-Time Uncertainty

This is the most interesting part because it comes directly from the tokenizer itself.

In the current `VectorQuantizer`, we already compute the distance from the latent to all codebook entries:

- [vq.py](/home/leejt/OneRec/src/onerec/sid/models/vq.py)

This means we can directly estimate how confident the code assignment is.

#### Candidate signals

- **assignment margin**
  - distance gap between nearest and second-nearest code
- **soft assignment entropy**
  - if we expose a soft score over codes
- **residual difficulty**
  - how large the residual remains after a quantization level

What these capture:

- “how hard it is for the current level to make a discrete decision”

Why this is powerful:

- no final SID is needed
- no downstream label is needed
- the uncertainty is measured exactly where the quantization decision is made

This is likely the cleanest **online** ambiguity signal.

---

## 7. Recommended Ambiguity Design for v2

The best current design is not to rely on a single ambiguity signal.

Instead:

### 7.1 Offline structural ambiguity prior

For each item `i`, estimate:

- `semantic_density(i)`
- `semantic_collab_disagreement(i)`
- `graph_competition(i)`

Then normalize and combine them into:

- `A_offline(i)`

This is a static prior.

### 7.2 Online level-wise ambiguity

During quantization at each level `l`, estimate:

- `margin_l(i)`
- optional `entropy_l(i)`
- `residual_norm_l(i)`

Then combine them into:

- `A_online(i, l)`

### 7.3 Final ambiguity score

For each item and each level:

- `A(i, l) = combine(A_offline(i), A_online(i, l))`

Interpretation:

- high `A(i, l)`:
  - graph supervision should act more strongly
- low `A(i, l)`:
  - semantic retention should dominate

This gives a clean train-time control signal without using post-hoc labels.

---

## 8. Ambiguity-Aware Graph Supervision

This is the first half of v2.

In `v1`, graph regularization is essentially:

- fixed level-to-graph assignment
- fixed global loss weight per level

In `v2`, we keep the level-to-graph assignment, but replace the fixed global weighting with item-aware weighting.

### Conceptually

- Level 1 still uses `G_coarse`
- Level 2 still uses `G_mid`
- Level 3 still uses `G_local`

But graph loss is no longer applied equally to every item.

Instead:

- high ambiguity items contribute more strongly to graph smoothness
- low ambiguity items contribute less strongly

### Why this is consistent with current evidence

Our top-k results already show:

- crowded hard cases benefit from hierarchy supervision
- easy cases can be over-corrected

So the cleanest training response is:

> do not reduce graph influence globally; reduce it selectively where ambiguity is low

---

## 9. Semantic-Structure Retention / Anti-Overcorrection Anchor

This is the second half of v2, and it is essential.

Current `v1` does not only improve hard cases; it also breaks some cases that the MiniOneRec baseline already handles correctly.

So we need an explicit mechanism that says:

> if the original semantic geometry already provides a stable local structure here, do not let graph supervision rewrite it too aggressively.

### What should be protected

Not everything in the semantic view should be frozen.

What should be protected is:

- coarse semantic structure
- already-stable local semantic neighborhood structure
- low-ambiguity items that do not need graph correction

### What should not be used as the main anchor

The main anchor should **not** be:

- the final discrete SID generated by the MiniOneRec baseline
- any post-hoc label such as final `same_l2`

Those are useful for diagnosis, but not the cleanest primary front-end supervision signal.

### How this can be implemented conceptually

Use the **original semantic geometry** available before final SID generation:

- semantic item embeddings
- semantic `k`-NN neighborhoods
- local semantic density / overlap structure

Then impose a semantic-structure retention term that is stronger on low-ambiguity items and weaker on high-ambiguity items.

This gives us exactly the behavior we want:

- low ambiguity:
  - preserve what already works
- high ambiguity:
  - allow graph supervision to reshape the structure

### Why this is not a post-hoc patch

Because the anchor is available **before** downstream evaluation:

- it comes from train-time semantic geometry
- it acts during tokenizer training
- it does not depend on final downstream outcomes

So it is a valid front-end method component.

---

## 10. v2 as a Full Training Objective

Conceptually, v2 balances three forces:

1. **reconstruction / quantization objective**
   - preserve the basic RQ-VAE tokenizer quality

2. **ambiguity-aware graph supervision**
   - inject collaborative structure where semantic SID is likely under-resolved

3. **ambiguity-aware semantic-structure retention**
   - preserve original semantic geometry where it is already stable

This can be thought of as:

- graph fixes the hard parts
- semantic-structure anchor protects the easy parts

This is exactly the behavior our current evidence says we need.

---

## 11. Suggested Implementation Path in This Repo

The current code already gives us clean insertion points.

### 11.1 Offline ambiguity prior

Likely file targets:

- [graph_bank.py](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/graph_bank.py)
- [transplanted_graph_bank.py](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/transplanted_graph_bank.py)

What to add:

- semantic density score
- semantic-collaborative disagreement score
- graph competition score
- a cached ambiguity-prior artifact

### 11.2 Online quantization uncertainty

Likely file target:

- [vq.py](/home/leejt/OneRec/src/onerec/sid/models/vq.py)

What to expose:

- top-1 / top-2 code distances
- assignment margin
- optional soft assignment entropy

### 11.3 Level-wise ambiguity collection

Likely file target:

- [train_v1.py](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/train_v1.py)

What to extend:

- collect level-wise ambiguity signals in `_forward_hierarchy`
- compute ambiguity-aware weights
- add semantic-structure retention loss

### 11.4 New trainer file

Recommended:

- keep `train_v1.py` untouched as the current reference implementation
- add a new experimental trainer, e.g. `train_v2_ambiguity.py`

This keeps the baseline and v1 evidence chain intact.

---

## 12. Why v2 Is Better Than Jumping Directly to a Free Learned Gate

At idea level, we did discuss a learned gate.

However, a fully free gate is not the best next move right now because:

- it adds too many degrees of freedom at once
- if it helps, attribution becomes messy
- if it fails, diagnosis becomes hard
- our current evidence already suggests the real issue is **where to trust graph**, not whether to completely relearn graph allocation

So v2 is a better next step because it is:

- simpler
- more interpretable
- directly driven by current evidence
- easier to compare against the MiniOneRec baseline

In short:

> before learning a free gate, learn **when graph should speak loudly and when it should stay quiet**.

---

## 13. Expected Behavioral Effect

If v2 works as intended, we should see:

### At tokenizer level

- final collision not worse than v1
- `same_l2` ambiguity further reduced on hard regions
- less collateral damage to easy / already-stable regions

### At SFT / evaluate level

- retain current `top3` / crowded-case gains
- reduce the number of samples that are currently broken by over-correction
- improve `top10+` relative to v1

The key target is not “another short-range gain only”.

The real target is:

> keep the hard-case local disambiguation benefit while preserving the original semantic neighborhood structure on easy cases.

---

## 14. Minimal v2 Experimental Plan

The first validation should stay small and disciplined.

### Step 1

Implement only:

- one offline ambiguity prior
- one online quantization proxy
- one semantic-structure retention anchor

Do **not** introduce:

- full free gate
- new graph families
- downstream reranking changes

### Step 2

Run tokenizer-only comparison:

- MiniOneRec baseline
- current `hierarchy_reg` v1
- `v2 ambiguity-aware`

Check:

- final collision
- local ambiguity metrics
- crowded-bucket behavior

### Step 3

Only if tokenizer results improve:

- run SFT + evaluate
- repeat the current top-k structural analysis

This is the most efficient path to determine whether the method is genuinely improving the right bottleneck.

---

## 15. Current Recommendation

Based on all current evidence, this is the best next-step method direction:

> **MGR-SID v2 = ambiguity-aware hierarchy graph supervision + semantic-structure retention anchor**

More concretely:

- keep the current graph hierarchy
- stop treating all items as equally ambiguous
- let graph act strongly where ambiguity is high
- let semantic structure dominate where the original geometry is already stable

This is a clean front-end method update, fully consistent with:

- our current experiments
- our top-k analysis
- our literature scan
- and the principle that post-hoc evidence should refine earlier training design, not directly patch earlier outputs
