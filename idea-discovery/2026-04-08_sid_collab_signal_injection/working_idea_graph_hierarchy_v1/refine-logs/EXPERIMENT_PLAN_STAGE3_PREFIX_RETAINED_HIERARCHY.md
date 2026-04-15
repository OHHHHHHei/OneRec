# Experiment Plan

**Problem**: stage-2 established that tokenizer-side structural improvement does not automatically produce a better downstream SID codebook space.  
**Method Thesis**: stage-3 should search for a better hierarchy-aware SID codebook space, while testing whether conservative upper-prefix constraints are one useful way to reach that better space.  
**Date**: 2026-04-13  
**Revision**: 2026-04-14 narrative update clarifying that the core issue is fresh-SFT learnability of the new SID space, not preservation of a previous SFT checkpoint

## Problem Anchor

## Narrative Update

This plan should be read with one narrative clarification:

- the goal is **not** to keep SID close to the current baseline
- the goal is to find a **better SID codebook space**
- the current baseline is only a useful anchor for diagnosis
- prefix stability and codebook drift are supporting diagnostics, not final optimization targets
- the hardest selection criterion is full downstream `SFT -> evaluate`

So the `R401*` family is best viewed as:

> one candidate design family for building a better SID codebook space, not a
> mandatory requirement that every good SID space must remain close to `v2`

We already know four things:

1. graph-structured collaborative supervision is useful for SID construction;
2. hierarchy-aware supervision is better than uniform graph regularization;
3. `v2_on_p05 -> RL` is already a valid end-to-end line;
4. the remaining gap is mainly a mid-beam retention gap, not a top-1 failure.

Stage-2 then showed a more specific failure mode:

- `R202a` improves tokenizer-side local structure;
- but it also causes near-global SID remapping;
- and the downstream model fails to benefit from that cleaner structure consistently.

So the next-stage question is no longer:

> can we further improve tokenizer structure at any cost?

It is now:

> can we use hierarchy-aware graph supervision to produce a better SID codebook
> space for downstream recommendation learning, and is conservative prefix
> control one effective way to do that?

## Main Claim

**Primary claim**: the current bottleneck is not that hierarchy-aware graph supervision is wrong, but that we still do not know what kind of SID codebook space is best for downstream learning. Prefix-retained refinement is one hypothesis for building a better space, not the only acceptable endpoint.

**Supporting claim**: this can be tested without collapsing the story into a leaf-only method. All three levels should still receive their own graph supervision; only the training dynamics of `L1/L2` should become more conservative.

### Anti-claims to rule out

- The only way to improve SID is to keep it close to the current baseline.
- The only way to make the new SID space learnable is to freeze `L1/L2` completely.
- The only way to fix the downstream gap is to abandon tokenizer-side refinement and move directly to RL-stage tuning.
- The stage-2 failure came only from token semantic confusion rather than large prefix reorganization.

## Motivation-to-Design Map

| Motivation | Evidence Anchor | Design Response | What this means experimentally |
|---|---|---|---|
| One plausible hypothesis is that a more conservative SID reorganization may help downstream learning. | `R301`: `R202a` changed `99.65%` of `l1`, `100%` of `l2`, and `100%` of full SIDs. | Add teacher-guided retention on upper prefixes as one candidate design. | We can test whether staying nearer to `v2` produces a better downstream SID space. |
| We should not freeze `L1/L2`, because that weakens the hierarchy-aware claim. | Current method story depends on `L1 <- G_coarse`, `L2 <- G_mid`, `L3 <- G_local`. | Keep all three graph losses active; add only soft upper-prefix retention. | `L1` and `L2` must still be trainable and graph-aware. |
| Easy/stable items need more protection than hard/ambiguous ones. | `R303`: worsened examples are associated with lower `l1` stability, while hard examples benefit from stronger local rewrite. | Weight retention by inverse ambiguity: stronger on easy items, weaker on hard items. | Hard cases can still move; easy cases should keep routing. |
| The leaf should retain the most freedom. | Current bottleneck remains local leaf ambiguity inside broadly correct prefixes. | Do not impose strong retention on `L3`. | We should still see real movement in local ambiguity metrics. |
| The first validation should isolate one concrete codebook-space hypothesis before exploring larger redesigns. | The current failure is that a structurally cleaner SID space was not easier for fresh SFT to learn. | Warm-start from the current best `v2` checkpoint and use it as teacher. | If this fails, we learn that “stay nearer to `v2`” is not by itself the answer. |
| Continuous representation matching does not automatically guarantee discrete prefix stability. | Prefix changes depend on representation-to-codebook relative geometry, not only representation MSE. | Make codebook drift a mandatory diagnostic and pre-plan a codebook-anchor fallback. | This helps interpret why a conservative branch succeeds or fails; it is not itself the final selection rule. |
| Stage-3 should not accidentally change two retention mechanisms at once. | Current `v2` already benefits from semantic retention terms. | Keep existing semantic retention losses on; add the new teacher-guided retention on top. | Improvement or failure can be attributed to the new prefix-stabilizing term rather than a changed base objective. |

## Proposed Method Family

### Shared setup for all `R401*` variants

- base tokenizer: current best `v2`
- initialization: warm-start from the best `v2` tokenizer checkpoint
- teacher: frozen current best `v2`
- graph bank: unchanged
- graph-role assignment: unchanged
  - `L1 <- G_coarse`
  - `L2 <- G_mid`
  - `L3 <- G_local`
- ambiguity prior: unchanged
- training recipe: unchanged upstream-aligned tokenizer recipe
- existing semantic retention losses: kept on
  - `semantic_coarse` and `semantic_mid` remain active exactly to avoid introducing a new confounding factor
- codebooks: remain trainable
- decoder: remains trainable
- initialization snapshots:
  - save level-wise initial codebooks at train start for drift diagnostics and anchor fallback

### Loss Composition

The first stage-3 round adds the new retention term on top of the existing `v2` tokenizer objective rather than replacing any current term.

`L_total = L_v2_base + L_retain`

where `L_v2_base` is the current best `v2` tokenizer recipe, including:

- reconstruction / quantization terms
- hierarchy-aware graph terms
- semantic retention terms already present in `v2`

This means stage-3 tests one new hypothesis only:

> does an additional prefix-stabilizing upper-prefix retention term create a
> better downstream SID codebook space without deleting hierarchy-aware
> structure gains?

### First implementation choice

For the first stage-3 round, retention is applied at the representation level rather than by introducing a new discrete token-classification head.

Let `h_s^(1), h_s^(2)` be the student cumulative representations at levels 1 and 2, and `h_t^(1), h_t^(2)` the frozen teacher representations for the same item.

We add:

`L_retain = gamma_1 * w_i * ||h_s^(1) - sg(h_t^(1))||^2 + gamma_2 * w_i * ||h_s^(2) - sg(h_t^(2))||^2`

where:

- `w_i` follows the inverse-ambiguity direction already used for semantic stabilization;
- concretely, it should follow the same convention as `semantic_item_scale`, i.e. scale with `1 - prior_i`, not `prior_i`;
- easy items receive larger retention weight;
- hard items receive smaller retention weight.

Recommended implementation convention:

`w_i = scale_from_prior(1.0 - prior_i, retention_scale_min, retention_scale_max)`

### Important caveat

This representation-level retention is intentionally the first version because it fits the current codepath with minimal disruption.

However, it does **not** guarantee discrete prefix stability by itself.

What actually determines whether a prefix changes is the relative geometry between:

- the cumulative representation at a level; and
- the corresponding level codebook.

So stage-3 must explicitly monitor:

- prefix stability;
- codebook drift; and
- reconstruction adaptation.

If representation retention looks healthy but prefix stability remains weak, the correct interpretation is **not** immediately “retention does not work”.

The more likely interpretation is:

> codebook movement is still large enough to break the representation-to-prefix link.

That is why a codebook-anchor fallback is pre-planned in this revision.

### Hyperparameter policy

For the first round:

- tie `gamma_1 = gamma_2 = gamma`
- search `gamma in {0.05, 0.10, 0.20}`

Only if this tied sweep shows a clear asymmetry need do we later untie `gamma_1` and `gamma_2`.

### Variant list

- `R401a`: `R202a + L1 retention only`
  - purpose: test whether protecting coarse routing alone is enough
- `R401b`: `R202a + L1/L2 retention`
  - purpose: test whether both upper levels need protection
- `R401c`: `R202a + ambiguity-aware L1/L2 retention`
  - purpose: recommended main candidate; keeps the hierarchy story while respecting easy vs hard differences
- `R401d`: `R401c + codebook anchor on L1/L2`
  - purpose: reserved fallback if representation retention is not translating into discrete prefix stability
  - not part of the first launch family unless Block 1 diagnostics justify it

### Reserved fallback term for `R401d`

If needed, add a light codebook anchor:

`L_anchor = alpha_1 * Drift^(1) + alpha_2 * Drift^(2)`

with:

`Drift^(l) = mean_k ||c_k^(l) - sg(c_k,init^(l))||^2`

The stage-3 default is still to test `R401a/b/c` first.
`R401d` exists so that a codebook-drift diagnosis can be acted on immediately rather than requiring a new planning cycle.

## Why Warm-Start Is the Right First Test

This stage is a mechanism-validation stage for one candidate family, not a final proof that every good tokenizer must stay near `v2`.

We therefore do **not** start with a from-scratch tokenizer run.

The reason is simple:

- from-scratch training changes initialization, codebook layout, and SID routing all at once;
- warm-start isolates the exact question we care about:
  - can a hierarchy-aware refinement improve the SID target space **relative to the current `v2` space that already trains cleanly under fresh SFT**?

If warm-start fails, we learn that even around a known downstream-learnable SID space the refinement is not good enough.
If warm-start succeeds, we can later add a stronger completeness experiment:

- `semantic warmup -> prefix-retained refinement`

That second experiment is important, but it should not be the first gate.

## Experiment Blocks

### Block 1: Tokenizer Mechanism Validation

- Claim tested:
  - soft upper-prefix retention may create a better SID codebook space by limiting one kind of over-aggressive reorganization
- Why this block exists:
  - this is the direct test of the next-stage hypothesis
- Dataset / split / task:
  - Industrial
  - tokenizer only: `sid-train -> sid-generate -> diagnostics`
- Compared systems:
  - `T0`: current `v2`
  - `T1`: `R202a`
  - `T2`: `R401a`
  - `T3`: `R401b`
  - `T4`: `R401c`
- Metrics:
  - prefix organization:
    - changed `l1/l2/full` rate
    - `l1/l2` pair retention
    - mean `l1/l2` neighbor Jaccard
    - `l1/l2` pair retention by ambiguity bucket
      - easy / medium / hard buckets by prior tertile
  - structure:
    - final generated collision
    - target-weighted mean `l2` leaf count
    - fraction of targets in deep crowded `l2>=4`
    - target-weighted `H(l3 | l1,l2)`
  - learnability:
    - `a`
    - `b|a`
    - `c|ab`
    - report mean and std over 3 probe seeds
  - stability diagnostics:
    - level-wise codebook drift
      - `Drift^(l) = mean_k ||c_k^(l) - c_k,init^(l)||^2`
    - early reconstruction loss trajectory after warm-start
- Success criterion:
  - the branch is tokenizer-health-safe:
    - no major reconstruction breakdown
    - no obviously pathological collision explosion
  - and it produces a plausible downstream candidate SID space worth pushing to full `SFT -> evaluate`

#### Block-1 interpretation rule

Using the current numbers:

- current `v2`: mean `l2` leaves = `4.3422`, `H(l3|l1,l2) = 1.1001`
- `R202a`: mean `l2` leaves = `3.6148`, `H(l3|l1,l2) = 1.0308`
- current `v2`: `b|a = 0.2392`, `c|ab = 0.4365`
- `R202a`: `b|a = 0.2134`, `c|ab = 0.4159`

The decision policy is split into a **health floor** and **diagnostic preference signals**.

**Health floor to remain a valid downstream candidate**

1. mean target `l2` leaves `<= 3.98`
   - preserves at least half of `R202a`'s gain over `v2`
2. target-weighted `H(l3|l1,l2) <= 1.065`
   - same logic as above
3. `b|a` mean over 3 seeds `>= 0.2134`
   - do not fall below `R202a`
4. `c|ab` mean over 3 seeds `>= 0.4159`
   - do not fall below `R202a`
5. codebook drift and reconstruction traces must be reported
   - even if not used as hard thresholds

**Diagnostic preference signals**

1. `l1` pair retention `>= 70%`
2. `l2` pair retention `>= 80%`
3. `b|a` mean over 3 seeds `>= 0.226`
   - halfway recovery from `R202a` back toward current `v2`
4. `c|ab` mean over 3 seeds `>= 0.426`
   - same halfway-recovery logic

Interpretation:

- stronger prefix retention is useful evidence **for this branch family**
- but weaker prefix retention does **not** by itself disqualify a tokenizer from downstream evaluation
- if a branch has low representation loss but weak prefix stability together with large codebook drift, that is strong evidence to also try `R401d`
- final branch selection should still be made by full downstream `SFT -> evaluate`

- Failure interpretation:
  - if stability improves but local structure collapses, retention is too strong;
  - if structure stays strong but `b|a` or `c|ab` remain weak, the next step may add a predictability-oriented term;
  - if representation retention looks good but prefix stability stays weak and codebook drift stays large, launch `R401d` in parallel;
  - tokenizer diagnostics alone should not be the final rejection rule for a downstream candidate.
- Table / figure target:
  - stage-3 tokenizer table
  - one prefix-stability table
  - one learnability table
  - one codebook-drift diagnostic table
- Priority:
  - `MUST-RUN`

### Block 1b: Conditional Tokenizer Extension Before SFT

This block is still tokenizer-side.
It exists to avoid a wasteful loop of:

> tokenizer looks partly fixed -> SFT still fails -> come back later for an obviously missing tokenizer term

Trigger this block immediately if Block 1 reveals a specific remaining bottleneck.

#### Case A: representation retention is healthy, but discrete stability is still weak

- symptom:
  - low representation drift but poor `l1/l2` retention
  - large `L1/L2` codebook drift
- action:
  - launch `R401d = R401c + codebook anchor`
- reason:
  - this directly addresses the representation-vs-codebook mismatch

#### Case B: stability and structure pass, but learnability still lags

- symptom:
  - `l1/l2` stability is good
  - structure metrics remain good
  - but `b|a` and `c|ab` fail to reach halfway recovery
- action:
  - add a light predictability-oriented regularizer on top of the best `R401*` branch
- reason:
  - this directly targets the `R304` failure mode before spending SFT budget

This conditional block is preferred over pushing predictability regularization into the very first launch family because the first question is still whether simple retention already works.

### Block 2: Fixed-Recipe SFT Screen

- Claim tested:
  - the best prefix-retained tokenizer branch improves downstream ranking under the same recipe, rather than by shifting recipe again
- Why this block exists:
  - the key question is whether the new SID codebook space actually produces stronger final downstream results
- Dataset / split / task:
  - Industrial
  - fixed recipe:
    - `title_history2sid_on + desc_align_p05`
- Compared systems:
  - `S0`: current `v2_on_p05 SFT`
  - `S1`: strongest original MiniOneRec SFT
  - `S2`: best stage-3 tokenizer candidate
- Metrics:
  - decisive:
    - `NDCG@10`
    - `HR@10`
  - secondary:
    - `NDCG@1/3/5`
    - `HR@1/3/5`
    - fanout-stratified top-k
    - same-prefix top-1 error
- Setup details:
  - keep SFT hyperparameters aligned with the current successful `v2_on_p05` run
  - run one seed first
  - if the main deltas are within `+-0.001` on `NDCG@10` or `HR@10`, add two confirmation seeds before making the final call
- Success criterion:
  - hard gate:
    - `S2` must at least match current `v2_on_p05 SFT` on both `NDCG@10` and `HR@10`
  - positive signal:
    - `S2` improves both metrics over current `v2_on_p05 SFT`
    - and does not give back the current `@1/@3` advantage materially
- Failure interpretation:
  - if tokenizer-side gains do not survive here, the remaining bottleneck is probably no longer mainly the SID target-space organization
- Table / figure target:
  - main downstream comparison table candidate
- Priority:
  - `MUST-RUN`, but only after Block 1 or Block 1b produces a promotable tokenizer

### Block 3: RL Confirmation

- Claim tested:
  - the downstream gain survives RL, not only SFT
- Why this block exists:
  - the end goal is still to improve the end-to-end line
- Dataset / split / task:
  - Industrial
  - RL launched only from the best positive SFT candidate
- Compared systems:
  - `R0`: current `v2_on_p05 RL`
  - `R1`: strongest original MiniOneRec RL
  - `R2`: best stage-3 tokenizer candidate -> RL
- Metrics:
  - decisive:
    - `NDCG@10`
    - `HR@10`
  - secondary:
    - `NDCG@5/@20`
    - `HR@5/@20`
    - collided-target top1 hit
    - `@1/@3`
- Success criterion:
  - `R2` improves the current `v2_on_p05 RL` on the mid-beam band without collapsing the current head-ranking profile
- Failure interpretation:
  - if SFT improves but RL does not, the next-stage bottleneck should move to RL objective / decoding rather than tokenizer redesign
- Table / figure target:
  - RL confirmation table
- Priority:
  - `CONDITIONAL`

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| `M0` | freeze the stage-3 hypothesis | no new training | agree that the next question is prefix-stabilized hierarchy refinement for a more learnable SID space | very low | drifting back into broad redesign |
| `M1` | tokenizer family | `R401a`, `R401b`, `R401c` | one branch is healthy enough to justify full downstream evaluation | comparable to one stage-2 tokenizer round per branch | retention may be too weak or too strong |
| `M1b` | tokenizer-side conditional fix | `R401d` or best `R401* + predictability` | launch in parallel if Block 1 identifies the corresponding bottleneck clearly | low-to-medium | adding extra complexity without a diagnosis |
| `M2` | fixed-recipe SFT | best promotable tokenizer branch only | beat or at least match `v2_on_p05 SFT` on both `NDCG@10` and `HR@10` | one SFT run | tokenizer gain may still fail to improve fresh-SFT learnability |
| `M3` | RL confirm | best positive `M2` branch only | improve current `v2_on_p05 RL` in mid-beam metrics | one RL run | RL may reintroduce the same gap |
| `M4` | completeness | `semantic warmup -> retained refinement` | only after positive evidence that stage-3 works mechanistically | medium | turning the first-stage test into a larger story too early |

## Must-Run vs Nice-to-Have

### Must-run

- `R401a`
- `R401b`
- `R401c`
- one tokenizer-side diagnostics pass including codebook drift
- one fixed-recipe SFT screen on the best promotable stage-3 branch

### Conditional

- `R401d` if Block 1 indicates representation-to-codebook mismatch
- best `R401* + predictability` if Block 1 indicates learnability-specific lag
- RL confirm if SFT is positive

### Nice-to-have

- `semantic warmup -> retained-refinement` completeness experiment
- token-level retention objective after the first representation-level test

## Risks and Mitigations

- **Risk**: representation-level retention may improve MSE to teacher states but still fail to preserve discrete routing.
  - **Mitigation**: report prefix stability and codebook drift as diagnostics, and pre-plan `R401d` as a parallel candidate rather than assuming this family must stay near `v2`.

- **Risk**: protecting `L2` too strongly may erase the very `G_mid` effect that makes the method interesting.
  - **Mitigation**: run `R401a` and `R401b` alongside `R401c` rather than assuming `L2` retention is always beneficial.

- **Risk**: the existing semantic retention loss and the new teacher-guided retention loss may pull in slightly different directions.
  - **Mitigation**: keep the semantic terms fixed, add only one new retention family, and monitor whether stability improves while structure collapses.

- **Risk**: warm-start success may be dismissed as a checkpoint trick.
  - **Mitigation**: treat warm-start as the mechanism-validation stage and explicitly plan a later completeness run only after positive evidence.

- **Risk**: decoder adaptation may lag behind `L3` movement even if `L1/L2` are stabilized.
  - **Mitigation**: keep decoder trainable and inspect early reconstruction loss traces after warm-start.

- **Risk**: the true bottleneck may now be downstream optimization rather than tokenizer-side SID organization.
  - **Mitigation**: stop at Block 2 if tokenizer-side gains still do not make the new SID space easier for SFT to use; do not keep stacking tokenizer complexity blindly.

## Reviewer-Facing Summary

This stage-3 plan is built to answer one precise question:

> can hierarchy-aware graph supervision produce a better downstream SID codebook space, and is conservative upper-prefix control one effective way to reach that space?

It is intentionally **not** a leaf-only fallback, but it is also **not** claiming that staying near the baseline is the only legitimate path.

The logic is:

1. keep all three hierarchy-aware graph roles;
2. test whether preserving `L1/L2` more carefully helps;
3. leave `L3` freer for local ambiguity repair;
4. explicitly monitor whether representation retention is actually becoming discrete prefix stability;
5. judge the branch primarily by full downstream `SFT -> evaluate`.

If this works, the next-stage paper story becomes:

> hierarchy-aware graph supervision is useful, and one promising path is to test whether conservative prefix control can help create a better downstream SID codebook space.

If it partially works, the revision already tells us how to proceed:

- add codebook anchoring if codebook drift breaks prefix stability;
- add predictability regularization if learnability lags after stability is fixed.

If this fails, we will have a much clearer justification for shifting the bottleneck to downstream objective or decoding rather than continuing blind tokenizer refinement.
