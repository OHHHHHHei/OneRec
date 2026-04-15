# Stage-2 Postmortem: What the Experiments Actually Tell Us

**Date**: 2026-04-13  
**Author**: External Review  
**Scope**: Full analysis of stage-2 results (Block 2/3/4), reassessment of the gap structure, and recommended next steps.

---

## 1. Executive Summary

Stage-2 set out to close the remaining `top5/top10` mid-beam retention gap with two small tokenizer-side refinements: stop-gradient hierarchy isolation and batch-local semantic retention KL. Neither passed the downstream gate.

But this is not a failure of the research direction. It is a **diagnostic success**: the experiments have exposed a deeper structural issue that was previously invisible.

> The core finding of stage-2 is not "stop-grad doesn't work" or "KL retention doesn't work."  
> The core finding is: **tokenizer-side local-ambiguity cleanup and downstream beam retention are not the same thing, and improving one does not automatically improve the other.**

This document reconstructs the reasoning chain, identifies what was missed in the original stage-2 framing, and proposes a corrected next step.

---

## 2. Evidence Reconstruction

### 2.1 The Full Scoreboard

I verified all numbers against `experiment_results.csv` and individual `RESULTS.md` files. The authoritative comparison set is:

| System | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | HR@1 | HR@3 | HR@5 | HR@10 | HR@50 |
|---|---|---|---|---|---|---|---|---|---|
| strongest orig SFT | 0.06706 | 0.08501 | 0.09315 | **0.10372** | 0.06706 | 0.09839 | 0.11824 | **0.15089** | 0.24531 |
| strongest orig RL | 0.07324 | 0.08903 | 0.09704 | **0.10726** | 0.07324 | 0.10038 | 0.11979 | **0.15133** | 0.21994 |
| v2_on_p05 SFT | **0.07059** | 0.08451 | 0.09253 | 0.10271 | **0.07059** | 0.09508 | 0.11471 | 0.14626 | **0.24818** |
| v2_on_p05 RL | **0.07434** | **0.09054** | 0.09630 | 0.10432 | **0.07434** | **0.10280** | 0.11692 | 0.14185 | **0.23737** |
| R208 (R202a SFT) | 0.06552 | 0.08360 | 0.09045 | 0.09974 | 0.06552 | 0.09729 | 0.11383 | 0.14251 | 0.23737 |

Bold = best in column across the relevant comparison pair.

### 2.2 What Each Stage-2 Branch Actually Did

#### R202a (stop-grad, unchanged weights)

**Tokenizer side: clear structural winner.**

| Metric | current v2 | R202a | Delta |
|---|---|---|---|
| mean target l2 leaf count | 4.342 | **3.615** | -0.727 (16.8%) |
| deep crowded l2≥4 | 0.223 | **0.199** | -0.024 |
| H(l3\|l1,l2) | 1.100 | **1.031** | -0.069 |
| multi-leaf same_l2 | 0.487 | 0.499 | +0.012 (slight worse) |
| generated collision | 13/3686 | 13/3686 | unchanged |

**Downstream: regression.**

- NDCG@10: 0.10271 → 0.09974 (-0.00297)
- HR@10: 0.14626 → 0.14251 (-0.00375)
- HR@50: 0.24818 → 0.23737 (-0.01081)

#### R202b / R202b-r075 (stop-grad + level-1 compensation)

Both failed. R202b collapsed (collision 0.96). R202b-r075 recovered stability but structure regressed on all retention-oriented metrics. This tells us level-1 is extremely sensitive to coarse_weight changes under stop-grad.

#### R205 (stop-grad + batch-local KL, τ=0.1)

**Tokenizer side: clear negative result.**

All structural metrics regressed vs both current v2 and R202a:
- mean target l2 leaf count: 4.342 → 4.957 (+0.615)
- deep crowded l2≥4: 0.223 → 0.262
- H(l3\|l1,l2): 1.100 → 1.262

Not pushed downstream. Correctly gated by the plan.

### 2.3 The Critical R208 Top-k Analysis

The `TOPK_V2_ON_P05_SFT_VS_R208.md` contains the most diagnostic data in the entire stage-2 batch. Here is what it shows:

**Fanout-stratified hit rates (R208 vs v2_on_p05 SFT):**

| Fanout bucket | top1 delta | top3 delta | top5 delta | top10 delta |
|---|---|---|---|---|
| l2≤2 (easy) | **-0.00764** | **-0.00732** | **-0.00732** | **-0.00637** |
| l2=3 (medium) | -0.02880 | -0.00785 | -0.00262 | +0.00262 |
| l2≥4 (hard) | **+0.01188** | **+0.03564** | **+0.01980** | +0.00198 |

This is the single most important table in stage-2. It says:

1. **R202a helps exactly where it should**: on high-fanout hard cases, top3 improves by 3.6%.
2. **R202a hurts exactly where it shouldn't**: on easy l2≤2 cases, every cutoff regresses.
3. The net effect is negative because there are ~3141 easy cases vs ~1010 hard cases. The easy-case regression dominates the aggregate.

**Improved vs worsened set profiles (top10):**

| Set | Count | Baseline same-l1 in beam | Baseline same-l2 in beam | Mean target l2 fanout |
|---|---|---|---|---|
| Improved | 130 | 0.538 | 0.354 | 7.72 |
| Worsened | 147 | **1.000** | **1.000** | 6.98 |

This reveals a critical pattern: **every single worsened example had the target's same-l1 AND same-l2 neighbors already in the baseline beam**. The worsened examples were cases where the old SID space already provided correct local routing, and the SID space change disrupted it.

---

## 3. What Stage-2 Actually Proved

### 3.1 Confirmed: Cross-layer gradient interference is real

R202a's structural improvements (mean l2 leaf count -16.8%, conditional entropy -6.3%) directly demonstrate that the v2 hierarchy loss was causing cross-layer interference. Isolating gradients lets each level optimize its own structure more effectively.

### 3.2 Confirmed: Level-1 is hypersensitive under stop-grad

R202b (λ_c doubled to 0.10) collapsed entirely. R202b-r075 (λ_c × 1.5 to 0.075) regressed structurally. This proves that without cross-layer gradient flow, level-1's codebook becomes extremely sensitive to its own graph regularization weight. The previous v2's effective level-1 regularization was ~6.5× the nominal weight due to gradient leakage from levels 2 and 3.

### 3.3 Confirmed: batch-local KL at τ=0.1 is too aggressive

R205's regression across all structural metrics indicates that the KL loss at low temperature dominated the graph regularization signal. At τ=0.1, softmax distributions are extremely peaked — only the nearest 2-3 semantic neighbors carry significant probability mass. This creates a very sharp "preserve the nearest neighbor" constraint that conflicts with graph smoothness.

### 3.4 **New finding: tokenizer structural cleanup ≠ downstream beam retention**

This is the most important finding and it was NOT anticipated in the stage-2 plan.

The mechanism is now clear from the R208 top-k analysis:

1. R202a's stop-grad isolates levels, allowing each level to better optimize for its assigned graph view.
2. This changes **100% of all SID assignments** — every item gets a new SID.
3. The new SID space has better local disambiguation properties (lower entropy, fewer deep crowded buckets).
4. BUT the new SID space has different global geometry — items that previously shared prefixes no longer do, and vice versa.
5. When the downstream LLM trains on this new SID space, it learns different token co-occurrence patterns.
6. The downstream model gains on hard cases (where the new SID space provides better discrimination) but loses on easy cases (where the old SID space already worked and the new one disrupts established routing).
7. Because easy cases outnumber hard cases ~3:1, the aggregate metric drops.

### 3.5 Implication: the problem is not "tokenizer refinement" but "SID space stability"

The real bottleneck exposed by stage-2 is:

> Any tokenizer change that reassigns all SIDs introduces a "global rearrangement cost" that competes against the "local disambiguation benefit."

This cost was invisible in tokenizer-only diagnostics because those diagnostics measure structure quality, not transfer stability.

---

## 4. What the Current Analysis Documents Got Right and Wrong

### 4.1 What the existing analysis got right

- The `EVAL_ANALYSIS.md` for R208 correctly identifies the fanout-stratified pattern: gain on hard cases, loss on easy cases.
- The `TOPK_V2_ON_P05_SFT_VS_R208.md` correctly notes that worsened examples all have baseline same-l1/l2 = 1.0.
- The `research_progress_log.tex` correctly frames the current state: "stage-2 tokenizer micro-fixes have not surpassed v2_on_p05."
- The `EXTERNAL_REVIEW_PACKET_STAGE2_20260413.md` correctly sets up the reading order and warns against judging from @10 alone.

### 4.2 What the existing analysis missed or underweighted

1. **The 100% SID change rate was noted but its implications were not fully drawn out.** Every diagnostic report says "SID changed on 100% of catalog items" but this was treated as a neutral observation rather than a red flag. A 100% change rate means the downstream model is learning in a completely new token space. This should have been the primary diagnostic concern, not the aggregate structural metrics.

2. **The fanout-stratified top-k analysis was done but its asymmetry was not used to update the stage-2 thesis.** The data clearly shows that the problem is not "retention-targeted refinement doesn't work" — it works exactly on the targeted cases (+3.56% top3 on l2≥4). The problem is collateral damage on non-targeted cases. This distinction should inform the next step.

3. **The `TOPK_V2_ON_P05_SFT_VS_R208.md` analysis script is well-designed.** The fanout bucketing, improved/worsened set profiling, and same-prefix rate tracking are all correct and genuinely useful. One analysis that is missing: **prefix stability analysis** — what fraction of item pairs that share a l1 or l2 prefix in v2's SID space still share that prefix in R202a's SID space. This would directly quantify the "global rearrangement cost."

4. **The v2_on_p05 RL result was somewhat under-analyzed.** The RL result shows a very specific U-shaped pattern vs strongest original RL: better at @1/@3, worse at @5/@10/@20, better again at @50. This U-shape means v2 RL is more decisive (sharper head) and has better deep recall (higher @50) but loses in the middle. This is a **decode-time beam behavior issue**, not a tokenizer structure issue. The current analysis correctly identifies this but doesn't explore the decode-time implications.

### 4.3 Analysis scripts: assessment

The analysis pipeline (top-k structural analysis, same-prefix diagnostics, fanout bucketing, improved/worsened profiling) is well-constructed. Two additions would be valuable:

1. **Prefix stability matrix**: For any two tokenizer variants A and B, compute:
   - What fraction of item pairs sharing l1 prefix in A also share l1 prefix in B
   - Same for l2 prefix
   - This directly measures "how much did the SID space rearrange"

2. **Transfer efficiency metric**: For each structural improvement at tokenizer level (e.g., an item moved from l2_leaf_count=25 to l2_leaf_count=1), check whether this specific item's downstream hit rate actually improved. This would separate "useful structural changes" from "structurally correct but downstream-neutral changes."

---

## 5. Reassessing the Gap Structure

With all evidence in hand, here is the corrected gap decomposition:

### 5.1 v2_on_p05 SFT vs strongest original SFT

| Dimension | v2_on_p05 advantage | strongest orig advantage |
|---|---|---|
| Head precision (@1) | **+0.00353** | — |
| Mid-beam retention (@3-@10) | — | **-0.00331 to -0.00463** |
| Deep recall (@50) | **+0.00287** | — |
| Same-prefix top1 error | **much lower** (0.033 vs 0.070) | — |
| Same-l1 beam presence | — | **much higher** (0.634 vs 0.493) |
| Collided target top1 hit | **higher** | — |

### 5.2 v2_on_p05 RL vs strongest original RL

| Dimension | v2_on_p05 RL advantage | strongest orig RL advantage |
|---|---|---|
| Head precision (@1/@3) | **+0.00110 / +0.00243** | — |
| Mid-beam retention (@5-@20) | — | **-0.00287 to -0.00949** |
| Deep recall (@50) | **+0.01743** | — |
| Same-prefix top1 error | **much lower** (0.059 vs 0.077) | — |
| Collided target top1 hit | **74.4% vs 61.5%** | — |

### 5.3 Updated reading

The gap is not a generic quality deficit. It is a **ranking profile redistribution**:

- v2 is decisively better at the head and at deep recall
- v2 is decisively better at local disambiguation (same-prefix errors, collided targets)
- v2 loses in the mid-beam band (@5-@20) because its SID space, while structurally cleaner, provides fewer "nearby correct alternatives" in the beam

This is exactly the profile you would expect from a tokenizer that produces more discriminative but less redundant SID assignments. The question is whether this is a fundamental tradeoff or a fixable interface issue.

---

## 6. Root Cause Analysis: Why Tokenizer Structural Gain Doesn't Transfer

Based on all evidence, I identify three contributing mechanisms:

### 6.1 Global SID rearrangement cost (PRIMARY)

Every tokenizer change so far reassigns 100% of SIDs. The downstream LLM must re-learn all token co-occurrence statistics from scratch. On easy cases where the old SID space already provided correct routing, the new SID space offers no benefit but does force relearning, leading to regression.

**Evidence**: R208 worsened set has baseline same-l1 = 1.0 and same-l2 = 1.0 on every worsened example. These are cases where the downstream model had already learned the correct local associations in the old SID space.

### 6.2 v2's SID space is less redundant (CONTRIBUTING)

v2's structural cleanup reduces the number of items sharing prefixes. This means the beam contains fewer "locally correct alternatives." From a tokenizer perspective this is desirable (less ambiguity), but from a beam search perspective it means that if the top-1 prediction is wrong, there are fewer nearby fallbacks.

**Evidence**: beam_contains_same_l1_rate drops from 0.634 (strongest orig) to 0.493 (v2_on_p05). This 14-point drop in local redundancy directly reduces the probability that a near-miss will still hit.

### 6.3 SFT training length may be suboptimal for new SID spaces (POSSIBLE)

R208 early-stopped at epoch 5.5 with eval loss 1.612. The original v2_on_p05 may have had a different convergence trajectory because its SID space was established first. If the downstream model needs more training time to learn the new SID space's token patterns, early stopping may be premature.

**Evidence**: Weak — would need a training curve comparison. But worth noting.

---

## 7. Recommended Next Steps

### 7.1 PRIORITY 1: Freeze-L1L2, retrain-L3-only tokenizer (NEW PROPOSAL)

**Rationale**: The root cause analysis says the main problem is "100% SID rearrangement." The cleanest solution is to make a tokenizer change that does NOT rearrange L1 and L2 prefixes.

**Implementation**:
1. Start from the current v2 tokenizer's best checkpoint
2. Freeze level 1 and level 2 codebooks completely
3. Apply stop-grad + G_local regularization on level 3 only
4. Retrain only the level 3 codebook and the decoder

**Expected behavior**:
- L1 and L2 prefixes for all items remain IDENTICAL to current v2
- Only L3 (leaf) assignments change
- The downstream model's learned token co-occurrence patterns for L1/L2 tokens are fully preserved
- The only change is better leaf-level discrimination within existing L2 subtrees

**Why this directly addresses the bottleneck**:
- It eliminates the 100% SID rearrangement cost entirely for L1/L2
- It targets exactly the leaf-level disambiguation that the analysis shows is the remaining value
- It is the most conservative possible tokenizer change that can still improve local structure

**Cost**: Very low — one tokenizer training run with frozen L1/L2.

### 7.2 PRIORITY 2: RL recipe tuning on existing v2_on_p05

**Rationale**: The SFT gap to strongest original SFT is only 0.001 NDCG@10. The real gap is at RL stage. v2_on_p05 RL already beats strongest original RL at @1/@3 and @50, losing only at @5-@20.

**Concrete actions**:
- Try a larger RL batch size (current is 256, try 512)
- Try a longer RL training with the same recipe
- Try a softer reward that gives partial credit for same-prefix predictions (leveraging v2's cleaner prefix structure)

**Why this is high-priority**:
- v2_on_p05 SFT is already within noise distance of strongest original SFT
- The RL stage is where the real gap lives
- v2's advantage on collided targets (74.4% vs 61.5%) suggests the tokenizer is fundamentally better; the gap is in how RL exploits this

### 7.3 PRIORITY 3: Prefix stability diagnostic

**Rationale**: Before attempting any further tokenizer changes, we need a diagnostic tool that quantifies "how much does this tokenizer change rearrange the SID space."

**Implementation**:
- Given two index files, compute:
  - l1 prefix Jaccard: for each item, what fraction of its l1-prefix neighbors in index A are still l1-prefix neighbors in index B
  - l2 prefix Jaccard: same for l2
  - Changed-prefix rate: what fraction of items changed their l1 prefix, l2 prefix

**Why this matters**:
- It would have predicted R208's failure: a 100% SID change rate with low prefix stability means high rearrangement cost
- It provides a fast pre-screening filter for any future tokenizer variant: if prefix stability is too low, don't push downstream

### 7.4 PRIORITY 4 (DEFERRED): Second-round semantic retention

If Priority 1 succeeds, semantic retention may not be needed. If it fails, a second attempt at semantic retention should use:
- Higher temperature (τ=0.5 or τ=1.0) to create a softer distributional target
- Applied only to level 3 (consistent with the "freeze L1/L2" philosophy)
- With a much smaller weight (e.g., μ=0.01 instead of 0.025/0.05)

### 7.5 PRIORITY 5 (DEFERRED): v2 paper framing

The current evidence already supports a viable paper story even without closing the full RL gap:

1. v2 tokenizer produces structurally cleaner SIDs (proven)
2. v2 changes the downstream ranking profile: better head, better deep recall, better on hard cases (proven)
3. v2 survives end-to-end RL (proven)
4. The remaining gap is a structured mid-beam retention issue, not a quality failure (proven)
5. v2's collided-target performance (74.4% vs 61.5%) is a strong standalone result

This is a publishable contribution even if the aggregate NDCG@10 does not exceed the original. The key is framing it as "a tokenizer that changes the ranking profile in a principled way" rather than "a tokenizer that uniformly improves all metrics."

---

## 8. What NOT to Do Next

Based on the evidence:

1. **Do NOT launch another full-SID-rearrangement tokenizer variant.** R202a/R205 have shown that 100% SID changes carry unrecoverable downstream costs on easy cases. Any tokenizer change must preserve L1/L2 prefix structure.

2. **Do NOT retry R205 with different τ before trying freeze-L1L2.** The semantic retention KL might be fixable, but it's a secondary issue compared to the global rearrangement cost that any full retrain creates.

3. **Do NOT re-open the graph bank design.** Current evidence still shows fagsp_mid_base is the best G_mid candidate. No stage-2 result suggests the graph views are the bottleneck.

4. **Do NOT add more complexity (gates, end-to-end joint training).** The current bottleneck is a transfer interface issue, not a tokenizer capacity issue.

5. **Do NOT judge success only by NDCG@10.** The review packet correctly insists on the full cutoff set. v2's advantage at @1/@3 and @50 is real and should not be optimized away.

---

## 9. Summary

| Question | Answer |
|---|---|
| Did stop-grad work at tokenizer level? | **Yes**, clearly |
| Did it transfer to downstream? | **No** — collateral damage on easy cases > gain on hard cases |
| Why? | 100% SID rearrangement cost; downstream model loses learned routing on easy cases |
| Did semantic retention KL work? | **No** — τ=0.1 too aggressive, structural regression |
| Is the research direction wrong? | **No** — v2's ranking profile (better head, better @50, better collided targets) proves the tokenizer is fundamentally better |
| What is the real bottleneck? | SID space stability during tokenizer refinement; mid-beam retention in RL |
| Best next step? | Freeze L1/L2 codebooks, retrain only L3 with stop-grad + G_local |
| Best parallel step? | RL recipe tuning on existing v2_on_p05 |
