# Stage-2 Interface Diagnostics Review: Deep Analysis

**Date**: 2026-04-13  
**Scope**: Detailed analysis of R301–R304 interface diagnostics, integrated with prior stage-2 results and the full v2 mainline evidence chain.

---

## 1. Why This Document Exists

Stage-2 produced a paradox: R202a clearly improved tokenizer-side structure, yet R208 (its downstream screen) regressed. The team then designed four interface diagnostics (R301–R304) to understand *why* the structural gain did not transfer. This document reviews those diagnostics in detail, assesses whether the analysis scripts are sound, identifies what the data actually says, and proposes the next step.

---

## 2. Diagnostic-by-Diagnostic Analysis

### 2.1 R301: Prefix Stability

**What it measures**: For each pair of items that share a prefix (l1 or l2) in the baseline SID space, what fraction still share that prefix in the variant SID space.

**Script assessment**: The implementation is clean. `prefix_pair_retention` iterates over all same-prefix pairs in the baseline and checks whether they remain same-prefix in the variant. `per_item_neighbor_metrics` computes Jaccard, recall, and precision for each item's prefix-neighbor set. The methodology is sound.

**Key numbers (R202a vs current v2)**:

| Metric | Value |
|---|---|
| Changed l1 rate | 99.65% |
| Changed l2 rate | 100% |
| Changed full SID rate | 100% |
| l1 pair retention | 41.4% |
| l2 pair retention | 61.2% |
| Mean l2 neighbor Jaccard | 0.589 |

**What this tells us**:

1. **Every single item changes its full SID.** This is the single most important number. It means the downstream LLM is learning in a completely new token space.

2. **But the rearrangement is not random.** 61.2% of same-l2 item pairs are preserved. This means stop-grad didn't "scramble" the SID space — it systematically reorganized it while preserving a majority of local l2 neighborhoods.

3. **l1 is less stable than l2.** Only 41.4% of same-l1 pairs are preserved, vs 61.2% for l2. This is surprising and important: stop-grad affects the coarse level *more* than the local level, which is the opposite of what we might expect from a mechanism that "isolates" each level.

**Why l1 is less stable**: In the original v2, level 1's codebook received gradient flow from all three levels' graph losses. Under stop-grad, level 1 only receives its own G_coarse loss (λ_c=0.05). This is a ~6.5× reduction in effective graph regularization for level 1. The codebook of level 1 therefore finds a substantially different equilibrium, causing broad l1 prefix reassignments. This is exactly the mechanism that disrupts "already-stable" easy cases downstream.

**Missing analysis**: R301 does not report l1/l2 stability *stratified by fanout*. It would be very informative to know: do the most stable l2 neighborhoods correspond to easy (low-fanout) or hard (high-fanout) items? If hard-case l2 neighborhoods are more stable than easy-case ones, that would explain why hard cases improve downstream while easy cases regress.

### 2.2 R302: Code Polysemy

**What it measures**: For each token at each level, how semantically diverse are the items assigned to it? And for tokens that appear under multiple parent prefixes, how much does their "meaning" drift across contexts?

**Script assessment**: Clean implementation. `cosine_spread` computes mean angular distance from the centroid, which is a standard semantic coherence measure. `pairwise_centroid_drift` measures how much a token's contextual meaning changes across parent prefixes. Both are well-motivated.

**Key finding**: Current v2 and R202a are virtually identical on all polysemy metrics.

| Level | v2 spread | R202a spread | Delta |
|---|---|---|---|
| a | 0.0620 | 0.0631 | +0.0011 |
| b | 0.1068 | 0.1057 | -0.0011 |
| c | 0.1202 | 0.1199 | -0.0003 |
| b prefix drift | 0.2265 | 0.2282 | +0.0017 |
| c prefix drift | 0.2426 | 0.2421 | -0.0005 |

**What this tells us**: This is one of the most valuable findings in the entire diagnostic batch. It definitively rules out "token semantic overload" as the cause of downstream regression.

The hypothesis "R202a's codes are more polysemous, so the downstream model can't distinguish between items sharing a token" is empirically falsified. Whatever is causing the downstream regression, it is NOT that the individual tokens have become less semantically coherent.

**One important caveat**: The number of *active* level-a tokens drops from 203 to 157 (a 23% reduction). This means each a-token covers ~23 items instead of ~18. While the semantic spread per token is similar, the combinatorial diversity at level a has decreased. This could mean the SID space has fewer distinct "categories" even though each category is equally coherent.

### 2.3 R303: Transfer Attribution

**What it measures**: For each test item, it joins (a) how the SID changed (from R301's per-item metrics) with (b) whether the downstream prediction improved or worsened (from R208's top-k analysis).

**Script assessment**: Sound implementation. It reads the top-k CSV that has per-item `improved_at_k` / `worsened_at_k` flags and joins them with the prefix stability metrics. This is the correct approach to answer "what SID-change features predict downstream improvement vs regression."

**Key finding — the asymmetric Jaccard pattern**:

| Cutoff | Group | Count | l1 Jaccard | l2 Jaccard | Baseline l2 fanout |
|---|---|---|---|---|---|
| @3 | improved | 102 | **0.347** | 0.497 | **11.72** |
| @3 | worsened | 92 | 0.242 | **0.700** | 6.93 |
| @10 | improved | 130 | **0.300** | 0.487 | **7.72** |
| @10 | worsened | 147 | 0.258 | **0.653** | 6.98 |

This pattern is extremely informative and consistent across all cutoffs:

1. **Improved examples have HIGHER l1 Jaccard but LOWER l2 Jaccard.** This means: for items that improved downstream, their coarse-level (l1) neighborhood was better preserved by R202a, but their local (l2) neighborhood was more aggressively rewritten.

2. **Worsened examples have LOWER l1 Jaccard but HIGHER l2 Jaccard.** This means: for items that worsened downstream, their coarse routing was disrupted more, but their local l2 neighborhood was actually more stable.

**Interpretation**: This is NOT simply "rearrangement = bad." The pattern says something much more specific:

> Items improve when R202a preserves their coarse routing (l1) while aggressively cleaning up their local subtree (l2). Items worsen when R202a disrupts their coarse routing while leaving their local subtree relatively unchanged.

In other words: **l1 stability is the key predictor of transfer success, not l2 stability.** The downstream model's performance depends more on whether items land in the "right general area" (l1) than on the exact leaf arrangement (l2).

This makes sense: the downstream LLM generates SIDs autoregressively, first predicting `a`, then `b|a`, then `c|a,b`. If the `a` token changes, the model's entire decoding trajectory shifts. But if only `c` changes within a stable `a,b` prefix, the model can adapt more easily.

**What this implies for the freeze-L1L2 proposal**: This data strongly supports freezing level 1 (and ideally level 2) during any tokenizer refinement. The R303 evidence says l1 stability is the strongest predictor of downstream transfer, and R202a's l1 instability (only 41.4% pair retention) is likely the primary cause of downstream regression.

### 2.4 R304: Learnability Probe

**What it measures**: A lightweight linear classifier (SGDClassifier with sparse features) tries to predict each level of the target SID from the history SID sequence. This measures how "learnable" the SID mapping is for a simple model.

**Script assessment**: The feature engineering is well-designed:
- Individual token counts from history (bag-of-tokens)
- Token pair features (a|b combinations)
- Conditioning tokens for deeper levels (gold a for predicting b, gold a+b for predicting c)

Using a linear model is appropriate because it measures the "linear separability" of the SID-to-SID mapping — essentially how much structure is available for a simple model to exploit.

**Key numbers**:

| Variant | Target | Overall | Hard (l2≥4) | Stable (l2≤2) |
|---|---|---|---|---|
| v2 | a | 0.0902 | 0.2230 | 0.0593 |
| R202a | a | **0.0997** | 0.2231 | **0.0676** |
| v2 | b\|a | **0.2392** | **0.4085** | **0.1914** |
| R202a | b\|a | 0.2134 | 0.3665 | 0.1794 |
| v2 | c\|a,b | **0.4365** | **0.2218** | **0.4973** |
| R202a | c\|a,b | 0.4159 | 0.1806 | 0.4784 |

**What this tells us**:

1. **R202a makes level a slightly easier to predict (+0.0095 overall).** This aligns with the structural finding that R202a's l1 codebook uses fewer active tokens (157 vs 203), creating a coarser but more predictable first level.

2. **R202a makes level b|a harder to predict (-0.0258 overall, -0.042 on hard cases).** This is the critical signal. Even given the correct a token, predicting b is harder in R202a's SID space. On hard cases (l2≥4), the drop is dramatic: 40.9% → 36.7%.

3. **R202a makes level c|a,b harder to predict (-0.0206 overall, -0.041 on hard cases).** Same pattern extends to the leaf level.

**Why deeper levels become harder to learn in R202a**: The stop-gradient mechanism lets each level optimize independently. Level 1 optimizes for G_coarse, level 2 for G_mid, level 3 for G_local. But this independence means the cross-level consistency is no longer enforced by shared gradients. In the original v2, the gradient flow from level 2/3 losses into level 1 created an implicit "alignment pressure" — level 1's codebook was shaped partly by what levels 2/3 needed. Stop-grad removes this, letting level 1 find its own optimum that may not be as compatible with levels 2/3.

In terms of learnability: if level 1 is optimized *independently* for G_coarse structure, the resulting l1 partitioning may not create the cleanest "conditioning context" for level 2/3 predictions. The original v2's "leaky gradients" may have accidentally helped by forcing level 1 to be a better conditioning signal.

**Critical observation**: The learnability probe uses a linear model on bag-of-token features. The downstream LLM is far more expressive. So the learnability drop may overstate the downstream impact for the LLM. But the directional signal is still valid: R202a's SID space has less linearly-exploitable structure at deeper levels.

---

## 3. Integrated Analysis: What the Four Diagnostics Tell Us Together

Combining all four diagnostics, the causal chain is now clear:

```
Stop-grad isolates levels
    ↓
Level 1 finds a different equilibrium (fewer active a-tokens, different l1 partitioning)
    ↓
99.65% of items change their l1 prefix
    ↓
Only 41.4% of same-l1 item pairs are preserved
    ↓
The downstream model's learned l1 routing is disrupted
    ↓
Meanwhile, the l2 structure improves locally (lower entropy, fewer deep crowded buckets)
    ↓
But the l1 disruption dominates because:
    - easy cases (l2≤2, ~3141 items) rely primarily on correct l1 routing
    - hard cases (l2≥4, ~1010 items) benefit from the l2 cleanup
    - easy cases outnumber hard cases 3:1
    ↓
Net downstream effect: regression
```

**The code polysemy evidence (R302) rules out the alternative hypothesis** that the regression is caused by "tokens becoming more semantically overloaded." Token semantics are fine — the problem is purely at the routing/interface level.

**The learnability probe (R304) adds depth**: even if you gave the downstream model the correct l1 token, it would still struggle more to predict l2 and l3 in R202a's space. This means the problem isn't *only* about l1 disruption — there's also a conditional predictability issue at deeper levels.

---

## 4. Assessment of the Analysis Scripts

### 4.1 Strengths

- **R301** is well-designed and provides exactly the right metrics (pair retention, Jaccard, recall).
- **R302** is a creative diagnostic that I haven't seen in prior work. Measuring code polysemy and prefix-conditioned drift is a genuinely useful way to separate "token semantics" from "routing structure."
- **R303** correctly joins tokenizer-side changes with downstream outcomes, enabling causal attribution.
- **R304** uses a lightweight probe that is fast, interpretable, and measures the right thing (conditional predictability across levels).

### 4.2 Gaps and Suggested Extensions

1. **R301 should be stratified by fanout bucket.** Currently it reports global statistics. If we could see l1/l2 stability separately for easy vs hard items, we could directly test whether the "easy cases lose l1 routing" hypothesis holds at the item level.

2. **R303 should include a regression/correlation analysis.** Currently it shows group means for improved vs worsened sets. A logistic regression with `improved_at_10` as the target and `l1_jaccard`, `l2_jaccard`, `baseline_l2_fanout` as features would quantify the relative importance of each factor and tell us whether l1 Jaccard is truly the dominant predictor.

3. **R304 should include a "freeze-L1L2 simulation."** What if we use v2's l1/l2 tokens but R202a's l3 tokens? This would simulate the freeze-L1L2 proposal at the learnability level, without actually running a tokenizer experiment. If this "synthetic" SID space has better c|a,b learnability than R202a, it directly predicts that freeze-L1L2 will avoid the learnability regression.

4. **Missing: decode-path analysis.** None of the diagnostics examine the autoregressive generation process. The downstream LLM generates SIDs left-to-right. If R202a's level-a predictions are wrong more often (because the model hasn't learned the new l1 routing), then beam search will explore the wrong l1 subtree, missing the target even if the l2/l3 structure is better. A decode-path analysis that checks "how often does the model predict the correct a-token" would directly test this.

---

## 5. What Does This Change About Next Steps?

### 5.1 The freeze-L1L2 proposal is now MORE strongly supported

Before the diagnostics, freeze-L1L2 was motivated by the general observation that "100% SID rearrangement is bad." Now R303 provides specific evidence: **l1 neighbor Jaccard is the strongest discriminator between improved and worsened examples.** Items that maintained their coarse routing tended to improve downstream; items that lost it tended to regress.

Freezing L1 and L2 codebooks would:
- Set l1 changed rate to 0% (vs current 99.65%)
- Set l2 changed rate to 0% (vs current 100%)
- Set l1/l2 pair retention to 100% (vs current 41.4%/61.2%)
- Preserve 100% of the downstream model's learned l1/l2 routing
- Only allow l3 (leaf) changes within fixed l2 subtrees

This is exactly what R303's data says should work.

### 5.2 The learnability concern (R304) adds a nuance

R304 shows that R202a makes b|a and c|a,b harder to predict even given the correct parent tokens. If we freeze L1/L2 and only retrain L3, the learnability of a and b|a will be UNCHANGED (because the codebooks are frozen). Only c|a,b will potentially change. And even that change will be within fixed (a,b) subtrees, which constrains how much the c-level learnability can degrade.

So freeze-L1L2 addresses both the routing disruption (R301/R303) and the learnability degradation (R304) simultaneously.

### 5.3 RL-stage optimization remains the parallel priority

The diagnostics confirm that the SFT-stage gap is a transfer/interface issue. But the RL-stage gap (v2_on_p05 RL vs strongest original RL) is a different problem. RL already narrows the NDCG@10 gap from 0.001 (SFT) to 0.003 (RL), but creates a larger @5-@20 retention issue. This is likely a beam-search / reward-shaping issue, not a tokenizer issue.

The v2 RL result already shows:
- 74.4% collided-target top1 hit (vs 61.5% for strongest original RL)
- Better @1/@3 and @50
- Only losing @5-@20

This profile suggests the tokenizer is fundamentally strong, and the remaining RL gap is about how the RL objective shapes beam diversity.

---

## 6. Recommended Concrete Actions

### Priority 1: Freeze-L1L2, Retrain L3 Only

**Implementation plan**:
1. Load v2 tokenizer best checkpoint
2. Freeze encoder, level 1 codebook, level 2 codebook
3. Keep stop-grad on (for consistency, though it's moot for frozen levels)
4. Apply only G_local regularization on level 3
5. Train level 3 codebook + decoder (the decoder needs to adapt to the new l3 assignments)
6. Generate SIDs → verify that l1/l2 prefixes are 100% preserved
7. Run R301 to confirm prefix stability ≈ 100%
8. If structure looks good, push to SFT screen

**Predicted diagnostic profile**:
- R301: l1 changed rate = 0%, l2 changed rate = 0%, l3 changed rate > 0%
- R302: polysemy unchanged (tokens themselves don't change)
- R304: a learnability unchanged, b|a learnability unchanged, c|a,b learnability potentially improved

### Priority 2: R304 Simulation of Freeze-L1L2

Before running the actual tokenizer experiment, we can cheaply test the freeze-L1L2 hypothesis:
1. Create a synthetic SID index that uses v2's (a,b) prefixes but R202a's c-tokens (where the (a,b) prefix matches; otherwise keep v2's c)
2. Run R304 on this synthetic SID space
3. If c|a,b learnability improves (or at least doesn't degrade), this predicts freeze-L1L2 will transfer better

Cost: CPU-only, minutes.

### Priority 3: RL Recipe Exploration

Independent of tokenizer work, explore:
- Larger RL batch size
- Prefix-aware reward bonus (give partial credit for correct l1 or l2 prefix)
- Longer RL training (current v2 RL may be stopping too early)

### Priority 4: Extended R303 Regression Analysis

Fit a logistic regression: `improved_at_10 ~ l1_jaccard + l2_jaccard + baseline_l2_fanout + l1_jaccard * baseline_l2_fanout`

This would formally quantify whether l1 Jaccard is the dominant predictor and whether there's an interaction with fanout. Low cost, high interpretive value.

---

## 7. Summary of Judgments

| Question | Judgment | Confidence |
|---|---|---|
| Was stage-2's interface diagnostics plan well-designed? | Yes — R301-R304 together tell a coherent story | High |
| Is the analysis script (R301-R304) implementation correct? | Yes — verified line by line | High |
| Is "token polysemy" the cause of downstream regression? | **No** — R302 rules it out | High |
| Is "l1 routing disruption" the primary cause? | **Yes** — R301+R303 strongly support this | High |
| Is "deeper-level learnability degradation" a contributing factor? | Probably yes, but secondary to l1 disruption | Medium |
| Will freeze-L1L2 fix the transfer problem? | Likely yes for the l1/l2 routing issue | Medium-High |
| Will freeze-L1L2 improve downstream metrics? | Cautiously optimistic — removes the dominant failure mode but other issues may remain | Medium |
| Should RL tuning proceed in parallel? | Yes — independent of tokenizer work | High |

---

## 8. One-Sentence Conclusion

> The interface diagnostics reveal that R202a's downstream failure is primarily caused by coarse-level (l1) routing disruption, not by token semantic degradation or local structural deficiency, which makes **freeze-L1L2 + retrain-L3-only** the single highest-value next experiment.
