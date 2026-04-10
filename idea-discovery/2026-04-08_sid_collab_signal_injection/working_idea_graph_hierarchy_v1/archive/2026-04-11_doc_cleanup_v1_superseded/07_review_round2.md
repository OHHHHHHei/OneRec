# Review Round 2

## Reviewer-style verdict

**Score**: `8.7 / 10`

## What improved

- the method is now tokenizer-native
- graph structure is central, not auxiliary
- denoising and hierarchy-awareness are integrated into one story

## Remaining concerns

### 1. Gains may come from graph priors in general, not level awareness

The paper must rule out:

- uniform graph regularization
- graph feature fusion without graph regularization
- swapped level-to-graph allocation

### 2. Local transition graph is sparse

The method must not depend on a brittle local graph alone.  
It should use:

- confidence-aware pruning
- soft weighting
- semantic anchor terms

### 3. Mid-scale graph remains the biggest technical decision

This is now the most important cheap pilot:

- which Level 2 graph construction gives the best ambiguity signal without becoming too noisy

## Final reviewer recommendation

Proceed with `MGR-SID`, but keep the first implementation narrow:

- one coarse graph
- one chosen mid graph
- one local graph
- one level-wise graph allocation module
- strong baselines that isolate hierarchy-awareness from generic graph enhancement
