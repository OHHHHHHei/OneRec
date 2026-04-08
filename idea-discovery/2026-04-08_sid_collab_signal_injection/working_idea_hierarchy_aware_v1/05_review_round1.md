# Review Round 1

## Reviewer verdict

### Score

`7.6 / 10`

### Main positive

The direction is meaningfully different from a generic collaborative tokenizer. The lightweight pilot is especially valuable because it shows:

- `coarse` is best globally
- `mid` is best for deep local ambiguity

This gives the hierarchy-aware claim a real empirical anchor.

## Main concerns

### 1. The current mapping is too hand-designed

If the method simply says:

- Level 1 uses coarse
- Level 2 uses mid
- Level 3 uses local

then it will look heuristic and under-justified.

### 2. The idea risks becoming too wide

Right now the method space includes:

- view bank
- denoising
- gating
- ambiguity
- maybe curriculum

This can easily become a bag of tricks.

### 3. Novelty must be protected carefully

The top idea is only differentiated if the paper strongly emphasizes:

- the SID level as the main unit of collaborative allocation

If the method is written as generic multi-view fusion, reviewers can compare it directly against PRISM, DiscRec, and PIT and conclude the contribution is incremental.

## Required refinement

### A. Replace hard mapping with learned allocation

The core method should learn level-wise weights over the collaborative views, rather than imposing a fixed assignment.

### B. Keep the first version simple

The first serious proposal should contain only:

- three views
- view-wise purification
- per-level learned gates

Avoid ambiguity-aware dynamic gating in the core version.

### C. Make the claim falsifiable

The paper must commit to:

1. learned allocation beats uniform global fusion
2. learned allocation beats fixed mapping
3. learned weights are measurably non-uniform across levels

## Round-1 conclusion

The top idea survives, but it must be simplified and sharpened.

