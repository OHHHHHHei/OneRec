# Review Round 2

## Reviewer verdict

### Score

`8.5 / 10`

### Main positive

The revised version is much cleaner. It now has one central point:

- the collaborative view allocation should be non-uniform across SID levels

This is easier to defend and easier to test.

## Remaining concerns

### 1. The gains could come from extra views, not from hierarchy-aware allocation

If `MRC-SID` beats a baseline simply because it uses more collaborative information than the baseline, then the main claim is not validated.

### 2. The method still needs strong controls

To defend the hierarchy claim, the experiments must include:

- uniform all-level fusion with the same view bank
- hard-coded fixed mapping
- swapped mapping
- no-purification version

### 3. The learned gates need interpretation

The paper should not just report improved HR/NDCG. It should also show:

- what each level learned to use
- whether those learned allocations match the pilot intuition

## Required refinement

### A. Add parameter-matched uniform baseline

Use the same three views, same total collaborative dimensionality, but the same fusion policy for all SID levels.

### B. Add swap controls

Examples:

- force local-heavy fusion at Level 1
- force coarse-heavy fusion at Level 3

This provides a direct challenge to the hierarchy claim.

### C. Add gate analysis as a first-class result

The paper should explicitly plot or report the learned allocation weights per SID level.

## Round-2 conclusion

The top idea is now strong enough to advance into a concrete proposal and experiment plan.

