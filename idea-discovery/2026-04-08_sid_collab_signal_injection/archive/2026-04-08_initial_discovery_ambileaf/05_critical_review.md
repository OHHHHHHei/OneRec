# Critical Review

## Reviewer-style verdict

### Idea 1: AmbiLeaf

**Score**: 8.3 / 10

### Strengths

- Strong problem-method alignment. The repo says the main residual failure is local leaf ambiguity, and the idea directly edits the leaf structure.
- It keeps front-end collaboration alive without repeating the collapsed `text + cf` recipe.
- Simpler than a full tokenizer rebuild, which improves feasibility and story clarity.

### Main concerns

- The ambiguity detector could accidentally leak evaluation information if not strictly defined from train or validation statistics.
- If only a small portion of the catalog is affected, overall gains may be modest.
- The method must show that it truly changes the SID structure, not just emulate reranking with a different name.

### Minimum viable form

- Freeze the prefix
- Recompute only the last token inside top-`M` ambiguous prefixes
- Use purified collaborative residuals plus semantic residuals
- Compare against:
  - baseline tokenizer
  - naive global fusion
  - semantic-only local leaf retokenization
  - shuffled-CF local leaf retokenization
  - ACLR-lite inference-only repair

### What would make it paper-worthy

- recovering meaningful HR/NDCG gain while keeping low global collision
- reducing same-prefix miss rates more than generic global methods
- showing that the gain is concentrated in ambiguous subtrees and not from arbitrary extra capacity

## Idea 2: PurifyThenQuantize

**Score**: 7.4 / 10

### Strengths

- High upside if it works
- Consistent with the field's current direction
- Easy to motivate from PRISM / PIT / ReSID-style concerns

### Main concerns

- Very crowded in 2026
- High engineering complexity
- Hard to prove differentiation unless the hierarchy-aware contribution is sharp and measurable

### Recommendation

Keep as a backup high-ceiling direction, or as a stronger baseline family to test against AmbiLeaf.

## Idea 3: Coarse2Fine Dual Signal

**Score**: 7.1 / 10

### Strengths

- Pragmatically strong
- Safest path if the goal is performance first

### Main concerns

- Story fragmentation
- Easy to look like system stitching
- Harder to defend as one clean idea

### Recommendation

Use as a contingency plan or later system extension, not as the first paper claim.

## Final reviewer recommendation

If only one direction is pushed next, it should be:

## Recommended next idea

**AmbiLeaf**

Not because global collaboration is wrong, but because AmbiLeaf gives the cleanest answer to all four constraints at once:

- collaboration should matter
- naive global fusion is unstable
- the remaining bottleneck is local leaf ambiguity
- the paper still needs a fresh and elegant method center

