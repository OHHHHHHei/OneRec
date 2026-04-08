# Review Round 1

## Reviewer-style verdict

**Score**: `7.6 / 10`

## Main concerns

### 1. Still too close to multi-view fusion

The initial graph version is better than the old window version, but it still risks sounding like:

- build several graph features
- mix them by level

That is not yet a strong enough tokenizer contribution.

### 2. `mid-scale` graph is under-defined

If the paper says:

- coarse graph
- mid graph
- local graph

but the mid graph is just “maybe 2-hop, maybe community, maybe band-pass,” the method will look vague.

### 3. Need a stronger tokenizer-native mechanism

The cleanest version should make graph structure participate in SID learning itself, not just in the item representation.

## Recommendations

- upgrade the main method from graph feature fusion to graph-regularized quantization
- define the method around a multiplex graph bank and level-specific structural preservation
- keep mid-scale design space explicit, but freeze one simple main instantiation for the first full version
