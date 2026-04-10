# Local Pilot

## Goal

Before designing a full tokenizer method, we first test a minimal question:

> Do collaborative signals with different temporal/local granularities behave differently on different error buckets?

If yes, then a hierarchy-aware collaborative tokenizer is more plausible than a uniform global fusion design.

## Setup

Using existing train/test/result files from the current repo, we built train-only collaborative compatibility scores at four resolutions:

- `coarse10`: use the last 10 history items
- `mid3`: use the last 3 history items
- `local2`: use the last 2 history items
- `fine1`: use only the most recent item

For each error case, we compare the target item's collaborative score against the strongest candidate under the predicted SID. This is the same conservative setting as the existing diagnostics.

## Pilot Result 1: Different views help different buckets

### Industrial

| View | All Errors | Same-L1 Errors | Same-L2 Errors |
|---|---:|---:|---:|
| `coarse10` | `0.0981` | `0.1903` | `0.2284` |
| `mid3` | `0.0888` | `0.1847` | `0.2346` |
| `local2` | `0.0843` | `0.1781` | `0.2191` |
| `fine1` | `0.0776` | `0.1648` | `0.2006` |

### Office

| View | All Errors | Same-L1 Errors | Same-L2 Errors |
|---|---:|---:|---:|
| `coarse10` | `0.1083` | `0.2171` | `0.4048` |
| `mid3` | `0.1072` | `0.2171` | `0.4167` |
| `local2` | `0.1002` | `0.2039` | `0.4048` |
| `fine1` | `0.0836` | `0.1820` | `0.3690` |

## Interpretation

These results support three important points.

### 1. A single collaborative view is probably not enough

- `coarse10` is strongest on the global error pool
- `mid3` becomes strongest on the deepest local ambiguity bucket (`same_l2`) on both datasets

This is exactly the kind of evidence we would expect if different SID levels need different collaborative resolutions.

### 2. “More local” does not automatically mean “better”

`fine1` is consistently the weakest view. So the right story is **not**:

- use the finest signal for the deepest layer because finer is always better

Instead, the right story is:

- each layer needs an appropriate level of locality, and overly local signals may be too sparse or too noisy

### 3. Mid-scale collaborative information looks especially relevant to leaf ambiguity

The strongest result in the deepest bucket comes from `mid3`, not from the globally pooled signal and not from the last-item-only signal.

This is a strong hint that the last SID level may need:

- local enough signal to resolve semantic neighbors
- but not so local that the signal becomes brittle

## Pilot Result 2: Finer views are much sparser

Target-score nonzero coverage on the test set:

### Industrial

| View | Nonzero Coverage |
|---|---:|
| `coarse10` | `0.3499` |
| `mid3` | `0.2815` |
| `local2` | `0.2447` |
| `fine1` | `0.1891` |

### Office

| View | Nonzero Coverage |
|---|---:|
| `coarse10` | `0.3921` |
| `mid3` | `0.3214` |
| `local2` | `0.2797` |
| `fine1` | `0.2111` |

## Interpretation of sparsity

This explains why the finest collaborative signal should not be injected everywhere:

- it has lower coverage
- it is more brittle
- it is less suitable as a universal front-end signal

This again supports a hierarchy-aware design:

- stable/coarse views can support upper levels
- more local views can be reserved for deeper levels where discrimination matters more

## Pilot Takeaway

The lightweight pilot does **not** yet prove a final method.

But it does give a strong empirical clue:

> collaborative signal utility is non-uniform across error buckets, and the best view for deep local ambiguity is not the same as the best view for the overall error pool.

That is enough to justify a next-stage idea centered on **multi-resolution collaborative allocation across SID levels**.

