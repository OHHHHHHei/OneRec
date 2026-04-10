# Idea Candidates

## Ranking Summary

| Rank | Idea | Main claim | Pilot support | Novelty outlook | Risk |
|---|---|---|---|---|---|
| 1 | `MRC-SID` | Different SID levels should learn different mixtures of coarse/mid/local collaborative views | Strong | Strong | Medium |
| 2 | `AAG-SID` | Collaborative allocation should also depend on ambiguity, not only level | Medium | Medium-High | High |
| 3 | `Prog-MRC` | Use progressive training to stabilize multi-resolution front-end fusion | Medium | Medium | Medium |
| 4 | `Spec-SID` | Use low-/band-/high-frequency graph signals explicitly as level-specific collaborative inputs | Weak-Medium | Medium | High |

## 1. `MRC-SID` — Recommended

### Working name

**Multi-Resolution Collaborative Allocation for Hierarchical Semantic IDs**

### Core idea

Construct several purified collaborative views with different resolutions:

- `coarse`
- `mid`
- `local`

Then let each SID level learn how much it should use from each view, instead of forcing the entire tokenizer to consume the same collaborative representation.

### Why it fits the evidence

- `coarse` performs best on the overall error pool
- `mid` performs best on `same_l2` ambiguity in the pilot
- `fine1` is too sparse to use everywhere

This suggests that the right design is not a single global collaborative feature, but a level-dependent allocation over multiple collaborative views.

### Why it is promising

- matches the repo's motivation
- directly addresses the failure of naive global fusion
- more method-like than post-hoc local repair
- more differentiated than another uniform collaborative tokenizer

### Main risk

If the final method becomes too complex, reviewers may say it is simply a multi-branch fusion system without a clean insight.

## 2. `AAG-SID` — Ambiguity-Aware Gated SID

### Core idea

Start from `MRC-SID`, but make the collaborative allocation depend not only on level, but also on:

- prefix ambiguity
- item density
- confidence of the collaborative views

This means some subtrees or items receive stronger local collaborative signals than others.

### Why it is attractive

- aligns strongly with the local leaf ambiguity story
- could be more adaptive than a purely global per-level gate

### Main risk

- easy to drift toward a personalized or dynamic tokenizer story close to `PIT` / `Pctx`
- much harder to train and explain

## 3. `Prog-MRC` — Progressive Multi-Resolution Collaborative Tokenization

### Core idea

Use the same multi-resolution view bank, but focus on training stability:

- early training: semantic + coarse views dominate
- later training: unlock more mid/local views, especially for deeper levels

### Why it matters

- directly addresses collapse/stability risk
- likely easy to implement

### Main risk

- may look more like a training trick than a new method

## 4. `Spec-SID` — Spectral Collaborative Signal Allocation

### Core idea

Replace coarse/mid/local temporal views with graph-frequency views:

- low-frequency signal
- band-pass signal
- high-frequency signal

and allocate them across SID levels.

### Why it is interesting

- has a stronger theory flavor
- aligns with the broader graph-signal literature

### Main risk

- implementation burden is higher
- relation to current repo evidence is less direct than `MRC-SID`

## Why `MRC-SID` ranks first

`MRC-SID` has the best balance of:

- repo fit
- novelty
- simplicity
- pilot support

It also provides a clean paper claim:

> collaborative signal should be allocated across SID levels according to resolution, instead of fused uniformly across the whole tokenizer.

## De-prioritized ideas

### Pure local post-hoc refinement

Useful as a baseline or prototype, but not strong enough as the main paper thesis.

### Another uniform global collaborative tokenizer

Scientifically plausible, but too crowded and too weakly differentiated.

