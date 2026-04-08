# Revision After Round 1

## Changes made

Based on the first review, the top idea is narrowed from a broad design space to a cleaner method:

## Refined top idea

### `MRC-SID`

- build exactly three collaborative views:
  - `coarse`
  - `mid`
  - `local`
- purify each view separately
- learn a **global per-level gate** over these views
- inject the resulting fused signal into the three SID levels

## What was removed from the core method

- no per-item dynamic gating
- no subtree-specific ambiguity branch in the first version
- no large curriculum story as the central contribution

These can still appear later as ablations or extensions, but not as the main method.

## Refined paper thesis

> Uniform collaborative fusion is suboptimal because different SID levels benefit from different collaborative resolutions. A tokenizer should therefore learn level-wise allocation over purified collaborative views.

## Why this is better

- cleaner than a hard-coded layer-to-view mapping
- simpler than a personalized/dynamic tokenizer
- easier to compare fairly against uniform fusion baselines

