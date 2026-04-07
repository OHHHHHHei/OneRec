# Round 1 Review

**Reviewer**: local critical review
**Score**: 6.8 / 10
**Verdict**: REVISE

## Strengths

- The proposal is aligned with the user's high-level motivation.
- It targets a real weakness: text-only SID does miss collaborative structure.
- The repo already has a modular embedding and SID pipeline, so some parts are implementable.

## Major Problems

### 1. Novelty overlap is too high

The broad recipe is now too close to the recent literature:

- `ReSID` already covers field-aware masked representation learning and globally aligned quantization.
- `PRISM` already covers collaborative denoising plus hierarchy-aware tokenization.
- `HiD-VAE` already covers hierarchical semantic IDs plus uniqueness regularization.
- `PIT`, `ETEGRec`, `DiscRec`, and `PRORec` all crowd the "inject collaboration into tokenizer construction" space.

As written, the method risks looking like a recombination of recent modules rather than a new paper.

### 2. The proposal drifts away from the repo's actual bottleneck

The repo's evidence does not say "global collision and global tokenizer failure are the main problem." It says:

- collision is low
- prefix ambiguity is high
- many misses are local same-prefix confusions
- collaborative evidence matters most inside those local confusions

The proposal is therefore too global for the observed failure mode.

### 3. Complexity sprawl

The initial design quietly contains several separate contributions:

- multi-field representation learning
- collaborative fusion
- global alignment quantization
- uniqueness regularization
- diversity regularization

That is contribution sprawl, not one clean paper.

### 4. Current code fit is weaker than it first appears

Global retokenization touches:

- embedding generation
- quantizer training
- `index.json`
- `convert`
- token extension
- all downstream training

That is possible, but it is not the smallest adequate intervention for this repo.

## Revision Mandate

Shrink the story to one dominant contribution:

> repair local leaf ambiguity inside already-correct semantic prefixes using train-only collaborative signal, instead of rebuilding the whole tokenizer.

This revised direction should:

- stay anchored on the observed same-prefix failures
- introduce at most one main trainable addition
- use diagnostics to decide where the method activates
- avoid global retokenization unless later evidence forces it

## What To Keep

- the collaborative signal should still matter
- the final method should still be compatible with MiniOneRec
- the evaluation should still include prefix diagnostics, not just HR/NDCG

## What To Remove

- full FAMAE-style representation stack
- GAOQ-style full quantizer replacement
- anti-collision side quests as a main story
- large graph modules and RL redesign
