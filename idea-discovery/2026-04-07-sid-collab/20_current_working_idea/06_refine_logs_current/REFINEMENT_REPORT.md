# Refinement Report

## Summary

This refinement pass started from a broad collaborative-tokenizer idea and ended with a narrower, more defensible proposal:

- **Start**: rebuild the global SID pipeline with collaborative fusion and new quantization
- **End**: repair only the ambiguous local leaf decision with a selective collaborative residual

## Why The Original Direction Was Reduced

The original plan had three problems:

1. **Novelty overlap**: too many recent papers already own the global collaborative-tokenizer story.
2. **Problem mismatch**: repo diagnostics show local prefix ambiguity is the actual remaining bottleneck.
3. **Complexity sprawl**: the method contained multiple paper-sized ideas at once.

## What Survived The Refinement

- collaborative signal still matters
- the repo's SID diagnostics remain central
- prefix-level analysis remains part of the evaluation story

## What Was Intentionally Rejected

- full retokenization as the main paper contribution
- GAOQ-style full quantizer replacement
- graph-heavy end-to-end feature fusion
- RL redesign

## Final Method

**Ambiguity-Aware Collaborative Leaf Refinement (ACLR)**

- detect ambiguous `(a,b)` prefixes from diagnostics
- build train-only collaborative leaf prototypes
- add a local leaf-level auxiliary loss during training
- add a local leaf residual during decoding

## Why The Final Method Is Stronger

- it targets the failure mode actually observed in this repo
- it has a positive lightweight pilot already
- it is easy to explain and easy to ablate
- it does not try to compete head-on with every collaborative-tokenizer paper

## Remaining Open Question

Whether ACLR alone is enough for a full paper depends on the size and stability of the gain after training-time integration. If the gain remains too small, the backup path is ambiguity-aware partial retokenization.
