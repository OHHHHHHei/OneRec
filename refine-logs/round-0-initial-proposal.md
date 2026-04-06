# Research Proposal: Collaborative-Aware Global SID Rebuild

## Problem Anchor

- Bottom-line problem: MiniOneRec's current SID construction is text-dominant and may underuse collaborative information that matters for recommendation.
- Must-solve bottleneck: the SID space should better separate behaviorally different but textually similar items.
- Non-goals: do not redesign the full recommendation stack or depend on a large industrial-scale retraining regime.
- Constraints: must stay compatible with the current MiniOneRec pipeline and data contracts; should be testable in the existing repo without a full restart from scratch.
- Success condition: the new SID construction improves prefix quality and downstream recommendation metrics.

## Technical Gap

The baseline tokenizer is text-first, while recommendation quality depends on both semantic and collaborative structure. A natural first idea is to rebuild the tokenizer around multi-field fusion, collaborative-aware quantization, and anti-collision regularization.

## Method Thesis

Learn a new globally collaborative tokenizer by fusing text, category, attribute, and collaborative signals before quantization, then enforce more stable and less colliding semantic IDs through globally aligned quantization and uniqueness regularization.

## Contribution Focus

- Dominant contribution: collaborative-aware semantic ID construction
- Optional supporting contribution: global alignment and anti-collision regularization
- Explicit non-contributions: no change to backbone LLM architecture

## Proposed Method

### Complexity Budget

- Frozen / reused backbone: current SFT and RL pipeline
- New trainable components: collaborative representation module, collaborative-aware quantizer
- Tempting additions intentionally not used: large graph stack, new RL objective, large multimodal system

### System Overview

1. Build richer item representations from text plus collaborative features.
2. Quantize them with a more structured SID generator.
3. Regenerate `index.json` and downstream data.
4. Retrain MiniOneRec on the new SIDs.

### Core Mechanism

- fuse text and collaborative features before quantization
- impose global code alignment across prefixes
- reduce collisions with uniqueness-style regularization

## Initial Minimal Validation

- compare new tokenizer against text-only SID
- evaluate collision rate, prefix entropy, same-prefix errors, and HR/NDCG
- ablate collaborative features and anti-collision loss

## Self-Assessment

This proposal is plausible, but it is still broad and may overlap with recent collaborative-tokenizer work. It also treats the whole tokenizer as the bottleneck before checking whether the repo's actual error pattern is more local.
