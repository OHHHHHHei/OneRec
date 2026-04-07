# Motivation And Scope

## User motivations

The user wants to investigate three concrete concerns in MiniOneRec:

1. SID construction and generation rely mainly on semantic information and discard collaborative information.
2. SID digits do not have globally fixed meaning. The same code can mean different things under different prefixes, reducing predictability.
3. SID collisions exist, meaning one SID can map to multiple items.

## What this discovery pass aims to answer

1. Are these motivations empirically supported in the current repo?
2. What do the latest arXiv papers in 2025-2026 already cover?
3. Which directions are already crowded?
4. What idea still looks both feasible in this repo and differentiated enough to matter?

## Scope

This pass focuses on:

- MiniOneRec and SID-based generative recommendation
- latest arXiv work, with emphasis on 2025-2026
- public-paper evidence plus local repo evidence
- lightweight local pilots that can be run quickly from existing code

This pass does not attempt:

- a full implementation of a new tokenizer
- long GPU training
- a finished paper claim

## Working principle

The goal is not to ask whether the motivations sound plausible in isolation. The goal is to identify a direction that is:

- motivated by real repo bottlenecks
- not already crowded by the latest literature
- implementable on top of the existing MiniOneRec codebase
- likely to produce a clean paper story rather than a large but derivative system
