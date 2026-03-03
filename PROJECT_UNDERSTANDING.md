# Project Understanding

## High-level view

This repository implements a generative recommendation workflow built around semantic item IDs.
The central idea is:

1. Convert each item into a text representation.
2. Encode that text into embeddings.
3. Quantize the embeddings into discrete SIDs.
4. Train an LLM to generate valid next-item SIDs from user history.
5. Improve the policy further with recommendation-oriented RL.
6. Evaluate using constrained decoding over the valid SID set.

## Mainline contracts

The refactor preserves the following contracts:

- preprocessing outputs interaction splits and item metadata
- SID construction outputs item-to-SID mappings in `*.index.json`
- conversion produces CSVs and `info.txt` files used by training and evaluation
- SFT extends the tokenizer with SID tokens
- RL and evaluation rely on constrained decoding over the same valid SID space

## Mainline modules

- `src/minionerec/preprocess/`: raw data to structured interaction files
- `src/minionerec/sid/`: embeddings, quantization, SID generation
- `src/minionerec/data/`: conversion and dataset builders
- `src/minionerec/training/sft/`: supervised training
- `src/minionerec/training/rl/`: reinforcement learning
- `src/minionerec/evaluation/`: constrained decoding and metrics

## Important implementation notes

- The dataset base layer filters `None` samples before training.
- Unsafe `eval` parsing in the mainline path was replaced with safe parsing.
- The SFT freeze-LLM path keeps the original vocabulary boundary so only new SID embeddings can
  remain trainable.
- RL semantic reward handling no longer crashes on invalid generated items.

## Operational reality

Local validation is meant for:
- config loading
- smoke tests
- conversion contract checks

Full training and evaluation should run on a remote machine with the required GPU environment.
