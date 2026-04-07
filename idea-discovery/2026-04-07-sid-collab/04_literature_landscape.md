# Literature Landscape

## Big picture

The user's three motivations are not wrong. In fact, the latest arXiv work strongly validates them.

However, the most important discovery is not "the motivations are true". It is:

- which parts are already crowded
- which parts remain under-explored
- which parts match this specific repo's strongest evidence

## Crowded region A: global collaborative tokenizer redesign

Representative papers:

- ReSID
- PIT
- PRISM
- UNGER

Shared pattern:

- semantic-only tokenization is insufficient
- collaborative information should enter the tokenizer
- representation learning and tokenization should be more recommendation-native

Implication:

- A proposal like "build a new collaborative-aware tokenizer for MiniOneRec" is no longer sharp enough.
- It may still work empirically, but it is hard to sell as a fresh paper anchor.

## Crowded region B: prefix semantics and prefix-aware generation

Representative papers:

- TrieRec
- APAO
- ReSID

Shared pattern:

- hierarchical SID induces a structure that should not be ignored
- prefix quality strongly affects beam-search success
- training should account for prefix-level failures

Implication:

- A proposal framed only as "prefixes are unstable, so let us make prefixes better" is also getting crowded.

## Crowded region C: collision-aware SID learning

Representative papers:

- QuaSID
- HiD-VAE

Shared pattern:

- collisions and entanglement are important
- supervision or regularization should explicitly reduce harmful collisions

Implication:

- A proposal framed mainly as "solve SID collision" risks feeling too close to recent work, especially if it touches tokenization directly.

## What still looks less crowded

The less crowded space is not "better tokenizer" in general. It is something more selective:

- keep the existing tokenizer mostly intact
- detect where the current prefix path is risky or ambiguous
- inject collaborative information only where the current semantic structure is not enough
- treat this as a local repair problem rather than a global re-indexing problem

This space is attractive for four reasons:

1. It matches the local repo evidence better than a global tokenizer rewrite.
2. It reuses the existing MiniOneRec pipeline instead of replacing it.
3. It naturally combines motivations 1 and 2.
4. It is more differentiated from the newest tokenizer papers.

## Repo-specific gap

The current repo has a particularly strong and unusual property:

- it already contains evaluation-time collaborative rerank utilities
- local diagnostics show same-prefix ambiguity is the practical bottleneck

That means this repo is unusually well positioned for a paper around:

- ambiguity detection
- local subtree repair
- train-only collaborative statistics
- selective intervention rather than global tokenizer retraining

## Landscape conclusion

The field now rewards precision.

If the paper says:

- "we inject collaboration into tokenizer learning"

it is probably too broad and too crowded.

If the paper says:

- "we identify that MiniOneRec's dominant failure mode is ambiguity inside semantically plausible local subtrees, and we introduce a calibrated local repair layer that only activates when semantic prefixes become unreliable"

that story is much cleaner and better aligned with both the latest literature and this repo's actual evidence.
