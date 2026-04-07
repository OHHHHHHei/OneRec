# Local Context

## Current repo understanding

The current MiniOneRec pipeline in this repo is effectively:

1. preprocess data
2. build text-driven item embeddings
3. quantize them into 3-level SIDs
4. train SFT / RL generation over SID sequences
5. evaluate with constrained decoding

The codebase already contains a lightweight collaborative rerank path:

- `src/onerec/evaluate/collaborative_rerank.py`
- `src/onerec/evaluate/pipeline.py`

This matters because it gives a low-cost way to test whether collaborative information would help without rewriting the tokenizer first.

## Existing diagnostic evidence from local notes

From the current repo diagnostics and notes:

- collision is present but small:
  - Industrial: about `0.4341%`
  - Office: about `0.4337%`
- same-prefix confusion is much larger than raw collision
- many wrong predictions still stay inside the same coarse subtree
- the beam often already contains candidates from the same local subtree as the target

Representative local diagnostic signals:

| Metric | Industrial | Office |
|---|---:|---:|
| top1 hit rate | 0.07324 | 0.08570 |
| collision rate | 0.00434 | 0.00434 |
| beam contains same l1 | 0.41805 | 0.42766 |
| beam contains same l2 | 0.25347 | 0.23510 |
| top1 error same l1 | 0.21519 | 0.10249 |
| top1 error same l2 | 0.07712 | 0.01888 |

Interpretation:

- The model often enters the right coarse semantic region.
- The bigger problem is not pure full-SID collision.
- The bigger problem is local confusion inside the coarse subtree.

## Lightweight local pilot: collaborative rerank on current results

Using the existing evaluation code, I ran a train-only collaborative rerank over the current saved predictions.

Modes:

- `baseline`: keep current ranking
- `global`: rerank the whole prediction list using collaborative scores
- `same_l1`: rerank only candidates sharing the top-1 first-level prefix
- `same_l2`: rerank only candidates sharing the top-1 second-level prefix
- `ambiguity_l2`: rerank only under prefixes marked ambiguous by leaf-count threshold

### Industrial

| Mode | Hit@1 | Absolute delta |
|---|---:|---:|
| baseline | 0.073241 | +0.000000 |
| global | 0.083830 | +0.010589 |
| same_l1 | 0.080079 | +0.006838 |
| same_l2 | 0.081624 | +0.008383 |
| ambiguity_l2 | 0.079197 | +0.005956 |

Activation counts:

- `global`: 4533 / 4533
- `same_l1`: 3679 / 4533
- `same_l2`: 2605 / 4533
- `ambiguity_l2`: 1242 / 4533

### Office

| Mode | Hit@1 | Absolute delta |
|---|---:|---:|
| baseline | 0.085697 | +0.000000 |
| global | 0.088368 | +0.002671 |
| same_l1 | 0.087752 | +0.002055 |
| same_l2 | 0.087957 | +0.002260 |
| ambiguity_l2 | 0.084875 | -0.000822 |

Activation counts:

- `global`: 4866 / 4866
- `same_l1`: 4498 / 4866
- `same_l2`: 2417 / 4866
- `ambiguity_l2`: 364 / 4866

## Local takeaways

### Motivation 1: missing collaborative information

Strongly supported.

Even simple train-only collaborative reranking improves top-1 hit rate on both datasets, especially on Industrial. This is direct evidence that the current system leaves collaborative information unused.

### Motivation 2: unstable prefix semantics / poor predictability

Also strongly supported.

The diagnostic profile says the system often reaches the right coarse subtree but fails within the subtree. That is exactly the pattern expected when prefix semantics are useful but not globally stable enough to make local discrimination easy.

### Motivation 3: collision

Real, but not dominant in the current repo.

Collision is measurable, and recent papers show it matters. But the local evidence here suggests collision alone is too small to explain the main error pattern.

### One subtle but important conclusion

The weak performance of the current `ambiguity_l2` heuristic on Office does not mean local repair is wrong. It means the current ambiguity trigger is crude. The likely missing piece is calibration:

- when should repair activate
- on which subset of beam candidates
- based on which notion of risk

That observation becomes the seed for the top idea in this folder.
