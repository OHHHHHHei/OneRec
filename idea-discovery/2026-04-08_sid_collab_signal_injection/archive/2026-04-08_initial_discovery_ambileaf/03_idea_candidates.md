# Idea Candidates

## Ranking Summary

| Rank | Idea | Main claim | Fit to repo evidence | Novelty outlook | Risk |
|---|---|---|---|---|---|
| 1 | **AmbiLeaf** | Freeze semantic prefix, repair only ambiguous leaves with purified collaborative tokenization | Very high | Strong | Medium |
| 2 | **PurifyThenQuantize** | Global collaborative fusion can work if denoised, level-wise, and curriculum-controlled | High | Medium | High |
| 3 | **Coarse2Fine Dual Signal** | Use a weak purified global prior plus a strong local leaf repair module | High | Medium | Medium |
| 4 | **Leaf Alias Tokens** | Keep one canonical SID and add context-conditioned optional leaf aliases | Medium | Medium | High |
| 5 | **Prefix Stability Distillation** | Train tokenization to reduce leaf uncertainty through prefix-stability supervision | Medium | Low-Medium | Medium |

## 1. AmbiLeaf: Prefix-Preserved Collaborative Leaf Retokenization

### Core hypothesis

The repo suggests the model usually routes to the correct semantic neighborhood, but fails to choose the right leaf. If so, the most useful place for collaborative information is **not the whole SID**, but **the final leaf decision inside ambiguous prefixes**.

### Main design

1. Start from the current semantic SID tree.
2. Detect ambiguous prefixes using train-only signals:
   - within-prefix item count
   - within-prefix semantic similarity
   - within-prefix collaborative disagreement
   - optional validation-time same-prefix miss statistics
3. Freeze the prefix assignment for all items.
4. For only the ambiguous prefixes, relearn the last token using:
   - semantic residual features
   - purified collaborative residual features
5. Keep all non-ambiguous prefixes unchanged.

### Why it fits this repo especially well

- It directly targets `prefix correct, leaf wrong`.
- It treats `v05` collapse as a warning against **global raw fusion**, not as proof that front-end collaboration is impossible.
- It is front-end enough to improve structure, but local enough to avoid rewriting the whole tree.

### Why it is not the same as inference-time selective reranking

- reranking changes scores after tokenization
- AmbiLeaf changes the **identifier structure itself**
- the selectivity here is **hierarchy-local structural repair**, not only inference-time triggering

### Why it looks novel

Recent papers already cover:

- global collaborative tokenizer redesign
- dynamic or end-to-end tokenization
- collapse prevention

But I did not find a recent paper centered on:

> preserving semantic prefixes and retokenizing only ambiguous leaf subtrees with collaborative supervision

That makes this a promising "new but still grounded" angle.

### Main risk

- If ambiguous prefixes cover too few items, gains may be too small.
- If too many prefixes are selected, the method drifts back toward a global redesign.
- The ambiguity detector must be defined carefully with train-only statistics.

## 2. PurifyThenQuantize: Denoised Global Collaborative Tokenization with Level-Wise Curriculum

### Core hypothesis

The failure of `v05` may come from the **form of fusion**, not from the idea of front-end collaboration itself. If collaborative signals are purified first and injected gradually, global front-end fusion may still be the highest-upside direction.

### Main design

1. Build two channels:
   - semantic channel
   - collaborative channel
2. Purify the collaborative channel:
   - low-rank denoising
   - agreement filtering
   - confidence weighting
3. Use level-wise fusion:
   - prefix levels: mostly semantic
   - lower levels / leaf: higher collaborative weight
4. Add collapse-resistant regularization:
   - code usage balance
   - optional exploration or uniformity regularization

### Why it matters

- It takes your pushback seriously: front-end fusion should remain a first-class candidate.
- It aligns with PRISM, PIT, ReSID, DIGER, and UniGRec, all of which suggest that collapse and volatility are the real obstacles, not the idea of collaboration itself.

### Main risk

- This space is already crowded by very recent papers.
- It may be scientifically valid but hard to differentiate unless the hierarchy-aware angle is very sharp.
- Implementation complexity is higher than AmbiLeaf.

## 3. Coarse2Fine Dual Signal

### Core hypothesis

The best practical solution may use collaboration twice:

- weakly and cleanly in the tokenizer, to stabilize structure
- strongly and locally at leaf time, to resolve ambiguity

### Main design

1. Use a mild purified collaborative prior in tokenization.
2. Keep semantic prefixes dominant.
3. Add an ambiguity-triggered leaf corrector:
   - local reranking
   - local classifier
   - or learned leaf expert

### Why it is attractive

- It respects both pieces of evidence:
  - front-end structure matters
  - local leaf repair already shows gains

### Main risk

- Easy for reviewers to see it as a system combination rather than one clean idea.
- Needs strong ablations to show each stage is necessary.

## 4. Leaf Alias Tokens

### Core hypothesis

Some items may need multiple leaf identities under the same semantic prefix, depending on user intent or sequence context.

### Main design

- Keep one canonical SID.
- For ambiguous subtrees, allow one or a few optional leaf aliases.
- Choose alias by user context or sequence state.

### Why it is interesting

- It directly attacks the assumption that one static leaf is enough for every context.

### Main risk

- Strong overlap risk with personalized/dynamic tokenizer papers such as Pctx and PIT.
- Harder to keep decoding simple.

## 5. Prefix Stability Distillation

### Core hypothesis

The current tokenizer is not only missing collaboration; it is also missing a training signal that explicitly says: "good prefixes should make the leaf easy to predict."

### Main design

- Add a prefix-stability objective:
  - low leaf entropy under good prefixes
  - higher separation between target leaf and confusable sibling leaves
- Optionally distill from collaborative neighborhoods or from a leaf expert.

### Main risk

- ReSID already moves in this direction through prefix-conditional predictability.
- This may end up sounding like a loss redesign, not a conceptually fresh method.

## Eliminated or de-prioritized ideas

### A. Collision-first tokenizer redesign

Not recommended as the main story because this repo's dominant error pattern is not raw full-SID collision.

### B. Pure global collaborative rerank as the main paper thesis

Useful as a strong baseline or upper bound, but weak as the whole scientific story. It does not answer where the structural bottleneck really is.

### C. Naive `text + cf` front-end fusion

Already falsified by current repo evidence.

