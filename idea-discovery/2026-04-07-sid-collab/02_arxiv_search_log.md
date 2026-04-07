# Latest arXiv Search Log

Search date: 2026-04-07

## Search goal

I searched for the latest arXiv work most relevant to the user's three motivations:

1. semantic-only SID and missing collaborative information
2. unstable prefix semantics and low SID predictability
3. collision in semantic IDs

## Search strategy

Priority was given to:

- arXiv papers from 2025-2026
- papers directly about generative recommendation or semantic IDs
- papers that intervene on tokenizer design, prefix modeling, or collision handling

I also cross-checked against local PDFs already present in `papers/`.

## Representative queries

- `site:arxiv.org Recommendation-native Semantic ID generative recommendation`
- `site:arxiv.org Purified Semantic IDs generative recommendation`
- `site:arxiv.org Personalized Item Tokenizer generative recommendation`
- `site:arxiv.org Trie-Aware Transformers generative recommendation`
- `site:arxiv.org Adaptive Prefix-Aware Optimization generative recommendation`
- `site:arxiv.org collision semantic ID generative recommendation`
- `site:arxiv.org unified code semantic collaborative generative recommendation`

## Selected latest papers

### 1. ReSID

- Title: `Rethinking Generative Recommender Tokenizer: Recsys-Native Encoding and Semantic Quantization Beyond LLMs`
- arXiv: `2602.02338`
- Date: submitted on 2026-02-02
- Link: https://arxiv.org/abs/2602.02338
- Why it matters:
  - directly attacks semantic-centric tokenizer design
  - explicitly optimizes recommendation-native representation learning and quantization
  - strong overlap with any broad "replace MiniOneRec tokenizer with a collaborative one" story

### 2. PIT

- Title: `PIT: A Dynamic Personalized Item Tokenizer for End-to-End Generative Recommendation`
- arXiv: `2602.08530`
- Date: submitted on 2026-02-09
- Link: https://arxiv.org/abs/2602.08530
- Why it matters:
  - dynamic tokenizer
  - collaborative signal alignment
  - end-to-end co-evolution of tokenizer and recommender
  - very close to "put collaboration into tokenization"

### 3. TrieRec

- Title: `Trie-Aware Transformers for Generative Recommendation`
- arXiv: `2602.21677`
- Date: submitted on 2026-02-25
- Link: https://arxiv.org/abs/2602.21677
- Why it matters:
  - focuses on the trie / prefix topology itself
  - directly relevant to prefix semantics and generation predictability

### 4. QuaSID

- Title: `Stop Treating Collisions Equally: Qualification-Aware Semantic ID Learning for Recommendation at Industrial Scale`
- arXiv: `2603.00632`
- Date: submitted on 2026-02-28
- Link: https://arxiv.org/abs/2603.00632
- Why it matters:
  - targets collision explicitly
  - adds collision-aware supervision and collaborative signal into SID learning
  - strong overlap with any collision-first tokenizer redesign

### 5. APAO

- Title: `APAO: Adaptive Prefix-Aware Optimization for Generative Recommendation`
- arXiv: `2603.02730`
- Date: submitted on 2026-03-03
- Link: https://arxiv.org/abs/2603.02730
- Why it matters:
  - directly attacks prefix-level training / inference mismatch
  - important prior art for any claim centered on "prefixes are not predictable enough"

### 6. PRISM

- Title: `PRISM: Purified Representation and Integrated Semantic Modeling for Generative Sequential Recommendation`
- arXiv: `2601.16556`
- Date: submitted on 2026-01-23
- Link: https://arxiv.org/abs/2601.16556
- Why it matters:
  - attacks unstable semantic tokenization
  - uses adaptive collaborative denoising and hierarchical semantic anchoring
  - another strong prior for global tokenizer improvement

### 7. UNGER

- Title: `UNGER: Generative Recommendation with A Unified Code via Semantic and Collaborative Integration`
- arXiv: `2502.06269`
- First version date: 2025-02-10
- Latest arXiv version seen: 2025-10-31
- Link: https://arxiv.org/abs/2502.06269
- Why it matters:
  - unified semantic + collaborative code
  - important 2025 bridge between semantic-only and collaborative-aware tokenization

### 8. HiD-VAE

- Title: `HiD-VAE: Interpretable Generative Recommendation via Hierarchical and Disentangled Semantic IDs`
- arXiv: `2508.04618`
- First version date: 2025-08-06
- Link: https://arxiv.org/abs/2508.04618
- Why it matters:
  - explicitly targets hierarchy and collision / entanglement
  - important prior art for any "hierarchical, disentangled, collision-reduced SID" proposal

## Older but still relevant local papers

These are not the newest arXiv papers, but they are important context and were already present in the local `papers/` folder:

- `ETEGRec`
- `UTGRec`
- `TokenRec`
- `LCRec`
- `VQ-Rec`

These papers help place the 2025-2026 work in context:

- earlier work established semantic-ID generative recommendation
- newer work increasingly adds collaborative alignment, prefix awareness, and collision control

## Search-level conclusion

The latest arXiv landscape says:

- adding collaboration into tokenization is no longer an open, empty space
- making prefixes more stable and better aligned with decoding has also become crowded
- collision-aware SID learning now has dedicated recent work

So the user's motivations are valid, but a naive response to them would likely be too close to recent work.
