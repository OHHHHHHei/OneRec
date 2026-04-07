# Latest Paper Notes

## ReSID

Link: https://arxiv.org/abs/2602.02338

Core message:

- Existing semantic-centric tokenizers are weakly coupled to collaborative prediction and inefficient for reducing sequential uncertainty.

What it adds:

- recommendation-native representation learning
- field-aware masked auto-encoding
- globally aligned orthogonal quantization
- direct emphasis on predictability and prefix-conditional uncertainty

Relevance to the user's motivations:

- directly answers motivation 1
- directly answers motivation 2
- partially touches motivation 3 through better quantization quality

Overlap risk:

- Very high for any proposal centered on "let us redesign the tokenizer to be recommendation-native and globally aligned".

## PIT

Link: https://arxiv.org/abs/2602.08530

Core message:

- Static, decoupled tokenization misses collaborative signals, and end-to-end co-evolution is needed.

What it adds:

- dynamic personalized tokenizer
- collaborative signal alignment
- co-evolution of tokenizer and recommender

Relevance:

- directly strengthens motivation 1

Overlap risk:

- Extremely high for any proposal whose main novelty is "put collaborative signal into tokenizer training".

## TrieRec

Link: https://arxiv.org/abs/2602.21677

Core message:

- The trie induced by hierarchical tokenization should be modeled explicitly rather than flattened into a plain token stream.

What it adds:

- trie-aware absolute positional encoding
- topology-aware relative positional encoding
- direct structural bias for prefix-tree reasoning

Relevance:

- directly strengthens motivation 2

Overlap risk:

- High for any proposal framed mainly as "prefixes have structure and the model should understand the trie".

## APAO

Link: https://arxiv.org/abs/2603.02730

Core message:

- Prefixes fail at inference because beam search prunes correct items too early; training should optimize vulnerable prefixes explicitly.

What it adds:

- prefix-level loss
- adaptive worst-prefix optimization
- train / inference consistency story

Relevance:

- directly strengthens motivation 2

Overlap risk:

- High for proposals whose main novelty is "prefix-aware training".

## QuaSID

Link: https://arxiv.org/abs/2603.00632

Core message:

- SID collisions matter, but not all collisions are equally harmful.

What it adds:

- collision qualification
- severity-scaled repulsion
- conflict-aware masking
- collaborative signal in tokenization

Relevance:

- directly strengthens motivation 3
- also touches motivation 1 because it injects collaborative signal

Overlap risk:

- Very high for any collision-first tokenizer redesign.

## PRISM

Link: https://arxiv.org/abs/2601.16556

Core message:

- Existing semantic tokenization is unstable and lossy; better token purity plus integrated semantics can improve generation.

What it adds:

- purified semantic quantizer
- adaptive collaborative denoising
- hierarchical semantic anchoring
- semantic structure alignment

Relevance:

- motivation 1 and 2

Overlap risk:

- High for global semantic-plus-collaborative tokenizer improvements.

## UNGER

Link: https://arxiv.org/abs/2502.06269

Core message:

- Semantic and collaborative modalities should be integrated into one unified code rather than split or naively concatenated.

What it adds:

- unified code via semantic + collaborative integration
- cross-modality alignment
- intra-modality distillation

Relevance:

- directly supports motivation 1

Overlap risk:

- High for unified-code or multimodal-code tokenization stories.

## HiD-VAE

Link: https://arxiv.org/abs/2508.04618

Core message:

- Unsupervised tokenization yields flat, entangled IDs; hierarchy and disentanglement improve interpretability and reduce collision.

What it adds:

- hierarchical supervision
- disentangled semantic IDs
- uniqueness loss for collision / entanglement reduction

Relevance:

- directly supports motivation 3
- indirectly touches motivation 2 through hierarchy

Overlap risk:

- High for proposals framed as "make SIDs more hierarchical and less collided" without a sharper angle.

## Older local baseline context

The local paper folder also shows the older evolution:

- `LCRec`: language and recommendation alignment
- `VQ-Rec`: quantized semantic IDs for recommendation
- `TokenRec`: collaborative information discretized into tokens
- `ETEGRec`: tighter end-to-end tokenizer and recommender coupling
- `UTGRec`: universal tokenizer direction

These older papers matter because they show that the field has been moving from semantic-only tokenization toward more recommendation-native and collaborative-aware designs for some time. The 2025-2026 papers are not isolated; they are the mature wave of that trend.
