# arXiv Related Work for MGR-SID

Date: 2026-04-09

## Search scope

This note focuses on arXiv papers most relevant to the current `MGR-SID` direction:

- graph neural networks or graph-structured collaborative modeling
- generative recommendation
- semantic ID or tokenizer learning
- hierarchy-aware or coarse-to-fine tokenization
- collaborative information injection into SID learning

The goal is not to list every graph-based recommender. The goal is to identify papers that are close enough to shape the positioning of:

- `Multiplex Graph-Regularized Hierarchical Semantic IDs`

## A. Directly relevant papers

### 1. ReSID

- arXiv: `2602.02338`
- Title: `Rethinking Generative Recommender Tokenizer: Recsys-Native Encoding and Semantic Quantization Beyond LLMs`
- Date: 2026-02-02
- Link: https://arxiv.org/abs/2602.02338

Why it matters:

- one of the strongest recent tokenizer papers
- moves from semantic-centric SID to recommendation-native representation learning
- explicitly optimizes quantization for sequential predictability

Relation to MGR-SID:

- supports the premise that semantic-only tokenization is insufficient
- overlap risk is high if our method is framed only as “better collaborative tokenizer”
- does not center graph-structured, level-specific supervision

### 2. PRISM

- arXiv: `2601.16556`
- Title: `PRISM: Purified Representation and Integrated Semantic Modeling for Generative Sequential Recommendation`
- Date: 2026-01-23
- Link: https://arxiv.org/abs/2601.16556

Why it matters:

- directly attacks unstable tokenization
- uses adaptive collaborative denoising and hierarchical semantic anchoring

Relation to MGR-SID:

- very relevant on the denoising side
- shows that “purified collaborative injection” is already occupied
- but still does not formulate collaboration as a multiplex graph bank allocated differently across SID levels

### 3. PIT

- arXiv: `2602.08530`
- Title: `PIT: A Dynamic Personalized Item Tokenizer for End-to-End Generative Recommendation`
- Date: 2026-02-09
- Link: https://arxiv.org/abs/2602.08530

Why it matters:

- dynamic collaborative tokenizer
- end-to-end co-evolution between tokenizer and recommender

Relation to MGR-SID:

- strong prior art against generic “inject collaboration into tokenizer” claims
- less close if we stay focused on graph structure preservation rather than dynamic co-evolution

### 4. TrieRec

- arXiv: `2602.21677`
- Title: `Trie-Aware Transformers for Generative Recommendation`
- Date: 2026-02-25
- Link: https://arxiv.org/abs/2602.21677

Why it matters:

- directly models the prefix-tree topology induced by hierarchical tokenization
- strong evidence that hierarchy should be treated structurally

Relation to MGR-SID:

- very relevant to the “hierarchy-aware” side
- but it improves the autoregressive model, not the tokenizer objective itself

### 5. CoFiRec

- arXiv: `2511.22707`
- Title: `CoFiRec: Coarse-to-Fine Tokenization for Generative Recommendation`
- Date: 2025-11-27
- Link: https://arxiv.org/abs/2511.22707

Why it matters:

- explicitly preserves coarse-to-fine semantic levels
- argues that flattening all item attributes into one latent space is suboptimal

Relation to MGR-SID:

- very relevant conceptually
- strongest support for treating SID levels differently
- still not graph-native

### 6. HiD-VAE

- arXiv: `2508.04618`
- Title: `HiD-VAE: Interpretable Generative Recommendation via Hierarchical and Disentangled Semantic IDs`
- Date: 2025-08-06
- Link: https://arxiv.org/abs/2508.04618

Why it matters:

- hierarchical supervision for SID learning
- uniqueness loss for reducing collision and entanglement

Relation to MGR-SID:

- relevant for hierarchy and collision handling
- but not a graph-structured collaborative supervision method

### 7. DiscRec

- arXiv: `2506.15576`
- Title: `DiscRec: Disentangled Semantic-Collaborative Modeling for Generative Recommendation`
- Date: 2025-06-18
- Link: https://arxiv.org/abs/2506.15576

Why it matters:

- disentangles semantic and collaborative signals
- localized collaborative attention within item tokens

Relation to MGR-SID:

- relevant as a collaborative signal disentanglement baseline
- still closer to embedding-level or token-level fusion than graph-regularized quantization

### 8. MoToRec

- arXiv: `2602.11062`
- Title: `MoToRec: Sparse-Regularized Multimodal Tokenization for Cold-Start Recommendation`
- Date: 2026-02-11
- Link: https://arxiv.org/abs/2602.11062

Why it matters:

- one of the clearest graph-related tokenization papers retrieved
- includes a hierarchical multi-source graph encoder
- uses discrete tokenization and graph signal fusion

Relation to MGR-SID:

- relevant to the “graph enters tokenizer” direction
- but closer to graph-encoder fusion for cold-start recommendation than to level-wise graph-regularized hierarchical SID learning

### 9. Hi-SAM

- arXiv: `2602.11799`
- Title: `Hi-SAM: A Hierarchical Structure-Aware Multi-modal Framework for Large-Scale Recommendation`
- Date: 2026-02-12
- Link: https://arxiv.org/abs/2602.11799

Why it matters:

- strong hierarchical tokenizer story
- treats semantic IDs as structured rather than flat

Relation to MGR-SID:

- relevant for hierarchy-aware design
- less relevant on graph-structured collaborative supervision

## B. Graph-related but more peripheral papers

### 10. GLTA

- arXiv: `2502.18757`
- Title: `Training Large Recommendation Models via Graph-Language Token Alignment`
- Date: 2025-02-26
- Link: https://arxiv.org/abs/2502.18757

Why it matters:

- aligns interaction-graph nodes with language tokens
- useful evidence that graph-to-token alignment is a live direction

Relation to MGR-SID:

- supportive background
- not a hierarchical SID tokenization paper

### 11. KGTB

- arXiv: `2509.12350`
- Title: `Knowledge Graph Tokenization for Behavior-Aware Generative Next POI Recommendation`
- Date: 2025-09-15
- Link: https://arxiv.org/abs/2509.12350

Why it matters:

- graph structure directly supervises tokenization
- one of the closest graph-native tokenization ideas retrieved, although in POI recommendation

Relation to MGR-SID:

- important for “graph structure can supervise tokenization” positioning
- domain is next-POI rather than standard item recommendation
- uses knowledge-graph tokenization instead of multiplex collaborative graph regularization

### 12. APAO

- arXiv: `2603.02730`
- Title: `APAO: Adaptive Prefix-Aware Optimization for Generative Recommendation`
- Date: 2026-03-03
- Link: https://arxiv.org/abs/2603.02730

Why it matters:

- focuses on vulnerable prefixes and training-inference mismatch

Relation to MGR-SID:

- not graph-based
- still relevant because it reinforces the importance of prefix-level structure in generative recommendation

### 13. DACT

- arXiv: `2603.29705`
- Title: `Drift-Aware Continual Tokenization for Generative Recommendation`
- Date: 2026-03-31
- Link: https://arxiv.org/abs/2603.29705

Why it matters:

- collaborative tokenization under drift and continual updates

Relation to MGR-SID:

- useful as a modern collaborative-tokenizer reference
- less central to the graph-hierarchy question

## C. What the current arXiv landscape says

### Clear consensus already formed

The literature already agrees on several points:

1. semantic-only SID/tokenization is insufficient
2. hierarchical tokenization structure matters
3. collaborative information should enter tokenization somehow
4. naive flat fusion is no longer enough for a strong paper

### What is already crowded

Crowded lanes:

- global semantic-collaborative tokenizer redesign
- dynamic collaborative tokenizer co-evolution
- hierarchy-aware but mostly non-graph tokenization
- collision-aware SID learning
- prefix-aware generation or training

### What still looks relatively open

What I did not find a direct arXiv match for is:

- a tokenizer that builds a denoised multiplex collaborative graph bank
- learns level-specific graph allocation across SID levels
- uses graph structure as a regularizer on hierarchical quantization itself

This is an inference from the current arXiv search results, not a proof of absence. But it does suggest that `MGR-SID` still has a plausible opening if it stays narrow and clearly distinguishes itself from:

- graph feature fusion
- one-shot collaborative denoising
- prefix-aware decoding improvements

## D. Positioning advice for MGR-SID

The cleanest positioning line is:

- `ReSID / PRISM / PIT / DiscRec` show why collaboration must matter
- `TrieRec / CoFiRec / HiD-VAE / Hi-SAM` show why hierarchy must matter
- `MoToRec / GLTA / KGTB` show that graph structure can be useful at token or language alignment time
- `MGR-SID` combines these insights by asking a sharper question:

`what graph structure should each SID level preserve, and how should that graph supervision be denoised and allocated across levels?`

## E. Baselines that now look mandatory

Given the current arXiv landscape, a serious `MGR-SID` paper should compare against:

1. semantic-only tokenizer
2. graph feature fusion
3. uniform graph regularization
4. hierarchy-aware but non-graph tokenizer baseline when available
5. swapped level-to-graph allocation control

Without these, reviewers can easily argue the gains come from generic graph priors rather than hierarchy-aware graph supervision.
