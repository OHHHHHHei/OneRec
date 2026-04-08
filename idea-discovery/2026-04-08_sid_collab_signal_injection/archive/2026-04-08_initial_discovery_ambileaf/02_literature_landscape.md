# Literature Landscape

## 1. Empirical anchor from this repo

This round starts from the repo's own evidence instead of from a paper-first narrative.

### What the current repo already tells us

- Full SID collision is real but small.
  - `Industrial_and_Scientific.index.json`: collision rate `0.0043407488`
- Naive front-end collaborative fusion is not usable in its current form.
  - `Industrial_and_Scientific.v05_e1.index.json`: collision rate `0.7403689636`
  - `Industrial_and_Scientific.v05_c1_shuffled.index.json`: collision rate `0.6283233858`
  - Since even shuffled CF also collapses, the problem is not only "bad collaborative structure"; it is also "bad injection form".
- Collaborative signal becomes more valuable in local same-prefix confusion.
  - Industrial `same_l2` target-better rate: `0.2284`
  - Office `same_l2` target-better rate: `0.4048`
  - Same-prefix errors also have extremely high text similarity:
    - Industrial `same_l2` cosine: `0.9821`
    - Office `same_l2` cosine: `0.9839`
- ACLR-lite suggests collaboration helps globally, but its marginal value is concentrated locally.
  - HR@1 gain over baseline:
    - `global`: `+0.010589`
    - `same_l2`: `+0.008383`
    - `ambiguity_l2`: `+0.005956`
  - Gain per 1000 activated samples:
    - `global`: `0.002336`
    - `same_l2`: `0.003218`
    - `ambiguity_l2`: `0.004796`

The combined reading is:

> collaborative information is useful, but raw global injection can destabilize tokenization; the strongest remaining error pattern is local leaf ambiguity under semantically correct prefixes.

## 2. Recent literature map

### A. Global collaborative or end-to-end tokenizer redesign

- **ETEGRec** (arXiv:2409.05546)  
  https://arxiv.org/abs/2409.05546  
  Unifies tokenization and generative recommendation with end-to-end alignment.

- **PIT** (arXiv:2602.08530)  
  https://arxiv.org/abs/2602.08530  
  Pushes dynamic personalized tokenization and co-evolution of tokenizer and recommender; explicitly warns that collaborative volatility makes tokenization unstable.

- **LCRec**  
  Integrates collaborative semantics into LLM-based recommendation through aligned item indexing and tuning tasks.

- **Align3GR**  
  Uses dual tokenization and multi-level alignment, including token-level collaborative/semantic fusion.

- **UniGRec** (arXiv:2601.17438)  
  https://arxiv.org/abs/2601.17438  
  Soft identifiers, end-to-end optimization, codeword uniformity regularization, and collaborative distillation.

### B. Purification, stability, and collapse control

- **PRISM** (arXiv:2601.16556)  
  https://arxiv.org/abs/2601.16556  
  The most directly relevant warning for this repo. It argues that interaction noise can cause impure tokenization and codebook collapse, and answers with adaptive collaborative denoising plus hierarchical semantic anchoring.

- **ReSID** (arXiv:2602.02338)  
  https://arxiv.org/abs/2602.02338  
  Argues that semantic-centric tokenization is misaligned with recommendation objectives and emphasizes prefix-conditional uncertainty.

- **DIGER** (arXiv:2601.19711)  
  https://arxiv.org/abs/2601.19711  
  Differentiable semantic IDs; highlights codebook collapse risk and proposes exploration-heavy early training.

- **QuaSID** (arXiv:2603.00632)  
  https://arxiv.org/abs/2603.00632  
  Focuses on collision heterogeneity and collision-qualified repulsion, with collaborative signals added into tokenization.

### C. Personalized or dynamic tokenization

- **Pctx** (arXiv:2510.21276)  
  https://arxiv.org/abs/2510.21276  
  Personalized context-aware tokenization. Very relevant as prior art for any idea that lets the same item take different identifiers depending on context.

- **PIT** again belongs here  
  It shows the field is already moving from static item tokenization toward dynamic or co-evolving tokenization.

### D. Hierarchy, structure, and disentanglement

- **HiD-VAE** (arXiv:2508.04618)  
  https://arxiv.org/abs/2508.04618  
  Emphasizes hierarchical, disentangled semantic IDs and interpretable paths.

## 3. What the literature does not cleanly answer yet

The recent literature is increasingly strong on:

- global collaborative tokenization
- end-to-end tokenizer-recommender coupling
- collapse prevention
- personalized or dynamic tokenization

But it still leaves an opening that is highly relevant to this repo:

### Open gap

Most recent methods ask:

- how to inject collaboration globally
- how to couple tokenizer and recommender end-to-end
- how to prevent collapse in a general sense

Much fewer works ask:

- **where in the SID hierarchy collaborative signal should enter**
- whether collaboration should mainly reshape **prefix routing** or mainly sharpen **leaf discrimination**
- how to exploit collaborative signal when the repo evidence says:
  - full collision is low
  - same-prefix leaf ambiguity is the real bottleneck
  - naive global fusion destabilizes the quantizer

## 4. Discovery takeaway

The cleanest research opportunity is not "add more collaboration everywhere".

It is:

> decide the correct stage and granularity for collaborative signal injection, under the constraint that global raw fusion is noisy and local leaf ambiguity is the dominant remaining failure mode.

This is why this discovery round focuses on three families on equal footing:

1. better **front-end global fusion**, but purified and controlled  
2. **front-end local leaf retokenization**, which changes structure only where ambiguity concentrates  
3. **hybrid coarse-to-fine** methods that combine weak global priors with strong local repair

