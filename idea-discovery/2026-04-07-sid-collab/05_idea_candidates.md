# Idea Candidates

## Ranking summary

### 1. ACLR-Plus: Ambiguity-Calibrated Local Collaborative Repair

Status: recommended

Core idea:

- Keep the current MiniOneRec SID mapping.
- Estimate when the current prefix path is risky or ambiguous.
- Only then, rerank or refine candidates inside the local subtree using train-only collaborative evidence.

Why it fits this repo:

- local diagnostics already show same-prefix confusion
- evaluation code already supports collaborative reranking
- a simple pilot already shows collaborative signal helps
- the current ambiguity trigger is weak, which creates room for a stronger calibrated design

Why it is differentiated:

- not a full tokenizer rewrite
- not a generic prefix-aware training method
- not a collision-first SID redesign
- specifically targets local semantic ambiguity after the coarse prefix is already mostly right

Proposed mechanism:

1. compute a risk score for the current prefix or beam state
2. trigger repair only when risk is high
3. rerank only within a restricted local candidate set
4. use train-only collaborative evidence so the method stays lightweight and clean

Best paper angle:

- semantics gets you into the right subtree
- collaboration resolves the final ambiguity when semantics alone is insufficient

Main risk:

- If the method remains a hand-crafted eval-time heuristic, reviewers may call it a decoding hack.

### 2. Selective Hot-Subtree Retokenization

Status: backup

Core idea:

- keep most existing SIDs
- only re-tokenize the small subset of hot ambiguous subtrees with severe same-prefix confusion or repeated collisions

Why it is interesting:

- more local than a full tokenizer rewrite
- directly targets the bad regions rather than the whole catalog

Why it is weaker than idea 1:

- more engineering complexity
- still overlaps with global tokenizer papers
- harder to keep the story simple

### 3. Collision-Qualified Alias Tokens

Status: backup

Core idea:

- keep the current prefix path
- add a sparse auxiliary alias or suffix only for harmful collided or high-risk items

Why it is interesting:

- targets motivation 3 without global re-indexing
- could be cheap to implement

Why it is weaker:

- collision is not the dominant repo bottleneck right now
- recent collision-aware papers make the novelty narrower

### 4. Full Collaborative Tokenizer Rebuild

Status: not recommended

Core idea:

- rebuild the MiniOneRec tokenizer with collaborative embeddings, structure fields, or joint semantic-collaborative learning

Why it is not recommended:

- strongest overlap with ReSID, PIT, PRISM, UNGER, QuaSID
- highest implementation cost
- least differentiated given the 2025-2026 literature

## Why idea 1 wins

It is the only idea that simultaneously does all of the following:

- fits the repo's strongest evidence
- avoids the most crowded literature lane
- builds on code that already exists
- can start from lightweight pilots
- still has room to become a real method rather than just a heuristic
