# Related Work and Module Map

This note merges the earlier “paper list by question” and “module mapping” notes into a smaller canonical entry.

Merged source notes are archived under:

- `archive/2026-04-12_doc_reorg_merged_sources/11_arxiv_related_work_by_question.md`
- `archive/2026-04-12_doc_reorg_merged_sources/12_modules_mapped_to_core_questions.md`

For the dedicated ambiguity-proxy scan, keep using:

- `17_ambiguity_proxy_literature_scan.md`

## Purpose

The goal here is not a broad recommender survey.
The goal is to answer one narrower design question:

> which prior works directly support the current graph-hierarchy tokenizer line, and what concrete module or principle do we borrow from each?

## Highest-Priority References

If we only keep a short core reading set, the most useful cluster is:

1. `PRISM`
2. `ReSID`
3. `GSPRec`
4. `FaGSP`
5. `Collaboration and Transition`
6. `DiscRec`

## Borrowed Ideas by Role

### 1. `PRISM`

What we borrow:

- collaborative signal should be denoised before it is trusted
- semantic hierarchy should remain the anchor
- tokenizer design needs anti-collapse protection

Where it lands in our method:

- semantic-structure retention
- purification mindset for graph carriers

### 2. `ReSID`

What we borrow:

- local uncertainty matters more than raw global collision
- SID quality should be interpreted through predictability and ambiguity, not only reconstruction

Where it lands in our method:

- the emphasis on `same_l2` / local ambiguity
- the motivation for ambiguity-aware supervision

### 3. `GSPRec` and `FaGSP`

What we borrow:

- useful collaborative graph signals live at different scales / frequencies
- `G_mid` should be built as a proper middle-resolution spectral view

Where it lands in our method:

- `fagsp_mid_base`
- the current `G_mid` story

### 4. `Collaboration and Transition`

What we borrow:

- collaboration and transition are distinct signals
- local transition structure still matters, but it should not dominate the whole tokenizer

Where it lands in our method:

- keeping `G_local` as a separate short-range carrier

### 5. `DiscRec`

What we borrow:

- semantic and collaborative signals should not be collapsed too early
- the right interface is controlled interaction, not blunt fusion

Where it lands in our method:

- graph supervision instead of early feature fusion
- semantic backbone + graph structural constraint

## Current Mapping to the Three Core Questions

### Q1. What graph should carry collaborative information?

Current best answer:

- `G_coarse` for broad consistency
- `G_mid` as the most important structure carrier
- `G_local` for transition-sensitive local signal

Main references:

- `GSPRec`
- `FaGSP`
- `Collaboration and Transition`

### Q2. How do we make the method hierarchy-aware?

Current best answer:

- hierarchy should be expressed through level-specific supervision, not generic graph mixing everywhere

Main references:

- `ReSID`
- `PRISM`
- graph-frequency / multi-scale papers

### Q3. How do we fuse graph-structured collaboration with MiniOneRec?

Current best answer:

- keep semantic SID as the backbone
- let graph act as structural supervision
- protect already-good semantic structure while adding collaborative help where ambiguity is high

Main references:

- `PRISM`
- `DiscRec`
- `PIT` as a cautionary reminder about instability

## How to Read the Current Method With These References in Mind

The simplest high-level synthesis is:

- `PRISM` gives the safety principle
- `ReSID` gives the ambiguity motivation
- `GSPRec / FaGSP` give the middle-scale graph design principle
- `DiscRec` explains why we do not want naive early fusion

That combination is the shortest literature-backed explanation of the current `v2` line.
