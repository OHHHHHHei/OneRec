# MiniOneRec Baseline Learnability Check (2026-04-14)

## What We Checked

We ran the same 3-seed `R304` learnability probe on the original MiniOneRec Industrial semantic SID space:

- train: `./data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv`
- valid: `./data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv`
- index: `./data/Amazon/index/Industrial_and_Scientific.index.json`
- seeds: `42, 43, 44`

Output files:

- [R304_original_semantic_learnability_probe.csv](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-14_mgr_sid_learnability_probe_baseline_check/R304_original_semantic_learnability_probe.csv)
- [R304_original_semantic_learnability_probe.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-14_mgr_sid_learnability_probe_baseline_check/R304_original_semantic_learnability_probe.json)

## Baseline Result

| Variant | a acc | b\|a acc | c\|ab acc |
|---|---:|---:|---:|
| original semantic | **0.1892** | 0.1515 | 0.3299 |

## Comparison Against Current MGR-SID Variants

For comparison, the current multi-seed values already summarized in
[DEEP_REVIEW_MGR_SID_PROJECT_STATE_20260414.md](/home/leejt/OneRec/research-progress-log/DEEP_REVIEW_MGR_SID_PROJECT_STATE_20260414.md)
are:

| Variant | a acc | b\|a acc | c\|ab acc |
|---|---:|---:|---:|
| original semantic | **0.1892** | 0.1515 | 0.3299 |
| current `v2` | 0.0908 | 0.2424 | 0.4365 |
| `R202a` | 0.0978 | 0.2118 | 0.4159 |
| `R401b` | 0.0780 | **0.2484** | 0.4712 |
| `R401d` | 0.0799 | 0.2455 | **0.4829** |

## Immediate Takeaway

This baseline check answers an important question:

> **The strongest MiniOneRec downstream baseline is not globally "most learnable" under the R304 probe.**

Its learnability profile is instead:

- **very strong at level `a`**
- **clearly weaker at conditional deeper prediction** (`b|a`, `c|ab`)

So "best NDCG/HR" does **not** imply "best learnability on every SID level."

## Interpretation

The result suggests a more nuanced picture:

1. The original semantic SID space likely gives the downstream model a **much easier first routing step**.
2. But once conditioned on that first token, its deeper hierarchy is **less clean / less predictable** than the current graph-informed spaces.
3. Therefore, if the original baseline still wins at some downstream cutoffs, the reason is **not simply that its full hierarchy is easier to learn everywhere**.

That shifts the question from:

> "Is the strongest baseline just more learnable?"

to:

> "Which part of learnability matters most for downstream ranking: early routing (`a`) or deeper conditional structure (`b|a`, `c|ab`)?"

## Why This Matters

This check makes one thing much clearer:

- Our current graph-informed tokenizer line (`v2`, `R401b`, `R401d`) has likely improved **deeper conditional structure**
- But it may still be paying a price in **coarse first-step routing simplicity**

That is a much sharper hypothesis than the vague statement:

> "baseline is just more learnable"

The stronger version is:

> **baseline is more learnable at the first token, while graph-informed tokenizers are more learnable at deeper conditional levels.**

This is now a concrete empirical finding, not just an intuition.

