# SID Quality Beyond Structure（超越结构的 SID 质量）

Status（状态）: `discussion-only（仅讨论）`

Discussion date: `2026-04-14`

This note is a reasoning document（推理文档）, not a current-state summary（当前状态摘要）.

Use it to understand why the project stopped treating structure-only metrics（仅结构指标） as the final objective（最终目标）.

If you need the live project state first, read:

1. [CURRENT_STATE.md](/home/leejt/OneRec/research-progress-log/CURRENT_STATE.md)
2. [experiment_results.csv](/home/leejt/OneRec/experiment_results.csv)

## Why this note exists

Stage-2 gave us an important new diagnosis:

> tokenizer-side structural cleanup does not automatically become downstream
> ranking gain.

This means our current notion of "better SID" is incomplete.

Until now, most of our SID diagnosis has focused on structure:

- collision
- local ambiguity
- `same_l2`
- `l2` fanout
- conditional entropy

These remain useful, but they are no longer sufficient.

## Core update

We should stop treating SID as only a discrete structural object.

A SID is also a **codebook space** used by the downstream LLM.

The project is therefore not trying to keep a new SID space close to the old
baseline by default.

The real objective is:

> to find a better SID codebook space for downstream recommendation learning.

If that better space happens to stay near the baseline, then conservative
constraints may be useful.
If it requires a larger reorganization, that is also acceptable.

So a high-quality SID must satisfy three layers at the same time:

1. **Structural quality**
   - does it reduce local ambiguity?
   - does it reduce harmful collision / deep crowded buckets?
   - does it create cleaner prefix-conditioned hierarchy?

2. **Semantic consistency / interpretability**
   - does a code token carry stable meaning?
   - or does the same code become highly polysemous under different prefixes?
   - are reused tokens expressing meaningful shared abstraction, or just mixing incompatible regions?

3. **Downstream learnability / usefulness**
   - can the downstream LLM learn this SID space effectively under full training?
   - does the tokenizer improvement survive into `SFT` and `RL`?
   - does the final `evaluate` result actually improve?

If a SID is structurally cleaner but semantically unstable or hard for the
downstream model to consume, it is not a truly better SID.

## What stage-2 changed in our understanding

`R202a` is the clearest example.

Tokenizer-side:

- lower mean target `l2` leaf count
- lower deep crowded `l2>=4`
- lower conditional entropy

But downstream `R208`:

- does not beat `v2_on_p05`
- helps hard crowded local cases
- hurts many already-stable easy cases

This strongly suggests that "cleaner structure" and "better downstream token
space" are not identical.

The remaining problem is not only:

> how to produce cleaner local SID structure

It is also:

> how to produce a SID space that remains learnable and task-compatible for the
> downstream LLM.

## New working hypothesis

The current stage-2 evidence supports a three-part working hypothesis.

### H1. Global SID rearrangement is a possible diagnostic, not a universal failure mode

When a tokenizer variant changes too much of the SID space, the downstream LLM
may face a different learning problem.

This can hurt downstream performance, but it does **not** mean that large SID
change is wrong by itself.

If a more reorganized SID space produces better final `evaluate`, then the
change was worthwhile.

### H2. Prefix-conditioned code polysemy

The same code token may express very different semantics under different
prefixes.

Examples:

- the same `<b_j>` may mean one type of refinement under `<a_3>`
- but a very different refinement under `<a_41>`

This is not automatically wrong for hierarchical quantization.
But for an LLM that learns a single shared embedding for `<b_j>`, excessive
prefix-conditioned polysemy may increase learning difficulty.

### H3. SID learnability is a separate axis from SID structure

Even when a SID space is structurally superior, the downstream model may not
use it better unless:

- useful prefixes remain stable enough,
- local alternatives remain recoverable enough,
- and the downstream objectives are compatible with the new token geometry.

## Practical implication

From now on, "better SID" should not mean only:

- lower collision
- lower entropy
- lower local ambiguity

It should mean:

> better structure, plus interpretable code semantics, plus stronger final
> downstream `evaluate`.

Among these, the hardest criterion is still the downstream result.
The other diagnostics are supporting evidence, not final judges.

## What we now need to measure

We need a new diagnostic layer on top of the current structural metrics.

### A. Prefix stability / rearrangement

We need to measure how much a new tokenizer variant rearranges the SID space:

- changed `l1` rate
- changed `l2` rate
- prefix-neighbor overlap between two SID spaces

This quantifies one possible source of downstream difficulty.
It should be treated as a diagnostic reference, not a hard rejection rule.

### B. Code polysemy / semantic consistency

We need to measure whether reused code tokens keep coherent semantics:

- semantic spread of each `a`, `b`, `c` token
- especially for `b` and `c`, semantic drift under different prefixes
- whether shared codes represent true abstraction or overloaded token reuse

### C. Downstream learnability

We need to test whether a SID space is easy for the downstream model to consume:

- next-token / next-level predictability
- prefix-to-leaf predictability
- transfer from tokenizer-side improvement to downstream hit-rate improvement

But the final decision must still come from full downstream training and final
`evaluate`, not from probe accuracy alone.

## Current decision impact

This note changes how we interpret stage-2:

- `R202a` is still a meaningful tokenizer-side structural gain
- but it is not yet a downstream-compatible SID improvement

So the next phase should not only ask:

> can we make SID structure cleaner?

It should ask:

> can we improve SID while preserving semantic coherence and downstream
> learnability?

## Related active follow-up

The current active execution plan is:

- `refine-logs/EXPERIMENT_PLAN_STAGE3_PREFIX_RETAINED_HIERARCHY.md`

The older stage-2 retention and interface-diagnostics execution documents were
archived here:

- `refine-logs/archive/2026-04-14_stage3_scope_cleanup/README.md`
