# 2026-04-12 MGR-SID V2 RL Result on Industrial

## Run

- valid RL run:
  `rl_industrial_mgr_tokenizer_v2_title_on_desc_p05_20260412_191449`
- source SFT checkpoint:
  `/data/leejt/OneRec/output_weights/experiments/mgr_sid_v2_recipe_isolation_20260411/title_on_desc_p05/sft/final_checkpoint`
- RL checkpoint:
  `/data/leejt/OneRec/output_weights/experiments/mgr_sid_v2_rl_on_p05_industrial_20260412/rl/final_checkpoint`
- final evaluate json:
  `/home/leejt/OneRec/results/experiments/mgr_sid_v2_rl_on_p05_industrial_20260412/final_result_rl_mgr_tokenizer_v2_title_on_desc_p05_Industrial_and_Scientific.json`
- RL log:
  `/home/leejt/OneRec/logs/experiment_mgr_sid_v2_rl_on_p05_industrial_20260412.log`
- evaluate log:
  `/home/leejt/OneRec/logs/experiment_mgr_sid_v2_eval_on_p05_rl_industrial_20260412.log`

Note:
- the first launch with `train_batch_size=8` failed because the current GRPO implementation requires the local batch shape to align with `num_generations=16`
- the valid aligned rerun used:
  - `train_batch_size=16`
  - `gradient_accumulation_steps=4`
  - `world_size=4`
  - `effective_global_batch=256`

## Raw Metrics

| Run | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | HR@1 | HR@3 | HR@5 | HR@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v2_on_p05 SFT` | 0.07059343 | 0.08451223 | 0.09253300 | 0.10270767 | 0.07059343 | 0.09508052 | 0.11471432 | 0.14626075 |
| `v2_on_p05 RL` | 0.07434370 | 0.09053678 | 0.09629833 | 0.10431921 | 0.07434370 | 0.10280168 | 0.11692036 | 0.14184867 |
| strongest original MiniOneRec SFT | 0.06706375 | 0.08500848 | 0.09315326 | 0.10372025 | 0.06706375 | 0.09838959 | 0.11824399 | 0.15089345 |
| strongest original MiniOneRec RL | 0.07324068 | 0.08903190 | 0.09704467 | 0.10726345 | 0.07324068 | 0.10037503 | 0.11978822 | 0.15133466 |

## Delta

### `v2_on_p05 RL - v2_on_p05 SFT`

- `NDCG@1`: `+0.00375027`
- `NDCG@3`: `+0.00602455`
- `NDCG@5`: `+0.00376533`
- `NDCG@10`: `+0.00161154`
- `HR@1`: `+0.00375027`
- `HR@3`: `+0.00772116`
- `HR@5`: `+0.00220604`
- `HR@10`: `-0.00441208`

### `v2_on_p05 RL - strongest original MiniOneRec SFT`

- `NDCG@1`: `+0.00727995`
- `NDCG@3`: `+0.00552830`
- `NDCG@5`: `+0.00314507`
- `NDCG@10`: `+0.00059896`
- `HR@1`: `+0.00727995`
- `HR@3`: `+0.00441209`
- `HR@5`: `-0.00132363`
- `HR@10`: `-0.00904478`

### `v2_on_p05 RL - strongest original MiniOneRec RL`

- `NDCG@1`: `+0.00110302`
- `NDCG@3`: `+0.00150488`
- `NDCG@5`: `-0.00074634`
- `NDCG@10`: `-0.00294424`
- `HR@1`: `+0.00110302`
- `HR@3`: `+0.00242665`
- `HR@5`: `-0.00286786`
- `HR@10`: `-0.00948599`

## Reading

1. RL is not a no-op for `v2_on_p05`.
It clearly improves head ranking quality over the source SFT checkpoint:
`NDCG@1/3/5/10` all increase, and `HR@1/3/5` also increase.

2. The old pattern still remains after RL.
The new RL model is sharper at the head, but `HR@10` is still lower than both the source `v2_on_p05` SFT and the strongest original MiniOneRec baselines.

3. This run already crosses one important line:
`v2_on_p05 RL` now beats the strongest original MiniOneRec SFT on `NDCG@10`, even though it still trails the strongest original RL overall.

4. The remaining gap is now very specific.
It is no longer “`v2` cannot survive downstream training”.
It is more like:
`v2` survives both SFT and RL, but the remaining weakness is still a deeper retention / mid-beam hit-rate gap.

## Current Claim

The strongest claim supported by this run is:

> `v2_on_p05` is not only an SFT-side improvement. Its gain survives into RL, improves head ranking further, and pushes `NDCG@10` beyond the strongest original MiniOneRec SFT baseline. However, it still does not beat the strongest original MiniOneRec RL overall because `HR@10`-style deeper retention remains weaker.
