# L2 Square Candidate（第二层平方图候选）

Status（状态）: `active-candidate（活跃候选）`

## Variants（变体）

- `r690b_lmh_l2_square_dominant_b025`: L2 graph（第二层图） uses `RowNorm(0.25 * A_local + A_local^2)`.
- `r690b_lmh_l2_square_only`: L2 graph（第二层图） uses `RowNorm(A_local^2)`.

## Current Diagnosis（当前诊断）

`square_dominant_b025`:

- generated collision rate（生成冲突率）: `0.002984`
- max conflict（最大冲突簇）: `2`
- structural profile（结构画像）: `separating-but-risky（拆分强但有风险）`
- interpretation（解释）: stronger S-near C-far separation（语义近协同远拆分更强）, but weaker S-near C-near preservation（语义近协同近保持更弱）.

`square_only`:

- generated collision rate（生成冲突率）: `0.032013`
- max conflict（最大冲突簇）: `41`
- interpretation（解释）: too risky for immediate SFT（监督微调）.

## Next Run（下一次运行）

The previous queue（队列） was stopped before launch. After directory cleanup（目录整理） and after the current L3 ranking SFT/eval（第三层排序监督微调/评测） finishes, start:

```bash
cd /home/leejt/OneRec
bash idea-discovery/2026-05-06_l2_local_multihop_rescue_tokenizers/active_candidates/l2_square/scripts/run_sft_eval_chain.sh 2,3,4,5
```
