# 2026-04-15 TAGCF `R510` SFT Evaluate Industrial

## 目的

把 `TAGCF` 支线当前最值得继续验证的 tokenizer（分词器）候选 `R510` 推到完整下游：

- `title_history2sid_on + desc_align_p05`
- `SFT -> evaluate`（监督微调到评测）

这轮不是为了证明 `R510` 已经是最优 tokenizer，而是为了回答：

> 一个 `generate collision`（生成后冲突率）不错、但结构结果 mixed（混合）的属性图纯替换分支，到底值不值得进入真正的下游比较。

## 候选来源

- tokenizer 运行：`R510`
- 生成索引：
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_tagcf_branch_20260415/generated_indices/Industrial_and_Scientific.tagcf_r510_attr_mid.index.json`
- 数据副本根目录：
  - `./data_experiment/Amazon/tagcf_r510_attr_mid`

## 配置

- SFT：
  - [sft_industrial_mgr_tagcf_r510_title_on_desc_p05.yaml](/home/leejt/OneRec/config/experiments/sft_industrial_mgr_tagcf_r510_title_on_desc_p05.yaml)
- Evaluate：
  - [evaluate_industrial_mgr_tagcf_r510_title_on_desc_p05.yaml](/home/leejt/OneRec/config/experiments/evaluate_industrial_mgr_tagcf_r510_title_on_desc_p05.yaml)

## 启动脚本

- [experiment_mgr_sid_tagcf_r510_sft_eval_chain.sh](/home/leejt/OneRec/scripts/experiment_mgr_sid_tagcf_r510_sft_eval_chain.sh)

## Runtime

- launch date：
  - `2026-04-15`
- tmux：
  - `mgr_tagcf_r510_sft_eval`
- GPUs：
  - `2,3,4,5`
- status：
  - `COMPLETED`

## Logs

- SFT：
  - `/home/leejt/OneRec/logs/experiment_mgr_sid_tagcf_r510_sft_20260415.log`
- Evaluate：
  - `/home/leejt/OneRec/logs/experiment_mgr_sid_tagcf_r510_eval_20260415.log`

## Output Targets

- SFT output：
  - `/data/leejt/OneRec/output_weights/experiments/mgr_sid_tagcf_sft_eval_20260415/r510_title_on_desc_p05/sft`
- Evaluate result：
  - `/home/leejt/OneRec/results/experiments/mgr_sid_tagcf_sft_eval_20260415/final_result_sft_mgr_tagcf_r510_title_on_desc_p05_Industrial_and_Scientific.json`

## Final Metrics

- `NDCG@1/3/5/10/20/50`
  - `0.07103 / 0.08505 / 0.08946 / 0.09758 / 0.10724 / 0.11835`
- `HR@1/3/5/10/20/50`
  - `0.07103 / 0.09530 / 0.10611 / 0.13148 / 0.16964 / 0.22568`
- `constraint_invalid_total`
  - `0`

## Quick Read

- `R510` 完成了第一轮完整 `SFT -> evaluate`（监督微调到评测）闭环。
- 相对当前最强 `v2_on_p05`，结果是负向：
  - `NDCG@10`: `0.09758` vs `0.10271`
  - `HR@10`: `0.13148` vs `0.14626`
- 相对 `R202a` 和 `R401b` 也更低。
- 这说明 `TAGCF` 纯属性图替换中图这条线，至少在当前实现下，还没有转化成更好的下游排名效果。
