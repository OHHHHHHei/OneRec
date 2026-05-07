Status（状态）: `reference（参考）`
Last updated（更新日期）: `2026-05-01`

# Attentive Residual RQ-VAE Tokenizer（注意力残差 RQ-VAE 分词器）

This folder records a new tokenizer-side（分词器侧） exploratory branch inspired by `Attention Residuals`（注意力残差）:

- Local paper（本地论文）: [2603.15031_Attention_Residuals.pdf](/home/leejt/OneRec/papers/2603.15031_Attention_Residuals.pdf)
- arXiv（论文编号）: `2603.15031`

## Research Question（研究问题）

Can the RQ-VAE tokenizer（残差量化变分自编码器分词器） improve downstream learnability（下游可学习性） by replacing fixed residual-code summation（固定残差码求和） with identity-preserving attentive residual composition（保持恒等初始化的注意力残差组合）?

## Method Sketch（方法草图）

The original OneRec tokenizer（原始 OneRec 分词器） encodes a semantic embedding（语义嵌入） \(x_i\), quantizes the latent representation（潜在表示） into residual code vectors（残差码向量） \(q_i^{(1)}, q_i^{(2)}, q_i^{(3)}\), and reconstructs from a fixed sum:

\[
h_i = q_i^{(1)} + q_i^{(2)} + q_i^{(3)}.
\]

This branch keeps the same semantic embedding input（语义嵌入输入）, the same code indices（码索引）, and the same residual quantization（残差量化） process, but changes the reconstruction path（重构路径） to:

\[
\tilde h_i = \sum_{l=1}^{3}\gamma_i^{(l)}q_i^{(l)},
\qquad
\gamma_i^{(l)} = 3 \cdot \mathrm{softmax}_l(a_i^{(l)}).
\]

The logits（打分） are computed with a lightweight pseudo-query（伪查询） over each residual code vector（残差码向量）:

\[
a_i^{(l)} = w_l^\top \mathrm{RMSNorm}(q_i^{(l)}).
\]

All \(w_l\) are initialized to zero, so the initial model is exactly the original fixed-sum tokenizer（固定求和分词器）:

\[
\gamma_i^{(1)}=\gamma_i^{(2)}=\gamma_i^{(3)}=1.
\]

## Scope（范围）

This is not a collaborative-graph injection（协同图注入） experiment. The first pilot（小试验） intentionally avoids graph loss（图损失） and target-aware post-processing（目标感知后处理）.

The purpose is to test a cleaner tokenizer hypothesis（分词器假设）:

> The mismatch between tokenizer loss（分词器损失） and downstream SFT loss（下游监督微调损失） may partly come from forcing all residual code levels（残差码层级） to contribute with fixed unit weights.

## Files（文件）

- [refine-logs/EXPERIMENT_PLAN.md](/home/leejt/OneRec/idea-discovery/2026-05-01_attentive_residual_rqvae_tokenizer/refine-logs/EXPERIMENT_PLAN.md): claim-driven experiment plan（面向主张的实验计划）.
- [refine-logs/EXPERIMENT_TRACKER.md](/home/leejt/OneRec/idea-discovery/2026-05-01_attentive_residual_rqvae_tokenizer/refine-logs/EXPERIMENT_TRACKER.md): compact run tracker（紧凑运行跟踪表）.
- `tools/attnrq_identity_sanity.py`: initialization-equivalence sanity check（初始化等价正确性检查） for R001.
- `tools/attnrq_gamma_diagnostics.py`: post-training residual-weight diagnostic（训练后残差权重诊断） for \(\gamma^{(1)},\gamma^{(2)},\gamma^{(3)}\).
- `configs/sid_train_rqbase_onerec_aligned.yaml`: clean RQ-VAE baseline（干净残差量化变分自编码器基线） with original OneRec tokenizer hyperparameters（原版 OneRec 分词器超参）.
- `configs/sid_train_attnrq_identity_lam0001.yaml`: dynamic AttnRQ-Identity（动态注意力残差量化）, \(\lambda_{\mathrm{attn}}=0.001\).
- `configs/sid_train_attnrq_identity_lam001.yaml`: dynamic AttnRQ-Identity（动态注意力残差量化）, \(\lambda_{\mathrm{attn}}=0.01\).
- `configs/sid_train_attnrq_static_lam0001.yaml`: StaticRQ（静态残差缩放） ablation（消融）.
- `configs/sid_train_attnrq_no_rmsnorm_lam0001.yaml`: no-RMSNorm（无均方根归一化） ablation（消融）.
- `configs/sid_train_attnrq_no_reg.yaml`: no-regularizer（无注意力正则） ablation（消融）.

## Current Decision（当前决策）

Start with `AttnRQ-Identity`（保持恒等初始化的注意力残差量化） as a low-disturbance tokenizer pilot（低扰动分词器小试验）. The first go / no-go（推进 / 停止） gate is downstream SFT/evaluate（下游监督微调/评测）, not tokenizer-only proxy metrics（仅分词器代理指标）.

R001 initialization sanity（初始化正确性检查） has passed locally: baseline RQ-VAE（基线残差量化变分自编码器） and AttnRQ-Identity（注意力残差量化） produce identical outputs, losses, and indices（索引） at initialization.

Tokenizer training configs（分词器训练配置） are aligned to the strong OneRec / prior tokenizer recipe（强 OneRec / 既有分词器配方） used in the archived SID experiments（已归档 SID 实验） unless explicitly stated otherwise:

- `epochs=10000`
- `batch_size=20480`
- `eval_step=50`
- `warmup_epochs=50`
- `learner=AdamW`
- `lr_scheduler_type=constant`
- `num_emb_list=[256,256,256]`
- `e_dim=32`
- `layers=[2048,1024,512,256,128,64]`
