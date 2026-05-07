Status（状态）: `reference（参考）`
Last updated（更新日期）: `2026-05-01`

# Experiment Plan（实验计划）

**Problem（问题）**: RQ-VAE tokenizer loss（残差量化变分自编码器分词器损失） optimizes reconstruction / quantization quality（重建 / 量化质量）, but the downstream SFT objective（下游监督微调目标） optimizes whether user histories can predict target SID sequences（目标语义 ID 序列）. These objectives may be misaligned（不对齐）.

**Method Thesis（方法主张）**: Replace fixed residual-code summation（固定残差码求和） in the RQ-VAE reconstruction path（重构路径） with identity-preserving attention weights（保持恒等初始化的注意力权重）, so each item（物品） can adaptively control how much each residual code level（残差码层级） contributes.

**Date（日期）**: `2026-05-01`

## Claim Map（主张地图）

| Claim（主张） | Why It Matters（重要性） | Minimum Convincing Evidence（最低说服证据） | Linked Blocks（对应实验块） |
|---|---|---|---|
| C1: Fixed residual-code summation（固定残差码求和） is a plausible tokenizer bottleneck（分词器瓶颈）. | The current RQ-VAE always uses \(q^{(1)}+q^{(2)}+q^{(3)}\), even though different items may need different residual levels（残差层级）. | AttnRQ-Identity（保持恒等初始化的注意力残差量化） keeps tokenizer health（分词器健康度） comparable and improves or does not damage downstream SFT/evaluate（下游监督微调/评测） against the strongest OneRec baseline（最强 OneRec 基线）. | B1, B2 |
| C2: Item-dependent attention（物品相关注意力） matters beyond static layer rescaling（静态层缩放）. | If static weights work equally well, the contribution is only a small calibration trick（校准技巧）, not adaptive residual composition（自适应残差组合）. | Dynamic AttnRQ（动态注意力残差量化） beats or meaningfully differs from StaticRQ（静态残差缩放） under matched parameters and training budget（训练预算）. | B3 |
| Anti-claim（反主张）: Any gain only comes from extra parameters or looser reconstruction（额外参数或更松重建）. | A reviewer（审稿人） will ask whether this is just more capacity（容量）. | Include StaticRQ and no-RMSNorm / no-regularization ablations（消融）; report reconstruction MSE（重建均方误差）, code usage（码使用）, and downstream metrics（下游指标） together. | B3, B4 |

## Method Definition（方法定义）

For each item \(i\), the baseline tokenizer（基线分词器） encodes semantic embedding（语义嵌入） \(x_i\) into:

\[
u_i = f_{\mathrm{enc}}(x_i).
\]

Residual quantization（残差量化） produces:

\[
r_i^{(1)} = u_i,
\qquad
q_i^{(l)} = Q_l(r_i^{(l)}),
\qquad
r_i^{(l+1)} = r_i^{(l)} - q_i^{(l)}.
\]

The original reconstruction representation（原始重构表示） is:

\[
h_i = \sum_{l=1}^{L} q_i^{(l)}, \qquad L=3.
\]

The proposed AttnRQ-Identity（保持恒等初始化的注意力残差量化） uses:

\[
\tilde h_i = \sum_{l=1}^{L}\gamma_i^{(l)}q_i^{(l)}.
\]

The residual weight（残差权重） is:

\[
\gamma_i^{(l)}
= L \cdot
\frac{\exp(a_i^{(l)})}
{\sum_{t=1}^{L}\exp(a_i^{(t)})},
\qquad
a_i^{(l)} = w_l^\top \mathrm{RMSNorm}(q_i^{(l)}).
\]

All pseudo-query vectors（伪查询向量） \(w_l\) are initialized to zero:

\[
w_l = 0
\Rightarrow
\gamma_i^{(l)} = 1
\Rightarrow
\tilde h_i = h_i.
\]

The training objective（训练目标） is:

\[
\mathcal L
=
\mathcal L_{\mathrm{rec}}\big(f_{\mathrm{dec}}(\tilde h_i), x_i\big)
+
\lambda_{\mathrm{rq}}\mathcal L_{\mathrm{rq}}
+
\lambda_{\mathrm{attn}}\mathcal L_{\mathrm{attn}}.
\]

The attention regularizer（注意力正则） keeps the pilot（小试验） close to baseline（基线） early in training:

\[
\mathcal L_{\mathrm{attn}}
=
\frac{1}{N L}
\sum_{i=1}^{N}\sum_{l=1}^{L}
\left(\gamma_i^{(l)} - 1\right)^2.
\]

Default pilot settings（默认小试验设置）:

- \(L=3\)
- \(\lambda_{\mathrm{rq}}=1.0\), matching the current OneRec tokenizer（当前 OneRec 分词器）
- \(\lambda_{\mathrm{attn}}\in\{0.001,0.01\}\)
- tokenizer training hyperparameters（分词器训练超参） aligned to the strong OneRec / prior tokenizer recipe（强 OneRec / 既有分词器配方）:
  `epochs=10000`, `batch_size=20480`, `eval_step=50`, `warmup_epochs=50`, `learner=AdamW`, `lr_scheduler_type=constant`
- RMSNorm（均方根归一化） enabled
- zero-initialized pseudo-query（零初始化伪查询）
- no graph loss（无图损失）
- no validation/test target leakage（无验证/测试目标泄露）

## Paper Storyline（论文叙事）

Main paper（主文） must prove:

- The fixed residual-composition assumption（固定残差组合假设） is testable and can affect downstream learnability（下游可学习性）.
- A low-disturbance attention residual（低扰动注意力残差） can preserve OneRec routeability（可路由性） while allowing item-specific residual weighting（物品特定残差加权）.

Appendix（附录） can support:

- Attention-weight distributions（注意力权重分布） by item frequency（物品频率）, prefix ambiguity（前缀歧义）, and reconstruction error（重建误差）.
- Alternative attention heads（注意力头） such as input-dependent query（输入相关查询）.

Experiments intentionally cut（暂不做实验）:

- Collaborative graph injection（协同图注入） into AttnRQ.
- Full downstream-aware bilevel tokenizer training（双层下游感知分词器训练）.
- Modifying the SFT backbone residuals（监督微调骨干残差）.

## Experiment Blocks（实验块）

### B0: Implementation Sanity（实现正确性检查）

- Claim tested（测试主张）: AttnRQ-Identity starts exactly as the original RQ-VAE（原始残差量化变分自编码器）.
- Why this block exists（存在原因）: If zero initialization（零初始化） does not reproduce fixed sum（固定求和）, downstream comparisons become invalid（无效）.
- Dataset / split / task（数据 / 划分 / 任务）: A small subset of `Industrial_and_Scientific` semantic embeddings（语义嵌入）.
- Compared systems（比较系统）: Original RQ-VAE vs AttnRQ-Identity at initialization.
- Metrics（指标）: max absolute difference（最大绝对差） between \(h_i\) and \(\tilde h_i\), reconstruction output difference（重构输出差异）, \(\gamma\) mean / std（均值 / 标准差）.
- Setup details（设置细节）: one forward pass（前向传播） before training.
- Success criterion（成功标准）: \(\gamma=1\) up to numerical tolerance（数值误差范围）, and outputs match baseline（基线）.
- Failure interpretation（失败解释）: The scaling or initialization is wrong; do not train.
- Table / figure target（表格 / 图）: development note（开发记录） only.
- Priority（优先级）: MUST-RUN（必跑）.

### B1: Tokenizer Health Pilot（分词器健康小试验）

- Claim tested（测试主张）: AttnRQ-Identity can train without destroying tokenizer geometry（分词器几何）.
- Why this block exists（存在原因）: Previous graph-loss variants（图损失变体） often improved local structure（局部结构） but hurt downstream routeability（下游可路由性）.
- Dataset / split / task（数据 / 划分 / 任务）: `Industrial_and_Scientific`, same semantic embeddings（语义嵌入） and train split（训练划分） as the OneRec tokenizer baseline（OneRec 分词器基线）.
- Compared systems（比较系统）:
  - RQBase（原始 RQ-VAE 基线）
  - AttnRQ-Identity, \(\lambda_{\mathrm{attn}}=0.001\)
  - AttnRQ-Identity, \(\lambda_{\mathrm{attn}}=0.01\)
- Metrics（指标）:
  - Primary diagnostic（主要诊断）: reconstruction MSE（重建均方误差）, RQ loss（残差量化损失）, final SID collision（最终语义 ID 冲突）
  - Secondary diagnostic（次要诊断）: active L1/L2/L3 code count（活跃码数量）, prefix entropy（前缀熵）, \(\gamma^{(1)},\gamma^{(2)},\gamma^{(3)}\) distribution（分布）
- Setup details（设置细节）: single seed（单随机种子） first; full baseline tokenizer schedule（基线分词器日程） if smoke test（冒烟测试） is healthy.
- Success criterion（成功标准）: No code collapse（码坍缩）; reconstruction and collision remain comparable to baseline; attention weights（注意力权重） show either meaningful but bounded deviation（有界偏离） or remain near identity（恒等）.
- Failure interpretation（失败解释）:
  - collapse（坍缩） means the attention path is too free or gradients are unstable;
  - \(\gamma\approx1\) with no downstream gain means fixed sum may not be the bottleneck.
- Table / figure target（表格 / 图）: tokenizer diagnostics table（分词器诊断表）.
- Priority（优先级）: MUST-RUN（必跑）.

### B2: Downstream Gate（下游门槛）

- Claim tested（测试主张）: Tokenizer-side attention residual（分词器侧注意力残差） improves downstream learnability（下游可学习性), not only tokenizer proxy metrics（分词器代理指标）.
- Why this block exists（存在原因）: The archived MGR-SID line（已归档 MGR-SID 线） showed tokenizer proxies（分词器代理指标） cannot promote methods by themselves.
- Dataset / split / task（数据 / 划分 / 任务）: Same `Industrial_and_Scientific` SFT/evaluate protocol（监督微调/评测协议） as the strongest OneRec baseline（最强 OneRec 基线）.
- Compared systems（比较系统）:
  - Strongest clean OneRec SID baseline（最强干净 OneRec 语义 ID 基线）
  - Best B1 AttnRQ-Identity tokenizer（B1 最优注意力残差量化分词器）
- Metrics（指标）:
  - Primary（主要）: `NDCG@1/@3/@5/@10`, `HR@1/@3/@5/@10`
  - Secondary（次要）: `NDCG@50`, `HR@50`, constraint invalid count（约束非法数量）, output collision（输出冲突）
- Setup details（设置细节）: use the same SFT hyperparameters（监督微调超参数）, same train/valid/test split（训练/验证/测试划分）, and same constrained decoding（约束解码） as the baseline.
- Success criterion（成功标准）: Any credible improvement（可信提升） on primary metrics（主要指标） without top-k regression（高位截断退化）; strongest success is a consistent win at `@1/@3/@5/@10`.
- Failure interpretation（失败解释）: If tokenizer health improves but downstream does not, residual weighting（残差加权） alone does not close tokenizer/downstream loss mismatch（分词器/下游损失不对齐）.
- Table / figure target（表格 / 图）: main downstream table（主下游表）.
- Priority（优先级）: MUST-RUN（必跑）.

### B3: Novelty Isolation Ablations（新颖性隔离消融）

- Claim tested（测试主张）: Item-dependent residual attention（物品相关残差注意力） matters beyond static layer calibration（静态层校准）.
- Why this block exists（存在原因）: Reviewers（审稿人） may argue that the method is just learnable scalar rescaling（可学习标量重缩放）.
- Dataset / split / task（数据 / 划分 / 任务）: Same tokenizer and downstream setting（分词器与下游设置） as B1/B2.
- Compared systems（比较系统）:
  - RQBase（原始基线）
  - StaticRQ: global learnable \(\gamma^{(l)}\), initialized at 1（全局可学习层权重）
  - AttnRQ-Identity（物品相关注意力）
  - AttnRQ without RMSNorm（无 RMSNorm）
  - AttnRQ without \(\mathcal L_{\mathrm{attn}}\)（无注意力正则）
- Metrics（指标）: same as B1 and B2.
- Setup details（设置细节）: run only after B2 shows non-negative signal（非负信号）.
- Success criterion（成功标准）: AttnRQ-Identity outperforms StaticRQ or gives clearer learnability behavior（可学习性行为） under comparable tokenizer health.
- Failure interpretation（失败解释）: If StaticRQ matches AttnRQ, the simpler method should be preferred（优先简单方法）.
- Table / figure target（表格 / 图）: ablation table（消融表）.
- Priority（优先级）: MUST-RUN only if B2 is promising（仅当 B2 有希望时必跑）.

### B4: Attention Weight Diagnosis（注意力权重诊断）

- Claim tested（测试主张）: Learned residual weights（学到的残差权重） expose when each SID level（语义 ID 层级） is useful.
- Why this block exists（存在原因）: The method must explain learnability（可学习性）, not merely report a number.
- Dataset / split / task（数据 / 划分 / 任务）: trained tokenizer outputs（已训练分词器输出） and downstream predictions（下游预测）.
- Compared systems（比较系统）: AttnRQ-Identity only, optionally compared with StaticRQ（静态残差缩放）.
- Metrics（指标）:
  - \(\gamma\) by item popularity（按物品流行度）
  - \(\gamma\) by prefix ambiguity（按前缀歧义）
  - \(\gamma\) by reconstruction error（按重建误差）
  - \(\gamma\) for downstream hit vs miss（命中与未命中）
- Setup details（设置细节）: offline diagnostic script（离线诊断脚本） after tokenizer and SFT/evaluate finish.
- Success criterion（成功标准）: interpretable non-random patterns（可解释非随机模式） that align with downstream behavior（下游行为）.
- Failure interpretation（失败解释）: The attention head may be acting as unstructured capacity（无结构容量）.
- Table / figure target（表格 / 图）: heatmap or grouped bar plot（热力图或分组柱状图）.
- Priority（优先级）: NICE-TO-HAVE（可选） for first pilot, MUST-RUN（必跑） if writing.

## Run Order and Milestones（运行顺序与里程碑）

| Milestone（里程碑） | Goal（目标） | Runs（运行） | Decision Gate（决策门槛） | Cost（成本） | Risk（风险） |
|---|---|---|---|---|---|
| M0 | Implementation sanity（实现正确性） | R001 | Exact baseline equivalence at initialization（初始化等价） | CPU / one small GPU, minutes（分钟级） | scaling bug（缩放错误） |
| M1 | Tokenizer smoke（分词器冒烟） | R002, R003 | no collapse（无坍缩）, \(\gamma\) finite（有限）, MSE comparable（均方误差可比） | one GPU, short subset run（短子集运行） | attention overfits reconstruction（注意力过拟合重建） |
| M2 | Full tokenizer（完整分词器） | R004, R005 | best AttnRQ selected by tokenizer health（分词器健康度） | one GPU per run（每次一张 GPU） | no meaningful \(\gamma\) movement（权重无变化） |
| M3 | Downstream gate（下游门槛） | R006 | primary `@1/@3/@5/@10` not worse, ideally better（主要指标不退化，最好提升） | same as baseline SFT/eval（同基线监督微调/评测） | tokenizer proxy/downstream mismatch remains（不对齐仍存在） |
| M4 | Ablation decision（消融决策） | R007-R010 | AttnRQ beats StaticRQ or simpler method wins（注意力优于静态，或采用更简单方法） | only after positive M3（仅 M3 正信号后） | too many variants（变体过多） |
| M5 | Diagnosis（诊断） | R011 | interpretable \(\gamma\) patterns（可解释权重模式） | CPU / light GPU（轻量） | patterns not robust（模式不稳定） |

## Compute and Data Budget（计算与数据预算）

- Total pilot budget（小试验总预算）: keep M0-M3 within the smallest feasible tokenizer + one downstream SFT/evaluate loop（一个下游监督微调/评测闭环）.
- Large outputs（大产物）: store checkpoints（检查点） under `/data/leejt/OneRec/output_weights`, not repository `./output`.
- Lightweight artifacts（轻量产物）: store logs（日志）, tokenizer diagnostics（分词器诊断）, and tables（表格） under repository `logs/`, `results/`, or this branch folder.
- Data preparation（数据准备）: reuse existing semantic embeddings（语义嵌入） and standard train/valid/test split（标准训练/验证/测试划分）.
- Biggest bottleneck（最大瓶颈）: downstream SFT/evaluate（下游监督微调/评测）, not tokenizer training（分词器训练）.

## Risks and Mitigations（风险与缓解）

- Risk（风险）: softmax scaling changes representation magnitude（表示幅值）.
  Mitigation（缓解）: multiply softmax by \(L=3\) and zero-initialize pseudo-query（零初始化伪查询）.

- Risk（风险）: attention weights（注意力权重） become a reconstruction-only trick（仅重建技巧）.
  Mitigation（缓解）: downstream gate（下游门槛） is mandatory; tokenizer-only wins do not promote the method（不因仅分词器胜利推进）.

- Risk（风险）: dynamic attention（动态注意力） is unnecessary.
  Mitigation（缓解）: compare with StaticRQ（静态残差缩放）.

- Risk（风险）: method damages SID routeability（语义 ID 可路由性）.
  Mitigation（缓解）: keep code assignment（码分配） unchanged structurally; start from identity（恒等）；monitor constrained decoding（约束解码） and collision（冲突）.

- Risk（风险）: no improvement because three residual levels（三个残差层） are too shallow.
  Mitigation（缓解）: treat negative result（负结果） as evidence that fixed residual summation（固定残差求和） is not the dominant tokenizer bottleneck（主要分词器瓶颈）.

## Final Checklist（最终检查清单）

- [ ] Main downstream table（主下游表） is covered.
- [ ] Original OneRec baseline（原始 OneRec 基线） is included.
- [ ] StaticRQ ablation（静态残差缩放消融） is included if M3 is promising.
- [ ] Tokenizer health（分词器健康度） and downstream metrics（下游指标） are reported together.
- [ ] Attention weight diagnosis（注意力权重诊断） is prepared if writing.
- [ ] No validation/test leakage（验证/测试泄露） is introduced.
