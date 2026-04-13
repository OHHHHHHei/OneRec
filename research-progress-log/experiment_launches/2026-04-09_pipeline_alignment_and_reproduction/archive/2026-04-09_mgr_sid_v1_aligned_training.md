# MGR-SID v1 对齐版训练实验（2026-04-09）

## 目的

这轮实验的目标不是继续扩大方法规模，而是先把训练语义和原始 `sid-train` 严格对齐，回答一个更基础的问题：

> 上一轮训练版结果差，到底是因为 `MGR-SID` 的 integration 不行，还是因为实验分支的训练语义和原始 RQ-VAE 根本没对齐？

## 本轮对齐内容

对齐后的实验分支位于：

- `scripts/experiment_mgr_sid_v1_train.py`
- `src/onerec/experiments/mgr_sid/train_v1.py`

这轮明确对齐了原始 `sid_train.yaml` / `onerec.sid.quantizers.rqvae` 的以下设置：

- `epochs = 10`
- `batch_size = 256`
- `num_workers = 0`
- `lr = 0.001`
- `learner = AdamW`
- `lr_scheduler_type = constant`
- `warmup_epochs = 50`
- `eval_step = 50`
- mini-batch `DataLoader` 训练语义
- 每个 batch 的 `optimizer.step()` 与 `scheduler.step()`

Industrial 当前数据规模为 `3686` 个 item embedding，因此：

- 原始训练：`15` 个 mini-batch / epoch
- 对齐版实验分支：也为 `15` 个 mini-batch / epoch

## 运行命令

### 对齐版实验分支

- baseline:
  - `python scripts/experiment_mgr_sid_v1_train.py --config config/experiments/sid_train_industrial_mgr_sid_v1_baseline.yaml --device cuda:2`
- uniform:
  - `python scripts/experiment_mgr_sid_v1_train.py --config config/experiments/sid_train_industrial_mgr_sid_v1_uniform.yaml --device cuda:3`
- hierarchy:
  - `python scripts/experiment_mgr_sid_v1_train.py --config config/experiments/sid_train_industrial_mgr_sid_v1_hierarchy.yaml --device cuda:4`

### 原始主线 sanity baseline

- `python -m onerec.main sid-train --config config/sid_train.yaml device=cuda:5 ckpt_dir=./output/rqvae_Industrial_and_Scientific_aligned_sanity`

## 结果位置

### 对齐版实验分支

- baseline summary:
  - `output/experiments/mgr_sid_v1/industrial_baseline/Apr-09-2026_21-31-04/summary.json`
- uniform summary:
  - `output/experiments/mgr_sid_v1/industrial_uniform_reg/Apr-09-2026_21-31-05/summary.json`
- hierarchy summary:
  - `output/experiments/mgr_sid_v1/industrial_hierarchy_reg/Apr-09-2026_21-31-04/summary.json`

日志：

- `logs/experiment_mgr_sid_v1_aligned_industrial_baseline_20260409.log`
- `logs/experiment_mgr_sid_v1_aligned_industrial_uniform_20260409.log`
- `logs/experiment_mgr_sid_v1_aligned_industrial_hierarchy_20260409.log`

### 原始主线 sanity baseline

- 输出目录：
  - `output/rqvae_Industrial_and_Scientific_aligned_sanity/Apr-09-2026_21-33-02`
- 日志：
  - `logs/sid_train_industrial_aligned_sanity_20260409.log`

## 主要结果

### 对齐版实验分支 best collision

- `baseline`: `0.998915`
- `uniform_reg`: `0.997016`
- `hierarchy_reg`: `0.996744`

### 原始主线 sanity baseline best collision

- `sid-train baseline`: `0.998915`

## 最关键的事实

1. 这轮对齐是成功的。

因为：

- 对齐版实验分支的 `baseline` 训练曲线与原始主线 `sid-train` 完全一致；
- 最终 `best loss` 和 `best collision` 也一致；
- 原始主线 sanity baseline 与实验分支 baseline 都得到 `0.998915`。

这说明：

> 现在 baseline 的差表现，不是实验分支训练语义错位导致的；至少在当前数据与超参数下，实验分支 baseline 已经和原始主线对齐。

2. 上一轮 full-batch 版本应视为无效轮次。

它可以证明“脚本可跑”，但不应该再被当作正式训练证据引用。

3. 在当前对齐版设置下，graph regularization 仍然没有给出正向结果。

虽然：

- `uniform_reg` 略好于 `baseline`
- `hierarchy_reg` 略好于 `uniform_reg`

但三者都处在极高 collision 区间，提升量非常有限，远不足以支持主张。

## 当前解释

这轮结果最重要的意义，是把问题进一步缩小了：

- 不是训练超参数没对齐
- 不是实验分支比原始主线少训练了很多步
- 更可能是当前 graph integration mechanism 本身还不对

更具体地说，当前版本的问题可能在于：

- graph smoothness 直接作用在量化后的 cumulative representation 上，过于粗糙
- batch 内子图 regularization 仍然过强或方向不对
- 没有显式保护 code usage balance
- 没有显式约束 prefix predictability
- 当前 graph loss 更像在鼓励过平滑，而不是提升可分性

## 当前结论

这轮训练实验的结论应该写成：

> 对齐版训练已经证明，当前 `MGR-SID v1` 的问题不在训练步数或 epoch 语义，而在 graph integration mechanism 仍然不够合适。

## 下一步

当前最合理的下一步不是继续扩大同一版训练，而是改 integration：

1. 引入更明确的 anti-collapse / code-balance 保护
2. 从表示平滑改成更 quantization-aware 的约束
3. 考虑把 graph signal 先作用在 encoder latent，而不是直接作用在 cumulative quantized representation
4. 重新考虑 graph loss 的时机或 schedule
5. 在 revised integration 上再比较 `fagsp_mid_base` 和 `gsprec_mid_prism`
