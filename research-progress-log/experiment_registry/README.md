# Experiment Registry（实验登记表）

Status（状态）: `registry（总账）`
Last updated（更新日期）: `2026-04-18`

## 目的

这里是新的 split registry（分表总账）入口，用来替代根目录旧 `experiment_results.csv` 的日常维护角色。

旧表的问题是 tokenizer（分词器）、SFT（监督微调）、RL（强化学习）三类实验被塞进同一个 wide table（宽表），导致列数膨胀到 `112`，追加新行时很容易 column drift（列错位）。新规则是：

- tokenizer（分词器）结果进入 `tokenizer_registry.csv`
- SFT（监督微调）结果进入 `sft_registry.csv`
- RL（强化学习）结果进入 `rl_registry.csv`
- 下游指标对比优先看 `downstream_scoreboard.csv`

根目录旧表：

- `/home/leejt/OneRec/experiment_results.csv`

现在保留为 legacy wide registry（历史宽表总账），用于迁移追溯和兼容旧脚本，不再作为人工追加新结果的首选入口。

## 当前表格

| 文件 | 角色 | 当前列数 | 说明 |
| --- | --- | ---: | --- |
| `tokenizer_registry.csv` | tokenizer registry（分词器登记表） | 22 | 只记录 SID/tokenizer（语义标识/分词器）的生成、冲突和产物路径 |
| `sft_registry.csv` | SFT registry（监督微调登记表） | 54 | 记录 SFT（监督微调）配方、模型路径、W&B（实验追踪）和 evaluate（评测）指标 |
| `rl_registry.csv` | RL registry（强化学习登记表） | 45 | 记录 RL（强化学习）链路、来源 SFT（监督微调）模型和 evaluate（评测）指标 |
| `downstream_scoreboard.csv` | scoreboard（下游指标榜单） | 24 | 只保留下游裁决最常用字段，按 `NDCG@10`（归一化折损累计增益@10）优先排序 |

## 字段口径

- `generated_collision_count`（生成冲突数）使用 duplicate excess（重复冗余数）口径，即每个重复 SID group（语义标识组）按 `group_size - 1` 计数。
- `generated_collision_rate`（生成冲突率）为 `generated_collision_count / num_items`（生成冲突数 / 物品数）。
- 它不是 collided item count（冲突物品总数），也不是 conflict group count（冲突组数量）。例如一个三物品冲突组的 duplicate excess（重复冗余数）是 `2`，collided item count（冲突物品数）是 `3`，conflict group count（冲突组数量）是 `1`。
- `downstream_scoreboard.csv`（下游指标榜单）按 dataset（数据集）分组后，以 `NDCG@10`（归一化折损累计增益@10）降序、`HR@10`（命中率@10）降序排序，不再按 SFT/RL（监督微调/强化学习）阶段硬分组。

## 生成方式

当前分表由 legacy wide registry（历史宽表总账）迁移生成：

```bash
source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec
cd /home/leejt/OneRec
python scripts/split_experiment_results_registry.py --overwrite
```

生成脚本：

- [scripts/split_experiment_results_registry.py](/home/leejt/OneRec/scripts/split_experiment_results_registry.py)

校验脚本：

- [scripts/validate_experiment_registry.py](/home/leejt/OneRec/scripts/validate_experiment_registry.py)

```bash
python scripts/validate_experiment_registry.py
```

## 维护规则

- 不再手写 `112` 列的 legacy wide registry（历史宽表总账）行。
- 新 finalized result（定稿结果）优先写入对应窄表。
- `split_experiment_results_registry.py` 是 migration helper（迁移辅助脚本）；如果分表里已经有新结果，只有确认要从 legacy wide registry（历史宽表总账）刷新时才使用 `--overwrite`（覆盖）。
- 每次更新 split registry（分表总账）后，都运行 `validate_experiment_registry.py` 做列数和重复 `record_id`（记录编号）校验。
- 如果还需要兼容旧分析脚本，可以在同一任务里同步更新 `experiment_results.csv`，但它不再是唯一入口。
- stage README（阶段快照）仍负责记录完整 config（配置）、日志、GPU（显卡）、W&B（实验追踪）和诊断产物路径。
- scoreboard（榜单）只服务快速比较，不替代 tokenizer/SFT/RL registry（分词器/监督微调/强化学习登记表）。

## 实验记录流水线

这套 registry（登记表）只记录 finalized result（定稿结果），不记录 running run（运行中任务）的中间状态。中间状态默认留在对话、日志和运行脚本里；只有当它属于 finalized method design（定稿方法设计）或改变 next-step decision（下一步决策）时，才写入长期文档。

### Stage 1: launch（启动）

当一个实验正式启动时：

1. 不写入 `tokenizer_registry.csv` / `sft_registry.csv` / `rl_registry.csv`，因为还没有 finalized result（定稿结果）。
2. 默认不更新 stage README（阶段快照）或 `CURRENT_STATE.md`（当前状态）。
3. 只有当 launch（启动）本身伴随新的 method design（方法设计）定稿、或改变 active mainline（活跃主线）/ next-step decision（下一步决策）时，才更新对应长期文档。
4. GPU（显卡）、tmux（终端复用器）、日志路径等运行细节优先在对话和脚本里同步，不作为长期文档必填项。

### Stage 2: running（运行中）

当实验仍在运行时：

1. 默认不维护长期 running status（运行状态）文档。
2. `tmux`（终端复用器） session（会话）、W&B（实验追踪） run id、当前日志和预计完成时间优先在对话中同步。
3. 不把 partial metric（中间指标）写入 split registry（分表总账）。

### Stage 3: tokenizer finalized（分词器定稿）

当 tokenizer/generate（分词器训练与生成）完成后：

1. 从 SID index（语义标识索引）或 generate summary（生成摘要）确认 `generated_collision_count`（生成冲突数）和 `generated_collision_rate`（生成冲突率）。
2. 更新 `tokenizer_registry.csv`。
3. 更新对应 stage README（阶段快照）的 result（结果）和 conclusion（结论）。
4. 运行：

```bash
python scripts/validate_experiment_registry.py
```

5. 如果该 tokenizer（分词器）进入下游验证，更新 `CURRENT_STATE.md`（当前状态）的 next step（下一步）。

### Stage 4: SFT/evaluate finalized（监督微调/评测定稿）

当 SFT -> evaluate（监督微调到评测）完成后：

1. 从 result JSON（结果文件）重算或读取 `NDCG/HR`（推荐指标）。
2. 更新 `sft_registry.csv`。
3. 同步更新 `downstream_scoreboard.csv`。
4. 更新对应 stage README（阶段快照）的 result（结果）和 verdict（裁决）。
5. 运行：

```bash
python scripts/validate_experiment_registry.py
```

6. 如果结果改变 strongest line（最强主线）、baseline comparison（基线比较）或 next step（下一步），更新 `CURRENT_STATE.md`（当前状态）。

### Stage 5: RL/evaluate finalized（强化学习/评测定稿）

当 RL -> evaluate（强化学习到评测）完成后：

1. 从 result JSON（结果文件）重算或读取 `NDCG/HR`（推荐指标）。
2. 更新 `rl_registry.csv`。
3. 同步更新 `downstream_scoreboard.csv`。
4. 更新对应 stage README（阶段快照）的 result（结果）和 verdict（裁决）。
5. 运行：

```bash
python scripts/validate_experiment_registry.py
```

6. 如果结果改变 strongest line（最强主线）或 paper claim（论文主张），更新 `CURRENT_STATE.md`（当前状态）和必要的 method/narrative docs（方法/叙事文档）。

### Stage 6: audit（审计）

每次 registry（登记表）更新后，至少做三类检查：

1. schema validation（表结构校验）：`validate_experiment_registry.py`
2. artifact existence（产物存在性）：确认非空 config/model/result/readme path（配置/模型/结果/说明路径）存在
3. metric consistency（指标一致性）：重要结果应从 result JSON（结果文件）重算 `NDCG/HR`（推荐指标）再入表

### 禁止项

- 不把 running run（运行中任务）写入 split registry（分表总账）。
- 不把 partial metric（中间指标）写成 finalized result（定稿结果）。
- 不直接手工追加 legacy wide registry（历史宽表总账）。
- 不用 `split_experiment_results_registry.py --overwrite` 覆盖已有分表，除非明确是在做 migration refresh（迁移刷新）。
