# MiniOneRec 复现进度表

配套文档：

- 这份文档主要记录“当前复现做到哪里、哪些结论已经成立、下一步做什么”。
- 如果你需要系统理解仓库代码结构、主链路、配置系统和数据契约，请同时阅读 [OneRec 项目理解报告](/home/leejt/OneRec/project_understanding_report.md)。

## 当前目标

当前仓库的目标是做 **研究型复现**，不是立刻去做“逐行逐配置完全一致”的严格论文复现。

当前策略如下：

- 直接使用原开源项目已经发布的 SID 产物和数据切分。
- 当前阶段接受 `Qwen3-1.7B` 作为研究骨干模型。
- `evaluate` 的 `num_beams=50` 先按原开源代码保持一致。
- RL 先暂缓，优先把 SFT 阶段的理解、对照实验和口径梳理清楚。

## 当前已确认事实

### 基础事实

- 当前 `Industrial_and_Scientific` 和 `Office_Products` 数据，来自原开源项目。
- SID 没有在本地重新训练，直接使用原项目已经生成好的 SID。
- 论文中主表 `MiniOneRec` 的结果，对应的是 **完整链路**，至少是 `SFT + RL + evaluate` 之后的结果。
- 当前我们已经完成的 2x2 对照实验，属于 **SFT -> Evaluate**，还不是完整论文主结果口径。

### 当前必须区分的三种口径

后续所有讨论都建议明确区分这三类：

- `paper target`
  论文正文和表格中声称的方法与最终指标。
- `repo-faithful`
  严格按原开源主线代码默认行为跑出来的结果。
- `research-enhanced`
  基于论文描述、代码理解或研究判断，对开源主线做的增强版本。

## 论文目标值

### 论文中 Industrial 的目标指标

| 设定 | HR@3 | NDCG@3 | HR@5 | NDCG@5 | HR@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| MiniOneRec（论文） | 0.1143 | 0.1011 | 0.1321 | 0.1084 | 0.1586 | 0.1167 |

### 当前解释

- 如果后面要声称“论文级复现成功”，最终应该和这组数字对比。
- 因为当前还没跑 RL，所以现在的 SFT-only 结果只能算“阶段性进展”，不能当作最终复现结果。

## 当前实验台账

台账来源文件：[experiment_results.csv](/home/leejt/OneRec/experiment_results.csv)

### Industrial 上的 2x2 SFT 对照实验

| Variant | TitleHistory2Sid | Desc Align | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | HR@1 | HR@3 | HR@5 | HR@10 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `title_history2sid_on__desc_align_off` | on | off | 0.06243106 | 0.07981905 | 0.08688172 | 0.09872236 | 0.06243106 | 0.09287448 | 0.11008162 | 0.14714317 |
| `title_history2sid_on__desc_align_p05` | on | on | 0.06353408 | 0.08273148 | 0.08944313 | 0.09870370 | 0.06353408 | 0.09684536 | 0.11317009 | 0.14206927 |
| `title_history2sid_off__desc_align_off` | off | off | 0.05691595 | 0.07390207 | 0.08378435 | 0.09380221 | 0.05691595 | 0.08603574 | 0.11008162 | 0.14118685 |
| `title_history2sid_off__desc_align_p05` | off | on | 0.06706375 | 0.08500848 | 0.09315326 | 0.10372025 | 0.06706375 | 0.09838959 | 0.11824399 | 0.15089345 |

### 当前最强的 SFT-only 配方

- 当前最强的 SFT-only 结果是：`title_history2sid_off__desc_align_p05`
- 这组在 `@1/@3/@5/@10` 上的 `NDCG` 和 `HR` 都是四组中最好的。

### 当前最强 SFT 与论文目标的差距

将当前最强的 SFT-only 结果与论文 Industrial 主结果相比：

| 指标 | 当前最强 SFT | 论文目标 | 差值 |
|---|---:|---:|---:|
| HR@3 | 0.09838959 | 0.1143 | -0.01591041 |
| NDCG@3 | 0.08500848 | 0.1011 | -0.01609152 |
| HR@5 | 0.11824399 | 0.1321 | -0.01385601 |
| NDCG@5 | 0.09315326 | 0.1084 | -0.01524674 |
| HR@10 | 0.15089345 | 0.1586 | -0.00770655 |
| NDCG@10 | 0.10372025 | 0.1167 | -0.01297975 |

### 当前结论

- 现在还**不能**说已经达到论文级复现。
- 但这个差距目前是在比较：
  - 我们的 `SFT -> Evaluate`
  - 与论文更可能对应的 `SFT + RL -> Evaluate`
- 所以当前结果更合理的定位是：
  - “SFT 阶段已经取得不错进展”
  - 不是“最终复现成功”或“最终复现失败”

## 对原开源代码的理解

### repo-faithful 的 SFT baseline

根据对原开源代码的核对：

- 原开源 `sft.py` 主线默认**不会**启用 `TitleHistory2SidSFTDataset`
- 原开源 `FusionSeqRecDataset` 里的 description 分支只是注释代码，默认**不会**真实运行

因此，当前最接近原开源主线 SFT baseline 的实验是：

- `title_history2sid_off__desc_align_off`

### repo-faithful 的 RL 方向

根据原开源 `rl.py` 主线：

- RL 并不是 recommendation-only
- RL 默认仍然混入了几类 alignment 相关任务：
  - recommendation SID 预测
  - `title2sid`
  - `description2sid`
  - `history-title2sid`
- 以原开源主线 Python 脚本为准，`num_generations=16`

所以从方向上看，当前仓库的 RL 设计和原开源主线是同一脉络。

### 论文与公开主线代码的不一致

当前最重要的判断是：

- 原开源公开主线代码，很可能**不能完整覆盖论文里对 alignment 的描述**

具体表现为：

- 论文写的是更广义的 full-process alignment
- 公开 `sft.py` 主线只稳定暴露出其中一部分
- 代码里存在更丰富的 task，但没有接到公开主线里
- 某些更像论文描述的实现，反而更接近实验支线，例如 `sft_gpr.py`

当前工作假设：

- 原开源公开主线不是论文 alignment 配方的完整实现
- 论文主实验更可能跑在主线 Python 脚本链：
  - `sft.py / rl.py / evaluate.py`
- `sft_gpr.py / rl_gpr.py` 更像实验旁支，而不是公开主实验主入口
- 但**不能**把论文主实验直接等同于当前公开 shell 默认值：
  - `sft.sh / rl.sh / evaluate.sh` 的默认参数彼此并不完全自洽

## 轻量验证补充

### `micro_batch_size=4` 的显存探针

为了判断 SFT 是否必须保留 `micro_batch_size=2`，已经做过一次轻量 OOM 探针：

- 数据：Industrial 小样本
  - train `64` 条
  - valid `32` 条
- 语义设置：
  - `title_history2sid_off__desc_align_off`
- 训练设置：
  - `batch_size=1024`
  - `micro_batch_size=4`
  - `world_size=4`
  - `grad_accum=64`
  - `num_epochs=1`

探针结果：

- 能正常完成模型加载、首个训练步、首轮验证
- 最终完整跑完 `1 epoch`
- 没有出现 CUDA OOM
- 训练中每张 `3090 24GB` 的显存占用大约在 `20.3GB - 20.4GB`

当前解释：

- 在这台机器的当前软件栈下，`micro_batch_size=4` **并不会天然触发 OOM**
- 当前配置里的 `micro_batch_size=2` 更像偏保守的安全设置
- 如果后面要向原开源主线 SFT 超参靠拢，`micro_batch_size=4` 是值得正式重跑验证的

## 当前研究解释

### 当前 2x2 实验各自意味着什么

- `title_history2sid_off__desc_align_off`
  - 最适合作为 `repo-faithful SFT baseline`
- `title_history2sid_off__desc_align_p05`
  - 最适合作为当前的 `research-enhanced SFT`
  - 也是当前最强的 SFT-only 配方
- `title_history2sid_on__desc_align_off`
  - 用来测试在 repo-faithful baseline 上额外打开 `TitleHistory2Sid` 是否有帮助
- `title_history2sid_on__desc_align_p05`
  - 用来测试在 description-style alignment 打开后，`TitleHistory2Sid` 是否仍然有帮助

### 当前最重要的发现

在当前仓库、当前 SFT-only 口径下：

- 打开 description-style alignment 明显有帮助
- 在更强的 description-style alignment 设置下，再加入 `TitleHistory2Sid` 反而没有帮助

因此当前最强研究方向是：

- `TitleHistory2Sid` 保持关闭
- `desc alignment` 保持开启

## 复现成功标准

### 短期标准

短期内不应宣称“论文复现成功”。

短期成功标准更适合定义为：

- 找到稳定的 `repo-faithful baseline`
- 找到稳定的 `research-enhanced SFT` 配方
- 所有实验结果都被结构化记录

### 后续 RL 之后

更严肃的论文级复现判断，至少要满足：

- 在 `repo-faithful baseline` 和当前最优 SFT 配方上都继续跑 RL
- 与论文目标值直接比对 `HR@3/5/10` 和 `NDCG@3/5/10`
- 最好对最强配置至少补若干不同 seed，避免单次波动误导结论

### 实用误差判断

后续如果做论文级结果判断，可以先用一个实用标准：

- 差距大约 `<= 0.005`
  - 可以叫“基本复现”
- 差距大约在 `0.005 - 0.01`
  - 可以叫“接近复现，但需谨慎”
- 差距明显大于 `0.01`
  - 通常说明存在实质性 recipe 差异

但由于当前实验都是单次运行：

- 小差距不能过度解读
- 大差距也必须先确认是不是训练阶段口径不同造成的

## 推荐的下一步

### RL 之前

1. 将 `title_history2sid_off__desc_align_off` 固定为官方的 `repo-faithful SFT baseline`
2. 将 `title_history2sid_off__desc_align_p05` 固定为当前的最佳 `research-enhanced SFT`
3. 用更贴近原开源主线的 SFT 超参重跑这两条线，优先考虑：
   - `micro_batch_size=4`
   - 其余训练语义尽量不变
4. 当前阶段不再把 `TitleHistory2Sid` 作为主变量继续投入

### RL 恢复之后

先只推进两条线：

1. repo-faithful 线：
   - `title_history2sid_off__desc_align_off -> RL`
2. research-enhanced 线：
   - `title_history2sid_off__desc_align_p05 -> RL`

这是当前信息下最小、最高价值的对照集合。

## 当前仍缺失的重要信息

最值得继续补的三类信息是：

1. 原仓库里更丰富的 alignment / reasoning datasets 是否参与了论文实验
2. 是否存在 shell 命令、日志、checkpoint 命名等证据，能把论文主表结果绑定到某一条具体训练链
3. RL 最终正式复现时，除了 `num_generations=16` 之外，是否还需要恢复更多“原开源主线实际运行值”

## 后续继续取证用 Prompt

下面这段 prompt 可以直接继续拿去查原开源代码：

```markdown
请继续系统检查原开源 MiniOneRec 代码，但这次重点不是 alignment 本身，而是“论文主实验到底用了哪套脚本和哪条训练链”。

请务必基于代码、shell 脚本、README、默认参数、日志命名规则来确认，并附上文件路径、关键代码片段和你的判断依据。

我需要你重点确认以下问题：

1. 论文主实验更可能使用哪套脚本？
- `sft.py / rl.py / evaluate.py`
- 还是 `sft_gpr.py / rl_gpr.py / evaluate.py`
- 请给出判断依据，不要只猜

2. shell 脚本默认实际运行的是哪套参数？
- `sft.sh`
- `rl.sh`
- `evaluate.sh`
- 如果存在其他实验脚本，也请一起说明

3. `sft_gpr.py` 和 `sft.py` 的真实差异是什么？
- 多了哪些 dataset / task
- 默认启用了哪些在主线里没有启用的东西
- 哪些更接近论文里的 alignment / reasoning 描述

4. `rl_gpr.py` 和 `rl.py` 的真实差异是什么？
- 数据集组成
- reward 设置
- sampling / beam / num_generations
- 是否更接近论文中的 full-process alignment

5. 有没有任何证据表明论文主表的结果来自“实验支线”而不是“公开主线”？
请检查：
- README
- 注释
- 脚本命名
- wandb run name 规则
- 默认 output_dir
- 任何写着 gpr / align / preference / reasoning 的文件

6. 如果你能找到，请顺手确认：
- 原仓库里是否已有 Industrial 的 RL 最终结果文件
- 这些结果更像来自哪套脚本
- 是否能从 output 命名、日志、checkpoint 结构推断训练链

输出要求：
- 按“主线脚本 / 实验支线 / shell 默认行为 / 证据 / 结论”分节
- 每个结论后必须带文件路径
- 如果不能确定，请明确写“待确认”
- 最后单独给一句总结：
  - 你认为论文主实验更像是跑在主线，还是跑在实验支线
```

## 当前本地状态快照

写下这份进度表时，本地状态如下：

- 当前 commit：`493b73a`
- 本地已修改但未提交的跟踪文件：
  - `config/sft.yaml`
  - `experiment_results.csv`
  - `src/onerec/config.py`
  - `src/onerec/sft/pipeline.py`
- 另外还有一个未跟踪的本地 PDF：
  - `MiniOneRec An Open-Source Framework for Scaling Generative Recommendation.pdf`
