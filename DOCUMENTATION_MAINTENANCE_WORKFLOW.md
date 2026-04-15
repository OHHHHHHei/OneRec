# Documentation Maintenance Workflow（文档维护工作流）

Status（状态）: `canonical-policy（权威规范）`

Last updated（更新日期）: `2026-04-15`

## 目的

这份文档定义 OneRec 当前研究工程的文档维护规范。

目标只有三个：

1. 让“现在我们在做什么”只有一个权威入口。
2. 让“结果到底是什么”只有一个权威总账。
3. 让其他文档都各自回到清晰、稳定、不互相冲突的角色。

这份规范既服务于人，也服务于 agent（智能体）。
后续所有文档维护，都应优先遵守这里的角色定义和更新流程。

## 核心原则

### 原则 1：只允许一个 current-state source（当前状态源）

当前唯一 current-state source（当前状态源）是：

- [research-progress-log/CURRENT_STATE.md](/home/leejt/OneRec/research-progress-log/CURRENT_STATE.md)

任何其他文档都不应再复制一份完整“当前主线摘要”。

### 原则 2：只允许一个 experiment registry（实验总账）

当前唯一 experiment registry（实验总账）是：

- [experiment_results.csv](/home/leejt/OneRec/experiment_results.csv)

其他文档可以引用结果，但不应承担“唯一结果源”的角色。

### 原则 3：README 主要做 navigation（导航），不是做第二份总结

除了 run folder（运行目录）里的阶段快照 `README.md` 之外，目录级 `README.md` 应尽量只承担：

- navigation（导航）
- reading order（阅读顺序）
- role description（角色说明）
- link routing（链接跳转）

不再承担完整 current-state summary（当前状态摘要）。

### 原则 4：dated note（带日期笔记）默认不是活文档

只要文件名里带明确日期，默认就应被视为：

- snapshot（快照）
- discussion-only（仅讨论）
- archived reference（归档参考）

除非它被显式指定为当前权威入口，否则不应继续把它当成 live status doc（实时状态文档）。

### 原则 5：新建文档之前，先判断是否应该更新旧文档

任何 agent（智能体）在新建文档前，都必须先问自己：

1. 这个信息是否应该更新到 `CURRENT_STATE.md`？
2. 这个结果是否应该写入 `experiment_results.csv`？
3. 这个变化是否只是某个已有文档的一次增量更新？

如果答案是“是”，优先更新已有权威文档，而不是新建一个旁支文档。

## 当前文档角色总表

### A. 高频维护文档

#### 1. `research-progress-log/CURRENT_STATE.md`

- 角色：`canonical（权威）`
- 用途：唯一 current-state document（当前状态文档）
- 更新频率：高
- 必须维护的内容：
  - 当前问题
  - 当前方法骨架
  - baseline（基线）口径
  - strongest validated line（最强已验证主线）
  - 已证结论
  - 未证结论
  - 当前进行中的实验
  - 下一步 1 到 3 条动作

#### 2. `experiment_results.csv`

- 角色：`registry（总账）`
- 用途：唯一实验结果总账
- 更新频率：高
- 必须维护的内容：
  - run id（运行编号）
  - variant（变体）
  - 对照
  - 关键指标
  - verdict（结论）
  - 是否进入主线

### B. 中低频维护文档

#### 3. `idea-discovery/.../RESEARCH_DIRECTION.md`

- 角色：`reference（参考）`
- 用途：固定 `motivation -> idea`（动机到想法）的顺向逻辑
- 更新频率：低
- 什么时候更新：
  - 问题定义真的变了
  - 主方法叙事真的变了

#### 4. `idea-discovery/.../CURRENT_TASK_ALIGNMENT.md`

- 角色：`reference（参考）`
- 用途：固定长期核心问题，不同步实验结果
- 更新频率：低
- 什么时候更新：
  - 核心研究问题发生重排
  - 主执行问题从 A 切换到 B

#### 5. `idea-discovery/.../18_mgr_sid_v2_ambiguity_aware_method.md`

- 角色：`reference（参考）`
- 用途：方法叙事版说明
- 更新频率：低
- 什么时候更新：
  - 方法定义真的变了

#### 6. `idea-discovery/.../19_mgr_sid_current_method_code_aligned_formulas.md`

- 角色：`reference（参考）`
- 用途：代码对齐公式说明
- 更新频率：低
- 什么时候更新：
  - 训练目标
  - 图构建方式
  - loss（损失）接口
  - 关键配置含义
  真正发生了实现变动

#### 7. `research-progress-log/experiment_launches/README.md`

- 角色：`stage-index（阶段索引）`
- 用途：阶段索引，不是当前状态摘要
- 更新频率：中
- 什么时候更新：
  - 新 stage（阶段）开始
  - 某 stage 的角色发生收口或结束

#### 8. `research-progress-log/research_progress_log.tex`

- 角色：`milestone-narrative（里程碑叙事）`
- 用途：长篇叙事、里程碑沉淀、论文前整理材料
- 更新频率：低
- 不再承担：
  - daily sync（日常同步）
  - current-state source（当前状态源）

### C. 低频 / 条件性维护文档

#### 9. `refine-logs/README.md`

- 角色：`plan-index（计划索引）`
- 用途：活跃计划索引
- 更新频率：中低
- 什么时候更新：
  - 活跃分支计划切换
  - 某 plan（计划）从 active（活跃）变成 archive（归档）

#### 10. `research-progress-log/experiment_launches/<run>/README.md`

- 角色：`stage-snapshot（阶段快照）`
- 用途：记录单个 stage / run（阶段 / 运行）的发起背景和产物位置
- 更新频率：仅在该 run 活跃时更新
- 结束后策略：
  - 保留
  - 冻结
  - 不再当作当前状态入口

### D. 不要求持续维护的文档

这些文档应该保留，但不要求和主线同步更新：

- `discussion-only（仅讨论）` 文档
- `snapshot（快照）` 文档
- `archive（归档）` 目录
- 侧支 paper reading note（论文阅读笔记）
- 历史 tracker（历史跟踪器）
- 历史 postmortem（历史复盘）

## 允许使用的状态标签

后续新文档或被重构过的旧文档，尽量在开头使用下列状态标签之一：

- `canonical（权威）`
- `registry（总账）`
- `navigation（导航页）`
- `pointer（指针页）`
- `reference（参考）`
- `plan-index（计划索引）`
- `stage-index（阶段索引）`
- `discussion-only（仅讨论）`
- `snapshot（快照）`
- `archived（归档）`
- `canonical-policy（权威规范）`

推荐头部模板：

```md
Status（状态）: `reference（参考）`
Last updated（更新日期）: `2026-04-15`
```

如果是快照文档，推荐：

```md
Status（状态）: `snapshot（快照）`
Snapshot date（快照日期）: `2026-04-14`
```

## 目录级规则

### 根目录

根目录应只保留少量“跨目录入口级”文档：

- `README.md`
- `AGENTS.md`
- `PROJECT_WORKSPACE_MAP.md`
- `DOCUMENTATION_MAINTENANCE_WORKFLOW.md`
- `experiment_results.csv`

根目录不应该继续堆积大量 dated summary（带日期总结）文档。

根目录里其他结果类 CSV（如 `legacy_experiment_results.csv`、`sid_diagnostic_results.csv`）默认视为：

- auxiliary registry（辅助总账）或
- legacy snapshot（历史快照）

它们不是当前主线的权威结果入口。

### `research-progress-log/`

这里放：

- 当前状态入口
- 里程碑叙事
- stage index（阶段索引）
- 阶段运行记录
- 研究复盘快照

这里不放：

- 大量 plan（计划）
- 大量 related work（相关工作）
- 方法设计草稿

### `idea-discovery/`

这里放：

- 方向定义
- 方法设计
- 计划索引
- 讨论文档
- 支线探索
- 归档材料

这里不放：

- 当前 strongest result（最强结果）的权威口径
- 完整实验总账

### `research-progress-log/experiment_launches/`

这里的每个子目录默认都视为：

- historical stage record（历史阶段记录）

即使它曾经是 active run（活跃运行），在阶段结束后也不应再承担当前状态入口角色。

## 文档维护流水线

### 流程 A：新实验启动

当一个新 experiment（实验）被正式启动时：

1. 在 `research-progress-log/experiment_launches/<date>_<name>/` 建立 run folder（运行目录）。
2. 创建该 run 的 `README.md`，记录：
   - 目的
   - 变体定义
   - 配置
   - 脚本
   - 输出目录
   - 初始状态
3. 如果它属于活跃分支，更新：
   - `refine-logs/README.md`
4. 只有在它改变当前主线判断时，才更新：
   - `CURRENT_STATE.md`

### 流程 B：实验完成

当一个 experiment（实验）跑完后：

1. 先更新 `experiment_results.csv`
2. 再更新对应 run folder（运行目录）的 `README.md` / `RESULTS.md`
3. 如果结果改变了：
   - strongest validated line（最强已验证主线）
   - baseline 口径
   - 当前 active exploration（当前活跃探索）
   - 下一步动作
   则更新 `CURRENT_STATE.md`
4. 如果阶段索引发生变化，再更新：
   - `research-progress-log/experiment_launches/README.md`
5. 如果这是里程碑，再决定是否补写：
   - `research_progress_log.tex`

### 流程 C：方法实现变化

当代码里的方法真正变了：

1. 更新 `18_mgr_sid_v2_ambiguity_aware_method.md`
2. 更新 `19_mgr_sid_current_method_code_aligned_formulas.md`
3. 如果方法变化导致主问题重排，再更新：
   - `CURRENT_TASK_ALIGNMENT.md`

### 流程 D：研究方向切换

当项目从一个主问题切到另一个主问题时：

1. 更新 `CURRENT_STATE.md`
2. 更新 `CURRENT_TASK_ALIGNMENT.md`
3. 必要时更新 `RESEARCH_DIRECTION.md`
4. 将旧的计划、tracker（跟踪器）、快照文档降级或归档

### 流程 E：文档清理与归档

每次大阶段结束后，执行一次 doc cleanup（文档清理）：

1. 检查是否出现新的重复 current-state summary（当前状态摘要）
2. 给仍保留的 dated note（带日期笔记）补状态标签
3. 把不再活跃的 plan / tracker（计划 / 跟踪器）移入 archive（归档）
4. 确认目录级 `README.md` 仍然只做导航

## 严格禁止事项

后续 agent（智能体）维护文档时，禁止下面这些做法：

1. 新建第二份“当前状态总结”
2. 在多个 README 里重复维护 strongest result（最强结果）数字
3. 把 dated snapshot（带日期快照）继续当成活文档写
4. 把 discussion note（讨论笔记）写成事实源
5. 方法没变却新建另一份 method spec（方法说明）
6. 实验结果没先落到 `experiment_results.csv`，就先在 prose（文字）里到处引用

## 对 agent（智能体）的执行要求

以后 agent（智能体）在维护文档时，应严格按下面顺序执行：

1. 先判断这次变化属于：
   - current state（当前状态）
   - experiment result（实验结果）
   - method change（方法变化）
   - discussion（讨论）
   - snapshot（快照）
2. 优先更新已有权威文档，而不是新建新文档
3. 如果必须新建文档，先为它指定状态标签
4. 新文档创建后，要在相应导航页中补入口
5. 若文档已过时，要么降级为 `snapshot（快照）` / `discussion-only（仅讨论）`，要么移入 `archive（归档）`

## 当前建议的最小阅读链

如果今天有一个新 agent（智能体）进入仓库，最小阅读链只应该是：

1. [DOCUMENTATION_MAINTENANCE_WORKFLOW.md](/home/leejt/OneRec/DOCUMENTATION_MAINTENANCE_WORKFLOW.md)
2. [research-progress-log/CURRENT_STATE.md](/home/leejt/OneRec/research-progress-log/CURRENT_STATE.md)
3. [experiment_results.csv](/home/leejt/OneRec/experiment_results.csv)
4. [PROJECT_WORKSPACE_MAP.md](/home/leejt/OneRec/PROJECT_WORKSPACE_MAP.md)

需要方法细节时，再补：

5. [RESEARCH_DIRECTION.md](/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/RESEARCH_DIRECTION.md)
6. [CURRENT_TASK_ALIGNMENT.md](/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/CURRENT_TASK_ALIGNMENT.md)
7. [18_mgr_sid_v2_ambiguity_aware_method.md](/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/18_mgr_sid_v2_ambiguity_aware_method.md)
8. [19_mgr_sid_current_method_code_aligned_formulas.md](/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/19_mgr_sid_current_method_code_aligned_formulas.md)
