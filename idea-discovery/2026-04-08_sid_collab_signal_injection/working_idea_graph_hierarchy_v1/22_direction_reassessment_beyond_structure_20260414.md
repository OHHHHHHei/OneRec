# 方向再审视：我们是不是把精力过度花在结构指标上了？（2026-04-14）

Status（状态）: `discussion-only（仅讨论）`

Discussion date: `2026-04-14`

这份文档记录的是一次 direction reassessment（方向再审视），不是 current-state summary（当前状态摘要）。

它适合在下面这些场景再读：

- 需要解释为什么项目从 structure-only thinking（仅结构思维）转向 codebook-space thinking（码本空间思维）
- 需要回看为什么 graph carrier（图载体）开始变成更高优先级

如果你只想知道“现在在做什么”，先读：

1. [CURRENT_STATE.md](/home/leejt/OneRec/research-progress-log/CURRENT_STATE.md)
2. [experiment_results.csv](/home/leejt/OneRec/experiment_results.csv)

## 写在前面

这份文档回应下面几类问题：

1. 我们现在是不是有点把研究重心放偏了？
2. 当前瓶颈到底更像是“SID 结构不够好”，还是“图设计 / 损失设计 / 系数设置还很粗糙”？
3. 我们之前很多系数是不是其实没有认真调过？
4. 现在继续沿着“结构指标更好”这条线盲目往下推，对不对？
5. 对一些和我们很接近的论文方法，是否应该从“受启发”升级到“直接复用模块”？
6. 如果要重排下一阶段重点，最该做什么？

这份文档不追求把所有细节都讲满，而是希望把判断逻辑讲清楚：  
**哪些结论已经有证据，哪些只是合理怀疑，哪些该进入下一步实验。**

---

## 一、先给总判断

### 结论 1：我们确实一度把重心放偏了

这个“偏”不是说研究方向彻底错了，而是说：

- 我们原本的核心目标应该是：  
  **利用图承载的协同信息，构造更好的 SID 码本空间，让下游 LLM 更好地完成推荐。**

- 但在最近一段时间里，我们有一段很明显的重心漂移：  
  **把 tokenizer 侧结构指标，当成了近似最终目标。**

具体表现是：

- 过于关注 `mean l2 leaves`、`H(l3|l1,l2)`、`pair retention`
- 一度把 `prefix stability`、`codebook drift` 当成近似 hard gate
- 讨论越来越多地围绕“空间离不离 baseline 近”，而不是“这个空间最后 evaluate 好不好”

这条偏移已经被纠正过一次，但我认为还需要再明确地写下来：

> **结构指标只是码本空间质量的一个观察角度，不是最终目标。**

---

### 结论 2：当前真正的瓶颈，很可能不只在“SID 结构”本身

我现在更倾向于这样理解当前局面：

- `v2` 已经证明：图协同信息注入 SID 是有价值的
- `R202a / R401b / R401d` 证明：单纯追 tokenizer 结构，可以把局部指标做得非常漂亮
- 但“结构更漂亮”并没有稳定地转成“下游更强”

这意味着：

> 当前瓶颈很可能不只是“SID 结构还不够好”，而是下面几类问题交织在一起：

1. 图本身设计还比较粗糙
2. 图信号注入方式还比较单一
3. 关键损失权重和图构建超参数几乎没认真探索
4. 我们借鉴论文时，吸收得偏保守，只拿了一个“灵感版本”

---

### 结论 3：如果继续只沿着“结构指标更强”这条线盲推，就不对了

这里我想把话说得比较明确：

**继续做 graph-informed SID 这条大方向，本身没有错。**  
因为 `v2` 已经证明，这条路能带来真实下游收益。

但如果下一步还是：

- 再造一个 tokenizer 变体
- 继续把结构指标压得更低
- 然后默认它应该更好

那就不太对了。

现在更合理的推进方式应该是：

1. 先让当前最强候选走完完整下游裁判
2. 然后把研究重点从“继续雕结构”切回到  
   **图设计、注入机制、和关键超参数是否根本没调对**

---

## 二、问题 1：我们是不是因为 SID 分析而把方向带偏了？

### 我的回答

**是，部分带偏了。**

但这个偏移是“方法重心临时失衡”，不是“主问题定义错了”。

### 为什么这么说

你们现在已经有非常明确的事实：

- strongest original SFT：`NDCG@10 = 0.10372`
- 当前 best `v2_on_p05 SFT`：`NDCG@10 = 0.10271`
- strongest original RL：`NDCG@10 = 0.10726`
- 当前 best `v2_on_p05 RL`：`NDCG@10 = 0.10432`

这些结果说明两件事：

1. **图协同信息注入 SID 不是没用。**  
   `v2` 至少已经把空间做到“接近原始 MiniOneRec baseline，且在 top-1/top-3 上更强”。

2. **当前真正没被解决的是：**
   为什么 tokenizer 结构继续改善以后，下游却没有稳定同步变强。

而我们最近一段时间做的很多事，实际上是在问：

- prefix 稳不稳
- pair retention 高不高
- codebook drift 大不大

这些问题当然有解释价值，但它们并不是你真正想回答的核心问题。

所以最准确的说法不是：

> “SID 分析错了”

而是：

> “SID 分析在最近这段时间被过度上升成了优化目标，而不是解释工具。”

---

## 三、问题 2：目前瓶颈更像是结构问题，还是图/损失/系数问题？

### 我的回答

我现在更倾向于：

> **当前瓶颈更像是“图设计 + 注入方式 + 超参数设置”这一整套还比较粗糙，而不只是‘结构本身还不够好’。**

### 证据 1：最强 `v2` 已经说明“结构不是全错”

`v2` 的 downstream 结果说明：

- 它不是一个失败空间
- 它已经能在 `@1/@3` 上超过 original
- 它在完整 `SFT -> RL` 链条里是能活下来的

这意味着你们当前主线不是“完全无效的方法”，而是：

> 已经能工作，但还没有把图协同信息的价值释放干净

如果是这样，最自然的怀疑就不该只是：

- “结构还不够好”

而应该包括：

- “图是不是构得太粗”
- “loss 是不是太单调”
- “系数是不是根本没调到位”

### 证据 2：结构指标和下游结果已经明显脱钩

你们现在已经看到一个很清楚的现象：

- `R202a` 结构好于 `v2`
- `R401b/R401d` 结构又明显好于 `R202a`
- 但我们并不能因此推出它们一定下游更强

这说明：

> **当前结构指标最多是在告诉我们“空间长什么样”，而不是在告诉我们“空间最后一定更适合 LLM 学推荐”。**

既然如此，就不能再把主要精力放在“继续把这些结构指标推高”上。

### 证据 3：当前图注入方式本身就很轻

当前 tokenizer 训练里，图信息进入主损失的方式，本质上还是图平滑：

- `coarse_weight * coarse_graph_loss`
- `mid_weight * mid_graph_loss`
- `local_weight * local_graph_loss`
- 再叠加两个 semantic retention 项

对应代码在：
[train_v2.py](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/train_v2.py:462)

这类图平滑的优点是：

- 简洁
- 便宜
- 容易和 RQ-VAE 兼容

但缺点也非常明显：

- 只能“拉近邻居”
- 很难“推远不该挤在一起的非邻居”
- 很难直接控制 codebook 的全局布局
- 对下游 token 学习难度没有直接约束

所以如果结构指标不再稳定转化为下游收益，这并不奇怪。

---

## 四、问题 3：我们之前很多系数是不是其实没有认真调过？

### 我的回答

**是，而且这是一个非常真实的问题。**

这不是一种感觉，而是从配置文件里能直接看出来。

### 现在反复沿用的核心数值

工业线很多 tokenizer 配置里，下面这组数值几乎是长期固定的：

- `coarse_weight = 0.05`
- `mid_weight = 0.15`
- `local_weight = 0.05`
- `semantic_coarse_weight = 0.05`
- `semantic_mid_weight = 0.025`
- `graph_scale_min = 0.5`
- `graph_scale_max = 1.5`
- `semantic_scale_min = 0.5`
- `semantic_scale_max = 1.5`
- `coarse_min_weight = 2.0`
- `local_min_weight = 1.0`
- `anchor_topk = 32`

这些值在下面这些配置里基本是一路沿用的：

- [sid_train_industrial_mgr_sid_tokenizer_v2.yaml](/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_tokenizer_v2.yaml)
- [sid_train_industrial_mgr_sid_stage2_r202a.yaml](/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_stage2_r202a.yaml)
- [sid_train_industrial_mgr_sid_stage2_r205.yaml](/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_stage2_r205.yaml)
- [sid_train_industrial_mgr_sid_stage3_r401b_g005.yaml](/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_stage3_r401b_g005.yaml)
- [sid_train_industrial_mgr_sid_stage3_r401d_g005_a005.yaml](/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_stage3_r401d_g005_a005.yaml)

### 真正改过的东西其实不多

你们这一路真正显式动过的，主要是：

- `R202b_retry075` 把 `coarse_weight` 从 `0.05` 调到 `0.075`
- Stage-3 里试了 `prefix retention` 和 `codebook anchor`

也就是说：

> **我们在设计 tokenizer 变体上花了很多精力，但在很多基础系数上，探索密度其实并不高。**

这会直接带来一个研究风险：

- 如果最后结果不好，
- 我们很容易误以为“方向不行”，
- 但其实也可能只是“这套图信号和损失权重根本没调到合适的工作区间”。

### 我的判断

我不认为下一步应该立刻去做“大规模盲扫参数”，那会很散。  
但我认为：

> **在当前 SFT 结果出来之后，做一组高价值、小规模、围绕图设计和 loss 配比的 sweep，是非常有必要的。**

---

## 五、问题 4：我们现在盲目往下推进的方向是正确的吗？

### 我的回答

如果“盲目往下推进”指的是：

- 继续以结构指标为主导
- 再造更多 tokenizer 变体
- 默认结构更强就应该更好

那我认为 **不正确**。

但如果“往下推进”指的是：

- 先让当前最强候选走完 full `SFT -> evaluate`
- 然后把重心转回图设计、损失设计、超参数和模块移植

那我认为 **是正确的**。

### 为什么不能继续结构优先

因为我们现在已经有足够多的证据说明：

1. 结构好，不等于下游一定更好
2. 离 baseline 近，不等于空间一定更好
3. tokenizer 侧的解释指标不能代替最终裁判

所以如果下一步继续停留在：

- `R401e`
- `R401f`
- `R402x`

这种“继续细修结构”的节奏上，
很可能就会进入一种低信息增量循环。

### 为什么也不能直接否掉这条大方向

因为 `v2` 已经给了你们一个非常重要的事实：

> 图协同信息确实可以把 SID 空间改造成一种对下游有帮助的空间。

所以真正该被否掉的不是“graph-informed SID”这条主方向，  
而是：

> **把 tokenizer 结构指标误当成最终主目标的推进方式。**

---

## 六、问题 5：论文借鉴是不是不该止步于 inspired，而要考虑直接复用？

### 我的回答

**完全同意。**

而且我觉得这可能正是你们下一阶段最值得做的事情之一。

不是所有论文都该直接搬，但有些和你们当前瓶颈高度对齐的模块，确实不该只停留在“受启发”。

### 我建议优先考虑“直接模块化复用”的几条线

#### 1. `FaGSP`：不要只拿一个中频图

你们现在拿到的是：

- `FaGSP` 的频率直觉
- 然后实现了一个 `fagsp_mid_base`

这当然有价值，但太轻了。

`FaGSP` 更完整的启发是：

- 不同频段承载不同协同成分
- 高频/低频/层次邻域应该联合考虑

所以如果以后要认真升级，我更建议：

> 不要只保留一个“中频图”，而是考虑把 `FaGSP` 的完整滤波思想更认真地移进 graph bank。

具体可以落成两种形式：

- 用完整的多频段视图重新定义 `G_coarse / G_mid / G_local`
- 或者直接用多频段线性组合作为图信号源，而不是单独一张 `fagsp_mid_base`

#### 2. `GraphDA`：直接借图去噪与增强，不要只在原图上做纯化

现在你们的图纯化还比较基础，更多是：

- 最小边权过滤
- 语义 anchor 混合

但 `GraphDA` 这类工作提醒了一件更系统的事：

> 协同图本身是 noisy（有噪声）的，先去噪、再增强图，可能比后面再补很多正则更值。

这类模块非常适合你们现在的阶段，因为它直接作用在图构建入口，而不会把整个 tokenizer pipeline 搞复杂。

#### 3. `CAGCN`：把拓扑可靠性直接写进边权

`CAGCN` 的价值不在于“要不要用它的 GNN”，而在于：

> 它给了一个很强的提醒：  
> 不同邻居边对协同推荐的价值不同，图边应该有可靠性分数，而不是一视同仁。

这和你们现在的图非常契合。

你们可以完全不引入它的整套模型，只移植它的核心思想：

- 先给 item-item 或 bipartite 邻居边算一个 topology-aware 可靠性
- 再用这个可靠性去重权 `G_coarse` 或 `G_mid`

这比现在单纯的权重阈值过滤更像“图设计升级”，而不是“小修小补”。

#### 4. `Collaboration and Transition`

这篇对你们尤其值得借。

因为你们现在其实是把：

- `coarse collaborative`
- `local transition`

分成了两张图，然后希望 `L1/L3` 去各自吸收。

这当然合理，但它也可能太分裂了。

这篇工作的核心提醒是：

> 协同关系和转移关系，应该是被联合校准的，不是简单并列摆着。

如果要做直接复用，我觉得最自然的切口不是搬它的整个下游模型，而是：

- 用它的“transition-aware distillation”思想，重构你们的 `G_local`
- 或者干脆构一张“collaboration-transition mixed mid/local graph”

#### 5. `ReSID`

这条我会谨慎一点。

`ReSID` 很值得借，但我更建议借它的**目标定义**，而不是一上来搬它整套 pipeline。

它真正对你们有价值的是这句话：

> tokenizer 不只要语义合理，还要让自回归预测更容易

你们现在的 `learnability probe` 已经在往这个方向靠了。  
下一步更合理的是：

- 先把这种思路升级成明确的训练约束
- 而不是急着把 `ReSID` 整套结构照搬进来

---

## 七、问题 6：如果现在要重排优先级，应该怎么排？

### 我的回答

我会把下一步重排成下面这个顺序。

### 第一优先级：让当前 `R401b / R401d` 跑完 full `SFT -> evaluate`

原因非常简单：

- 它们已经是当前最强 tokenizer 候选
- 结构和条件可预测性都已经很强
- 再不看 downstream，所有讨论都只是半截

这一步不是“可选项”，而是必须完成的裁判。

### 第二优先级：做一组小而狠的图/损失/系数重排实验

不是做海量 sweep，而是做几组高信息量实验。

我最建议优先扫的，不是 retention，而是下面这些：

1. 图损失配比  
   先围绕 `coarse / mid / local / semantic_coarse / semantic_mid` 做很小规模矩阵

2. `G_mid` 构建参数  
   比如：
   - `band_low`
   - `band_high`
   - `spectral_rank`
   - 是否用 `gsprec_mid_prism` 代替 `fagsp_mid_base`

3. 图纯化与重权  
   - `coarse_min_weight`
   - `local_min_weight`
   - `anchor_topk`
   - `semantic_mix`

这些比继续造新的 retention 分支更接近你现在真正怀疑的瓶颈。

### 第三优先级：认真做一次“模块移植型”图设计升级

如果当前 SFT 结果说明：

- 现有 tokenizer 线已经很接近天花板
- 但还差最后一点

那我认为下一条最值得押注的线不是“更多结构微调”，而是：

> **做一个真正的 graph bank 升级版。**

我个人最看好的顺序是：

1. `GraphDA / CAGCN` 风格的图边去噪与重权
2. 更完整的 `FaGSP` 多频段模块
3. collaboration + transition 的联合图设计

### 第四优先级：再回来审问 claim

等结果出来以后，再认真做：

- `v2_uniform`
- hierarchy claim 是否真的成立
- warm-start 到底贡献了多少

这些都重要，但我不认为它们是现在最先要打的牌。

---

## 八、把你的问题压成一句最核心的话

如果把你这次的问题压缩成一句，我觉得其实就是：

> 我们现在是不是把“SID 结构长什么样”当得太重，而忽略了“为什么这套图设计、损失设计和参数设置未必已经给了 SID 最好的训练条件”？

我的回答是：

**是。**

而这件事带来的真正行动含义不是：

- 放弃 graph-informed SID

而是：

- 不再把结构指标当最终目标
- 不再默认当前图设计和损失设计已经足够成熟
- 对更接近问题本体的论文模块，开始考虑“直接复用”，而不是只停在 inspired

---

## 九、最终结论

### 9.1 我现在最明确的判断

1. 我们当前这条大方向没有错。  
   `v2` 已经证明图协同信息可以改善 SID 码本空间，并给下游带来真实收益。

2. 但我们最近一段时间的推进方式有点偏。  
   过度围绕结构指标和稳定性指标打转，弱化了“最终以 evaluate 选空间”的主线。

3. 当前瓶颈很可能不只是结构本身，而是图设计、注入方式和超参数设置都还比较粗。  
   尤其很多关键系数几乎没有认真扫过。

4. 下一阶段最值得做的，不是再继续盲造 tokenizer 结构变体。  
   而是：
   - 先完成当前最强候选的 full downstream 裁判
   - 然后把注意力切回 graph bank、loss 配比和高价值模块移植

### 9.2 如果只保留一句工作建议

> **完成当前 `R401b / R401d` 的 full `SFT -> evaluate` 之后，不要再默认“更好的结构指标”就是主要突破口；下一阶段更应该认真检查图设计、关键损失权重和可直接移植的论文模块，是不是才是真正卡住我们的瓶颈。**

---

## 附：文中直接相关的本地证据入口

- 当前图设计复盘：
  [21_graph_design_review_20260414.md](/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/21_graph_design_review_20260414.md)

- 当前方法与代码对齐公式：
  [19_mgr_sid_current_method_code_aligned_formulas.md](/home/leejt/OneRec/idea-discovery/2026-04-08_sid_collab_signal_injection/working_idea_graph_hierarchy_v1/19_mgr_sid_current_method_code_aligned_formulas.md)

- 深度项目状态复盘：
  [DEEP_REVIEW_MGR_SID_PROJECT_STATE_20260414.md](/home/leejt/OneRec/research-progress-log/DEEP_REVIEW_MGR_SID_PROJECT_STATE_20260414.md)

- 统一实验台账：
  [experiment_results.csv](/home/leejt/OneRec/experiment_results.csv)

- 当前图移植实现：
  [paper_transplants.py](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/paper_transplants.py)
  [transplanted_graph_bank.py](/home/leejt/OneRec/src/onerec/experiments/mgr_sid/transplanted_graph_bank.py)
