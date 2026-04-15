# TAGCF 支链 M0：离线属性拓扑构图小实验

## 目的

这轮不是训练新的 `SID tokenizer`（SID 分词器），而是先回答一个更基础的问题：

> 我们能不能仅用 `title + description`（标题 + 描述），离线构出一张健康的 `item-attribute-item`（物品-属性-物品）属性拓扑图？

由于当前环境没有可直接使用的 `OpenAI API`（开放人工智能接口）密钥，
这一轮先跑的是 **offline text-phrase surrogate**（离线文本短语替代版）：

- `R500_attr_raw_textphrase`
- `R501_attr_fused_textphrase`
- `R502_attr_heuristic_title`

它们不是最终 `LLM-attribute`（大语言模型属性）版本，但足够回答：

- 属性拓扑图能不能先构起来
- `raw / fused / heuristic` 三种方式，哪种更像样

## 运行脚本

- [experiment_mgr_sid_tagcf_m0_attribute_graphs.py](/home/leejt/OneRec/scripts/experiment_mgr_sid_tagcf_m0_attribute_graphs.py)

运行命令：

```bash
source /home/leejt/miniconda3/etc/profile.d/conda.sh
conda activate MiniOneRec
python scripts/experiment_mgr_sid_tagcf_m0_attribute_graphs.py
```

## 输出目录

- 总目录：
  [results/tagcf_branch/20260415_m0_attribute_graphs](/home/leejt/OneRec/results/tagcf_branch/20260415_m0_attribute_graphs)

- 总表：
  [summary_table.csv](/home/leejt/OneRec/results/tagcf_branch/20260415_m0_attribute_graphs/summary_table.csv)

- 三个子运行：
  - [R500_attr_raw_textphrase](/home/leejt/OneRec/results/tagcf_branch/20260415_m0_attribute_graphs/R500_attr_raw_textphrase)
  - [R501_attr_fused_textphrase](/home/leejt/OneRec/results/tagcf_branch/20260415_m0_attribute_graphs/R501_attr_fused_textphrase)
  - [R502_attr_heuristic_title](/home/leejt/OneRec/results/tagcf_branch/20260415_m0_attribute_graphs/R502_attr_heuristic_title)

每个子目录都包含：

- `item_attributes.jsonl`
- `attribute_vocab.json`
- `item_attribute_graph.npz`
- `attribute_preview.csv`
- `summary.json`

## 核心结果

| Variant | Coverage（覆盖率） | Unique Attrs（唯一属性数） | Connected Rate（连通率） | Largest Component（最大连通分量占比） | Neighbor Overlap with `fagsp_mid_base`（与当前 `G_mid` 邻居重合） |
|---|---:|---:|---:|---:|---:|
| `R500_attr_raw_textphrase` | `0.9997` | `9602` | `0.9916` | `0.9824` | `0.0110` |
| `R501_attr_fused_textphrase` | `0.9927` | `3254` | `0.9927` | `0.9835` | `0.0110` |
| `R502_attr_heuristic_title` | `1.0000` | `9681` | `0.7911` | `0.6356` | `0.0080` |

## 结果解读

### 1. `R501 fused`（融合版）是目前最像样的属性拓扑候选

它的特点是：

- 覆盖率仍然很高：`99.27%`
- 唯一属性数明显收缩：`3254`
- 图保持高连通：
  - `connected_item_rate = 0.9927`
  - `largest_component_ratio = 0.9835`

这说明：

> 过滤与融合（filtering and fusion，过滤与融合）是有必要的，而且能把原始文本短语压成一张更紧凑、更像图的结构。

### 2. `R500 raw`（原始短语版）能成图，但噪声更重

它的图健康性其实不差，说明：

- `title + description` 本身就足够提供大量属性候选

但它的问题是：

- 唯一属性太多：`9602`
- 属性空间明显更碎

也就是说：

> `raw` 可以作为上游候选池，但不适合直接拿来当最终属性图。

### 3. `R502 heuristic`（启发式控制组）明显更弱

它虽然表面覆盖率是 `100%`，但图结构明显差：

- 连通率只有 `0.7911`
- 最大连通分量只有 `0.6356`

这说明：

> 只用简单标题短语，确实很容易得到一张“每个 item（物品）都有点标签，但图整体不成形”的弱图。

这对后面很重要，因为它支持了一个方向性判断：

> 价值可能不在“随便加点文本标签”，而在“更认真地构造属性拓扑”。

## 当前判断

这轮 `M0` 的结论是：

- 属性拓扑图 **可以构出来**
- 而且 `R501 fused`（融合版）已经像一张真正可用的候选 `G_mid`（中尺度图）
- 所以下一步最自然的是：
  - `R510`
  - 把 `G_attr_fused` 直接替换当前 `G_mid`

## 当前限制

这一轮还不是最终 `TAGCF` 式 `LLM-attribute`（大语言模型属性）版本。

当前限制是：

- 没有在线 `LLM API`（大语言模型接口）可直接调用
- 所以 `R500/R501` 是离线文本短语替代版

这不影响当前判断，因为这轮的目标本来就是：

> 先验证“属性拓扑图能不能成立”，而不是先证明“必须由 `LLM` 来抽属性”。

## 下一步建议

1. 先用 `R501_attr_fused_textphrase` 推 `R510`
   - `G_mid <- G_attr_fused`
2. 如果 `R510` 看起来太激进
   - 再推 `R511`
   - `G_mid <- mix(fagsp_mid_base, G_attr_fused)`
3. 只有在这两步有正信号后
   - 再考虑真正的 `LLM-attribute` 版本

