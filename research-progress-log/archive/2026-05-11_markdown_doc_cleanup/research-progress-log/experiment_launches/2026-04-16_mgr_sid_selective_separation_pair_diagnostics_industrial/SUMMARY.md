# D600 Pair Diagnostics Summary（D600 物品对诊断摘要）

- semantic pair count（语义近邻物品对数量）: `82596`
- graph-weak threshold（图弱连接阈值）: `0.002001`

## semantic-near + graph-non-neighbor（语义接近 + 图上无邻接）
- pair count（物品对数量）: `74332`
- pair ratio（物品对比例）: `0.8999`
- item coverage rate（物品覆盖率）: `0.9995`
- mean semantic sim（平均语义相似度）: `0.0322`
- mean user overlap（平均用户重叠）: `0.0021`
- mean reliability（平均可靠性）: `0.0321`

## semantic-near + graph-weak（语义接近 + 图弱连接）
- pair count（物品对数量）: `2066`
- pair ratio（物品对比例）: `0.0250`
- item coverage rate（物品覆盖率）: `0.3733`
- mean semantic sim（平均语义相似度）: `0.0322`
- mean graph affinity（平均图亲和度）: `0.001478`
- mean reliability（平均可靠性）: `0.0320`

## Recommendation（推荐）
- preferred first pair rule（优先物品对规则）: `semantic_near_graph_weak`
- reason（原因）: graph-non-neighbor pairs are too broad for a first training pass, so the better initial rule is semantic-near + graph-weak, which stays closer to the current collaborative support boundary

## Files（文件）
- `D600_all_non_neighbor_pairs.csv`
- `D600_all_graph_weak_pairs.csv`
- `D600_pair_summary.json`
- `D600_top_non_neighbor_pairs.csv`
- `D600_top_graph_weak_pairs.csv`
- `D600_top_semantic_pairs.csv`
