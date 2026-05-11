# D640 Seq2Graph-lite Graph Audit（图审计）

## Scope（范围）

- dataset（数据集）: `Industrial_and_Scientific`
- status（状态）: `completed（已完成）`
- role（角色）: engineering filter（工程过滤）, not scientific verdict（不是科学裁决）

## Settings（设置）

- `seq2g_mix_alpha = 0.35`
- `seq2g_context_topk = 32`
- `seq2g_candidate_topm = 32`
- `seq2g_direct_tau = 0.5`

## Graph Summary（图摘要）

| graph | nnz | connected_rate | overlap | novel_edge_ratio | rescue_edge_ratio | topk_expansion | |
|---|---:|---:|---:|---:|---:|---:|---|
| `coarse_purified` | 55336 | 0.9721 | 0.9721 | 0.0000 | 0.0000 | 0.0000 | |
| `coarse_seq2g_ctx_only` | 115907 | 0.9891 | 0.3640 | 0.5980 | 0.5407 | 1.2525 | |
| `coarse_seq2g_rel` | 115907 | 0.9891 | 0.3731 | 0.5902 | 0.5325 | 1.2362 | |
| `coarse_seq2g_rel_masked` | 114654 | 0.9891 | 0.3527 | 0.6049 | 0.6049 | 1.2533 | |

## Hotspot Visibility（热点可见性）

| graph | visible_fraction | direct_weak_visible | direct_zero_visible | predecessor_visible | predecessor_direct_zero_visible | |
|---|---:|---:|---:|---:|---:|---|
| `coarse_purified` | 0.1667 | 0.0000 | 0.0000 | 0.3333 | 0.0000 | |
| `coarse_seq2g_ctx_only` | 0.3333 | 0.1875 | 0.1875 | 0.7500 | 0.6000 | |
| `coarse_seq2g_rel` | 0.3667 | 0.2500 | 0.2500 | 0.8333 | 0.8000 | |
| `coarse_seq2g_rel_masked` | 0.2667 | 0.2500 | 0.2500 | 0.5833 | 0.8000 | |

## Top Rescued Hotspot Pairs（最典型补盲热点对）

### `coarse_seq2g_ctx_only`

| anchor | neighbor | semantic_sim | context_affinity | direct_support | baseline | delta | |
|---|---|---:|---:|---:|---:|---:|---|
| 3475: 3D Solutech Real Orange 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3469: 3D Solutech Real Pink 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 0.9935 | 0.3602 | 0.0000 | 0.0000 | 0.0185 | |
| 3494: 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 1552: 3D Solutech Teal Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 0.9906 | 0.3882 | 1.5000 | 0.0000 | 0.0127 | |
| 3494: 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 2660: 3D Solutech Real Grey 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 0.9908 | 0.3490 | 0.0000 | 0.0000 | 0.0114 | |
| 3494: 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3466: 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - RA-I00I-DQUY | 0.9918 | 0.3115 | 4.0000 | 0.0000 | 0.0102 | |
| 3494: 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3599: 3D Solutech Aqua Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 0.9910 | 0.2856 | 0.0000 | 0.0000 | 0.0094 | |
| 3522: 3D Solutech Real Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - ST176BLPLA | 2909: 3D Solutech Silver Metal 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - PLA175TCMS | 0.9920 | 0.2704 | 5.6000 | 0.0204 | 0.0084 | |
| 3522: 3D Solutech Real Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - ST176BLPLA | 3112: 3D Solutech Real Black 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - PLA175RBLK | 0.9937 | 0.3276 | 6.5000 | 0.0358 | 0.0063 | |
| 2697: 3D Solutech Real White 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3466: 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - RA-I00I-DQUY | 0.9912 | 0.2837 | 5.2143 | 0.0183 | 0.0046 | |
| 2697: 3D Solutech Real White 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3494: 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 0.9902 | 0.4822 | 9.9750 | 0.0484 | 0.0017 | |
| 3494: 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3453: 3D Solutech Real Gold 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 0.9923 | 0.2574 | 0.0000 | 0.0000 | 0.0000 | |
| 3494: 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 2993: 3D Solutech Blue 3D Printer Ultra PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 0.9909 | 0.2427 | 1.5000 | 0.0000 | 0.0000 | |
| 2697: 3D Solutech Real White 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 2660: 3D Solutech Real Grey 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 0.9943 | 0.2262 | 0.0000 | 0.0000 | 0.0000 | |

### `coarse_seq2g_rel`

| anchor | neighbor | semantic_sim | context_affinity | direct_support | baseline | delta | |
|---|---|---:|---:|---:|---:|---:|---|
| 3475: 3D Solutech Real Orange 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3469: 3D Solutech Real Pink 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 0.9935 | 0.3602 | 0.0000 | 0.0000 | 0.0321 | |
| 3494: 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 1552: 3D Solutech Teal Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 0.9906 | 0.3882 | 1.5000 | 0.0000 | 0.0208 | |
| 3522: 3D Solutech Real Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - ST176BLPLA | 3112: 3D Solutech Real Black 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - PLA175RBLK | 0.9937 | 0.3276 | 6.5000 | 0.0358 | 0.0168 | |
| 3494: 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3466: 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - RA-I00I-DQUY | 0.9918 | 0.3115 | 4.0000 | 0.0000 | 0.0148 | |
| 3494: 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 2660: 3D Solutech Real Grey 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 0.9908 | 0.3490 | 0.0000 | 0.0000 | 0.0144 | |
| 2697: 3D Solutech Real White 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3494: 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 0.9902 | 0.4822 | 9.9750 | 0.0484 | 0.0119 | |
| 3494: 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3599: 3D Solutech Aqua Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 0.9910 | 0.2856 | 0.0000 | 0.0000 | 0.0104 | |
| 3522: 3D Solutech Real Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - ST176BLPLA | 2909: 3D Solutech Silver Metal 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - PLA175TCMS | 0.9920 | 0.2704 | 5.6000 | 0.0204 | 0.0082 | |
| 3494: 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3453: 3D Solutech Real Gold 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 0.9923 | 0.2574 | 0.0000 | 0.0000 | 0.0076 | |
| 2697: 3D Solutech Real White 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3466: 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - RA-I00I-DQUY | 0.9912 | 0.2837 | 5.2143 | 0.0183 | 0.0038 | |
| 3494: 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 2993: 3D Solutech Blue 3D Printer Ultra PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 0.9909 | 0.2427 | 1.5000 | 0.0000 | 0.0000 | |
| 2697: 3D Solutech Real White 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 2660: 3D Solutech Real Grey 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 0.9943 | 0.2262 | 0.0000 | 0.0000 | 0.0000 | |

### `coarse_seq2g_rel_masked`

| anchor | neighbor | semantic_sim | context_affinity | direct_support | baseline | delta | |
|---|---|---:|---:|---:|---:|---:|---|
| 3494: 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 2660: 3D Solutech Real Grey 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 0.9908 | 0.3490 | 0.0000 | 0.0000 | 0.0393 | |
| 3475: 3D Solutech Real Orange 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3469: 3D Solutech Real Pink 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 0.9935 | 0.3602 | 0.0000 | 0.0000 | 0.0387 | |
| 3494: 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3599: 3D Solutech Aqua Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 0.9910 | 0.2856 | 0.0000 | 0.0000 | 0.0284 | |
| 3494: 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3453: 3D Solutech Real Gold 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 0.9923 | 0.2574 | 0.0000 | 0.0000 | 0.0208 | |
| 3494: 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 1552: 3D Solutech Teal Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 0.9906 | 0.3882 | 1.5000 | 0.0000 | 0.0000 | |
| 3494: 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3466: 3D Solutech Real Red 3D Printer PLA Filament 1.75MM, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) - RA-I00I-DQUY | 0.9918 | 0.3115 | 4.0000 | 0.0000 | 0.0000 | |
| 3494: 3D Solutech Real Yellow 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 2993: 3D Solutech Blue 3D Printer Ultra PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 0.9909 | 0.2427 | 1.5000 | 0.0000 | 0.0000 | |
| 2697: 3D Solutech Real White 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 2660: 3D Solutech Real Grey 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 0.9943 | 0.2262 | 0.0000 | 0.0000 | 0.0000 | |
| 3475: 3D Solutech Real Orange 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3442: 3D Solutech Real Purple 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 0.9922 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | |
| 2697: 3D Solutech Real White 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 2993: 3D Solutech Blue 3D Printer Ultra PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 0.9915 | 0.0000 | 2.0000 | 0.0000 | 0.0000 | |
| 2697: 3D Solutech Real White 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 3599: 3D Solutech Aqua Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 0.9912 | 0.0000 | 3.0000 | 0.0000 | 0.0000 | |
| 2697: 3D Solutech Real White 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 1552: 3D Solutech Teal Blue 3D Printer PLA Filament 1.75MM Filament, Dimensional Accuracy +/- 0.03 mm, 2.2 LBS (1.0KG) | 0.9911 | 0.0000 | 2.3333 | 0.0000 | 0.0000 | |

## Quick Read（快速结论）

- 如果 `coarse_seq2g_rel_masked`（带掩码的可靠性感知补盲粗图）在 `direct_zero_visible_fraction`（直接零连接可见率）和 `predecessor_sharing_direct_zero_visible_fraction`（前驱共享且直接零连接可见率）上明显更高，就说明它确实在补 blind spot（盲区），而不是只做全局加边。
- 如果 `coarse_seq2g_ctx_only`（仅上下文补盲粗图）有更高的 `topk_expansion_ratio`（邻域扩张率）但更低的 `rescue_edge_ratio`（补盲边比例），就说明 reliability（可靠性）和 mask（掩码）是必要的。
