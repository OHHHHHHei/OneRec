# OneRec（二次重构版）

本仓库当前以 **SFT -> RL -> Evaluate** 主链路为优先，目标是与原始 `E:/MiniOneRec` 保持训练语义一致，并降低目录割裂。

## 1. 主线目录

核心实现集中在：

- `src/minionerec/flows/sft`
- `src/minionerec/flows/rl`
- `src/minionerec/flows/evaluate`

旧路径（`training/*`、`data/datasets/*`、`evaluation/*`）保留为兼容 re-export，真实实现源已收敛到 `flows/*`。

## 2. 配置目录

主配置路径：

- `flows/sft/default.yaml`
- `flows/rl/default.yaml`
- `flows/evaluate/default.yaml`

旧配置路径 `configs/stages/*` 仍可用，但属于兼容路径。

## 3. 运行方式

兼容方式（推荐你当前服务器继续用）：

```bash
bash sft.sh
bash rl.sh
bash evaluate.sh
```

统一 CLI：

```bash
python -m minionerec.cli.main sft --config flows/sft/default.yaml
python -m minionerec.cli.main rl --config flows/rl/default.yaml
python -m minionerec.cli.main evaluate --config flows/evaluate/default.yaml
```

## 4. 一致性检查

新增：

- `parity_matrix.md`：记录与原版关键参数对齐基线。
- `parity_check.py`：静态检查 flow yaml 与 legacy 参数映射是否满足基线。

执行：

```bash
python parity_check.py
```

## 5. Legacy 归档

非主线历史文件已迁入 `legacy/`，说明见：

- `legacy/README.md`

主链路禁止 import `legacy/`。

