# OneRec（第三轮重构版）

本仓库当前主目标是：在保持原版 `E:/MiniOneRec` 训练语义的前提下，固定使用 `bash + yaml` 启动主链路，并降低目录复杂度。

主链路为：

`preprocess -> embed -> sid -> convert -> sft -> rl -> evaluate`

其中高频训练入口固定为三条命令：

```bash
bash sft.sh
bash rl.sh
bash evaluate.sh
```

## 1. 现在的项目结构

主实现集中在 `src/minionerec/flows`：

- `src/minionerec/flows/sft`：SFT 配置、数据集、训练管线
- `src/minionerec/flows/rl`：RL 配置、数据集、奖励、训练管线
- `src/minionerec/flows/evaluate`：约束解码、评估、split/merge/metrics

入口与兼容层：

- `src/minionerec/cli`：统一 CLI 调度
- `src/minionerec/compat`：旧参数到新配置的映射
- 根目录 `sft.py / rl.py / evaluate.py`：保留兼容入口（薄包装）
- 根目录 `sft.sh / rl.sh / evaluate.sh`：推荐入口（固定）

配置主路径：

- `flows/sft/default.yaml`
- `flows/rl/default.yaml`
- `flows/evaluate/default.yaml`

## 2. 你只需要改哪里

日常训练只改 YAML，不改命令。

1. 改 `flows/sft/default.yaml` 的模型、数据路径、训练超参
2. 改 `flows/rl/default.yaml` 的奖励与 RL 超参
3. 改 `flows/evaluate/default.yaml` 的评估超参与输出路径

多卡由 YAML 控制（`runtime.cuda_visible_devices`、`runtime.nproc_per_node` / `runtime.num_processes`）。

## 3. 运行方式

默认运行：

```bash
bash sft.sh
bash rl.sh
bash evaluate.sh
```

指定配置文件运行：

```bash
bash sft.sh flows/sft/default.yaml
bash rl.sh flows/rl/default.yaml
bash evaluate.sh flows/evaluate/default.yaml
```

如需 override，仍可在命令末尾追加（兼容保留，不是主推荐）。

## 4. 校验与排障

本地静态校验：

```bash
python parity_check.py
python -m unittest discover -s tests/unit -v
python -m minionerec.cli.main --help
```

远程排障时请优先提供：

1. 启动命令（完整一行）
2. 使用的 YAML 文件内容（可脱敏路径）
3. 首个报错堆栈（从 `Traceback` 开始）
4. 训练/评估 summary 行（脚本启动时打印的那一行）

## 5. Legacy 说明

非主线历史代码统一归档在 `legacy/`，不参与主链路执行。  
主链路禁止 import `legacy/`。
