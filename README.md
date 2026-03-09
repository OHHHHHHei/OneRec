# OneRec

这是当前正式主仓。项目只保留一套主实现，主流程为：

`preprocess -> embed -> sid-train -> sid-generate -> convert -> sft -> rl -> evaluate`

正式使用方式固定为 `bash + yaml`。你主要修改的是 `config/*.yaml`，主要运行的是根目录这些脚本：

```bash
bash preprocess_amazon18.sh
bash preprocess_amazon23.sh
bash text2emb.sh
bash sid_train.sh
bash sid_generate.sh
bash convert.sh
bash sft.sh
bash rl.sh
bash evaluate.sh
```

## 目录结构

核心目录只有三块：

- `src/onerec/`
  - `preprocess/`：原始 Amazon 数据处理
  - `sid/`：文本 embedding、量化训练、SID 生成
  - `convert/`：把 `.inter/.item.json/.index.json` 转成训练与评估 CSV
  - `sft/`：SFT 数据集、token 扩展、训练流程
  - `rl/`：RL 数据集、reward、训练器、约束生成
  - `evaluate/`：评估数据集、约束解码、merge、metrics
  - `utils/`：IO、日志、随机种子、缓存、模板配置渲染
  - `main.py`：统一 stage runner
- `config/`
  - `sft.yaml`
  - `rl.yaml`
  - `evaluate.yaml`
  - `datasets.yaml`
  - `preprocess_amazon18.yaml`
  - `preprocess_amazon23.yaml`
  - `embed.yaml`
  - `sid_train.yaml`
  - `sid_generate.yaml`
  - `convert.yaml`
  - `zero2_opt.yaml`
- `data/`
  - 数据、索引、`info` 文件、训练与评估 CSV

## 配置方式

SFT、RL、Evaluate 现在使用模板化 YAML。你会在配置里看到这种写法：

```yaml
data:
  train_file: ./data/Amazon/train/%{split_stem}.csv
  category: "%{category}"
output:
  output_dir: "./output/sft_%{category}_refactor"
```

这些占位符会在 shell 入口里根据 `config/datasets.yaml` 自动展开。

当前内置了两个数据集 key：

- `industrial`
- `office`

映射定义在 `config/datasets.yaml`。

## 如何修改配置

直接改对应 YAML：

- SFT：`config/sft.yaml`
- RL：`config/rl.yaml`
- Evaluate：`config/evaluate.yaml`
- 数据预处理与 SID 阶段：改 `config/` 下对应文件

常改参数：

- `model.base_model`
- `training.batch_size / micro_batch_size / num_epochs / learning_rate`
- `training.eval_step`
- `runtime.cuda_visible_devices`
- `runtime.nproc_per_node` 或 `runtime.num_processes`
- `output.output_dir`

如果你传了数据集 key，数据集相关路径会按模板自动替换；训练超参数仍然来自对应 stage 的 YAML。

## 如何运行

默认数据集是 `industrial`：

```bash
bash sft.sh
bash rl.sh
bash evaluate.sh
```

显式指定数据集：

```bash
bash sft.sh industrial
bash rl.sh industrial
bash evaluate.sh industrial
```

评估时也可以显式指定要评哪个阶段的 checkpoint：

```bash
bash evaluate.sh sft industrial
bash evaluate.sh rl industrial
```

也可以继续追加 override：

```bash
bash sft.sh office training.num_epochs=3 training.batch_size=512
bash rl.sh industrial training.num_generations=4
bash evaluate.sh rl office num_beams=48
```

## 日志排查

如果你要把服务器日志发给我定位问题，优先保留这些内容：

- 脚本开头的 summary 行
- 第一段 `Traceback`
- 进度条附近的 warning
- evaluate 的 `constraint mismatch summary`
- 你当前使用的 YAML 关键字段

## 本地静态检查

```bash
python parity_check.py
python -m unittest discover -s tests/unit -v
PYTHONPATH=./src python -m onerec.main --help
```

当前主仓不再保留旧 `minionerec` 兼容入口，只服务当前 OneRec 主链路。
