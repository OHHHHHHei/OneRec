# OneRec 主线仓库

这是基于原始 MiniOneRec 项目重构后的独立主仓库。

当前保留的核心流程不变：

```text
preprocess -> embed -> SID -> convert -> SFT -> RL -> evaluate
```

重构的重点是：

- 将主线实现迁移到 `src/minionerec`
- 提供统一 CLI 和 YAML 配置
- 保留旧入口兼容层，方便服务器渐进迁移
- 将非主线代码剥离到归档区域
- 让本地开发以契约检查和 smoke test 为主

## 1. 统一入口

新的统一入口如下：

```bash
python -m minionerec.cli.main <stage> --config <yaml> [overrides...]
```

支持的阶段：

- `preprocess`
- `embed`
- `sid-train`
- `sid-generate`
- `convert`
- `sft`
- `rl`
- `evaluate`

示例：

```bash
python -m minionerec.cli.main sft --config configs/stages/sft/default.yaml
python -m minionerec.cli.main rl --config configs/stages/rl/default.yaml
```

## 2. 目录结构

主线源码位于：

```text
src/minionerec/
  cli/           统一命令入口
  common/        IO、日志、路径解析、校验、tokenizer 辅助
  config/        YAML 配置加载和 schema
  preprocess/    Amazon 数据预处理
  sid/           文本 embedding、量化训练、SID 生成
  data/          数据契约、convert、dataset 构造
  training/sft/  SFT 主流程
  training/rl/   RL 主流程
  training/cf/   SASRec 相关实现
  evaluation/    约束解码和指标计算
  compat/        旧参数和旧入口兼容层
```

另外保留：

- `archive/`：非主线代码和历史文档
- `scripts/`：便于服务器调用的薄包装脚本
- 根目录旧脚本：兼容原始运行方式

## 3. 兼容入口

这些旧入口仍然可用，但内部已经转发到新主线：

- `sft.py`
- `rl.py`
- `evaluate.py`
- `convert_dataset.py`
- `split.py`
- `merge.py`
- `calc.py`

新开发优先使用统一 CLI，兼容入口仅用于平滑迁移。

## 4. 文档

仓库内主要文档包括：

- `REFACTORED_PROJECT_GUIDE.md`：当前项目结构、运行逻辑、操作方式
- `PROJECT_UNDERSTANDING.md`：项目主线理解摘要
- `archive/README.md`：归档目录说明

## 5. 本地与远程使用建议

本地主要用于：

- 修改代码和配置
- 跑单元测试
- 做契约验证和 smoke test

远程服务器主要用于：

- SFT 正式训练
- RL 正式训练
- evaluate 正式评估

推荐本地检查：

```bash
python -m minionerec.cli.main --help
python -m unittest discover -s tests/unit -v
python -m compileall src minionerec
```

## 6. 版本控制约定

仓库默认不跟踪以下内容：

- 原始数据
- 处理后的大体积中间产物
- embedding 和 checkpoint
- 训练输出和实验结果

保留在版本库中的主要是：

- 源代码
- 配置文件
- 文档
- 小型测试夹具

