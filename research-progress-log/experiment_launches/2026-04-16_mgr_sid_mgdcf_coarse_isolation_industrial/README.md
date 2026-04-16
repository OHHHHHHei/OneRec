# 2026-04-16 `MGDCF` Coarse Isolation（`MGDCF` 粗图隔离实验）

## 目的

这是 `R542a` 之后的直接跟进实验。

它回答的问题是：

> `R542a` 之所以偏弱，到底是因为 `MGDCF` coarse（`MGDCF` 粗图）本身不够好，
> 还是因为我们把 `L2` 也换成了 `fagsp_mid_mgdcf`（基于 `MGDCF` 的中尺度图）之后，
> 整条链一起变得太激进了。

所以这轮实验只做一件事：

- 保留 `L1 <- coarse_mgdcf`
- 把 `L2` 恢复成 baseline `fagsp_mid_base`
- `L3` 保持 `local_purified`

也就是做一个严格的 coarse-only isolation（仅粗图隔离）。

## 运行定义

### `R542b`

- `L1 <- coarse_mgdcf`
- `L2 <- fagsp_mid_base`
- `L3 <- local_purified`
- `mgdcf_keep_ratio = 0.20`

### `R542c`

- `L1 <- coarse_mgdcf`
- `L2 <- fagsp_mid_base`
- `L3 <- local_purified`
- `mgdcf_keep_ratio = 0.10`

## 配置

- [sid_train_industrial_mgr_sid_r542b_mgdcf_coarse_iso_r020.yaml](/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_r542b_mgdcf_coarse_iso_r020.yaml)
- [sid_train_industrial_mgr_sid_r542c_mgdcf_coarse_iso_r010.yaml](/home/leejt/OneRec/config/experiments/sid_train_industrial_mgr_sid_r542c_mgdcf_coarse_iso_r010.yaml)

## 启动脚本

- [experiment_mgr_sid_r542b_mgdcf_coarse_iso_r020_train_generate.sh](/home/leejt/OneRec/scripts/experiment_mgr_sid_r542b_mgdcf_coarse_iso_r020_train_generate.sh)
- [experiment_mgr_sid_r542c_mgdcf_coarse_iso_r010_train_generate.sh](/home/leejt/OneRec/scripts/experiment_mgr_sid_r542c_mgdcf_coarse_iso_r010_train_generate.sh)

## Runtime（运行时）

- launch date（启动日期）:
  - `2026-04-16`
- `R542b` tmux（终端复用）:
  - `mgr_r542b_mgdcf_coarse_iso_r020`
- `R542c` tmux（终端复用）:
  - `mgr_r542c_mgdcf_coarse_iso_r010`
- GPU（图形处理器）:
  - `R542b -> 2`
  - `R542c -> 3`
- status（状态）:
  - `RUNNING`

## Logs（日志）

- `R542b`:
  - `/home/leejt/OneRec/logs/experiment_mgr_sid_r542b_mgdcf_coarse_iso_r020_20260416.log`
- `R542c`:
  - `/home/leejt/OneRec/logs/experiment_mgr_sid_r542c_mgdcf_coarse_iso_r010_20260416.log`

## Output Targets（输出目标）

- `R542b` generated summary（生成摘要）:
  - [R542b_generate_summary.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_mgdcf_coarse_isolation_industrial/R542b_generate_summary.json)
- `R542c` generated summary（生成摘要）:
  - [R542c_generate_summary.json](/home/leejt/OneRec/research-progress-log/experiment_launches/2026-04-16_mgr_sid_mgdcf_coarse_isolation_industrial/R542c_generate_summary.json)

## 当前判断

这轮实验比继续盲调 `MGDCF` family（`MGDCF` 家族）别的细节更优先，因为它先回答一个更基本的问题：

> `R542a` 的主要问题，是不是来自把 `mid`（中层图）也一起换掉了。

如果 coarse-only isolation（仅粗图隔离）明显比 `R542a` 更好，那我们下一步应该优先优化：

- `coarse_mgdcf`
- 与 baseline `mid`（中层图）的搭配

而不是继续把 `coarse + mid` 一起绑定推进。
