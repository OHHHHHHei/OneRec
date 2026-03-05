# Legacy 目录说明

`legacy/` 用于存放不参与当前主链路执行的历史代码与文档，按“可回溯、不可误用”组织：

- `gpr/`：GPR 与偏好实验相关脚本
- `old_root_scripts/`：原 root 目录下的历史实现与重复脚本
- `old_docs/`：历史分析文档与重构草稿文档
- `old_experiments/`：临时实验记录

约束：

1. `src/minionerec` 主链路禁止 import `legacy/`
2. 兼容入口仅负责参数转发到主链路，不再调用本目录脚本
3. 本目录内容默认归档保留，未经评估不直接删除
