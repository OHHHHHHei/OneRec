# Archive（归档）

Status（状态）: `navigation（导航）`

This directory is for superseded launch helpers（已被替代的启动辅助脚本）, old queue wrappers（旧队列脚本）, and obsolete summaries（过期摘要）.

Current policy（当前策略）:

- Do not move original execution paths（原始执行路径） while related jobs are running.
- Prefer symlink pointers（软链接指针） first.
- Physical moves（物理移动） should happen only after active SFT/eval jobs（活跃监督微调/评测任务） finish.
