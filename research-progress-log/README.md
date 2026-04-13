# Research Progress Log

这个目录现在承担两件事：

1. 维护当前主线的一份持续更新 LaTeX 日志
2. 为每个关键实验阶段提供一个简洁、可回溯的记录入口

## Canonical Files

- `research_progress_log.tex`
- `research_progress_log.pdf`
- `experiment_launches/README.md`

## Recommended Reading Order

1. `research_progress_log.tex`
2. `experiment_launches/README.md`
3. 再进入具体实验阶段目录读 `README.md` / `RESULTS.md`

## Compile

```bash
cd /home/leejt/OneRec/research-progress-log
pdflatex -interaction=nonstopmode -halt-on-error research_progress_log.tex
```

## Usage Policy

- 每次出现有意义的实验结果或方向变化后更新
- 主结论优先写进 `research_progress_log.tex`
- 每个实验阶段只保留少数 canonical prose 文档
- 细粒度 raw artifacts 保留在对应 run 目录下，不再在根目录平铺
