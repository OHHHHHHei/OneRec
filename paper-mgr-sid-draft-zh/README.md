# MGR-SID 中文论文初稿

这个目录是当前 `paper-mgr-sid-draft/` 的中文平行版本。

## 编译

```bash
cd /home/leejt/OneRec/paper-mgr-sid-draft-zh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## 当前范围

- tokenizer 阶段论文初稿
- 上游 MiniOneRec tokenizer 训练制度复现
- graph-bank probe 结果
- MGR-SID v1 tokenizer 训练结果
- local ambiguity 分析

当前版本还没有纳入下游 SFT 或 RL 结果。
