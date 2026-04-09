# MGR-SID Paper Draft

This directory contains the current standalone LaTeX paper draft for the tokenizer-side MGR-SID work. It is separate from the running research log in `research-progress-log/`.

## Compile

```bash
cd /home/leejt/OneRec/paper-mgr-sid-draft
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Current scope

- tokenizer-stage paper draft
- fair upstream-aligned MiniOneRec reproduction
- graph-bank probe evidence
- MGR-SID v1 training-time tokenizer results
- local ambiguity analysis

The current draft does not yet include downstream SFT or RL results.
