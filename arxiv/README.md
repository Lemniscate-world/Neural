# arXiv Submission Instructions — NeuralDBG Paper

## Quick Submit (Overleaf — Recommended)

1. Go to https://www.overleaf.com (free account)
2. Create a new blank project
3. Upload `paper.tex` as the main file
4. Click "Recompile" → downloads PDF automatically
5. Go to https://arxiv.org/submit
6. Upload the PDF + source (.tex)

## Local Compilation (Windows)

### Option A: Install MiKTeX (lightweight)
```powershell
choco install miktex
# Then:
pdflatex paper.tex
```

### Option B: Docker (any OS)
```bash
docker run --rm -v ${PWD}:/workdir texlive/texlive pdflatex paper.tex
```

## Files
- `paper.tex` — LaTeX source (28 references, 10 sections, ~8 pages)
- `paper_draft.md` — Markdown source (docs/paper_draft.md)
- `benchmark_honest.py` — Reproducible benchmark script
- `benchmark_honest.json` — Benchmark results

## Paper Stats
- Title: "Causal Debugging of Deep Learning Training Failures"
- Author: Jacques-Charles Gad Senouvo (LambdaSection)
- Sections: 10 (Intro, Related Work, Method, Architecture Improvements, Experiments, NeuralPrune, Post-Mortems, Discussion, Limitations, Conclusion)
- References: 28 (proper BibTeX, DOIs)
- Detection rates: 95% combinatorial, 100% OOS, 100% RL, 100% stress
- License: MIT (code), CC-BY 4.0 (paper)

## arXiv Categories
- cs.LG (Machine Learning) — primary
- cs.SE (Software Engineering) — secondary
- stat.ML (Machine Learning) — secondary
