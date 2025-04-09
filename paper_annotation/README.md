# Paper Annotation Workflow

This directory contains Jupyter notebooks for annotating research papers and preparing blog content for Medium.

## Structure

```
paper_annotation/
├── notebooks/           # Jupyter notebooks for paper annotations
├── assets/              # Images, diagrams, and other media
├── templates/           # Templates for Medium posts
└── exports/             # Exported HTML/Markdown for Medium
```

## Workflow

1. Create a new notebook in `notebooks/` for each paper
2. Use the Neural DSL to implement and annotate key concepts
3. Export to Markdown for Medium using `nbconvert`
4. Publish to Medium

## Usage

```bash
# Create a new annotation notebook
python create_annotation.py --paper "Paper Title" --authors "Author1, Author2" --year 2023

# Export notebook to Medium-ready markdown
jupyter nbconvert notebooks/paper_title.ipynb --to markdown --output-dir exports/
```