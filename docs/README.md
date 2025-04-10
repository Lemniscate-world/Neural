# Neural Documentation

<p align="center">
  <img src="images/documentation_structure.png" alt="Documentation Structure" width="600"/>
</p>

## Overview

This directory contains comprehensive documentation for the Neural framework, including reference guides, tutorials, examples, and API documentation. The documentation is designed to help users understand and use the Neural DSL and related tools effectively.

## Documentation Structure

The documentation is organized into the following sections:

### 1. Getting Started

- [Introduction to Neural](getting-started.md)
- [Installation Guide](installation.md)
- [Quick Start Tutorial](quick-start.md)
- [Basic Concepts](concepts.md)

### 2. Neural DSL Reference

- [DSL Syntax](DSL.md)
- [Layer Reference](layers.md)
- [Optimizer Reference](optimizers.md)
- [Training Configuration](training.md)
- [Hyperparameter Specification](hyperparameters.md)

### 3. CLI Reference

- [Command-Line Interface](cli-reference.md)
- [Command Reference](commands.md)
- [Configuration Options](configuration.md)
- [Environment Variables](environment.md)

### 4. API Reference

- [Parser API](api/parser.md)
- [Code Generation API](api/code-generation.md)
- [Shape Propagation API](api/shape-propagation.md)
- [Visualization API](api/visualization.md)
- [Dashboard API](api/dashboard.md)
- [HPO API](api/hpo.md)

### 5. Tutorials

- [Basic Tutorials](tutorials/basic/)
  - [Creating Your First Model](tutorials/basic/first-model.md)
  - [Training a Model](tutorials/basic/training.md)
  - [Visualizing a Model](tutorials/basic/visualization.md)
  - [Debugging a Model](tutorials/basic/debugging.md)
- [Advanced Tutorials](tutorials/advanced/)
  - [Custom Layers](tutorials/advanced/custom-layers.md)
  - [Hyperparameter Optimization](tutorials/advanced/hpo.md)
  - [Multi-GPU Training](tutorials/advanced/multi-gpu.md)
  - [Distributed Training](tutorials/advanced/distributed.md)
- [Framework-Specific Tutorials](tutorials/frameworks/)
  - [TensorFlow Integration](tutorials/frameworks/tensorflow.md)
  - [PyTorch Integration](tutorials/frameworks/pytorch.md)
  - [JAX Integration](tutorials/frameworks/jax.md)

### 6. Examples

- [Basic Examples](examples/basic/)
- [Computer Vision Examples](examples/computer-vision/)
- [Natural Language Processing Examples](examples/nlp/)
- [Reinforcement Learning Examples](examples/reinforcement-learning/)
- [Generative Models Examples](examples/generative/)

### 7. Guides

- [Best Practices](guides/best-practices.md)
- [Performance Optimization](guides/performance.md)
- [Debugging Guide](guides/debugging.md)
- [Deployment Guide](guides/deployment.md)
- [Contributing Guide](guides/contributing.md)

### 8. Blog

- [Release Notes](blog/releases/)
- [Feature Spotlights](blog/features/)
- [Case Studies](blog/case-studies/)
- [Tutorials](blog/tutorials/)

## Documentation Formats

The documentation is available in multiple formats:

- **Markdown**: The primary format for all documentation
- **HTML**: Generated from Markdown for web viewing
- **PDF**: Generated from Markdown for offline reading
- **Interactive Notebooks**: Jupyter notebooks for tutorials and examples

## Contributing to Documentation

We welcome contributions to the documentation! Here's how you can help:

1. **Fix Typos and Errors**: If you find a typo or error, please submit a pull request with the fix.
2. **Improve Existing Documentation**: If you think a section could be clearer or more detailed, feel free to improve it.
3. **Add New Documentation**: If you'd like to add new tutorials, examples, or guides, please submit a pull request.
4. **Translate Documentation**: Help make Neural accessible to more people by translating documentation.

Please follow these guidelines when contributing:

- Use clear, concise language
- Include code examples where appropriate
- Add diagrams and images to illustrate complex concepts
- Follow the existing documentation structure
- Test code examples to ensure they work

## Documentation Tools

The documentation is built using the following tools:

- **MkDocs**: Static site generator for documentation
- **Material for MkDocs**: Theme for MkDocs
- **Mermaid**: Diagramming and charting tool
- **Jupyter Book**: For interactive notebooks
- **Sphinx**: For API documentation

## Building the Documentation

To build the documentation locally:

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Build the documentation
mkdocs build

# Serve the documentation locally
mkdocs serve
```

Then open your browser to `http://localhost:8000` to view the documentation.

## Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs Documentation](https://squidfunk.github.io/mkdocs-material/)
- [Mermaid Documentation](https://mermaid-js.github.io/mermaid/#/)
- [Jupyter Book Documentation](https://jupyterbook.org/en/stable/intro.html)
- [Sphinx Documentation](https://www.sphinx-doc.org/en/master/)
