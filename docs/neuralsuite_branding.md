# NeuralSuite Branding Guide

## Brand Name

**NeuralSuite** — the complete toolkit for diagnosing and fixing deep learning training failures.

## Tagline

"Catch training problems before they waste your GPU hours."

## Components

| Component | Role | Package | Never call it... |
|-----------|------|---------|-----------------|
| NeuralDBG | Diagnostic engine | `neuraldbg` | "Neural DBG", "Neural-DBG" |
| Neural-Agent | Auto-corrector | `neural-agent` | "Neural Agent", "NeuralAgent" |
| Aquarium | Visualizer | Desktop app | "Neural Aquarium" |

## Usage Rules

### In titles and headers
- Use "NeuralSuite" as the primary name
- Component names as subtitles when needed
- Example: "NeuralSuite: Causal diagnostics for PyTorch training"

### In social posts
- Primary hashtag: `#NeuralSuite`
- Secondary: `#NeuralDBG` `#PyTorch` `#MLOps`
- Mention the component when discussing a specific feature

### In README
- Title: "NeuralSuite" or "NeuralDBG (part of NeuralSuite)"
- First paragraph: explain the suite, then the component
- Install: `pip install neuraldbg` (individual package)

### In PyPI descriptions
- "Part of the NeuralSuite ecosystem"
- "NeuralDBG: the diagnostic engine of NeuralSuite"

## What NOT to do

- Don't rename packages (keep `neuraldbg`, `neural-agent`)
- Don't create a `neuralsuite` PyPI package (unnecessary wrapper)
- Don't say "NeuralSuite DBG" or "NeuralSuite Agent" (component names are standalone)
- Don't use "NS" as abbreviation (too generic)

## Visual Identity (future)

- Logo: three interconnected nodes (diagnostic -> fix -> visualize)
- Colors: to be defined
- Font: to be defined
