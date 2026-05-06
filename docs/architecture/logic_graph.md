# NeuralDbg Logic Graph

This document provides a visual representation of the NeuralDbg debugging workflow and the SemanticEvent class logic.

## Overall System Logic Flow

```mermaid
graph TD
    A[Start Neural Network Training] --> B[Wrap Model with NeuralDbg]
    B --> C[Forward Pass]
    C --> D[Extract Activation Stats]
    D --> E{Activation Regime Shift?}
    E -->|Yes| F[Create SemanticEvent]
    E -->|No| G[Continue]
    F --> H[Store in NeuralDbg.events]
    G --> I[Backward Pass]
    I --> J[Extract Gradient Norm]
    J --> K{Gradient Health Transition?}
    K -->|Yes| L[Create SemanticEvent]
    K -->|No| M[Continue]
    L --> H
    H --> N[Post-Mortem Causal Analysis]
    N --> O[Generate Ranked Hypotheses]
```

## SemanticEvent Class Logic

```mermaid
graph TD
    A[New SemanticEvent Created] --> B[Store event_type]
    B --> C[Store layer_name]
    C --> D[Store step number]
    D --> E[Store from_state / to_state stats]
    E --> F[Store confidence & metadata]
    F --> G[Ready for Causal Reasoning]
```

## Causal Inference Workflow

```mermaid
graph LR
    A[Events Stream] --> B[Pattern Matching]
    B --> C[Causal Chain Tracing]
    C --> D[Hypothesis Ranking]
    D --> E[Human-Readable Explanation]
```

This logic graph shows how NeuralDbg extracts high-level semantic events from training dynamics, focusing on transitions and shifts rather than raw tensor storage, enabling efficient causal inference.
