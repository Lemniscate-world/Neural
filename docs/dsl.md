```markdown
# Neural DSL Documentation

## Table of Contents
- [Syntax Reference](#syntax-reference)
- [Core Components](#core-components)
- [Validation Rules](#validation-rules)
- [CLI Reference](#cli-reference)
- [Error Handling](#error-handling)
- [Examples](#examples)
- [Development Guide](#development-guide)

---

## Syntax Reference

### Network Structure
```yaml
network <ModelName> {
  input: <Shape>                # e.g., (224, 224, 3) or multiple inputs
  layers:
    <LayerType>(<Parameters>)   # Supports ordered and named params
    <LayerType>*<Count>         # Layer repetition syntax
  loss: <LossFunction>
  optimizer: <Optimizer>
  train {
    epochs: <Int>
    batch_size: <Int|HPO-range>
    validation_split: <0.0-1.0>
    search_method: "random"     # Hyperparameter optimization
  }
  execution {
    device: <String>            # "cpu", "cuda", "tpu"
  }
}
```

### Parameter Types
```yaml
# Ordered parameters
Conv2D(32, (3,3), "relu")

# Named parameters
TransformerEncoder(num_heads=8, ff_dim=512)

# HPO parameters
Dense(HPO(choice(128, 256, 512)))

# Device placement
LSTM(units=128) @ "cuda:0"
```

---

## Core Components

### Layer Types
| Category         | Layers                                                                 |
|------------------|-----------------------------------------------------------------------|
| **Convolution**  | `Conv1D`, `Conv2D`, `DepthwiseConv2D`, `SeparableConv2D`              |
| **Recurrent**    | `LSTM`, `GRU`, `Bidirectional`, `ConvLSTM2D`                          |
| **Transformer**  | `TransformerEncoder`, `TransformerDecoder`                            |
| **Regularization**| `Dropout`, `SpatialDropout2D`, `GaussianNoise`, `BatchNormalization` |
| **Utility**      | `Flatten`, `Lambda`, `TimeDistributed`, `ResidualConnection`          |

---

## Validation Rules

### Parameter Constraints
| Parameter           | Rule                                  | Error Example              |
|---------------------|---------------------------------------|----------------------------|
| `num_heads`         | Must be > 0                          | `num_heads=0` → ERROR      |
| `filters`           | Positive integer                     | `filters=-32` → ERROR      |
| `kernel_size`       | Tuple of positive integers           | `(0,3)` → ERROR            |
| `rate`              | 0 ≤ value ≤ 1                        | `Dropout(1.2)` → ERROR     |
| `validation_split`  | 0 ≤ value ≤ 1                        | `1.1` → ERROR              |
| `device`            | Valid device identifier              | `device:"npu"` → CRITICAL  |

### Type Coercions
- Float → Int conversion with warning (e.g., `Dense(256.0)`)
- Automatic tuple wrapping for single values

---

## CLI Reference

### Command Overview
```bash
# Core Commands
neural compile <file> [--backend tensorflow|pytorch] [--hpo]
neural run <file> [--device cpu|cuda]
neural debug <file> [--gradients] [--dead-neurons] [--step]

# Analysis & Visualization
neural visualize <file> [--format png|svg|html]
neural profile <file> [--memory] [--latency]

# Project Management
neural clean  # Remove generated files
neural version  # Show version info
```

### Key Options
- `--dry-run`: Validate without code generation
- `--hpo`: Enable hyperparameter optimization
- `--step`: Interactive debugging mode
- `--port`: Specify GUI port for no-code interface

---

## Error Handling

### Severity Levels
| Level     | Description                          | Example Case                      |
|-----------|--------------------------------------|-----------------------------------|
| CRITICAL  | Fatal configuration error           | Invalid device specification      |
| ERROR     | Structural/model error              | Missing required parameter        |
| WARNING   | Non-fatal issue                     | Type coercion                     |
| INFO      | Diagnostic message                  | Shape propagation update          |

### Example Messages
```text
CRITICAL at line 5, column 12: 
Invalid device 'npu' - must be one of [cpu, cuda, tpu]

ERROR at line 12, column 8:
Conv2D kernel_size requires positive integers. Got (0,3)

WARNING at line 8, column 15:
Implicit conversion of 256.0 to integer 256
```

---

## Examples

### Vision Transformer
```yaml
network ViT {
  input: (224, 224, 3)
  layers:
    Conv2D(64, (7,7), strides=2) @ "cuda:0"
    TransformerEncoder() * 12     # 12 transformer blocks
    GlobalAveragePooling2D()
    Dense(1000, "softmax")
  loss: "categorical_crossentropy"
  optimizer: Adam(learning_rate=1e-4)
  train {
    epochs: 300
    batch_size: HPO(range(128, 512, step=64))
  }
}
```

### Hyperparameter Optimization
```yaml
network HPOExample {
  layers:
    Dense(HPO(choice(128, 256, 512)))
    Dropout(HPO(range(0.3, 0.7, step=0.1)))
  optimizer: Adam(
    learning_rate=HPO(log_range(1e-4, 1e-2))
  )
  train {
    epochs: 100
    search_method: "bayesian"
  }
}
```

---

## Development Guide

### Setup & Testing
```bash
# Install dependencies
pip install -r requirements.txt
pre-commit install

# Run tests
pytest test_parser.py -v
pytest -k "transformer_validation" --log-level=DEBUG

# Generate coverage report
coverage run -m pytest && coverage html
```

### Linting Rules
- Type hint enforcement
- Parameter validation checks
- HPO syntax verification
- Device configuration validation

### Contribution Requirements
1. Unit tests for new features
2. Documentation updates
3. Backward compatibility checks
4. Pre-commit hook validation

[Full API Docs](https://neural-dsl.dev/api) | 
[Report Issues](https://github.com/neural-dsl/issues)
```
