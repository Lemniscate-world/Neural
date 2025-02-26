# Neural DSL Documentation

## Table of Contents
- [Neural DSL Documentation](#neural-dsl-documentation)
  - [Table of Contents](#table-of-contents)
  - [Syntax Basics](#syntax-basics)
    - [Key Components](#key-components)
  - [Validation Rules](#validation-rules)
  - [CLI Commands](#cli-commands)
  - [Error Handling](#error-handling)
    - [Example Error Message](#example-error-message)
    - [Severity Levels](#severity-levels)
  - [Examples](#examples)
    - [MNIST Classifier](#mnist-classifier)
    - [Transformer with Defaults](#transformer-with-defaults)
  - [Contributing](#contributing)
    - [Linting Setup](#linting-setup)
    - [Running Tests](#running-tests)

---

## Syntax Basics
```yaml
network <ModelName> {
  input: <Shape>           # e.g., (28, 28, 1)
  layers:
    <LayerType>(<Params>)  # e.g., Conv2D(filters=32, kernel=(3,3))
  loss: <LossFunction>
  optimizer: <Optimizer>
  train { epochs: <Int>, batch_size: <Int> }
}
```

### Key Components
1. **Layers**:  
   ```yaml
   # Ordered params (units first)
   Dense(128, activation="relu")
   
   # Named params
   Conv2D(filters=32, kernel_size=(3,3), activation="relu")
   ```

2. **Advanced Layers**:  
   ```yaml
   TransformerEncoder(num_heads=8)  # Requires num_heads > 0
   Attention()                      # No mandatory params
   ```

3. **Training Config**:  
   ```yaml
   train {
     epochs: 10
     batch_size: 32
     validation_split: 0.2  # Must be 0-1
   }
   ```

---

## Validation Rules
| Parameter           | Rule                          | Error Example              |
|---------------------|-------------------------------|----------------------------|
| `Dropout(rate)`     | `0 ≤ rate ≤ 1`                | `Dropout(1.5)` → ERROR     |
| `Conv2D(filters)`   | Must be positive integer      | `filters=-32` → ERROR      |
| `kernel_size`       | Tuple of positive integers    | `(0,0)` → ERROR            |
| `validation_split`  | `0 ≤ value ≤ 1`               | `1.1` → ERROR              |
| Transformer params  | `num_heads > 0`, `ff_dim > 0` | `num_heads=0` → ERROR      |

---

## CLI Commands
```bash
# Compile to framework code
neural compile model.neural --backend tensorflow

# Dry-run validation
neural compile model.neural --dry-run

# Debug with live monitoring
neural debug model.neural --backend pytorch

# Launch no-code GUI
neural no-code --port 8051
```

---

## Error Handling
### Example Error Message
```text
ERROR at line 7, column 15:
Dropout rate must be between 0 and 1. Got 1.2
```

### Severity Levels
- **ERROR**: Parsing stops (e.g., invalid `kernel_size=(0,0)`).  
- **WARNING**: Logged but parsing continues (e.g., `Dense(128.0)` → coerced to 128).  

---

## Examples
### MNIST Classifier
```yaml
network MNISTClassifier {
  input: (28, 28, 1)
  layers:
    Conv2D(32, (3,3), activation="relu")
    MaxPooling2D(pool_size=(2,2))
    Flatten()
    Dense(128, activation="relu")
    Dropout(0.5)
    Output(10, activation="softmax")
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.001)
  train { epochs: 10, batch_size: 32 }
}
```

### Transformer with Defaults
```yaml
network ViT {
  layers:
    TransformerEncoder()  # Auto-sets num_heads=8, ff_dim=512
}
```

---

## Contributing
### Linting Setup
```bash
pip install pre-commit
pre-commit install  # Auto-run Black/Flake8 on commit
```

### Running Tests
```bash
# Run all tests
pytest test_parser.py -v

# Test specific layer
pytest test_parser.py -k "test_conv2d_validation"
```

[Full API Reference](https://github.com/your-repo/docs) | 
[Report Issues](https://github.com/your-repo/issues)
```