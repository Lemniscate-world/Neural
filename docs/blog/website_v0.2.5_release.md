# Neural DSL v0.2.5: Multi-Framework HPO Support & More

![Neural DSL Logo](../assets/images/neural-logo.png)

*Posted on March 24, 2025 by Lemniscate-SHA-256*

We're excited to announce the release of Neural DSL v0.2.5! This update brings significant improvements to hyperparameter optimization (HPO), making it seamlessly work across both PyTorch and TensorFlow backends, along with several other enhancements and fixes.

## Multi-Framework HPO Support

The standout feature in v0.2.5 is the unified hyperparameter optimization system that works consistently across both PyTorch and TensorFlow backends. This means you can:

- Define your model and HPO parameters once
- Run optimization with either backend
- Compare results across frameworks
- Leverage the strengths of each framework

Here's how easy it is to use:

```yaml
network HPOExample {
  input: (28, 28, 1)
  layers:
    Conv2D(filters=HPO(choice(32, 64)), kernel_size=(3,3))
    MaxPooling2D(pool_size=(2,2))
    Flatten()
    Dense(HPO(choice(128, 256, 512)))
    Output(10, "softmax")
  optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
  train {
    epochs: 10
    search_method: "bayesian"
  }
}
```

Run with either backend:

```bash
# PyTorch backend
neural compile model.neural --backend pytorch --hpo

# TensorFlow backend
neural compile model.neural --backend tensorflow --hpo
```

## Enhanced Optimizer Handling

We've significantly improved how optimizers are handled in the DSL:

- **No-Quote Syntax**: Cleaner syntax for optimizer parameters without quotes
- **Nested HPO Parameters**: Full support for HPO within learning rate schedules
- **Scientific Notation**: Better handling of scientific notation (e.g., `1e-4` vs `0.0001`)

Before:
```yaml
optimizer: "Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))"
```

After:
```yaml
optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
```

Advanced example with learning rate schedules:
```yaml
optimizer: SGD(
  learning_rate=ExponentialDecay(
    HPO(range(0.05, 0.2, step=0.05)),  # Initial learning rate
    1000,                              # Decay steps
    HPO(range(0.9, 0.99, step=0.01))   # Decay rate
  ),
  momentum=HPO(range(0.8, 0.99, step=0.01))
)
```

## Precision & Recall Metrics

Training loops now report precision and recall alongside loss and accuracy, giving you a more comprehensive view of your model's performance:

```python
loss, acc, precision, recall = train_model(model, optimizer, train_loader, val_loader)
```

## Other Improvements

- **Error Message Enhancements**: More detailed error messages with line/column information
- **Layer Validation**: Better validation for MaxPooling2D, BatchNormalization, Dropout, and Conv2D layers
- **TensorRT Integration**: Added conditional TensorRT setup in CI pipeline for GPU environments
- **VSCode Snippets**: Added code snippets for faster Neural DSL development in VSCode
- **CI/CD Pipeline**: Enhanced GitHub Actions workflows with better error handling and reporting

## Bug Fixes

- Fixed parsing of optimizer HPO parameters without quotes
- Corrected string representation handling in HPO parameters
- Resolved issues with nested HPO parameters in learning rate schedules
- Enhanced validation for various layer types
- Fixed parameter handling in Concatenate, Activation, Lambda, and Embedding layers

## Installation

```bash
pip install neural-dsl
```

## Get Involved

- [GitHub Repository](https://github.com/Lemniscate-SHA-256/Neural)
- [Documentation](https://github.com/Lemniscate-SHA-256/Neural/blob/main/docs/dsl.md)
- [Discord Community](https://discord.gg/KFku4KvS)

If you find Neural DSL useful, please consider giving us a star on GitHub ⭐ and sharing this project with your friends and colleagues. The more developers we reach, the more likely we are to build something truly revolutionary together!
