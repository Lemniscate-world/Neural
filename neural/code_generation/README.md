# Neural Code Generation

<p align="center">
  <img src="../../docs/images/code_generation_flow.png" alt="Code Generation Flow" width="600"/>
</p>

## Overview

The Code Generation module is responsible for transforming Neural DSL model representations into executable code for various backend frameworks. It takes the intermediate representation produced by the parser and generates optimized code that implements the neural network architecture.

## Supported Backends

The Code Generation module currently supports the following backends:

1. **TensorFlow/Keras**: Generates TensorFlow 2.x code using the Keras API
2. **PyTorch**: Generates PyTorch code with nn.Module classes
3. **JAX**: Generates JAX code with Flax modules
4. **ONNX**: Generates ONNX model definitions for cross-framework compatibility

## Components

### 1. Code Generator (`code_generator.py`)

The main entry point for code generation that:
- Takes a model representation from the parser
- Selects the appropriate backend generator
- Orchestrates the code generation process
- Returns the generated code

### 2. Backend Generators

Specialized generators for each supported backend:

- **TensorFlow Generator** (`tensorflow_generator.py`): Generates TensorFlow/Keras code
- **PyTorch Generator** (`pytorch_generator.py`): Generates PyTorch code
- **JAX Generator** (`jax_generator.py`): Generates JAX code
- **ONNX Generator** (`onnx_generator.py`): Generates ONNX model definitions

### 3. Template Engine (`templates/`)

A template-based approach for generating code:
- Templates for each backend
- Layer-specific templates
- Training configuration templates

### 4. Optimizers (`optimizers/`)

Code optimization utilities:
- Dead code elimination
- Constant folding
- Layer fusion
- Memory optimization

## Usage

```python
from neural.code_generation.code_generator import generate_code

# Model data from the parser
model_data = {
    "name": "MNIST",
    "input": {"shape": [28, 28, 1]},
    "layers": [
        {"type": "Conv2D", "filters": 32, "kernel_size": 3, "activation": "relu"},
        {"type": "MaxPooling2D", "pool_size": 2},
        {"type": "Flatten"},
        {"type": "Dense", "units": 128, "activation": "relu"},
        {"type": "Output", "units": 10, "activation": "softmax"}
    ],
    "loss": "sparse_categorical_crossentropy",
    "optimizer": {"type": "Adam", "learning_rate": 0.001},
    "metrics": ["accuracy"]
}

# Generate TensorFlow code
tensorflow_code = generate_code(model_data, backend="tensorflow")

# Generate PyTorch code
pytorch_code = generate_code(model_data, backend="pytorch")

# Generate code with hyperparameter optimization results
optimized_code = generate_code(model_data, backend="tensorflow", best_params={
    "learning_rate": 0.0005,
    "batch_size": 128
})
```

## Hyperparameter Optimization Integration

The Code Generation module integrates with the Hyperparameter Optimization (HPO) module to generate code that incorporates optimized hyperparameters:

```python
from neural.code_generation.code_generator import generate_optimized_dsl

# Original Neural DSL code
neural_code = """
network MNIST {
  input: (28, 28, 1)
  layers:
    Conv2D(32, kernel_size=3, activation="relu")
    MaxPooling2D(pool_size=2)
    Flatten()
    Dense(128, activation="relu")
    Output(10, activation="softmax")

  optimizer: Adam(learning_rate=0.001)
  batch_size: 64
}
"""

# Optimized hyperparameters from HPO
best_params = {
    "learning_rate": 0.0005,
    "batch_size": 128
}

# Generate optimized DSL code
optimized_dsl = generate_optimized_dsl(neural_code, best_params)
```

## Extension Points

The Code Generation module is designed to be extensible:

1. **New Backends**: Add support for new frameworks by creating a new backend generator.
2. **Custom Layers**: Extend existing backends to support custom layer types.
3. **Optimization Strategies**: Implement new code optimization strategies.
4. **Template Customization**: Modify templates to generate code with specific patterns or optimizations.

## Related Components

- **Parser**: Provides the model representation used for code generation.
- **Shape Propagation**: Provides shape information used in the generated code.
- **HPO**: Provides optimized hyperparameters for code generation.

## Resources

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs/python/tf)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [JAX Documentation](https://jax.readthedocs.io/en/latest/)
- [ONNX Documentation](https://onnx.ai/onnx/index.html)
- [Neural DSL Reference](../../docs/DSL.md)
