# Neural Shape Propagation

<p align="center">
  <img src="../../docs/images/shape_propagation.png" alt="Shape Propagation Diagram" width="600"/>
</p>

## Overview

The Shape Propagation module is responsible for inferring and validating tensor shapes throughout a neural network model. It analyzes the model architecture and propagates input shapes through each layer, ensuring that the dimensions are compatible and catching shape-related errors before runtime.

## Key Features

1. **Automatic Shape Inference**: Automatically calculates output shapes for each layer based on input shapes and layer parameters.
2. **Shape Validation**: Validates that tensor shapes are compatible between connected layers.
3. **Error Detection**: Identifies shape mismatches and provides detailed error messages.
4. **Visualization**: Generates visualizations of tensor shapes throughout the network.
5. **Performance Estimation**: Estimates memory usage and computational requirements based on tensor shapes.

## Components

### 1. Shape Propagator (`shape_propagator.py`)

The main component that:
- Takes a model representation and input shape
- Propagates shapes through each layer
- Validates shape compatibility
- Generates a shape propagation report

### 2. Layer Shape Calculators

Specialized calculators for different layer types:
- Convolutional layers (Conv1D, Conv2D, Conv3D)
- Pooling layers (MaxPooling, AveragePooling)
- Recurrent layers (LSTM, GRU, RNN)
- Dense layers
- Normalization layers
- Activation layers
- Reshape and transpose operations

### 3. Visualization Generators

Components for visualizing shape propagation:
- Tensor flow diagrams
- Shape transformation animations
- Memory usage charts

## Usage

```python
from neural.shape_propagation.shape_propagator import ShapePropagator

# Create a shape propagator
propagator = ShapePropagator()

# Define input shape
input_shape = [28, 28, 1]  # Height, Width, Channels

# Define layers
layers = [
    {"type": "Conv2D", "filters": 32, "kernel_size": 3, "padding": "valid"},
    {"type": "MaxPooling2D", "pool_size": 2},
    {"type": "Conv2D", "filters": 64, "kernel_size": 3, "padding": "valid"},
    {"type": "MaxPooling2D", "pool_size": 2},
    {"type": "Flatten"},
    {"type": "Dense", "units": 128},
    {"type": "Dense", "units": 10}
]

# Propagate shapes through the network
shape_history = []
current_shape = input_shape

for layer in layers:
    current_shape = propagator.propagate(current_shape, layer)
    shape_history.append({
        "layer": layer["type"],
        "output_shape": current_shape
    })
    print(f"Layer: {layer['type']}, Output Shape: {current_shape}")

# Generate a report
report = propagator.generate_report()
```

## Shape Propagation Report

The shape propagation report includes:
- Tensor shapes at each layer
- Memory requirements for each tensor
- Computational complexity (FLOPs) for each operation
- Visualization of tensor flow
- Potential shape-related issues

## Integration with Other Components

The Shape Propagation module integrates with:

1. **Parser**: Uses the model representation from the parser.
2. **Code Generation**: Provides shape information for generated code.
3. **Visualization**: Provides data for architecture visualizations.
4. **Debugging**: Helps identify shape-related issues during debugging.

## Extension Points

The Shape Propagation module is designed to be extensible:

1. **Custom Layers**: Add support for new layer types by implementing shape calculation functions.
2. **Framework-Specific Rules**: Implement framework-specific shape calculation rules.
3. **Visualization Formats**: Add new visualization formats for shape propagation.

## Resources

- [TensorFlow Shape Inference](https://www.tensorflow.org/guide/tensor#shape_of_a_tensor)
- [PyTorch Size Operations](https://pytorch.org/docs/stable/tensor_attributes.html#torch.Tensor.size)
- [Neural DSL Reference](../../docs/DSL.md)
- [Shape Propagation Tutorial](../../docs/tutorials/shape_propagation.md)
