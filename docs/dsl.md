# Neural DSL Documentation

## Overview
The Neural domain-specific language (DSL) is a YAML-like syntax for defining, training, and debugging neural networks. It’s designed to be intuitive for beginners and powerful for experts, supporting TensorFlow, PyTorch, and ONNX.

## Syntax
Neural uses a declarative, block-based structure. Here’s the basic format:

```yaml
network <ModelName> {
  input: <ShapeTuple>  # E.g., (28, 28, 1)
  layers:
    <LayerType>(<Parameters>)
    <LayerType>(<Parameters>)
  loss: <LossFunction>
  optimizer: <Optimizer>(<Params>)
  train {
    epochs: <Number>
    batch_size: <Number>
  }
}
Key Components
1. Network Definition
Syntax: network <Name> { ... }
Example:
yaml
network MNISTClassifier {
  # ... layers, configs ...
}
Purpose: Defines the model name and contains all configurations.
2. Input Layer
Syntax: input: (<Dimension1>, <Dimension2>, ...)
Example:
yaml
input: (28, 28, 1)  # Image with height 28, width 28, 1 channel
Notes: Supports NONE for dynamic dimensions (e.g., input: (NONE, 28, 1)).
3. Layers
Syntax: <LayerType>(<NamedParams> | <OrderedParams>)
Common Layers:
Conv2D: Conv2D(filters=32, kernel=(3,3), activation="relu")
Dense: Dense(128, activation="relu") or Dense(units=128, activation="relu")
Dropout: Dropout(rate=0.5)
Output: Output(units=10, activation="softmax")
Advanced Layers: TransformerEncoder(num_heads=8, ff_dim=2048), Attention(), GraphConv().
Parameters:
Named (e.g., filters=32) or ordered (e.g., 32 for Dense units).
Supports tuples (e.g., kernel=(3,3)), strings (e.g., activation="relu"), numbers, and booleans.
4. Loss and Optimizer
Syntax: loss: "<LossName>", optimizer: <OptimizerName>(<Params>)
Example:
yaml
loss: "sparse_categorical_crossentropy"
optimizer: Adam(learning_rate=0.001)
Supported Losses: cross_entropy, mean_squared_error, etc.
Supported Optimizers: Adam, SGD, with parameters like learning_rate.
5. Training Configuration
Syntax: train { epochs: <Number>, batch_size: <Number>, ... }
Example:
yaml
train {
  epochs: 10
  batch_size: 32
  validation_split: 0.2
}
Optional: checkpoint, data (e.g., MNIST).
6. Advanced Features
Shape Inference: Automatically validates tensor shapes (e.g., Conv2D outputs (26,26,32) for input (28,28,1)).
Math Integration: Custom layers with forward() and auto-differentiation (e.g., CustomAttention).
Hyperparameter Tuning: hyperparams { learning_rate: [0.1, 0.01, 0.001] } for grid search.
Hardware Acceleration: compile { target: cuda, precision: float16 }.
Interoperability: export to onnx { file: "model.onnx" }, using python { np = import("numpy") }.
Examples
MNIST Classifier
yaml
network MNISTClassifier {
  input: (28, 28, 1)
  layers:
    Conv2D(filters=32, kernel=(3,3), activation="relu")
    MaxPooling2D(pool_size=(2,2))
    Flatten()
    Dense(128, activation="relu")
    Dropout(0.5)
    Output(10, activation="softmax")
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.001)
  train {
    epochs: 10
    batch_size: 32
  }
}
GAN (Generative Adversarial Network)
yaml
network GAN {
  generator: {
    input: (100)  # Latent dim
    layers:
      Dense(256, activation="leaky_relu")
      Conv2DTranspose(128, kernel=(3,3), stride=2)
      Output(784, activation="tanh")
  }
  discriminator: {
    input: (28, 28, 1)
    layers:
      Conv2D(64, kernel=(3,3), activation="leaky_relu")
      Flatten()
      Output(1, activation="sigmoid")
  }
  train {
    loop epochs(1000) {
      batch real_data in MNIST {
        noise = sample_noise(batch_size)
        fake_images = generator(noise)
        d_loss = discriminator.train_on_batch([real_data, fake_images], [1, 0])
        g_loss = generator.train_on_batch(noise, 1)
      }
    }
  }
}
CNN with Shape Validation
yaml
network CNN {
  input: (28, 28, 1)
  layers:
    Conv2D(filters=32, kernel=(3,3))  # Output shape inferred as (26,26,32)
    MaxPooling2D(pool_size=(2,2))     # Output: (13,13,32)
    Flatten()                         # Output: (5408)
    Dense(10)                         # Error: Incompatible shape (5408 → 10)
}
Custom Attention Layer
yaml
layer CustomAttention {
  forward(query, key, value) {
    scores = softmax(query @ key.T / sqrt(d_k))
    return scores @ value
  }
  # Auto-derivative for backward pass
}
Hyperparameter Tuning
yaml
hyperparams {
  learning_rate: [0.1, 0.01, 0.001]
  batch_size: [32, 64]
}

experiment {
  grid_search over hyperparams {
    train Model on MNIST {
      epochs: 10
      metrics: [accuracy, f1_score]
    }
  }
}
Hardware Acceleration
yaml
compile Model {
  target: cuda  # Options: cpu, cuda, tpu
  precision: float16
  optimizations: [fusion, mem_reuse]
}
Interoperability
yaml
export Model to onnx {
  file: "model.onnx"
}

using python {
  np = import("numpy")
  plt.plot(np.log(loss_history))
}
Best Practices
Use named parameters for clarity (e.g., filters=32 over 32).
Validate shapes before compilation with neural visualize.
Test with neural debug for real-time monitoring.
Document custom layers in comments or using python blocks.
Known Limitations
Some advanced layers (e.g., QuantumLayer) require manual Python implementation.
Shape inference may fail with dynamic shapes (NONE)—use CustomShape explicitly.
Troubleshooting
Shape Mismatch: Use neural visualize to debug shape errors.
Parse Errors: Check syntax (e.g., missing commas, quotes) with neural compile --verbose.
Bugs: Report in GitHub Issues—v0.1.x is a WIP with known issues.