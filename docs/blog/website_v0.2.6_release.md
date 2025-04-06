# Neural DSL v0.2.6: Enhanced Dashboard UI & Blog Support

![Neural DSL Logo](../assets/images/neural-logo.png)

*Posted on March 25, 2025 by Lemniscate-SHA-256*

We're excited to announce the release of Neural DSL v0.2.6! This update brings significant improvements to the NeuralDbg dashboard with a more aesthetic design, along with blog support and several other enhancements and fixes.

## Enhanced Dashboard UI

The standout feature in v0.2.6 is the completely redesigned NeuralDbg dashboard with a sleek dark theme and improved visualization components. The new dashboard provides:

- **Dark Mode Theme**: A modern, eye-friendly dark interface using Dash Bootstrap components
- **Responsive Design**: Better layout that adapts to different screen sizes
- **Improved Visualizations**: Enhanced tensor flow animations and shape propagation charts
- **Real-time Updates**: Fixed WebSocket connectivity for smoother data streaming

These improvements make debugging and visualizing your neural networks more intuitive and aesthetically pleasing, helping you better understand model behavior during training and inference.

### Using the New Dashboard

```bash
# Basic usage with default dark theme
neural debug my_model.neural

# Explicitly specify dark theme
neural debug my_model.neural --theme dark

# Or use light theme if preferred
neural debug my_model.neural --theme light
```

### Dashboard Components

The dashboard now includes several enhanced visualization components:

```python
# Example model to visualize in the dashboard
network MNISTClassifier {
  input: (28, 28, 1)
  layers:
    Conv2D(filters=32, kernel_size=(3,3), activation="relu")
    MaxPooling2D(pool_size=(2,2))
    Conv2D(filters=64, kernel_size=(3,3), activation="relu")
    MaxPooling2D(pool_size=(2,2))
    Flatten()
    Dense(128, activation="relu")
    Dropout(0.5)
    Output(10, "softmax")
  optimizer: Adam(learning_rate=0.001)
}
```

With this model, you can explore various dashboard features:

```bash
# Run with gradient analysis enabled
neural debug my_model.neural --gradients

# Run with dead neuron detection
neural debug my_model.neural --dead-neurons

# Run with anomaly detection
neural debug my_model.neural --anomalies

# Run with step-by-step debugging
neural debug my_model.neural --step
```

## Blog Support & Documentation

We've added infrastructure for blog content with markdown support, making it easier to:

- Share updates about Neural DSL development
- Provide tutorials and examples
- Publish content both on our website and Dev.to
- Engage with the community through detailed technical content

This release also includes enhanced documentation with more detailed examples for HPO usage and error handling, making it easier for new users to get started with Neural DSL.

### Blog Directory Structure

```
docs/
  blog/
    README.md             # Blog overview and guidelines
    blog-list.json        # Metadata for all blog posts
    website_*.md          # Posts for the website
    devto_*.md            # Posts formatted for Dev.to
```

### Creating a Blog Post

Here's an example of how to create a new blog post:

```markdown
# Title of Your Blog Post

![Optional Image](../assets/images/your-image.png)

*Posted on Month Day, Year by Your Name*

First paragraph of your blog post...

## Section Heading

Content of your section...
```

### Dev.to Integration

For posts that will also be published on Dev.to, use the following frontmatter format:

```markdown
---
title: "Your Title Here"
published: true
description: "Brief description of your post"
tags: machinelearning, python, deeplearning, opensource
cover_image: https://url-to-your-cover-image.png
---

# Your Content Here
```

## Advanced HPO Examples

For users working with hyperparameter optimization, we've added comprehensive examples demonstrating:

- Complex nested HPO configurations
- Multi-framework optimization strategies
- Advanced parameter search spaces
- Integration with training loops

These examples make it easier to leverage Neural DSL's powerful HPO capabilities across both PyTorch and TensorFlow backends.

### Example: Complex Nested HPO Configuration

```python
network AdvancedHPOExample {
  input: (28, 28, 1)
  layers:
    # Convolutional layers with HPO parameters
    Conv2D(filters=HPO(choice(32, 64)), kernel_size=(3,3), activation="relu")
    MaxPooling2D(pool_size=(2,2))

    # Another conv block with HPO
    Conv2D(filters=HPO(choice(64, 128)), kernel_size=(3,3), activation="relu")
    MaxPooling2D(pool_size=(2,2))

    # Flatten and dense layers
    Flatten()
    Dense(HPO(choice(128, 256, 512)), activation="relu")
    Dropout(HPO(range(0.3, 0.7, step=0.1)))
    Output(10, "softmax")

  # Advanced optimizer configuration with HPO
  optimizer: SGD(
    learning_rate=ExponentialDecay(
      HPO(range(0.05, 0.2, step=0.05)),  # Initial learning rate
      1000,                              # Decay steps
      HPO(range(0.9, 0.99, step=0.01))   # Decay rate
    ),
    momentum=HPO(range(0.8, 0.99, step=0.01))
  )

  # Training configuration with HPO
  train {
    epochs: 20
    batch_size: HPO(choice(32, 64, 128))
    validation_split: 0.2
    search_method: "bayesian"  # Use Bayesian optimization
  }
}
```

### Running HPO Optimization

```bash
# Run HPO with 50 trials
neural optimize my_model.neural --trials 50 --backend tensorflow

# Run HPO with PyTorch backend
neural optimize my_model.neural --trials 30 --backend pytorch

# Generate optimized model with best parameters
neural optimize my_model.neural --generate optimized_model.neural
```

## Other Improvements

- **CLI Version Display**: Updated version command to dynamically fetch package version
- **Error Reporting**: Improved error context with precise line/column information
- **Performance Optimizations**: Faster shape propagation and tensor flow visualization
- **CI/CD Pipeline**: Streamlined GitHub Actions workflows with better error reporting
- **Test Suite Stability**: Resolved flaky tests in dashboard and HPO components

### CLI Version Command Example

```bash
# Run the version command to see details
neural version

# Output:
# Neural CLI v0.2.6
# Python: 3.10.12
# Click: 8.1.7
# Lark: 1.1.7
# Torch: 2.1.0
# Tensorflow: 2.15.0
# Optuna: 3.4.0
```

### Performance Improvements

The shape propagation and tensor flow visualization have been optimized for better performance:

```python
# Before optimization: ~500ms for complex models
# After optimization: ~150ms for the same models

# Example of visualizing shape propagation
neural visualize my_model.neural --format html --show-shapes
```

## Bug Fixes

- Fixed edge cases in HPO parameter validation and parsing
- Resolved WebSocket connection issues in the dashboard
- Improved error context in validation messages
- Enhanced validation for layer parameters
- Fixed test suite stability issues

### HPO Parameter Validation Example

Previously, certain nested HPO configurations would cause validation errors. Now they work correctly:

```python
# This would previously fail with a validation error
network ComplexHPO {
  input: (28, 28, 1)
  layers:
    Dense(HPO(choice(HPO(range(64, 256, step=64)), HPO(choice(512, 1024)))))
    Output(10)
  optimizer: Adam(learning_rate=0.001)
}
```

### WebSocket Connection Fix

The dashboard now maintains stable WebSocket connections for real-time updates:

```javascript
// Internal implementation improvement
// Before: Connection would drop after ~30 seconds of inactivity
// After: Connections remain stable with proper ping/pong mechanism

// Example of how to connect to the dashboard API
const socket = new WebSocket('ws://localhost:8050/socket');
socket.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received real-time update:', data);
};
```

## Installation

```bash
pip install neural-dsl
```

## Get Involved

- [GitHub Repository](https://github.com/Lemniscate-SHA-256/Neural)
- [Documentation](https://github.com/Lemniscate-SHA-256/Neural/blob/main/docs/dsl.md)
- [Discord Community](https://discord.gg/KFku4KvS)

If you find Neural DSL useful, please consider giving us a star on GitHub ⭐ and sharing this project with your friends and colleagues. The more developers we reach, the more likely we are to build something truly revolutionary together!
