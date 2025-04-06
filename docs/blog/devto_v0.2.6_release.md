---
title: "Neural DSL v0.2.6: Enhanced Dashboard UI & Blog Support"
published: true
description: "Announcing Neural DSL v0.2.6 with a redesigned dashboard featuring a sleek dark theme, blog support infrastructure, and improved error reporting."
tags: machinelearning, python, deeplearning, opensource
cover_image: https://github.com/user-attachments/assets/f92005cc-7b1c-4020-aec6-0e6922c36b1b
---

# Neural DSL v0.2.6: Enhanced Dashboard UI & Blog Support

![Neural DSL Logo](https://github.com/user-attachments/assets/f92005cc-7b1c-4020-aec6-0e6922c36b1b)

We're excited to announce the release of Neural DSL v0.2.6! This update brings significant improvements to the NeuralDbg dashboard with a more aesthetic design, along with blog support and several other enhancements and fixes.

## 🚀 Spotlight Feature: Enhanced Dashboard UI

The standout feature in v0.2.6 is the completely redesigned NeuralDbg dashboard with a sleek dark theme and improved visualization components. The new dashboard provides:

- **Dark Mode Theme**: A modern, eye-friendly dark interface using Dash Bootstrap components
- **Responsive Design**: Better layout that adapts to different screen sizes
- **Improved Visualizations**: Enhanced tensor flow animations and shape propagation charts
- **Real-time Updates**: Fixed WebSocket connectivity for smoother data streaming

These improvements make debugging and visualizing your neural networks more intuitive and aesthetically pleasing, helping you better understand model behavior during training and inference.

![NeuralDbg Dashboard](https://github.com/user-attachments/assets/dashboard-dark-theme.png)

## 📝 Blog Support & Documentation

We've added infrastructure for blog content with markdown support, making it easier to:

- Share updates about Neural DSL development
- Provide tutorials and examples
- Publish content both on our website and Dev.to
- Engage with the community through detailed technical content

This release also includes enhanced documentation with more detailed examples for HPO usage and error handling, making it easier for new users to get started with Neural DSL.

## 🔍 Advanced HPO Examples

For users working with hyperparameter optimization, we've added comprehensive examples demonstrating:

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

## ✨ Other Improvements

- **CLI Version Display**: Updated version command to dynamically fetch package version
- **Error Reporting**: Improved error context with precise line/column information
- **Performance Optimizations**: Faster shape propagation and tensor flow visualization
- **CI/CD Pipeline**: Streamlined GitHub Actions workflows with better error reporting
- **Test Suite Stability**: Resolved flaky tests in dashboard and HPO components

## 🐛 Bug Fixes

- Fixed edge cases in HPO parameter validation and parsing
- Resolved WebSocket connection issues in the dashboard
- Improved error context in validation messages
- Enhanced validation for layer parameters
- Fixed test suite stability issues

## 📦 Installation

```bash
pip install neural-dsl
```

## 🔗 Links

- [GitHub Repository](https://github.com/Lemniscate-SHA-256/Neural)
- [Documentation](https://github.com/Lemniscate-SHA-256/Neural/blob/main/docs/dsl.md)
- [Discord Community](https://discord.gg/KFku4KvS)

## 🙏 Support Us

If you find Neural DSL useful, please consider giving us a star on GitHub ⭐ and sharing this project with your friends and colleagues. The more developers we reach, the more likely we are to build something truly revolutionary together!
