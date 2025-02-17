

  ![Design sans titre (1) (1)](https://github.com/user-attachments/assets/cb351971-ba3d-4391-9401-25cf2ccfeca9)


# Neural: A Neural Network Programming Language

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![Discord](https://img.shields.io/badge/Chat-Discord-7289DA)](https://discord.gg/your-invite-link)

Neural is a domain-specific language (DSL) designed for defining, training, and deploying neural networks. With **declarative syntax** and **cross-framework support**, it simplifies building complex architectures while automating error-prone tasks like shape validation.

![Network Visualization Demo]()  
*Example: Auto-generated architecture diagram and shape propagation report*

## 🚀 Features

- **YAML-like Syntax**: Define models intuitively without framework boilerplate.
- **Shape Propagation**: Catch dimension mismatches *before* runtime.
- **Multi-Backend Export**: Generate code for **TensorFlow**, **PyTorch**, or **ONNX**.
- **Training Orchestration**: Configure optimizers, schedulers, and metrics in one place.
- **Visual Debugging**: Render interactive 3D architecture diagrams.
- **Extensible**: Add custom layers/losses via Python plugins.

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/neural.git
cd neural

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

**Prerequisites**: Python 3.8+, pip

## 🛠️ Quick Start

### 1. Define a Model

Create `mnist.neural`:
```yaml
network MNISTClassifier {
  input: (28, 28, 1)  # Channels-last format
  layers:
    Conv2D(filters=32, kernel_size=(3,3), activation="relu")
    MaxPooling2D(pool_size=(2,2))
    Flatten()
    Dense(units=128, activation="relu")
    Dropout(rate=0.5)
    Output(units=10, activation="softmax")
  
  loss: "sparse_categorical_crossentropy"
  optimizer: Adam(learning_rate=0.001)
  metrics: ["accuracy"]
  
  train {
    epochs: 15
    batch_size: 64
    validation_split: 0.2
  }
}
```

### 2. Generate Framework Code

```bash
# For TensorFlow
python neural.py compile mnist.neural --backend tensorflow --output mnist_tf.py

# For PyTorch
python neural.py compile mnist.neural --backend pytorch --output mnist_torch.py
```

### 3. Visualize Architecture

```bash
python neural.py visualize mnist.neural --format png
```
![MNIST Architecture]()

## 🌟 Why Neural?

| Feature               | Neural      | Raw TensorFlow/PyTorch |
|-----------------------|-------------|-------------------------|
| Shape Validation      | ✅ Auto     | ❌ Manual               |
| Framework Switching   | 1-line flag | Days of rewriting       |
| Architecture Diagrams | Built-in    | Third-party tools       |
| Training Config       | Unified     | Fragmented configs      |

## 📚 Documentation

Explore advanced features:
- [Custom Layers Guide]()
- [ONNX Export Tutorial]()
- [Training Configuration]()

## 🤝 Contributing

We welcome contributions! See our:
- [Contributing Guidelines](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Roadmap](ROADMAP.md)

To set up a development environment:
```bash
git clone https://github.com/yourusername/neural.git
cd neural
pip install -r requirements-dev.txt  # Includes linter, formatter, etc.
pre-commit install  # Auto-format code on commit
```

## 📬 Community

- [Discord Server](https://discord.gg/your-invite-link): Chat with developers
- [Twitter @NeuralLang](https://twitter.com/NeuralLang): Updates & announcements
