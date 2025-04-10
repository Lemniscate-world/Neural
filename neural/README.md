

  ![Design sans titre (1) (1)](https://github.com/user-attachments/assets/cb351971-ba3d-4391-9401-25cf2ccfeca9)


## Status
Neural-dsl v0.1.0 is live on PyPI (`pip install neural-dsl`)! It’s a work-in-progress with bugs—see [CHANGELOG.md](CHANGELOG.md) for details. Feedback welcome!

# Neural: A Neural Network Programming Language

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![Discord](https://img.shields.io/badge/Chat-Discord-7289DA)](https://discord.gg/your-invite-link)
[![Pylint](https://github.com/Lemniscate-SHA-256/Neural/actions/workflows/pylint.yml/badge.svg?branch=main)](https://github.com/Lemniscate-SHA-256/Neural/actions/workflows/pylint.yml)
[![Python package](https://github.com/Lemniscate-SHA-256/Neural/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/Lemniscate-SHA-256/Neural/actions/workflows/python-package.yml)
[![CodeQL Advanced](https://github.com/Lemniscate-SHA-256/Neural/actions/workflows/codeql.yml/badge.svg)](https://github.com/Lemniscate-SHA-256/Neural/actions/workflows/codeql.yml)
[![Tests](https://github.com/Lemniscate-SHA-256/Neural/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/Lemniscate-SHA-256/Neural/actions/workflows/pytest.yml)
[![Coverage](https://img.shields.io/codecov/c/github/Lemniscate-SHA-256/Neural)](https://codecov.io/gh/Lemniscate-SHA-256/Neural)


Neural is a domain-specific language (DSL) designed for defining, training, debugging, and deploying neural networks. With **declarative syntax**, **cross-framework support**, and **built-in execution tracing (NeuralDbg)**, it simplifies deep learning development.


![Network Visualization Demo]()
*Example: Auto-generated architecture diagram and shape propagation report*

## 🚀 Features

- **YAML-like Syntax**: Define models intuitively without framework boilerplate.
- **Shape Propagation**: Catch dimension mismatches *before* runtime.
- **Multi-Backend Export**: Generate code for **TensorFlow**, **PyTorch**, or **ONNX**.
- **Training Orchestration**: Configure optimizers, schedulers, and metrics in one place.
- **Visual Debugging**: Render interactive 3D architecture diagrams.
- **Extensible**: Add custom layers/losses via Python plugins.

### **🛠 NeuralDbg: Built-in Neural Network Debugger**
NeuralDbg provides **real-time execution tracing, profiling, and debugging**, allowing you to visualize and analyze deep learning models in action.

✅ **Real-Time Execution Monitoring** – Track activations, gradients, memory usage, and FLOPs.
![test_trace_graph](https://github.com/user-attachments/assets/15b1edd2-2643-4587-9843-aa4697ed2e4b)
![test_flops_memory_chart](https://github.com/user-attachments/assets/de1f6504-787b-4948-b543-fe3d2f8bfd74)

✅ **Shape Propagation Debugging** – Visualize tensor transformations at each layer.
✅ **Gradient Flow Analysis** – Detect **vanishing & exploding gradients**.
✅ **Dead Neuron Detection** – Identify inactive neurons in deep networks.
✅ **Anomaly Detection** – Spot **NaNs, extreme activations, and weight explosions**.
✅ **Step Debugging Mode** – Pause execution and inspect tensors manually.


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

### 3. Run Or Compile The Model

```bash
neural run mnist.neural --backend tensorflow --output mnist_tf.py
# Or for PyTorch:
neural run mnist.neural --backend pytorch --output mnist_torch.py
```

```bash
neural compile mnist.neural --backend tensorflow --output mnist_tf.py
# Or for PyTorch:
neural compile mnist.neural --backend pytorch --output mnist_torch.py
```

### 4. Visualize Architecture

```bash
neural visualize mnist.neural --format png
```
This will create architecture.png, shape_propagation.html, and tensor_flow.html for inspecting the network structure and shape propagation.

![MNIST Architecture]()

### 5. Debug with NeuralDbg

```bash
neural debug mnist.neural
```
Open your browser to http://localhost:8050 to monitor execution traces, gradients, and anomalies interactively.

### 6. Use The No-Code Interface

```bash
neural --no_code
```
Open your browser to http://localhost:8051 to build and compile models via a graphical interface.

---

## **🛠 Debugging with NeuralDbg**

### **🔹 1️⃣ Start Real-Time Execution Tracing**
```bash
python neural.py debug mnist.neural
```
**Features:**
✅ Layer-wise execution trace
✅ Memory & FLOP profiling
✅ Live performance monitoring

### **🔹 2️⃣ Analyze Gradient Flow**
```bash
python neural.py debug --gradients mnist.neural
```
🚀 **Detect vanishing/exploding gradients** with interactive charts.

### **🔹 3️⃣ Identify Dead Neurons**
```bash
python neural.py debug --dead-neurons mnist.neural
```
🛠 **Find layers with inactive neurons (common in ReLU networks).**

### **🔹 4️⃣ Detect Training Anomalies**
```bash
python neural.py debug --anomalies mnist.neural
```
🔥 **Flag NaNs, weight explosions, and extreme activations.**

### **🔹 5️⃣ Step Debugging (Interactive Tensor Inspection)**
```bash
python neural.py debug --step mnist.neural
```
🔍 **Pause execution at any layer and inspect tensors manually.**

---

## 🌟 Why Neural?

| Feature               | Neural      | Raw TensorFlow/PyTorch |
|-----------------------|-------------|-------------------------|
| Shape Validation      | ✅ Auto     | ❌ Manual               |
| Framework Switching   | 1-line flag | Days of rewriting       |
| Architecture Diagrams | Built-in    | Third-party tools       |
| Training Config       | Unified     | Fragmented configs      |


### **🔄 Cross-Framework Code Generation**
| Neural DSL          | TensorFlow Output          | PyTorch Output            |
|---------------------|----------------------------|---------------------------|
| `Conv2D(filters=32)`| `tf.keras.layers.Conv2D(32)`| `nn.Conv2d(in_channels, 32)` |
| `Dense(units=128)`  | `tf.keras.layers.Dense(128)`| `nn.Linear(in_features, 128)`|

## 🏆 Benchmarks
| Task                 | Neural | Baseline (TF/PyTorch) |
|----------------------|--------|-----------------------|
| MNIST Training       | 1.2x ⚡| 1.0x                  |
| Debugging Setup      | 5min 🕒| 2hr+                  |

## 📚 Documentation

Explore advanced features:
- [Custom Layers Guide]()
- [ONNX Export Tutorial]()
- [Training Configuration]()
- [NeuralDbg Debugging Features]()

## 📚 Examples

Explore common use cases in `examples/` with step-by-step guides in `docs/examples/`:
- [MNIST Classifier Guide](docs/examples/mnist_guide.md)
- [Sentiment Analysis Guide](docs/examples/sentiment_guide.md)
- [Transformer for NLP Guide](docs/examples/transformer_guide.md)


---


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

## 🌐 Supported Integrations
| Service               | Status | Docs                   |
|-----------------------|--------|------------------------|
| TensorBoard           | ✅     | [Link]()              |
| Weights & Biases      | Beta   | [Link]()              |
| AWS SageMaker         | Q3'24  | Roadmap               |
| NVIDIA Triton         | Q4'24  | Roadmap               |

## 📬 Community

- [Discord Server](https://discord.gg/your-invite-link): Chat with developers
- [Twitter @NLang4438](https://x.com/NLang4438): Updates & announcements
