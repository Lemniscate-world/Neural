
  ![Design sans titre (2)](https://github.com/user-attachments/assets/8738c025-7081-47da-b825-9cc5e038a86c)


  ⚠️ WARNING: Neural-dsl is a WIP DSL and debugger—bugs exist, feedback welcome!
  This project is under active development and not yet production-ready!

# Neural: A Neural Network Programming Language

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![Discord](https://img.shields.io/badge/Chat-Discord-7289DA)](https://discord.gg/KFku4KvS)
[![Pylint](https://github.com/Lemniscate-SHA-256/Neural/actions/workflows/pylint.yml/badge.svg?branch=main)](https://github.com/Lemniscate-SHA-256/Neural/actions/workflows/pylint.yml)
[![Python package](https://github.com/Lemniscate-SHA-256/Neural/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/Lemniscate-SHA-256/Neural/actions/workflows/python-package.yml)
[![CodeQL Advanced](https://github.com/Lemniscate-SHA-256/Neural/actions/workflows/codeql.yml/badge.svg)](https://github.com/Lemniscate-SHA-256/Neural/actions/workflows/codeql.yml)
[![Tests](https://github.com/Lemniscate-SHA-256/Neural/actions/workflows/pytest.yml/badge.svg?branch=main)](https://github.com/Lemniscate-SHA-256/Neural/actions/workflows/pytest-to-issues.yml)
[![Coverage](https://img.shields.io/codecov/c/github/Lemniscate-SHA-256/Neural)](https://codecov.io/gh/Lemniscate-SHA-256/Neural)
![Dev.to blog](https://img.shields.io/badge/dev.to-0A0A0A?style=for-the-badge&logo=dev.to&logoColor=white)
![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)
![PyPi](https://img.shields.io/badge/pypi-%23ececec.svg?style=for-the-badge&logo=pypi&logoColor=1f73b7)

![design-01jmphv5f1-1740433387](https://github.com/user-attachments/assets/ecbcce19-73df-4696-ace2-69e32d02709f)

## 🎯 Pain Points Solved

Neural addresses deep learning challenges across **Criticality** (how essential) and **Impact Scope** (how transformative):

| Criticality / Impact | Low Impact                  | Medium Impact                       | High Impact                         |
|----------------------|-----------------------------|-------------------------------------|-------------------------------------|
| **High**             |                             |                                     | - **Shape Mismatches**: Pre-runtime validation stops runtime errors.<br>- **Debugging Complexity**: Real-time tracing & anomaly detection. |
| **Medium**           |                             | - **Steep Learning Curve**: No-code GUI eases onboarding. | - **Framework Switching**: One-flag backend swaps.<br>- **HPO Inconsistency**: Unified tuning across frameworks. |
| **Low**              | - **Boilerplate**: Clean DSL syntax saves time. | - **Model Insight**: FLOPs & diagrams.<br>- **Config Fragmentation**: Centralized setup. |                                     |

### Why It Matters
- **Core Value**: Fix critical blockers like shape errors and debugging woes with game-changing tools.  
- **Strategic Edge**: Streamline framework switches and HPO for big wins.  
- **User-Friendly**: Lower barriers and enhance workflows with practical features.

Neural is a domain-specific language (DSL) designed for defining, training, debugging, and deploying neural networks. With **declarative syntax**, **cross-framework support**, and **built-in execution tracing (NeuralDbg)**, it simplifies deep learning development.


## 🚀 Features

- **YAML-like Syntax**: Define models intuitively without framework boilerplate.
- **Shape Propagation**: Catch dimension mismatches *before* runtime.
  - ✅ Interactive shape flow diagrams included.
- **Multi-Framework HPO**: Optimize hyperparameters for both PyTorch and TensorFlow with a single DSL config (#434).
- **Multi-Backend Export**: Generate code for **TensorFlow**, **PyTorch**, or **ONNX**.
- **Training Orchestration**: Configure optimizers, schedulers, and metrics in one place.
- **Visual Debugging**: Render interactive 3D architecture diagrams.
- **Extensible**: Add custom layers/losses via Python plugins.

### **🛠 NeuralDbg: Built-in Neural Network Debugger**
NeuralDbg provides **real-time execution tracing, profiling, and debugging**, allowing you to visualize and analyze deep learning models in action.

✅ **Real-Time Execution Monitoring** – Track activations, gradients, memory usage, and FLOPs.  
![test_trace_graph](https://github.com/user-attachments/assets/15b1edd2-2643-4587-9843-aa4697ed2e4b)
![test_flops_memory_chart](https://github.com/user-attachments/assets/de1f6504-787b-4948-b543-fe3d2f8bfd74)
![test_trace_graph_stacked](https://github.com/user-attachments/assets/529fc487-fb31-48ad-bb11-b0c64ab330ed)
![test_trace_graph_heatmap](https://github.com/user-attachments/assets/debef7d5-9989-45da-ae91-7cef19aac2b0)
![test_anomaly_chart](https://github.com/user-attachments/assets/b57d3142-6da8-4d57-94f0-486d1797e92c)
![test_dead_neurons](https://github.com/user-attachments/assets/f4629b4f-2988-410e-8b49-3dde225f926f)
![test_gradient_chart](https://github.com/user-attachments/assets/ca6b9f20-7dd8-4c72-9ee8-a3f35af6208b)


✅ **Shape Propagation Debugging** – Visualize tensor transformations at each layer.  
✅ **Gradient Flow Analysis** – Detect **vanishing & exploding gradients**.  
✅ **Dead Neuron Detection** – Identify inactive neurons in deep networks.  
✅ **Anomaly Detection** – Spot **NaNs, extreme activations, and weight explosions**.  
✅ **Step Debugging Mode** – Pause execution and inspect tensors manually.


## 📦 Installation

# Clone the repository
git clone https://github.com/yourusername/neural.git
cd neural

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies

```bash
pip install -r requirements.txt
```

```bash
pip install neural-dsl
```

see v0.2.3 for bug fixes

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

- [DSL Documentation](docs/dsl.md)

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

## 🕸Architecture Graphs (Zoom Alot For Some🫩)

![classes](https://github.com/Lemniscate-SHA-256/Neural/blob/main/classes.png)
![packages](https://github.com/Lemniscate-SHA-256/Neural/blob/main/packages.png)



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

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Lemniscate-SHA-256/Neural&type=Timeline)](https://www.star-history.com/#Lemniscate-SHA-256/Neural&Timeline)

## Support
Please give us a star ⭐️ to increase our chances of getting into GitHub trends - the more attention we get, the higher our chances of actually making a difference.
Please share this project with your friends! Every share helps us reach more developers and grow our community. The more developers we reach, the more likely we are to build something truly revolutionary together. 🚀

## 📬 Community

- [Discord Server](https://discord.gg/KFku4KvS): Chat with developers
- [Twitter @NLang4438](https://x.com/NLang4438): Updates & announcements
