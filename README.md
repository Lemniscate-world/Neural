# Neural: A Neural Network Programming Language

Neural is a domain-specific language (DSL) designed for defining, training, and deploying neural networks. With an intuitive syntax and powerful abstractions, Neural makes it easy to experiment with complex architectures like CNNs, RNNs, GANs, and Transformers.

## Features

- **Declarative Syntax**: Define neural networks with a clean, YAML-like syntax.
- **Shape Validation**: Automatic tensor shape propagation and validation.
- **Multi-Backend Support**: Generate code for TensorFlow, PyTorch, and ONNX.
- **Training Configuration**: Built-in support for epochs, batch size, and optimizers.
- **Extensible**: Easily add custom layers, activations, and loss functions.

## Installation

To use Neural, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/neural.git
cd neural
pip install -r requirements.txt
```

## Usage

### Define a Neural Network

Create a `.neural` file to define your model:

```plaintext
network MyModel {
    input: (28, 28, 1)
    layers:
        Conv2D(filters=32, kernel_size=(3, 3), activation="relu")
        MaxPooling2D(pool_size=(2, 2))
        Flatten()
        Dense(units=128, activation="relu")
        Dropout(rate=0.5)
        Output(units=10, activation="softmax")
    loss: "categorical_crossentropy"
    optimizer: "adam"
    train {
        epochs: 10
        batch_size: 32
    }
}
```

### Parse and Generate Code

Use the Neural CLI to parse the model and generate code for your preferred backend:

```bash
python neural.py compile --input my_model.neural --backend tensorflow
```

This will generate the following TensorFlow code:

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(data, epochs=10, batch_size=32)
```

### Supported Backends

- **TensorFlow**: Generate TensorFlow/Keras code.
- **PyTorch**: Generate PyTorch code.
- **ONNX**: Export models to ONNX for interoperability.

## Examples

Check out the `examples/` directory for sample `.neural` files and their generated code:

- [MNIST Classifier](examples/mnist.neural)
- [Simple GAN](examples/gan.neural)
- [Transformer Model](examples/transformer.neural)

## Contributing

We welcome contributions! Here’s how you can help:

1. **Report Issues**: Found a bug? Open an issue on GitHub.
2. **Suggest Features**: Have an idea for a new feature? Let us know!
3. **Submit Pull Requests**: Fix a bug or add a feature? Submit a PR.

### Development Setup

1. Fork the repository and clone it locally.
2. Install development dependencies:

   ```bash
   pip install -r requirements-dev.txt
   ```

3. Run tests:

   ```bash
   pytest tests/
   ```

4. Submit your changes as a pull request.

## Roadmap

- [x] Basic CNN support
- [x] TensorFlow code generation
- [ ] PyTorch code generation
- [ ] ONNX export
- [ ] Hyperparameter tuning
- [ ] IDE integration (VSCode, PyCharm)

## License

Neural is released under the [MIT License](LICENSE).
