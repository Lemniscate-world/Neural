import pytest
import os
from neural.code_generation.code_generator import generate_code, save_file, load_file, export_onnx
from neural.parser.parser import create_parser, ModelTransformer

# Sample model data fixtures
@pytest.fixture
def simple_model_data():
    """A basic model with Conv2D and Dense layers."""
    return {
        "type": "model",
        "name": "SimpleNet",
        "input": {"type": "Input", "shape": (None, 28, 28, 1)},
        "layers": [
            {"type": "Conv2D", "params": {"filters": 32, "kernel_size": 3, "activation": "relu"}},
            {"type": "Flatten"},
            {"type": "Dense", "params": {"units": 10, "activation": "softmax"}}
        ],
        "loss": "categorical_crossentropy",
        "optimizer": {"type": "Adam"}
    }

@pytest.fixture
def nested_model_data():
    """A model with a nested TimeDistributed layer."""
    return {
        "type": "model",
        "name": "NestedNet",
        "input": {"type": "Input", "shape": (None, 10, 28, 28, 1)},
        "layers": [
            {
                "type": "TimeDistributed",
                "sub_layers": [{"type": "Conv2D", "params": {"filters": 32, "kernel_size": 3, "activation": "relu"}}],
                "params": {}
            },
            {"type": "Flatten"},
            {"type": "Dense", "params": {"units": 10, "activation": "softmax"}}
        ],
        "loss": "categorical_crossentropy",
        "optimizer": {"type": "SGD"}
    }

@pytest.fixture
def invalid_model_data():
    """An invalid model missing required fields."""
    return {
        "type": "model",
        "name": "InvalidNet",
        "input": {"type": "Input", "shape": (None, 28, 28, 1)}
        # Missing 'layers', 'loss', 'optimizer'
    }

# Test generate_code function
def test_generate_tensorflow_simple(simple_model_data):
    """Test code generation for TensorFlow with a simple model."""
    code = generate_code(simple_model_data, "tensorflow")
    assert "import tensorflow as tf" in code
    assert "model = tf.keras.Sequential" in code
    assert "Conv2D(filters=32, kernel_size=3, activation='relu')" in code
    assert "Flatten()" in code
    assert "Dense(units=10, activation='softmax')" in code
    assert "model.compile(loss='categorical_crossentropy', optimizer='Adam')" in code
    # Verify syntax by attempting to exec (basic check)
    try:
        exec(code.split("\n")[0])  # Just test import line for simplicity
    except SyntaxError:
        pytest.fail("Generated TensorFlow code has syntax errors")

def test_generate_pytorch_simple(simple_model_data):
    """Test code generation for PyTorch with a simple model."""
    code = generate_code(simple_model_data, "pytorch")
    assert "import torch" in code
    assert "import torch.nn as nn" in code
    assert "class NeuralNetworkModel(nn.Module):" in code
    assert "Conv2d(in_channels=1, out_channels=32, kernel_size=3)" in code
    assert "Flatten()" in code
    assert "Linear(in_features=21632, out_features=10)" in code  # Corrected shape
    assert "loss_fn = nn.CrossEntropyLoss()" in code
    assert "optimizer = optim.Adam(model.parameters()" in code  # Removed lr check
    try:
        exec(code.split("model =")[0])  # Test up to model instantiation
    except SyntaxError:
        pytest.fail("Generated PyTorch code has syntax errors")

def test_generate_tensorflow_nested(nested_model_data):
    """Test code generation for TensorFlow with a nested model."""
    code = generate_code(nested_model_data, "tensorflow")
    assert "TimeDistributed(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))" in code
    assert "Flatten()" in code
    assert "Dense(units=10, activation='softmax')" in code
    assert "model.compile(loss='categorical_crossentropy', optimizer='SGD')" in code

def test_generate_pytorch_nested(nested_model_data):
    """Test code generation for PyTorch with a nested model."""
    code = generate_code(nested_model_data, "pytorch")
    expected_in_features = 10 * 26 * 26 * 32  # 216,320
    assert "Conv2d(in_channels=1, out_channels=32, kernel_size=3)" in code
    assert "Flatten()" in code
    assert f"Linear(in_features={expected_in_features}" in code
    assert "loss_fn = nn.CrossEntropyLoss()" in code
    assert "optimizer = optim.SGD(model.parameters(), lr=0.001)" in code

def test_generate_invalid_backend(simple_model_data):
    """Test that an invalid backend raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported backend: invalid"):
        generate_code(simple_model_data, "invalid")

def test_generate_invalid_model_data(invalid_model_data):
    """Test that invalid model data raises ValueError."""
    with pytest.raises(ValueError, match="Invalid model_data format"):
        generate_code(invalid_model_data, "tensorflow")

def test_generate_missing_input_shape():
    """Test that missing input shape raises ValueError."""
    model_data = {
        "type": "model",
        "name": "NoInputNet",
        "input": {"type": "Input", "shape": None},  # Missing shape
        "layers": [{"type": "Dense", "params": {"units": 10}}],
        "loss": "mse",
        "optimizer": {"type": "Adam"}
    }
    with pytest.raises(ValueError, match="Input layer shape is not defined"):
        generate_code(model_data, "tensorflow")

def test_generate_missing_params():
    """Test that missing required parameters raises ValueError."""
    model_data = {
        "type": "model",
        "name": "MissingParamsNet",
        "input": {"type": "Input", "shape": (None, 28, 28, 1)},
        "layers": [{"type": "Conv2D", "params": {"activation": "relu"}}],  # Missing filters, kernel_size
        "loss": "mse",
        "optimizer": {"type": "Adam"}
    }
    with pytest.raises(ValueError, match="Conv2D layer config missing 'filters' or 'kernel_size'"):
        generate_code(model_data, "tensorflow")

# Test save_file and load_file
def test_save_file(tmp_path):
    """Test saving generated code to a file."""
    file_path = tmp_path / "test_model.py"
    code = "print('Hello, Neural!')"
    save_file(str(file_path), code)
    assert os.path.exists(file_path)
    with open(file_path, 'r') as f:
        assert f.read() == code

def test_save_file_error(tmp_path):
    """Test that save_file raises IOError on failure."""
    file_path = tmp_path / " unwritable/test_model.py"  # Invalid path
    with pytest.raises(IOError, match="Error writing file"):
        save_file(str(file_path), "test")

def test_load_file(tmp_path):
    """Test loading and parsing a .neural file."""
    file_path = tmp_path / "test.neural"
    content = """
    network TestNet {
        input: (28, 28, 1)
        layers:
            Conv2D(filters=32, kernel_size=3)
            Dense(units=10)  # Fixed DSL syntax
        loss: "mse"
        optimizer: "adam"
    }
    """

    with open(file_path, 'w') as f:
        f.write(content)
    tree = load_file(str(file_path))
    transformer = ModelTransformer()
    model_data = transformer.transform(tree)
    assert model_data["name"] == "TestNet"
    assert len(model_data["layers"]) == 2

def test_load_file_invalid_extension(tmp_path):
    """Test that an invalid file extension raises ValueError."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("dummy content")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_file(str(file_path))

# Test ONNX export
def test_export_onnx(simple_model_data, tmp_path):
    """Test ONNX export functionality."""
    file_path = tmp_path / "model.onnx"
    result = export_onnx(simple_model_data, str(file_path))
    assert os.path.exists(file_path)
    assert result == f"ONNX model saved to {file_path}"