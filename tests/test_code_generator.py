import pytest
import os
import onnx
from onnx import checker
from neural.code_generation.code_generator import generate_code, save_file, load_file, export_onnx, to_number
from neural.parser.parser import create_parser, ModelTransformer

# Setup for temporary file handling
@pytest.fixture
def tmp_path(tmp_path):
    return tmp_path

# Fixtures for model data
@pytest.fixture
def simple_model_data():
    return {
        "input": {"shape": (None, 32, 32, 3)},
        "layers": [
            {"type": "Conv2D", "params": {"filters": 16, "kernel_size": 3}},
            {"type": "Output", "params": {"units": 10}}
        ],
        "loss": "mse",
        "optimizer": "Adam"
    }

@pytest.fixture
def complex_model_data():
    """Model with multiple layer types and nested structures"""
    return {
        "type": "model",
        "name": "ComplexNet",
        "input": {"type": "Input", "shape": (None, 64, 64, 3)},
        "layers": [
            {
                "type": "Residual",
                "sub_layers": [
                    {"type": "Conv2D", "params": {"filters": 64, "kernel_size": 3, "padding": "same"}},
                    {"type": "BatchNormalization"}
                ]
            },
            {"type": "MaxPooling2D", "params": {"pool_size": 2}},
            {"type": "Flatten"},
            {"type": "Dense", "params": {"units": 256, "activation": "relu"}},
            {"type": "Dropout", "params": {"rate": 0.5}},
            {"type": "Output", "params": {"units": 10, "activation": "softmax"}}
        ],
        "loss": {"value": "categorical_crossentropy"},
        "optimizer": {"type": "Adam", "params": {"lr": 0.001}}
    }

@pytest.fixture
def channels_first_model_data():
    """Model with channels_first data format"""
    return {
        "type": "model",
        "name": "ChannelsFirstNet",
        "input": {"type": "Input", "shape": (None, 3, 32, 32)},
        "layers": [
            {"type": "Conv2D", "params": {"filters": 32, "kernel_size": 3, "data_format": "channels_first"}},
            {"type": "MaxPooling2D", "params": {"pool_size": 2}},
            {"type": "Flatten"},
            {"type": "Dense", "params": {"units": 10}}
        ],
        "loss": "mse",
        "optimizer": "SGD"
    }

@pytest.fixture
def transformer_model_data():
    return {
        "type": "model",
        "input": {"shape": (None, 128)},
        "layers": [
            {
                "type": "TransformerEncoder",
                "params": {"num_heads": 4, "ff_dim": 256, "dropout": 0.1}
            },
            {"type": "Dense", "params": {"units": 10}}
        ],
        "loss": "categorical_crossentropy",
        "optimizer": "Adam"
    }

@pytest.fixture
def multiplied_layers_model():
    return {
        "input": {"shape": (None, 32)},
        "layers": [
            {"type": "Dense", "params": {"units": 64}, "multiply": 3},
            {"type": "Dropout", "params": {"rate": 0.5}, "multiply": 2}
        ],
        "loss": "mse",
        "optimizer": "Adam"
    }

@pytest.fixture
def rnn_model_data():
    return {
        "input": {"shape": (None, 10, 32)},  # Timesteps, features
        "layers": [
            {"type": "LSTM", "params": {"units": 128, "return_sequences": True}},
            {"type": "GRU", "params": {"units": 64}},
            {"type": "Dense", "params": {"units": 10}}
        ],
        "loss": "mse",
        "optimizer": "Adam"
    }

# Parameterized test cases for layers
layer_test_cases = [
    ("Conv2D", {"filters": 64, "kernel_size": (3, 3)}, "Conv2D(filters=64, kernel_size=(3, 3)"),
    ("LSTM", {"units": 128, "return_sequences": True}, "LSTM(units=128, return_sequences=True"),
    ("BatchNormalization", {}, "BatchNormalization()"),
    ("Dropout", {"rate": 0.3}, "Dropout(rate=0.3"),
    ("Dense", {"units": 256, "activation": "tanh"}, "Dense(units=256, activation='tanh'")
]

# Existing Tests (Enhanced)
def test_generate_tensorflow_complex(complex_model_data):
    """Test complex model generation for TensorFlow"""
    code = generate_code(complex_model_data, "tensorflow")
    assert "layers.Conv2D(filters=64, kernel_size=3, padding='same')" in code
    assert "layers.BatchNormalization()" in code
    assert "layers.Add()" in code
    assert "MaxPooling2D(pool_size=2)" in code
    assert "Dense(units=256, activation='relu')" in code
    assert "Dropout(rate=0.5)" in code
    assert "model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001))" in code

def test_generate_pytorch_complex(complex_model_data):
    """Test complex model generation for PyTorch"""
    code = generate_code(complex_model_data, "pytorch")
    assert "self.layer0_residual = nn.Sequential(" in code
    assert "nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)" in code
    assert "nn.BatchNorm2d(num_features=64)" in code
    assert "x = x + self.layer0_residual(x)" in code
    assert "x = self.layer2_flatten(x)" in code
    assert "x = self.layer4_dropout(x)" in code

def test_generate_pytorch_channels_first(channels_first_model_data):
    """Test channels_first data format handling in PyTorch"""
    code = generate_code(channels_first_model_data, "pytorch")
    assert "nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)" in code
    assert "nn.MaxPool2d(kernel_size=2)" in code

def test_onnx_model_structure(simple_model_data, tmp_path):
    """Test ONNX model structure validation"""
    file_path = tmp_path / "model.onnx"
    export_onnx(simple_model_data, str(file_path))
    model = onnx.load(str(file_path))
    checker.check_model(model)
    inputs = [input.name for input in model.graph.input]
    outputs = [output.name for output in model.graph.output]
    assert "input" in inputs
    assert "output" in outputs

@pytest.mark.parametrize("layer_type,params,expected", layer_test_cases)
def test_tensorflow_layer_generation(layer_type, params, expected):
    """Test generation of individual layer types for TensorFlow"""
    model_data = {
        "input": {"shape": (None, 32, 32, 3)},
        "layers": [{"type": layer_type, "params": params}],
        "loss": "mse",
        "optimizer": "Adam"
    }
    code = generate_code(model_data, "tensorflow")
    assert expected in code

@pytest.mark.parametrize("layer_type,params,expected", layer_test_cases)
def test_pytorch_layer_generation(layer_type, params, expected):
    """Test generation of individual layer types for PyTorch"""
    model_data = {
        "input": {"shape": (None, 3, 32, 32)},
        "layers": [{"type": layer_type, "params": params}],
        "loss": "mse",
        "optimizer": "SGD"
    }
    code = generate_code(model_data, "pytorch")
    if layer_type == "LSTM":
        assert "nn.LSTM(input_size=32" in code  # Adjust for input shape
    elif layer_type == "Conv2D":
        assert "nn.Conv2d(in_channels=3" in code
    elif layer_type == "BatchNormalization":
        assert "nn.BatchNorm2d(" in code

def test_invalid_activation_handling():
    """Test handling of invalid activation functions"""
    model_data = {
        "input": {"shape": (None, 32)},
        "layers": [{"type": "Dense", "params": {"units": 64, "activation": "invalid"}}],
        "loss": "mse",
        "optimizer": "Adam"
    }
    tf_code = generate_code(model_data, "tensorflow")
    assert "activation='invalid'" in tf_code
    pt_code = generate_code(model_data, "pytorch")
    assert "nn.Identity()" in pt_code

def test_shape_propagation():
    """Test end-to-end shape propagation"""
    model_data = {
        "input": {"shape": (None, 28, 28, 1)},
        "layers": [
            {"type": "Conv2D", "params": {"filters": 32, "kernel_size": 3}},
            {"type": "MaxPooling2D", "params": {"pool_size": 2}},
            {"type": "Flatten"},
            {"type": "Dense", "params": {"units": 10}}
        ],
        "loss": "mse",
        "optimizer": "Adam"
    }
    tf_code = generate_code(model_data, "tensorflow")
    assert "input_shape=(28, 28, 1)" in tf_code
    pt_code = generate_code(model_data, "pytorch")
    assert "in_features=5408" in pt_code  # (28-3+1)/2 = 13 -> 13x13x32 = 5408

def test_custom_optimizer_params():
    """Test handling of custom optimizer parameters"""
    model_data = {
        "input": {"shape": (None, 32)},
        "layers": [{"type": "Dense", "params": {"units": 64}}],
        "loss": "mse",
        "optimizer": {"type": "Adam", "params": {"lr": 0.01, "weight_decay": 0.001}}
    }
    pt_code = generate_code(model_data, "pytorch")
    assert "optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)" in pt_code

def test_to_number():
    """Test string to number conversion"""
    assert to_number("42") == 42
    assert to_number("3.14") == 3.14
    assert to_number("-15") == -15
    with pytest.raises(ValueError):  # Fix: Should raise ValueError for invalid input
        to_number("invalid")

def test_file_handling_errors(tmp_path):
    """Test file handling edge cases"""
    invalid_path = tmp_path / "invalid_dir" / "test.py"
    with pytest.raises(IOError):
        save_file(str(invalid_path), "test")
    valid_path = tmp_path / "test.nr"
    valid_path.write_text("invalid content")
    with pytest.raises(ValueError):
        load_file(str(valid_path))

def test_layer_multiplication(multiplied_layers_model):
    """Test layer multiplication"""
    tf_code = generate_code(multiplied_layers_model, "tensorflow")
    pt_code = generate_code(multiplied_layers_model, "pytorch")
    assert tf_code.count("Dense(units=64") == 3
    assert tf_code.count("Dropout(rate=0.5") == 2
    assert pt_code.count("self.layer0_dense") == 1
    assert pt_code.count("self.layer2_dense") == 1
    assert "x = self.layer0_dense(x)" in pt_code
    assert "x = self.layer3_dropout(x)" in pt_code

def test_transformer_generation(transformer_model_data):
    """Test TransformerEncoder generation"""
    tf_code = generate_code(transformer_model_data, "tensorflow")
    assert "MultiHeadAttention" in tf_code
    assert "LayerNormalization" in tf_code
    pt_code = generate_code(transformer_model_data, "pytorch")
    assert "TransformerEncoderLayer(" in pt_code
    assert "dim_feedforward=256" in pt_code
    assert "nhead=4" in pt_code

# New Tests
def test_different_pooling_configs():
    """Test various pooling configurations"""
    model_data = {
        "input": {"shape": (None, 32, 32, 3)},
        "layers": [
            {"type": "MaxPooling2D", "params": {"pool_size": (3, 3), "strides": 2}},
            {"type": "AveragePooling2D", "params": {"pool_size": 2}},
            {"type": "Dense", "params": {"units": 10}}
        ],
        "loss": "mse",
        "optimizer": "Adam"
    }
    tf_code = generate_code(model_data, "tensorflow")
    assert "MaxPooling2D(pool_size=(3, 3)" in tf_code  # Strides not supported yet in your code
    assert "AveragePooling2D(pool_size=2)" in tf_code
    pt_code = generate_code(model_data, "pytorch")
    assert "nn.MaxPool2d(kernel_size=(3, 3)" in pt_code
    assert "nn.AvgPool2d(kernel_size=2)" in pt_code

def test_rnn_types_and_configs(rnn_model_data):
    """Test various RNN types and configurations"""
    tf_code = generate_code(rnn_model_data, "tensorflow")
    assert "LSTM(units=128, return_sequences=True)" in tf_code
    assert "GRU(units=64" in tf_code
    pt_code = generate_code(rnn_model_data, "pytorch")
    assert "nn.LSTM(input_size=32, hidden_size=128, batch_first=True)" in pt_code
    assert "nn.GRU(input_size=128, hidden_size=64, batch_first=True)" in pt_code

def test_batch_norm_params():
    """Test batch normalization with parameters"""
    model_data = {
        "input": {"shape": (None, 32, 32, 3)},
        "layers": [
            {"type": "BatchNormalization", "params": {"momentum": 0.9, "epsilon": 0.001}},
            {"type": "Dense", "params": {"units": 10}}
        ],
        "loss": "mse",
        "optimizer": "Adam"
    }
    tf_code = generate_code(model_data, "tensorflow")
    assert "BatchNormalization(momentum=0.9, epsilon=0.001)" in tf_code
    pt_code = generate_code(model_data, "pytorch")
    assert "nn.BatchNorm2d(num_features=3, momentum=0.9, eps=0.001)" in pt_code

def test_custom_layer_handling():
    """Test handling of unsupported/custom layer types"""
    model_data = {
        "input": {"shape": (None, 32)},
        "layers": [{"type": "QuantumLayer", "params": {"qubits": 4}}],
        "loss": "mse",
        "optimizer": "Adam"
    }
    with pytest.warns(UserWarning, match="Unsupported layer type 'QuantumLayer'"):
        tf_code = generate_code(model_data, "tensorflow")
    assert "model = tf.keras.Model" in tf_code  # Still generates a model
    with pytest.warns(UserWarning, match="Unsupported layer type 'QuantumLayer'"):
        pt_code = generate_code(model_data, "pytorch")
    assert "class NeuralNetworkModel(nn.Module)" in pt_code

def test_mixed_precision_training():
    """Test mixed precision training support"""
    model_data = {
        "input": {"shape": (None, 32, 32, 3)},
        "layers": [{"type": "Dense", "params": {"units": 10}}],
        "loss": "mse",
        "optimizer": "Adam",
        "training_config": {"mixed_precision": True}
    }
    tf_code = generate_code(model_data, "tensorflow")
    assert "from tensorflow.keras.mixed_precision import set_global_policy" in tf_code
    assert "set_global_policy('mixed_float16')" in tf_code
    pt_code = generate_code(model_data, "pytorch")
    assert "torch.cuda.amp.autocast()" in pt_code  # Add to training loop

def test_model_saving_loading(simple_model_data, tmp_path):
    """Test model saving and loading in generated code"""
    model_data = simple_model_data.copy()
    model_data["training_config"] = {"save_path": "model.h5"}
    tf_code = generate_code(model_data, "tensorflow")
    assert "model.save('model.h5')" in tf_code
    pt_code = generate_code(model_data, "pytorch")
    assert "torch.save(model.state_dict(), 'model.h5')" in pt_code