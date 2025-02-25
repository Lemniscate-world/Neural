# test_code_generator.py (expanded)
import pytest
import os
import onnx
from onnx import checker
from neural.code_generation.code_generator import generate_code, save_file, load_file, export_onnx, to_number
from neural.parser.parser import create_parser, ModelTransformer

# Existing fixtures...

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
        "optimizer": {"type": "Adam", "params": {"learning_rate": 0.001}}
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

# Parameterized test cases
layer_test_cases = [
    ("Conv2D", {"filters": 64, "kernel_size": (3,3)}, "Conv2D(filters=64, kernel_size=(3, 3)"),
    ("LSTM", {"units": 128, "return_sequences": True}, "LSTM(units=128, return_sequences=True"),
    ("BatchNormalization", {}, "BatchNormalization()"),
    ("Dropout", {"rate": 0.3}, "Dropout(rate=0.3"),
    ("Dense", {"units": 256, "activation": "tanh"}, "Dense(units=256, activation='tanh'")
]

# Enhanced test cases
def test_generate_tensorflow_complex(complex_model_data):
    """Test complex model generation for TensorFlow"""
    code = generate_code(complex_model_data, "tensorflow")
    
    # Verify model structure
    assert "Conv2D(filters=64, kernel_size=3, padding='same'" in code
    assert "BatchNormalization()" in code
    assert "MaxPooling2D(pool_size=2)" in code
    assert "Dense(units=256, activation='relu'" in code
    assert "Dropout(rate=0.5)" in code
    
    # Verify compilation
    assert "model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001)" in code

def test_generate_pytorch_complex(complex_model_data):
    """Test complex model generation for PyTorch"""
    code = generate_code(complex_model_data, "pytorch")
    
    # Verify residual block
    assert "self.layer0_residual = nn.Sequential(" in code
    assert "nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)" in code
    assert "nn.BatchNorm2d(num_features=64)" in code
    
    # Verify forward pass
    assert "x = x + self.layer0_residual(x)" in code
    assert "x = self.layer2_flatten(x)" in code
    assert "x = self.layer4_dropout(x)" in code

def test_generate_pytorch_channels_first(channels_first_model_data):
    """Test channels_first data format handling in PyTorch"""
    code = generate_code(channels_first_model_data, "pytorch")
    
    # Verify Conv2D parameters
    assert "nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)" in code
    assert "nn.MaxPool2d(kernel_size=2)" in code

def test_onnx_model_structure(simple_model_data, tmp_path):
    """Test ONNX model structure validation"""
    file_path = tmp_path / "model.onnx"
    export_onnx(simple_model_data, str(file_path))
    
    # Load and validate ONNX model
    model = onnx.load(str(file_path))
    checker.check_model(model)
    
    # Verify input/output shapes
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
        "optimizer": "adam"
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
        "optimizer": "sgd"
    }
    code = generate_code(model_data, "pytorch")
    
    # Verify layer initialization
    if layer_type == "LSTM":
        assert f"nn.{layer_type}(input_size=3" in code
    elif layer_type == "Conv2D":
        assert "nn.Conv2d(" in code
    elif layer_type == "BatchNormalization":
        assert "nn.BatchNorm2d(" in code

def test_invalid_activation_handling():
    """Test handling of invalid activation functions"""
    model_data = {
        "input": {"shape": (None, 32)},
        "layers": [{"type": "Dense", "params": {"units": 64, "activation": "invalid"}}],
        "loss": "mse",
        "optimizer": "adam"
    }
    
    # TensorFlow should use the string directly
    tf_code = generate_code(model_data, "tensorflow")
    assert "activation='invalid'" in tf_code
    
    # PyTorch should fall back to Identity
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
        "optimizer": "adam"
    }
    
    # TensorFlow shape propagation
    tf_code = generate_code(model_data, "tensorflow")
    assert "input_shape=(28, 28, 1)" in tf_code
    
    # PyTorch shape propagation
    pt_code = generate_code(model_data, "pytorch")
    assert "in_features=5408" in pt_code  # (28-3+1)/2 = 13, 13x13x32=5408

def test_custom_optimizer_params():
    """Test handling of custom optimizer parameters"""
    model_data = {
        "input": {"shape": (None, 32)},
        "layers": [{"type": "Dense", "params": {"units": 64}}],
        "loss": "mse",
        "optimizer": {"type": "Adam", "params": {"lr": 0.01, "weight_decay": 0.001}}
    }
    
    # PyTorch optimizer configuration
    pt_code = generate_code(model_data, "pytorch")
    assert "optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)" in pt_code

def test_to_number():
    """Test string to number conversion"""
    assert to_number("42") == 42
    assert to_number("3.14") == 3.14
    assert to_number("-15") == -15
    assert to_number("invalid") == "invalid"  # Should this raise instead?

def test_file_handling_errors(tmp_path):
    """Test file handling edge cases"""
    # Invalid path
    invalid_path = tmp_path / "invalid_dir" / "test.py"
    with pytest.raises(IOError):
        save_file(str(invalid_path), "test")
        
    # Unreadable file
    valid_path = tmp_path / "test.nr"
    valid_path.write_text("invalid content")
    with pytest.raises(ValueError):
        load_file(str(valid_path))

def test_unsupported_layer_handling():
    """Test handling of unsupported layer types"""
    model_data = {
        "input": {"shape": (None, 32)},
        "layers": [{"type": "QuantumLayer"}],
        "loss": "mse",
        "optimizer": "adam"
    }
    
    # Should generate warning but still produce code
    tf_code = generate_code(model_data, "tensorflow")
    assert "Warning: Unsupported layer type 'QuantumLayer'" in tf_code
    assert "Sequential" in tf_code

def test_activation_function_mapping():
    """Test correct activation function mapping"""
    model_data = {
        "input": {"shape": (None, 32)},
        "layers": [
            {"type": "Dense", "params": {"units": 64, "activation": "gelu"}},
            {"type": "Dense", "params": {"units": 10, "activation": "softmax"}}
        ],
        "loss": "categorical_crossentropy",
        "optimizer": "adam"
    }
    
    # PyTorch specific activations
    pt_code = generate_code(model_data, "pytorch")
    assert "nn.GELU()" in pt_code
    assert "nn.Softmax(dim=1)" in pt_code

def test_multi_input_handling():
    """Test models with multiple inputs"""
    model_data = {
        "input": [
            {"shape": (None, 32)},
            {"shape": (None, 64)}
        ],
        "layers": [
            {"type": "Concatenate"},
            {"type": "Dense", "params": {"units": 128}}
        ],
        "loss": "mse",
        "optimizer": "adam"
    }
    
    # TensorFlow implementation
    tf_code = generate_code(model_data, "tensorflow")
    assert "tf.keras.layers.Concatenate()" in tf_code
    assert "input_shape=[(32,), (64,)]" in tf_code

@pytest.fixture
def multiplied_layers_model():
    return {
        "input": {"shape": (None, 32)},
        "layers": [
            {"type": "Dense", "params": {"units": 64}, "*": 3},
            {"type": "Dropout", "params": {"rate": 0.5}, "*": 2}
        ],
        "loss": "mse",
        "optimizer": "adam"
    }

def test_layer_multiplication(multiplied_layers_model):
    tf_code = generate_code(multiplied_layers_model, "tensorflow")
    pt_code = generate_code(multiplied_layers_model, "pytorch")
    
    # Verify TensorFlow
    assert tf_code.count("Dense(units=64") == 3
    assert tf_code.count("Dropout(rate=0.5") == 2
    
    # Verify PyTorch
    assert pt_code.count("self.layer0_dense") == 1
    assert pt_code.count("self.layer3_dropout") == 1
    assert "x = self.layer0_dense(x)" in pt_code
    assert "x = self.layer1_dense(x)" in pt_code
    assert "x = self.layer3_dropout(x)" in pt_code


@pytest.fixture
def transformer_model_data():
    return {
        "type": "model",
        "input": {"shape": (None, 128)},
        "layers": [
            {
                "type": "TransformerEncoder",
                "params": {
                    "num_heads": 4,
                    "ff_dim": 256,
                    "dropout": 0.1
                }
            },
            {"type": "Dense", "params": {"units": 10}}
        ],
        "loss": "categorical_crossentropy",
        "optimizer": "adam"
    }

def test_transformer_generation(transformer_model_data):
    # TensorFlow
    tf_code = generate_code(transformer_model_data, "tensorflow")
    assert "MultiHeadAttention" in tf_code
    assert "LayerNormalization" in tf_code
    
    # PyTorch
    pt_code = generate_code(transformer_model_data, "pytorch")
    assert "TransformerEncoderLayer(" in pt_code
    assert "dim_feedforward=256" in pt_code
    assert "nhead=4" in pt_code


# Add more tests for:
# - Different pooling configurations
# - Various RNN types and configurations
# - Batch norm parameters
# - Custom layer handling
# - Mixed precision training
# - Model saving/loading