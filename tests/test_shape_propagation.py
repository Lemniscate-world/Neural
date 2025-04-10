import sys
import os

# Add the parent directory of 'neural' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
from neural.shape_propagation.shape_propagator import ShapePropagator

import pytest
import numpy as np
from neural.shape_propagation.shape_propagator import ShapePropagator

#########################################
# 1. Conv2D with 'valid' padding (channels_last)
#########################################
def test_conv2d_valid_padding():
    propagator = ShapePropagator()
    input_shape = (1, 28, 28, 3)
    layer = {
        "type": "Conv2D",
        "params": {
            "filters": 16,
            "kernel_size": (3, 3),
            "padding": "valid",
            "stride": 1
        }
    }
    # (28 - 3)//1 + 1 = 26 spatial dims
    expected = (1, 26, 26, 16)
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    assert output_shape == expected

#########################################
# 2. Conv2D with stride=2 and 'same' padding (channels_last)
#########################################
def test_conv2d_stride_2():
    propagator = ShapePropagator()
    input_shape = (1, 28, 28, 3)
    layer = {
        "type": "Conv2D",
        "params": {
            "filters": 16,
            "kernel_size": (3, 3),
            "padding": "same",
            "stride": 2
        }
    }
    # For stride 2, expected spatial dims: ceil(28/2)=14 (assuming TensorFlow semantics)
    expected = (1, 14, 14, 16)
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    assert output_shape == expected

#########################################
# 3. Conv2D with integer kernel size (channels_first)
#########################################
def test_conv2d_channels_first_int_kernel():
    propagator = ShapePropagator()
    input_shape = (1, 3, 28, 28)
    layer = {
        "type": "Conv2D",
        "params": {
            "filters": 16,
            "kernel_size": 3,  # integer kernel size
            "padding": "same",
            "stride": 1,
            "data_format": "channels_first"
        }
    }
    # For channels_first, same padding maintains spatial dims: (28, 28)
    expected = (1, 16, 28, 28)
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    assert output_shape == expected

#########################################
# 4. MaxPooling2D with pool_size and stride as integer (channels_last)
#########################################
def test_maxpooling2d_stride_int():
    propagator = ShapePropagator()
    input_shape = (1, 28, 28, 8)
    layer = {
        "type": "MaxPooling2D",
        "params": {
            "pool_size": (2, 2),
            "stride": 2,
            "data_format": "channels_last"
        }
    }
    expected = (1, 14, 14, 8)
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    assert output_shape == expected

#########################################
# 5. MaxPooling2D with channels_first data_format
#########################################
def test_maxpooling2d_channels_first():
    propagator = ShapePropagator()
    input_shape = (1, 16, 28, 28)
    layer = {
        "type": "MaxPooling2D",
        "params": {
            "pool_size": (2, 2),
            "stride": 2,
            "data_format": "channels_first"
        }
    }
    expected = (1, 16, 14, 14)
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    assert output_shape == expected

#########################################
# 6. Flatten layer with 4D input (channels_last)
#########################################
def test_flatten_layer_channels_last():
    propagator = ShapePropagator()
    input_shape = (1, 4, 4, 8)
    layer = {"type": "Flatten", "params": {}}
    expected = (1, 4 * 4 * 8)
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    assert output_shape == expected

#########################################
# 7. Flatten layer without explicit batch dimension (treat first dim as batch)
#########################################
def test_flatten_without_batch():
    propagator = ShapePropagator()
    input_shape = (4, 4, 8)  # Here, 4 is treated as batch size
    layer = {"type": "Flatten", "params": {}}
    expected = (4, 4 * 8)
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    assert output_shape == expected

#########################################
# 8. Dense layer with batched input
#########################################
def test_dense_with_batch():
    propagator = ShapePropagator()
    input_shape = (1, 128)
    layer = {"type": "Dense", "params": {"units": 64}}
    expected = (1, 64)
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    assert output_shape == expected

#########################################
# 9. Dense layer with 1D input (non-batched)
#########################################
def test_dense_without_batch():
    propagator = ShapePropagator()
    input_shape = (256,)  # 1D input
    layer = {"type": "Dense", "params": {"units": 10}}
    expected = (10,)
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    assert output_shape == expected

#########################################
# 10. Output layer functioning like Dense layer
#########################################
def test_output_layer():
    propagator = ShapePropagator()
    input_shape = (1, 64)
    layer = {"type": "Output", "params": {"units": 5}}
    expected = (1, 5)
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    assert output_shape == expected

#########################################
# 11. TransformerEncoder propagation in TensorFlow (shape preserved)
#########################################
def test_transformer_encoder_tf():
    propagator = ShapePropagator()
    input_shape = (1, 10, 64)
    layer = {"type": "TransformerEncoder", "params": {}}
    # For TensorFlow, shape remains unchanged
    expected = input_shape
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    assert output_shape == expected

#########################################
# 12. TransformerEncoder propagation in PyTorch (sequence length & d_model)
#########################################
def test_transformer_encoder_pytorch():
    propagator = ShapePropagator()
    input_shape = (1, 10, 64)
    layer = {"type": "TransformerEncoder", "params": {}}
    # For PyTorch, returns (batch, seq_len)
    expected = (1, 10)
    output_shape = propagator.propagate(input_shape, layer, framework="pytorch")
    assert output_shape == expected

#########################################
# 13. Unknown layer type uses default handler (returns same shape)
#########################################
def test_unknown_layer_type():
    propagator = ShapePropagator()
    input_shape = (1, 20, 20, 3)
    layer = {"type": "CustomLayer", "params": {"foo": 42}}
    expected = input_shape
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    assert output_shape == expected

#########################################
# 14. Conv2D with custom integer padding
#########################################
def test_conv2d_custom_padding():
    propagator = ShapePropagator()
    input_shape = (1, 28, 28, 3)
    layer = {
        "type": "Conv2D",
        "params": {
            "filters": 8,
            "kernel_size": (3, 3),
            "padding": 2,  # explicit integer padding
            "stride": 1
        }
    }
    # Calculate output: (28 + 2*2 - 3)//1 + 1 = (28 + 4 - 3) + 1 = 30
    expected = (1, 30, 30, 8)
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    assert output_shape == expected

#########################################
# 15. Complete network: Two Conv2D layers with 'valid' padding
#########################################
def test_two_conv2d_layers():
    propagator = ShapePropagator()
    input_shape = (1, 28, 28, 3)
    layers = [
        {"type": "Conv2D", "params": {"filters": 16, "kernel_size": (3, 3), "padding": "valid", "stride": 1}},
        {"type": "Conv2D", "params": {"filters": 32, "kernel_size": (3, 3), "padding": "valid", "stride": 1}},
    ]
    shape = input_shape
    # First conv: (28-3)//1+1 = 26; second conv: (26-3)//1+1 = 24
    expected = (1, 24, 24, 32)
    for layer in layers:
        shape = propagator.propagate(shape, layer, framework="tensorflow")
    assert shape == expected

#########################################
# 16. Conv2D with kernel_size provided as a list
#########################################
def test_conv2d_kernel_list():
    propagator = ShapePropagator()
    input_shape = (1, 32, 32, 3)
    layer = {
        "type": "Conv2D",
        "params": {
            "filters": 10,
            "kernel_size": [5, 5],  # provided as list
            "padding": "same",
            "stride": 1
        }
    }
    # With same padding, output spatial dims remain 32
    expected = (1, 32, 32, 10)
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    assert output_shape == expected

#########################################
# 17. Layer using default handler (unknown type) returns input shape unchanged
#########################################
def test_default_handler_layer():
    propagator = ShapePropagator()
    input_shape = (1, 15, 15, 3)
    layer = {"type": "MysteryLayer", "params": {"bar": 100}}
    expected = input_shape
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    assert output_shape == expected

#########################################
# 18. Error when Conv2D kernel size exceeds input dimensions (using 'valid' padding)
#########################################
def test_conv2d_kernel_too_large_error():
    propagator = ShapePropagator()
    input_shape = (1, 28, 28, 3)
    layer = {
        "type": "Conv2D",
        "params": {
            "filters": 16,
            "kernel_size": (30, 30),
            "padding": "valid",
            "stride": 1
        }
    }
    with pytest.raises(ValueError):
        propagator.propagate(input_shape, layer, framework="tensorflow")

#########################################
# 19. Verify that debug mode accumulates execution trace entries
#########################################
def test_debug_mode_execution_trace():
    propagator = ShapePropagator(debug=True)
    input_shape = (1, 28, 28, 1)
    layer = {"type": "Flatten", "params": {}}
    _ = propagator.propagate(input_shape, layer, framework="tensorflow")
    # Expect one entry in the execution trace after one propagation
    assert len(propagator.execution_trace) == 1

#########################################
# 20. Multiple sequential propagations accumulate shape history and trace entries
#########################################
def test_sequential_propagations_accumulate_history():
    propagator = ShapePropagator()
    input_shape = (1, 28, 28, 1)
    layers = [
        {"type": "Conv2D", "params": {"filters": 8, "kernel_size": (3, 3), "padding": "same", "stride": 1}},
        {"type": "MaxPooling2D", "params": {"pool_size": (2, 2)}},
        {"type": "Flatten", "params": {}},
        {"type": "Dense", "params": {"units": 32}},
    ]
    shape = input_shape
    for layer in layers:
        shape = propagator.propagate(shape, layer, framework="tensorflow")
    # Expect shape_history to have one entry per layer propagation
    expected_history_length = len(layers)  # updated from 1 + len(layers) to len(layers)
    assert len(propagator.shape_history) == expected_history_length
    # Also, execution_trace should have an entry per layer propagation
    assert len(propagator.execution_trace) == len(layers)
