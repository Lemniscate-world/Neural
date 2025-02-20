import pytest
import numpy as np
from neural.shape_propagation.shape_propagator import ShapePropagator

# Test 1: Test Conv2D propagation (channels_last)
def test_conv2d_shape():
    propagator = ShapePropagator()
    # Input shape: (batch, height, width, channels)
    input_shape = (1, 28, 28, 1)
    layer = {
        "type": "Conv2D", 
        "params": {
            "filters": 32, 
            "kernel_size": (3, 3), 
            "padding": "same", 
            "stride": 1
        }
    }
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    # With "same" padding and stride=1, spatial dims should remain 28.
    # Expected output: (1, 28, 28, 32)
    assert output_shape == (1, 28, 28, 32)

# Test 2: Test MaxPooling2D propagation (channels_last)
def test_maxpooling2d_shape():
    propagator = ShapePropagator()
    input_shape = (1, 28, 28, 32)
    layer = {
        "type": "MaxPooling2D", 
        "params": {
            "pool_size": (2, 2)
        }
    }
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    # For channels_last, height and width should be halved; channels remain same.
    # Expected: (1, 14, 14, 32)
    assert output_shape == (1, 14, 14, 32)

# Test 3: Test Flatten layer preserves batch dimension
def test_flatten_shape():
    propagator = ShapePropagator()
    input_shape = (1, 14, 14, 32)  # (batch, H, W, channels)
    layer = {"type": "Flatten", "params": {}}
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    # Expected flattened shape: (1, 14*14*32)
    expected = (1, 14 * 14 * 32)
    assert output_shape == expected

# Test 4: Test Dense layer preserves batch dimension
def test_dense_shape():
    propagator = ShapePropagator()
    # Simulate flattened input from previous layers
    input_shape = (1, 6272)  # (batch, features)
    layer = {"type": "Dense", "params": {"units": 128}}
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    # Expected: (1, 128)
    assert output_shape == (1, 128)

# Test 5: Test Output layer works as Dense (updates feature dimension)
def test_output_shape():
    propagator = ShapePropagator()
    input_shape = (1, 128)
    layer = {"type": "Output", "params": {"units": 10}}
    output_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    # Expected: (1, 10)
    assert output_shape == (1, 10)

# Test 6: Test complete network propagation with all layers
def test_complete_network_propagation():
    # Starting with a complete network: Conv2D -> MaxPooling2D -> Flatten -> Dense -> Output
    input_shape = (1, 28, 28, 1)
    layers = [
        {"type": "Conv2D", "params": {"filters": 32, "kernel_size": (3, 3), "padding": "same", "stride": 1}},
        {"type": "MaxPooling2D", "params": {"pool_size": (2, 2)}},
        {"type": "Flatten", "params": {}},
        {"type": "Dense", "params": {"units": 128}},
        {"type": "Output", "params": {"units": 10}}
    ]
    
    propagator = ShapePropagator()
    shape = input_shape
    for layer in layers:
        shape = propagator.propagate(shape, layer, framework="tensorflow")
    # Expected final output shape: (1, 10)
    assert shape == (1, 10)

# Test 7: Test visualization report contains expected keys
def test_visualization_report():
    propagator = ShapePropagator(debug=True)
    input_shape = (1, 28, 28, 1)
    layers = [
        {"type": "Conv2D", "params": {"filters": 32, "kernel_size": (3, 3), "padding": "same", "stride": 1}},
        {"type": "MaxPooling2D", "params": {"pool_size": (2, 2)}},
        {"type": "Flatten", "params": {}},
        {"type": "Dense", "params": {"units": 128}},
        {"type": "Output", "params": {"units": 10}}
    ]
    for layer in layers:
        input_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    
    report = propagator.generate_report()
    # The report should have keys for 'dot_graph', 'plotly_chart', and 'shape_history'
    assert "dot_graph" in report
    assert "plotly_chart" in report
    assert "shape_history" in report
    # And shape_history should record one entry per propagated layer plus the initial input.
    # In our network, we visualize the initial input plus each layer.
    # Here, we expect 1 (input) + 5 (layers) = 6 entries.
    assert len(report["shape_history"]) == 6

if __name__ == "__main__":
    pytest.main([__file__])
