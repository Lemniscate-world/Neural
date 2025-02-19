import os
import sys


# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from neural.shape_propagation.shape_propagator import ShapePropagator

def test_shape_propagation():
    input_shape = (1, 28, 28, 1)  # Add batch dimension (1)
    layers = [
        # Example Conv2D layer with explicit padding
        {"type": "Conv2D", "params": {"filters": 32, "kernel_size": (3, 3), "padding": "same"}},
        {"type": "MaxPooling2D", "params": {"pool_size": (2, 2)}},
        {"type": "Flatten", "params": {}},
        {"type": "Dense", "params": {"units": 128}},
        {"type": "Output", "params": {"units": 10}}
    ]
    
    propagator = ShapePropagator()
    for layer in layers:
        input_shape = propagator.propagate(input_shape, layer, framework="tensorflow")
    
    assert input_shape == (1, 10)  # Output shape includes batch dimension

def test_calculate_shape_propagation():
    input_shape = (28, 28, 1)
    layers = [
        {"type": "Conv2D", "filters": 32, "kernel_size": (3, 3)},
        {"type": "MaxPooling2D", "pool_size": (2, 2)},
        {"type": "Flatten"},
        {"type": "Dense", "units": 128},
        {"type": "Output", "units": 10}
    ]
    
def test_conv2d_shape():
    propagator = ShapePropagator()
    input_shape = (1, 1, 28, 28)  # PyTorch format (batch=1, channels=1)
    output = propagator.propagate(
        input_shape,
        {"type": "Conv2D", "params": {"out_channels": 32, "kernel_size": 3}},
        framework="pytorch"
    )
    assert output == (1, 32, 26, 26)