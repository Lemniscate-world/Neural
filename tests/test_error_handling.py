import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from neural.shape_propagation.shape_propagator import ShapePropagator
import pytest



def test_invalid_shape():
    input_shape = (28, 28)
    layers = [{"type": "Conv2D", "filters": 32, "kernel_size": (3, 3)}]
    with pytest.raises(ValueError):
        propagate_shape = ShapePropagator.propagate()
        propagate_shape(input_shape, layers[0])
