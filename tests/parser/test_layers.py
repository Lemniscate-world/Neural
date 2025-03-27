import os
import sys
import pytest
from lark import exceptions

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neural.parser.parser import ModelTransformer, create_parser, DSLValidationError

# Fixtures
@pytest.fixture
def layer_parser():
    return create_parser('layer')

@pytest.fixture
def transformer():
    return ModelTransformer()

# Layer Parsing Tests
@pytest.mark.parametrize(
    "layer_string, expected, test_id",
    [
        # Basic layers
        ('Dense(10)', {'type': 'Dense', 'params': {'units': 10}, 'sublayers': []}, "dense-basic"),
        ('Conv2D(32, (3, 3))', {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': (3, 3)}, 'sublayers': []}, "conv2d-basic"),
        
        # With activation
        ('Dense(10, "relu")', {'type': 'Dense', 'params': {'units': 10, 'activation': 'relu'}, 'sublayers': []}, "dense-with-activation"),
        
        # Named parameters
        ('Dense(units=10, activation="relu")', {'type': 'Dense', 'params': {'units': 10, 'activation': 'relu'}, 'sublayers': []}, "dense-named-params"),
        
        # Multiple nested layers with comments
        ('''Residual() {  # Outer comment
            Conv2D(32, (3, 3))  # Inner comment 1
            BatchNormalization()  # Inner comment 2
        }''', 
        {'type': 'Residual', 'params': None, 'sublayers': [
            {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': (3, 3)}, 'sublayers': []},
            {'type': 'BatchNormalization', 'params': None, 'sublayers': []}
        ]}, "residual-with-comments"),
    ],
    ids=["dense-basic", "conv2d-basic", "dense-with-activation", "dense-named-params", "residual-with-comments"]
)
def test_layer_parsing(layer_parser, transformer, layer_string, expected, test_id):
    tree = layer_parser.parse(layer_string)
    result = transformer.transform(tree)
    assert result == expected, f"Failed for {test_id}: expected {expected}, got {result}"