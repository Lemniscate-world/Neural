import os
import sys
import pytest
from lark import Lark

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser import ModelTransformer, create_parser

@pytest.fixture
def layer_parser():
    return create_parser('layer')

@pytest.fixture
def network_parser():
    return create_parser('network')

@pytest.fixture
def research_parser():
    return create_parser('research')

@pytest.fixture
def transformer():
    return ModelTransformer()

@pytest.mark.parametrize("layer_string,expected", [
    (
        'Dense(128, "relu")',
        {'type': 'Dense', 'units': 128, 'activation': 'relu'}
    ),
    (
        'Conv2D(32, 3, 3, "relu")',
        {'type': 'Conv2D', 'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'}
    ),
    (
        'Flatten()',
        {'type': 'Flatten'}
    ),
    (
        'Dropout(0.5)',
        {'type': 'Dropout', 'rate': 0.5}
    ),
])
def test_layer_parsing(layer_parser, transformer, layer_string, expected):
    tree = layer_parser.parse(layer_string)
    result = transformer.transform(tree)
    assert result == expected
def test_network_parsing(network_parser, transformer):
    network_string = """
    network TestModel {
        input: (28, 28, 1)
        layers:
            Conv2D(filters=32, kernel_size=(3,3), activation="relu")
            Flatten()
            Dense(units=128, activation="relu")
        loss: "categorical_crossentropy"
        optimizer: "adam"
    }
    """
    tree = network_parser.parse(network_string)
    result = transformer.transform(tree)
    
    assert result['type'] == 'model'
    assert result['name'] == 'TestModel'
    assert result['input_shape'] == (28, 28, 1)
    assert len(result['layers']) == 3

def test_invalid_layer(layer_parser):
    with pytest.raises(Exception):
        layer_parser.parse("InvalidLayer()")

def test_invalid_network(network_parser):
    with pytest.raises(Exception):
        network_parser.parse("invalid network syntax")