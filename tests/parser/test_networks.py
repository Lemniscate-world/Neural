import os
import sys
import pytest
from lark import exceptions
from lark.exceptions import VisitError

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neural.parser.parser import ModelTransformer, create_parser, DSLValidationError


class TestNetworkParsing:
    @pytest.fixture
    def network_parser(self):
        return create_parser('network')

    @pytest.fixture
    def transformer(self):
        return ModelTransformer()

    # Basic Network Tests
    @pytest.mark.parametrize(
        "network_string, expected, raises_error, test_id",
        [
            # Simple network
            (
                """
                network SimpleNet {
                    input: (28, 28, 1)
                    layers: Dense(10)
                }
                """,
                {
                    'name': 'SimpleNet',
                    'input': {'type': 'Input', 'shape': (28, 28, 1)},
                    'layers': [{'type': 'Dense', 'params': {'units': 10}, 'sublayers': []}],
                    'framework': 'tensorflow',
                    'shape_info': [],
                    'warnings': []
                },
                False,
                "simple-network"
            ),
            
            # Complex network - with corrected expected output
            (
                """
                network TestModel {
                    input: (None, 28, 28, 1)
                    layers:
                        Conv2D(32, (3,3), "relu")
                        MaxPooling2D((2, 2))
                        Flatten()
                        Dense(128, "relu")
                        Dense(10, "softmax")
                    loss: "categorical_crossentropy"
                    optimizer: Adam(learning_rate=0.001)
                    train { epochs: 10 batch_size: 32 }
                }
                """,
                {
                    'name': 'TestModel',
                    'input': {'type': 'Input', 'shape': (None, 28, 28, 1)},
                    'layers': [
                        {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'}, 'sublayers': []},
                        {'type': 'MaxPooling2D', 'params': {'pool_size': (2, 2)}, 'sublayers': []},
                        {'type': 'Flatten', 'params': None, 'sublayers': []},
                        {'type': 'Dense', 'params': {'units': 128, 'activation': 'relu'}, 'sublayers': []},
                        {'type': 'Dense', 'params': {'units': 10, 'activation': 'softmax'}, 'sublayers': []}
                    ],
                    'loss': 'categorical_crossentropy',
                    'optimizer': {'type': 'Adam', 'params': {'learning_rate': 0.001}},
                    'training': {'epochs': 10, 'batch_size': 32},
                    'framework': 'tensorflow',
                    'shape_info': [],
                    'warnings': []
                },
                False,
                "complex-model"
            ),
        ],
        ids=["simple-network", "complex-model"]
    )
    def test_network_parsing(self, network_parser, transformer, network_string, expected, raises_error, test_id):
        if raises_error:
            with pytest.raises((exceptions.UnexpectedCharacters, exceptions.UnexpectedToken, DSLValidationError)):
                transformer.parse_network(network_string)
        else:
            result = transformer.parse_network(network_string)
            assert result == expected, f"Failed for {test_id}: expected {expected}, got {result}"
