import os
import sys
import pytest
from lark import exceptions

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neural.parser.parser import ModelTransformer, create_parser, DSLValidationError
from neural.exceptions import VisitError

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
                    'type': 'model',
                    'name': 'SimpleNet',
                    'input': {'shape': (28, 28, 1)},
                    'output_layer': {'type': 'Dense', 'params': {'units': 10}, 'sublayers': []},
                    'output_shape': 10,
                    'loss': 'categorical_crossentropy',  # Default
                    'optimizer': {'type': 'Adam', 'params': {}},  # Default
                    'training_config': {'epochs': 10, 'batch_size': 32},  # Default
                    'execution_config': {'device': 'auto'},  # Default
                    'framework': 'tensorflow',
                    'shape_info': [],
                    'warnings': []
                },
                False,
                "simple-network"
            ),
            
            # Complex network
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
                    optimizer: "adam"
                    train { epochs: 10 batch_size: 32 }
                }
                """,
                {
                    'type': 'model',
                    'name': 'TestModel',
                    'input': {'shape': (None, 28, 28, 1)},
                    'output_layer': {'type': 'Dense', 'params': {'units': 10, 'activation': 'softmax'}, 'sublayers': []},
                    'output_shape': 10,
                    'loss': 'categorical_crossentropy',
                    'optimizer': {'type': 'Adam', 'params': {}},
                    'training_config': {'epochs': 10, 'batch_size': 32},
                    'execution_config': {'device': 'auto'},
                    'framework': 'tensorflow',
                    'shape_info': [],
                    'warnings': []
                },
                False,
                "complex-model"
            ),
            # Add more network test cases here
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
