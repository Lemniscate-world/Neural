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

    # Advanced Network Tests
    @pytest.mark.parametrize(
        "network_string, expected, raises_error, test_id",
        [
            # Learning rate schedule with HPO
            (
                """
                network LRScheduleModel {
                    input: (28, 28, 1)
                    layers:
                        Conv2D(32, (3,3), "relu")
                        MaxPooling2D((2, 2))
                        Flatten()
                        Dense(128, "relu")
                        Dense(10, "softmax")
                    optimizer: SGD(
                        learning_rate=ExponentialDecay(
                            HPO(range(0.05, 0.2, step=0.05)),
                            1000,
                            HPO(range(0.9, 0.99, step=0.01))
                        ),
                        momentum=0.9
                    )
                }
                """,
                {
                    'name': 'LRScheduleModel',
                    'input': {'type': 'Input', 'shape': (28, 28, 1)},
                    'layers': [
                        {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'}, 'sublayers': []},
                        {'type': 'MaxPooling2D', 'params': {'pool_size': (2, 2)}, 'sublayers': []},
                        {'type': 'Flatten', 'params': None, 'sublayers': []},
                        {'type': 'Dense', 'params': {'units': 128, 'activation': 'relu'}, 'sublayers': []},
                        {'type': 'Dense', 'params': {'units': 10, 'activation': 'softmax'}, 'sublayers': []}
                    ],
                    'optimizer': {
                        'type': 'SGD',
                        'params': {
                            'learning_rate': {
                                'type': 'ExponentialDecay',
                                'params': {
                                    'initial_learning_rate': {'hpo': {'type': 'range', 'min': 0.05, 'max': 0.2, 'step': 0.05}},
                                    'decay_steps': 1000,
                                    'decay_rate': {'hpo': {'type': 'range', 'min': 0.9, 'max': 0.99, 'step': 0.01}}
                                }
                            },
                            'momentum': 0.9
                        }
                    },
                    'framework': 'tensorflow',
                    'shape_info': [],
                    'warnings': []
                },
                False,
                "lr-schedule-hpo"
            ),
            
            # Multi-input network
            (
                """
                network MultiInputModel {
                    input: {
                        image: (224, 224, 3),
                        metadata: (10)
                    }
                    layers:
                        # Image branch
                        image: {
                            Conv2D(32, (3,3), "relu")
                            MaxPooling2D((2, 2))
                            Flatten()
                            Dense(64, "relu")
                        }
                        # Metadata branch
                        metadata: {
                            Dense(32, "relu")
                        }
                        # Merge branches
                        Concatenate()
                        Dense(128, "relu")
                        Dense(1, "sigmoid")
                    loss: "binary_crossentropy"
                }
                """,
                {
                    'name': 'MultiInputModel',
                    'input': {
                        'image': {'type': 'Input', 'shape': (224, 224, 3)},
                        'metadata': {'type': 'Input', 'shape': (10)}
                    },
                    'layers': [
                        {'type': 'Branch', 'name': 'image', 'sublayers': [
                            {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'}, 'sublayers': []},
                            {'type': 'MaxPooling2D', 'params': {'pool_size': (2, 2)}, 'sublayers': []},
                            {'type': 'Flatten', 'params': None, 'sublayers': []},
                            {'type': 'Dense', 'params': {'units': 64, 'activation': 'relu'}, 'sublayers': []}
                        ]},
                        {'type': 'Branch', 'name': 'metadata', 'sublayers': [
                            {'type': 'Dense', 'params': {'units': 32, 'activation': 'relu'}, 'sublayers': []}
                        ]},
                        {'type': 'Concatenate', 'params': None, 'sublayers': []},
                        {'type': 'Dense', 'params': {'units': 128, 'activation': 'relu'}, 'sublayers': []},
                        {'type': 'Dense', 'params': {'units': 1, 'activation': 'sigmoid'}, 'sublayers': []}
                    ],
                    'loss': 'binary_crossentropy',
                    'framework': 'tensorflow',
                    'shape_info': [],
                    'warnings': []
                },
                False,
                "multi-input-model"
            ),
            
            # Layer repetition syntax
            (
                """
                network RepetitionModel {
                    input: (28, 28, 1)
                    layers:
                        Conv2D(32, (3,3), "relu") * 2
                        MaxPooling2D((2, 2))
                        Dense(64, "relu") * 3
                        Dense(10, "softmax")
                }
                """,
                {
                    'name': 'RepetitionModel',
                    'input': {'type': 'Input', 'shape': (28, 28, 1)},
                    'layers': [
                        {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'}, 'sublayers': []},
                        {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'}, 'sublayers': []},
                        {'type': 'MaxPooling2D', 'params': {'pool_size': (2, 2)}, 'sublayers': []},
                        {'type': 'Dense', 'params': {'units': 64, 'activation': 'relu'}, 'sublayers': []},
                        {'type': 'Dense', 'params': {'units': 64, 'activation': 'relu'}, 'sublayers': []},
                        {'type': 'Dense', 'params': {'units': 64, 'activation': 'relu'}, 'sublayers': []},
                        {'type': 'Dense', 'params': {'units': 10, 'activation': 'softmax'}, 'sublayers': []}
                    ],
                    'framework': 'tensorflow',
                    'shape_info': [],
                    'warnings': []
                },
                False,
                "layer-repetition"
            ),
            
            # Device placement
            (
                """
                network DevicePlacementModel {
                    input: (28, 28, 1)
                    layers:
                        Conv2D(32, (3,3), "relu") @ "cuda:0"
                        MaxPooling2D((2, 2)) @ "cuda:0"
                        Flatten() @ "cuda:1"
                        Dense(128, "relu") @ "cuda:1"
                        Dense(10, "softmax") @ "cpu"
                    execution {
                        device: "cuda"
                    }
                }
                """,
                {
                    'name': 'DevicePlacementModel',
                    'input': {'type': 'Input', 'shape': (28, 28, 1)},
                    'layers': [
                        {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'}, 'device': 'cuda:0', 'sublayers': []},
                        {'type': 'MaxPooling2D', 'params': {'pool_size': (2, 2)}, 'device': 'cuda:0', 'sublayers': []},
                        {'type': 'Flatten', 'params': None, 'device': 'cuda:1', 'sublayers': []},
                        {'type': 'Dense', 'params': {'units': 128, 'activation': 'relu'}, 'device': 'cuda:1', 'sublayers': []},
                        {'type': 'Dense', 'params': {'units': 10, 'activation': 'softmax'}, 'device': 'cpu', 'sublayers': []}
                    ],
                    'execution': {'device': 'cuda'},
                    'framework': 'tensorflow',
                    'shape_info': [],
                    'warnings': []
                },
                False,
                "device-placement"
            ),
            
            # Comprehensive HPO with activation functions
            (
                """
                network ActivationHPOModel {
                    input: (28, 28, 1)
                    layers:
                        Conv2D(32, (3,3), activation=HPO(choice("relu", "elu", "selu")))
                        MaxPooling2D((2, 2))
                        Flatten()
                        Dense(128, activation=HPO(choice("relu", "tanh", "sigmoid")))
                        Dense(10, "softmax")
                    optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
                    train {
                        epochs: 10
                        batch_size: HPO(choice(32, 64, 128))
                        search_method: "bayesian"
                    }
                }
                """,
                {
                    'name': 'ActivationHPOModel',
                    'input': {'type': 'Input', 'shape': (28, 28, 1)},
                    'layers': [
                        {'type': 'Conv2D', 'params': {
                            'filters': 32, 
                            'kernel_size': (3, 3), 
                            'activation': {'hpo': {'type': 'categorical', 'values': ['relu', 'elu', 'selu']}}
                        }, 'sublayers': []},
                        {'type': 'MaxPooling2D', 'params': {'pool_size': (2, 2)}, 'sublayers': []},
                        {'type': 'Flatten', 'params': None, 'sublayers': []},
                        {'type': 'Dense', 'params': {
                            'units': 128, 
                            'activation': {'hpo': {'type': 'categorical', 'values': ['relu', 'tanh', 'sigmoid']}}
                        }, 'sublayers': []},
                        {'type': 'Dense', 'params': {'units': 10, 'activation': 'softmax'}, 'sublayers': []}
                    ],
                    'optimizer': {'type': 'Adam', 'params': {
                        'learning_rate': {'hpo': {'type': 'log_range', 'min': 1e-4, 'max': 1e-2}}
                    }},
                    'training': {
                        'epochs': 10,
                        'batch_size': {'hpo': {'type': 'categorical', 'values': [32, 64, 128]}},
                        'search_method': 'bayesian'
                    },
                    'framework': 'tensorflow',
                    'shape_info': [],
                    'warnings': []
                },
                False,
                "activation-hpo"
            ),
            
            # ResidualConnection test
            (
                """
                network ResNetModel {
                    input: (224, 224, 3)
                    layers:
                        Conv2D(64, (7,7), "relu", strides=(2,2))
                        MaxPooling2D((3, 3), strides=(2,2))
                        
                        ResidualConnection {
                            Conv2D(64, (3,3), "relu", padding="same")
                            Conv2D(64, (3,3), padding="same")
                        }
                        
                        ResidualConnection {
                            Conv2D(128, (3,3), "relu", strides=(2,2))
                            Conv2D(128, (3,3), padding="same")
                        }
                        
                        GlobalAveragePooling2D()
                        Dense(1000, "softmax")
                }
                """,
                {
                    'name': 'ResNetModel',
                    'input': {'type': 'Input', 'shape': (224, 224, 3)},
                    'layers': [
                        {'type': 'Conv2D', 'params': {'filters': 64, 'kernel_size': (7, 7), 'activation': 'relu', 'strides': (2, 2)}, 'sublayers': []},
                        {'type': 'MaxPooling2D', 'params': {'pool_size': (3, 3), 'strides': (2, 2)}, 'sublayers': []},
                        {'type': 'ResidualConnection', 'params': None, 'sublayers': [
                            {'type': 'Conv2D', 'params': {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu', 'padding': 'same'}, 'sublayers': []},
                            {'type': 'Conv2D', 'params': {'filters': 64, 'kernel_size': (3, 3), 'padding': 'same'}, 'sublayers': []}
                        ]},
                        {'type': 'ResidualConnection', 'params': None, 'sublayers': [
                            {'type': 'Conv2D', 'params': {'filters': 128, 'kernel_size': (3, 3), 'activation': 'relu', 'strides': (2, 2)}, 'sublayers': []},
                            {'type': 'Conv2D', 'params': {'filters': 128, 'kernel_size': (3, 3), 'padding': 'same'}, 'sublayers': []}
                        ]},
                        {'type': 'GlobalAveragePooling2D', 'params': None, 'sublayers': []},
                        {'type': 'Dense', 'params': {'units': 1000, 'activation': 'softmax'}, 'sublayers': []}
                    ],
                    'framework': 'tensorflow',
                    'shape_info': [],
                    'warnings': []
                },
                False,
                "residual-connection"
            )
        ],
        ids=["lr-schedule-hpo", "multi-input-model", "layer-repetition", "device-placement", "activation-hpo", "residual-connection"]
    )
    def test_advanced_network_features(self, network_parser, transformer, network_string, expected, raises_error, test_id):
        """Test advanced network features from the DSL documentation."""
        if raises_error:
            with pytest.raises((exceptions.UnexpectedCharacters, exceptions.UnexpectedToken, DSLValidationError, VisitError)):
                transformer.parse_network(network_string)
        else:
            result = transformer.parse_network(network_string)
            assert result == expected, f"Failed for {test_id}: expected {expected}, got {result}"

    # Device Execution Tests
    @pytest.mark.parametrize(
        "network_string, expected, raises_error, test_id",
        [
            # Multi-device network
            (
                """
                network MultiDeviceModel {
                    input: (28, 28, 1)
                    layers:
                        Conv2D(32, (3,3), "relu") @ "cuda:0"
                        Conv2D(64, (3,3), "relu") @ "cuda:1"
                        Flatten() @ "cpu"
                        Dense(128, "relu") @ "cuda:0"
                        Dense(10, "softmax") @ "cpu"
                    execution {
                        device: "auto"
                    }
                }
                """,
                {
                    'name': 'MultiDeviceModel',
                    'input': {'type': 'Input', 'shape': (28, 28, 1)},
                    'layers': [
                        {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'}, 'device': 'cuda:0', 'sublayers': []},
                        {'type': 'Conv2D', 'params': {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'}, 'device': 'cuda:1', 'sublayers': []},
                        {'type': 'Flatten', 'params': None, 'device': 'cpu', 'sublayers': []},
                        {'type': 'Dense', 'params': {'units': 128, 'activation': 'relu'}, 'device': 'cuda:0', 'sublayers': []},
                        {'type': 'Dense', 'params': {'units': 10, 'activation': 'softmax'}, 'device': 'cpu', 'sublayers': []}
                    ],
                    'execution': {'device': 'auto'},
                    'framework': 'tensorflow',
                    'shape_info': [],
                    'warnings': []
                },
                False,
                "multi-device-model"
            ),
            
            # TPU device specification
            (
                """
                network TPUModel {
                    input: (28, 28, 1)
                    layers:
                        Conv2D(32, (3,3), "relu") @ "tpu:0"
                        MaxPooling2D((2, 2))
                        Flatten()
                        Dense(128, "relu") @ "tpu:0"
                        Dense(10, "softmax")
                    execution {
                        device: "tpu"
                    }
                }
                """,
                {
                    'name': 'TPUModel',
                    'input': {'type': 'Input', 'shape': (28, 28, 1)},
                    'layers': [
                        {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'}, 'device': 'tpu:0', 'sublayers': []},
                        {'type': 'MaxPooling2D', 'params': {'pool_size': (2, 2)}, 'sublayers': []},
                        {'type': 'Flatten', 'params': None, 'sublayers': []},
                        {'type': 'Dense', 'params': {'units': 128, 'activation': 'relu'}, 'device': 'tpu:0', 'sublayers': []},
                        {'type': 'Dense', 'params': {'units': 10, 'activation': 'softmax'}, 'sublayers': []}
                    ],
                    'execution': {'device': 'tpu'},
                    'framework': 'tensorflow',
                    'shape_info': [],
                    'warnings': []
                },
                False,
                "tpu-model"
            ),
            
            # Invalid device specification
            (
                """
                network InvalidDeviceModel {
                    input: (28, 28, 1)
                    layers:
                        Conv2D(32, (3,3), "relu") @ "quantum"
                        MaxPooling2D((2, 2))
                        Dense(10, "softmax")
                }
                """,
                None,
                True,
                "invalid-device-model"
            )
        ],
        ids=["multi-device-model", "tpu-model", "invalid-device-model"]
    )
    def test_device_specifications(self, network_parser, transformer, network_string, expected, raises_error, test_id):
        """Test device specification features in the DSL."""
        if raises_error:
            with pytest.raises((exceptions.UnexpectedCharacters, exceptions.UnexpectedToken, DSLValidationError)):
                transformer.parse_network(network_string)
        else:
            result = transformer.parse_network(network_string)
            assert result == expected, f"Failed for {test_id}: expected {expected}, got {result}"
