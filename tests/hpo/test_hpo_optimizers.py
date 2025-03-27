import pytest
import sys
import os
from lark import exceptions
from lark.exceptions import VisitError

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import from the neural package
from neural.parser.parser import layer_parser, ModelTransformer, DSLValidationError

class TestHPOOptimizers:
    @pytest.fixture
    def transformer(self):
        return ModelTransformer()

    @pytest.mark.parametrize(
        "network_string, expected_optimizer, test_id",
        [
            # Basic optimizer tests
            (
                '''
                network BasicAdam {
                    input: (28, 28, 1)
                    layers:
                        Dense(128)
                        Output(10)
                    loss: "categorical_crossentropy"
                    optimizer: Adam(learning_rate=0.001, beta_1=0.9)
                }
                ''',
                {
                    'type': 'Adam',
                    'params': {'learning_rate': 0.001, 'beta_1': 0.9}
                },
                "adam-basic"
            ),
            # HPO in learning rate
            (
                '''
                network AdamLrHPO {
                    input: (28, 28, 1)
                    layers:
                        Dense(128)
                        Output(10)
                    loss: "categorical_crossentropy"
                    optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
                }
                ''',
                {
                    'type': 'Adam',
                    'params': {'learning_rate': {'hpo': {'type': 'log_range', 'low': 1e-4, 'high': 1e-2}}}
                },
                "adam-lr-hpo"
            ),
            # HPO in multiple parameters
            (
                '''
                network SGDMultipleHPO {
                    input: (28, 28, 1)
                    layers:
                        Dense(128)
                        Output(10)
                    loss: "categorical_crossentropy"
                    optimizer: SGD(learning_rate=HPO(log_range(1e-4, 1e-1)), momentum=0.9)
                }
                ''',
                {
                    'type': 'SGD',
                    'params': {
                        'learning_rate': {'hpo': {'type': 'log_range', 'low': 1e-4, 'high': 1e-1}},
                        'momentum': 0.9
                    }
                },
                "sgd-multiple-hpo"
            ),
            # Learning rate schedule
            (
                '''
                network SGDLrSchedule {
                    input: (28, 28, 1)
                    layers:
                        Dense(128)
                        Output(10)
                    loss: "categorical_crossentropy"
                    optimizer: SGD(learning_rate=ExponentialDecay(0.1, 1000, 0.96), momentum=0.9)
                }
                ''',
                {
                    'type': 'SGD',
                    'params': {
                        'learning_rate': {
                            'type': 'ExponentialDecay',
                            'args': [0.1, 1000, 0.96]
                        },
                        'momentum': 0.9
                    }
                },
                "sgd-lr-schedule"
            ),
            # HPO in learning rate schedule
            (
                '''
                network SGDLrScheduleHPO {
                    input: (28, 28, 1)
                    layers:
                        Dense(128)
                        Output(10)
                    loss: "categorical_crossentropy"
                    optimizer: SGD(learning_rate=ExponentialDecay(HPO(range(0.05, 0.2, step=0.05)), 1000, HPO(range(0.9, 0.99, step=0.01))))
                }
                ''',
                {
                    'type': 'SGD',
                    'params': {
                        'learning_rate': {
                            'type': 'ExponentialDecay',
                            'args': [
                                {'hpo': {'type': 'range', 'low': 0.05, 'high': 0.2, 'step': 0.05}},
                                1000,
                                {'hpo': {'type': 'range', 'low': 0.9, 'high': 0.99, 'step': 0.01}}
                            ]
                        }
                    }
                },
                "sgd-lr-schedule-hpo"
            ),
        ],
        ids=[
            "adam-basic", "adam-lr-hpo", "sgd-multiple-hpo", 
            "sgd-lr-schedule", "sgd-lr-schedule-hpo"
        ]
    )
    def test_optimizer_parsing(self, transformer, network_string, expected_optimizer, test_id):
        """Test parsing of optimizer configurations with HPO within complete networks."""
        model_dict, hpo_params = transformer.parse_network_with_hpo(network_string)
        
        # Check that the optimizer matches the expected configuration
        assert model_dict['optimizer'] == expected_optimizer, \
            f"Failed for {test_id}: expected {expected_optimizer}, got {model_dict['optimizer']}"

    @pytest.mark.parametrize(
        "network_string, expected_optimizer_type, expected_hpo_param, test_id",
        [
            (
                """
                network OptimizerTest {
                    input: (28, 28, 1)
                    layers:
                        Dense(128)
                        Output(10)
                    loss: "categorical_crossentropy"
                    optimizer: "Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))"
                }
                """,
                "Adam",
                "learning_rate",
                "network-adam-lr-hpo"
            ),
            (
                """
                network ScheduleTest {
                    input: (28, 28, 1)
                    layers:
                        Dense(128)
                        Output(10)
                    loss: "categorical_crossentropy"
                    optimizer: "SGD(learning_rate=ExponentialDecay(HPO(choice(0.1, 0.01)), 1000, 0.96))"
                }
                """,
                "SGD",
                "learning_rate",
                "network-sgd-schedule-hpo"
            ),
        ],
        ids=["network-adam-lr-hpo", "network-sgd-schedule-hpo"]
    )
    def test_optimizer_in_network(self, transformer, network_string, expected_optimizer_type, expected_hpo_param, test_id):
        """Test HPO in optimizers within complete network definitions."""
        model_dict, hpo_params = transformer.parse_network_with_hpo(network_string)
        
        # Check optimizer type
        assert model_dict['optimizer']['type'] == expected_optimizer_type, \
            f"Expected optimizer type {expected_optimizer_type}, got {model_dict['optimizer']['type']}"
        
        # Check that we have HPO parameters in the optimizer
        found_hpo = False
        for param in hpo_params:
            if param.get('path', '').startswith('optimizer.params'):
                found_hpo = True
                break
        
        assert found_hpo, f"No HPO parameters found in optimizer for {test_id}"
