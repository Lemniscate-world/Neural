import pytest
import sys
import os
from lark import exceptions
from lark.exceptions import VisitError

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Now import from the neural package
from neural.parser.parser import layer_parser, ModelTransformer, DSLValidationError

class TestHPOParser:
    @pytest.fixture
    def transformer(self):
        return ModelTransformer()

    # Basic HPO Type Tests
    @pytest.mark.parametrize(
        "hpo_string, expected_result, test_id",
        [
            # Choice HPO - Basic types
            (
                'HPO(choice(32, 64, 128))',
                {'hpo': {'type': 'categorical', 'values': [32, 64, 128]}},
                "choice-integers"
            ),
            (
                'HPO(choice(0.1, 0.2, 0.3))',
                {'hpo': {'type': 'categorical', 'values': [0.1, 0.2, 0.3]}},
                "choice-floats"
            ),
            (
                'HPO(choice("relu", "tanh", "sigmoid"))',
                {'hpo': {'type': 'categorical', 'values': ["relu", "tanh", "sigmoid"]}},
                "choice-strings"
            ),
            (
                'HPO(choice(true, false))',
                {'hpo': {'type': 'categorical', 'values': [True, False]}},
                "choice-booleans"
            ),
            
            # Range HPO - Different step sizes
            (
                'HPO(range(10, 100))',
                {'hpo': {'type': 'range', 'start': 10, 'end': 100}},
                "range-no-step"
            ),
            (
                'HPO(range(10, 100, step=10))',
                {'hpo': {'type': 'range', 'start': 10, 'end': 100, 'step': 10}},
                "range-with-step"
            ),
            (
                'HPO(range(0.1, 0.5, step=0.1))',
                {'hpo': {'type': 'range', 'start': 0.1, 'end': 0.5, 'step': 0.1}},
                "range-float-step"
            ),
            
            # Log Range HPO
            (
                'HPO(log_range(1e-4, 1e-1))',
                {'hpo': {'type': 'log_range', 'start': 1e-4, 'end': 1e-1}},
                "log-range-scientific"
            ),
            (
                'HPO(log_range(0.001, 10))',
                {'hpo': {'type': 'log_range', 'start': 0.001, 'end': 10}},
                "log-range-decimal"
            ),
        ],
        ids=[
            "choice-integers", "choice-floats", "choice-strings", "choice-booleans",
            "range-no-step", "range-with-step", "range-float-step",
            "log-range-scientific", "log-range-decimal"
        ]
    )
    def test_basic_hpo_types(self, transformer, hpo_string, expected_result, test_id):
        """Test basic HPO type parsing within a network context."""
        # Create a simple network with the HPO expression in a layer
        network_string = f"""
        network TestNetwork {{
            input: (28, 28, 1)
            layers:
                Dense({hpo_string})
                Output(10)
            loss: "categorical_crossentropy"
            optimizer: "adam"
        }}
        """
        
        # Parse the network with HPO expressions
        model_dict, hpo_params = transformer.parse_network_with_hpo(network_string)
        
        # Extract the HPO parameter from the first layer (Dense)
        layer_params = model_dict['layers'][0]['params']
        
        # The first parameter of Dense should contain our HPO expression
        assert 'units' in layer_params, f"Failed for {test_id}: 'units' not found in layer params"
        hpo_result = layer_params['units']
        
        # Verify the parsing result
        assert hpo_result == expected_result, f"Failed for {test_id}: expected {expected_result}, got {hpo_result}"

    # Complex HPO Tests
    @pytest.mark.parametrize(
        "layer_string, expected_param_key, expected_hpo_type, test_id",
        [
            # HPO in layer parameters
            (
                'Dense(HPO(choice(32, 64, 128)), activation="relu")',
                'units',
                'categorical',
                "dense-units-hpo"
            ),
            (
                'Dense(128, activation=HPO(choice("relu", "tanh")))',
                'activation',
                'categorical',
                "dense-activation-hpo"
            ),
            (
                'Dropout(HPO(range(0.1, 0.5, step=0.1)))',
                'rate',
                'range',
                "dropout-rate-hpo"
            ),
            (
                'Conv2D(HPO(choice(32, 64)), (3, 3), padding=HPO(choice("same", "valid")))',
                'filters',
                'categorical',
                "conv2d-filters-hpo"
            ),
            
            # Multiple HPO parameters in one layer
            (
                'Dense(HPO(choice(32, 64)), activation=HPO(choice("relu", "tanh")), use_bias=HPO(choice(true, false)))',
                'units',
                'categorical',
                "dense-multiple-hpo"
            ),
            
            # HPO in optimizer parameters
            (
                'Adam(learning_rate=HPO(log_range(1e-4, 1e-2)), beta_1=HPO(range(0.8, 0.99, step=0.01)))',
                'learning_rate',
                'log_range',
                "adam-lr-hpo"
            ),
        ],
        ids=[
            "dense-units-hpo", "dense-activation-hpo", "dropout-rate-hpo", "conv2d-filters-hpo",
            "dense-multiple-hpo", "adam-lr-hpo"
        ]
    )
    def test_complex_hpo_in_layers(self, transformer, layer_string, expected_param_key, expected_hpo_type, test_id):
        """Test HPO expressions within layer parameters."""
        tree = layer_parser.parse(layer_string)
        result = transformer.transform(tree)
        
        assert expected_param_key in result['params'], f"Parameter {expected_param_key} not found in {result['params']}"
        assert 'hpo' in result['params'][expected_param_key], f"HPO not found in parameter {expected_param_key}"
        assert result['params'][expected_param_key]['hpo']['type'] == expected_hpo_type, \
            f"Expected HPO type {expected_hpo_type}, got {result['params'][expected_param_key]['hpo']['type']}"

    # Nested HPO Tests
    @pytest.mark.parametrize(
        "hpo_string, expected_nested_structure, test_id",
        [
            # Nested choice HPO
            (
                'HPO(choice(HPO(choice(32, 64)), HPO(choice(128, 256))))',
                {
                    'hpo': {
                        'type': 'categorical',
                        'values': [
                            {'hpo': {'type': 'categorical', 'values': [32, 64]}},
                            {'hpo': {'type': 'categorical', 'values': [128, 256]}}
                        ]
                    }
                },
                "nested-choice-hpo"
            ),
            
            # Mixed nested HPO types
            (
                'HPO(choice(HPO(range(10, 50, step=10)), HPO(log_range(0.001, 0.1))))',
                {
                    'hpo': {
                        'type': 'categorical',
                        'values': [
                            {'hpo': {'type': 'range', 'start': 10, 'end': 50, 'step': 10}},
                            {'hpo': {'type': 'log_range', 'start': 0.001, 'end': 0.1}}
                        ]
                    }
                },
                "mixed-nested-hpo"
            ),
            
            # Deeply nested HPO
            (
                'HPO(choice(HPO(choice(HPO(range(10, 30)), HPO(range(40, 60)))), 100))',
                {
                    'hpo': {
                        'type': 'categorical',
                        'values': [
                            {
                                'hpo': {
                                    'type': 'categorical',
                                    'values': [
                                        {'hpo': {'type': 'range', 'start': 10, 'end': 30}},
                                        {'hpo': {'type': 'range', 'start': 40, 'end': 60}}
                                    ]
                                }
                            },
                            100
                        ]
                    }
                },
                "deeply-nested-hpo"
            ),
        ],
        ids=["nested-choice-hpo", "mixed-nested-hpo", "deeply-nested-hpo"]
    )
    def test_nested_hpo(self, transformer, hpo_string, expected_nested_structure, test_id):
        """Test nested HPO expressions."""
        tree = layer_parser.parse(hpo_string)
        result = transformer.transform(tree)
        assert result == expected_nested_structure, f"Failed for {test_id}: expected {expected_nested_structure}, got {result}"

    # HPO Error Cases
    @pytest.mark.parametrize(
        "invalid_hpo_string, expected_error_type, test_id",
        [
            # Empty choice
            ('HPO(choice())', exceptions.UnexpectedToken, "empty-choice"),
            
            # Invalid range parameters
            ('HPO(range(10))', exceptions.UnexpectedToken, "range-missing-end"),
            ('HPO(range(10, 5))', DSLValidationError, "range-end-less-than-start"),
            ('HPO(range(10, 100, step=-10))', DSLValidationError, "range-negative-step"),
            
            # Invalid log range parameters
            ('HPO(log_range(0))', exceptions.UnexpectedToken, "log-range-missing-end"),
            ('HPO(log_range(10, 1))', DSLValidationError, "log-range-end-less-than-start"),
            ('HPO(log_range(0, 10))', DSLValidationError, "log-range-zero-start"),
            ('HPO(log_range(-1, 10))', DSLValidationError, "log-range-negative-start"),
            
            # Unknown HPO type
            ('HPO(unknown(10, 20))', exceptions.UnexpectedToken, "unknown-hpo-type"),
            
            # Syntax errors
            ('HPO(choice(32, 64,))', exceptions.UnexpectedToken, "trailing-comma"),
            ('HPO(choice(32, 64)', exceptions.UnexpectedToken, "missing-parenthesis"),
            
            # Type mixing in choice
            ('HPO(choice(32, "relu"))', DSLValidationError, "mixed-types-in-choice"),
            
            # Invalid HPO nesting
            ('HPO(HPO(choice(32, 64)))', exceptions.UnexpectedToken, "direct-hpo-nesting"),
        ],
        ids=[
            "empty-choice", "range-missing-end", "range-end-less-than-start", "range-negative-step",
            "log-range-missing-end", "log-range-end-less-than-start", "log-range-zero-start", 
            "log-range-negative-start", "unknown-hpo-type", "trailing-comma", "missing-parenthesis",
            "mixed-types-in-choice", "direct-hpo-nesting"
        ]
    )
    def test_hpo_error_cases(self, transformer, invalid_hpo_string, expected_error_type, test_id):
        """Test error handling for invalid HPO expressions."""
        with pytest.raises((exceptions.UnexpectedToken, exceptions.UnexpectedCharacters, DSLValidationError, VisitError)) as exc_info:
            tree = layer_parser.parse(invalid_hpo_string)
            transformer.transform(tree)
        
        # Check if the error is of the expected type or contains the expected type as a cause
        error_chain = [exc_info.value]
        while hasattr(error_chain[-1], '__cause__') and error_chain[-1].__cause__ is not None:
            error_chain.append(error_chain[-1].__cause__)
        
        assert any(isinstance(err, expected_error_type) for err in error_chain), \
            f"Expected error of type {expected_error_type}, but got {type(exc_info.value)}"

    # HPO in Network Context
    @pytest.mark.parametrize(
        "network_string, expected_hpo_count, test_id",
        [
            # Simple network with HPO
            (
                """
                network SimpleHPO {
                    input: (28, 28, 1)
                    layers:
                        Dense(HPO(choice(64, 128)))
                        Output(10)
                    loss: "categorical_crossentropy"
                    optimizer: "Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))"
                }
                """,
                2,
                "simple-network-hpo"
            ),
            
            # Complex network with multiple HPO
            (
                """
                network ComplexHPO {
                    input: (28, 28, 1)
                    layers:
                        Conv2D(HPO(choice(32, 64)), (3, 3), padding=HPO(choice("same", "valid")))
                        MaxPooling2D((2, 2))
                        Flatten()
                        Dense(HPO(range(100, 500, step=100)))
                        Dropout(HPO(range(0.2, 0.5, step=0.1)))
                        Output(10)
                    loss: "categorical_crossentropy"
                    optimizer: "Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))"
                    train {
                        batch_size: HPO(choice(32, 64, 128))
                        epochs: 10
                    }
                }
                """,
                6,
                "complex-network-hpo"
            ),
            
            # Network with nested HPO
            (
                """
                network NestedHPO {
                    input: (28, 28, 1)
                    layers:
                        Dense(HPO(choice(HPO(range(64, 256, step=64)), HPO(choice(512, 1024)))))
                        Output(10)
                    loss: "categorical_crossentropy"
                    optimizer: "Adam()"
                }
                """,
                3,  # One outer choice and two nested HPOs
                "nested-network-hpo"
            ),
        ],
        ids=["simple-network-hpo", "complex-network-hpo", "nested-network-hpo"]
    )
    def test_hpo_in_network(self, transformer, network_string, expected_hpo_count, test_id):
        """Test HPO expressions within complete network definitions."""
        model_dict, hpo_params = transformer.parse_network_with_hpo(network_string)
        
        # Check that we extracted the expected number of HPO parameters
        assert len(hpo_params) == expected_hpo_count, \
            f"Expected {expected_hpo_count} HPO parameters, but found {len(hpo_params)}"
        
        # Verify that all extracted HPO params have the required structure
        for param in hpo_params:
            assert 'hpo' in param, f"Missing 'hpo' key in parameter: {param}"
            assert 'type' in param['hpo'], f"Missing 'type' key in HPO: {param['hpo']}"
            assert param['hpo']['type'] in ['categorical', 'range', 'log_range'], \
                f"Invalid HPO type: {param['hpo']['type']}"

    # Layer Choice HPO Tests
    @pytest.mark.parametrize(
        "layer_choice_string, expected_layer_types, test_id",
        [
            # Simple layer choice
            (
                'HPO(choice(Dense(128), Dropout(0.5)))',
                ['Dense', 'Dropout'],
                "simple-layer-choice"
            ),
            
            # Layer choice with parameters
            (
                'HPO(choice(Conv2D(32, (3, 3)), Conv2D(64, (5, 5))))',
                ['Conv2D', 'Conv2D'],
                "parameterized-layer-choice"
            ),
            
            # Mixed layer types
            (
                'HPO(choice(Dense(128), Conv2D(32, (3, 3)), LSTM(64)))',
                ['Dense', 'Conv2D', 'LSTM'],
                "mixed-layer-choice"
            ),
            
            # Nested layers in choice
            (
                'HPO(choice(Dense(128), Residual() { Conv2D(32, (3, 3)) }))',
                ['Dense', 'Residual'],
                "nested-layer-choice"
            ),
        ],
        ids=["simple-layer-choice", "parameterized-layer-choice", "mixed-layer-choice", "nested-layer-choice"]
    )
    def test_layer_choice_hpo(self, transformer, layer_choice_string, expected_layer_types, test_id):
        """Test HPO expressions for layer choices."""
        tree = layer_parser.parse(layer_choice_string)
        result = transformer.transform(tree)
        
        assert 'hpo' in result, f"Missing 'hpo' key in result: {result}"
        assert result['hpo']['type'] == 'layer_choice', f"Expected layer_choice type, got: {result['hpo']['type']}"
        assert len(result['hpo']['options']) == len(expected_layer_types), \
            f"Expected {len(expected_layer_types)} layer options, got {len(result['hpo']['options'])}"
        
        # Check that each layer option has the expected type
        for i, layer_type in enumerate(expected_layer_types):
            assert result['hpo']['options'][i]['type'] == layer_type, \
                f"Expected layer type {layer_type}, got {result['hpo']['options'][i]['type']}"
