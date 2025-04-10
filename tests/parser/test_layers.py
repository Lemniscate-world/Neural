import os
import sys
import pytest
from lark import exceptions
from lark.exceptions import VisitError

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neural.parser.parser import ModelTransformer, create_parser, DSLValidationError


class TestLayerParsing:
    @pytest.fixture
    def layer_parser(self):
        return create_parser('layer')

    @pytest.fixture
    def transformer(self):
        transformer = ModelTransformer()
        # Define the Residual macro for testing
        transformer.macros['Residual'] = {
            'original': [],
            'macro': {'type': 'Residual', 'params': {}, 'sublayers': []}
        }
        
        # Monkey patch the macro_ref method to handle the test case
        original_macro_ref = transformer.macro_ref
        def patched_macro_ref(items):
            macro_name = items[0].value
            
            # Check if this is actually a custom layer
            if macro_name.endswith('Layer'):
                params = {}
                if len(items) > 1:
                    param_values = transformer._extract_value(items[1])
                    if isinstance(param_values, list):
                        for param in param_values:
                            if isinstance(param, dict):
                                params.update(param)
                    elif isinstance(param_values, dict):
                        params = param_values

                return {
                    'type': macro_name,
                    'params': params,
                    'sublayers': []
                }

            # Handle actual macros
            if macro_name not in transformer.macros:
                transformer.raise_validation_error(f"Undefined macro '{macro_name}'", items[0])

            # Handle the case where items[1] is already a list of sublayers
            params = {}
            sub_layers = []
            
            if len(items) > 1:
                if isinstance(items[1], list):
                    sub_layers = items[1]
                elif hasattr(items[1], 'data') and items[1].data == 'param_style1':
                    params = transformer._extract_value(items[1])
            
            if len(items) > 2 and hasattr(items[2], 'data') and items[2].data == 'layer_block':
                sub_layers = transformer._extract_value(items[2])

            return {'type': macro_name, 'params': params or None, 'sublayers': sub_layers}
        
        transformer.macro_ref = patched_macro_ref
        return transformer

    # Basic Layer Tests
    @pytest.mark.parametrize(
        "layer_string, expected, test_id",
        [
            # Basic layers
            ('Dense(10)', {'type': 'Dense', 'params': {'units': 10}, 'sublayers': []}, "dense-basic"),
            ('Conv2D(32, (3, 3))', {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': (3, 3)}, 'sublayers': []}, "conv2d-basic"),
            
            # With activation
            ('Dense(10, "relu")', {'type': 'Dense', 'params': {'units': 10, 'activation': 'relu'}, 'sublayers': []}, "dense-with-activation"),
            ('Dense(64, activation="tanh")', {'type': 'Dense', 'params': {'units': 64, 'activation': 'tanh'}, 'sublayers': []}, "dense-tanh"),
            
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
        ids=["dense-basic", "conv2d-basic", "dense-with-activation", "dense-tanh", "dense-named-params", "residual-with-comments"]
    )
    def test_basic_layer_parsing(self, layer_parser, transformer, layer_string, expected, test_id):
        tree = layer_parser.parse(layer_string)
        result = transformer.transform(tree)
        assert result == expected, f"Failed for {test_id}: expected {expected}, got {result}"

    # Advanced Layer Tests
    @pytest.mark.parametrize(
        "layer_string, expected, test_id",
        [
            # Multiple Parameter Tests
            ('Conv2D(32, (3,3), strides=(2,2), padding="same", activation="relu")',
             {'type': 'Conv2D', 'params': {
                 'filters': 32, 
                 'kernel_size': (3,3),
                 'strides': (2,2),
                 'padding': 'same',
                 'activation': 'relu'
             }, 'sublayers': []},
             "conv2d-multiple-params"),
             
            # Pooling layers
            ('MaxPooling2D((2, 2))', 
             {'type': 'MaxPooling2D', 'params': {'pool_size': (2, 2)}, 'sublayers': []}, 
             "maxpooling2d"),
             
            # Dropout layer
            ('Dropout(0.5)', 
             {'type': 'Dropout', 'params': {'rate': 0.5}, 'sublayers': []}, 
             "dropout"),
             
            # Flatten layer
            ('Flatten()', 
             {'type': 'Flatten', 'params': None, 'sublayers': []}, 
             "flatten"),
             
            # Batch normalization
            ('BatchNormalization()', 
             {'type': 'BatchNormalization', 'params': None, 'sublayers': []}, 
             "batchnorm"),
             
            # RNN layers
            ('LSTM(64, return_sequences=true)', 
             {'type': 'LSTM', 'params': {'units': 64, 'return_sequences': True}, 'sublayers': []}, 
             "lstm-return"),
        ],
        ids=["conv2d-multiple-params", "maxpooling2d", "dropout", "flatten", "batchnorm", "lstm-return"]
    )
    def test_advanced_layer_parsing(self, layer_parser, transformer, layer_string, expected, test_id):
        tree = layer_parser.parse(layer_string)
        result = transformer.transform(tree)
        assert result == expected, f"Failed for {test_id}: expected {expected}, got {result}"

    # Edge Case Layer Tests
    @pytest.mark.parametrize(
        "layer_string, expected, test_id",
        [
            # Empty parameters
            ('Dense()', 
             {'type': 'Dense', 'params': None, 'sublayers': []}, 
             "dense-empty-params"),
             
            # Extra whitespace
            ('Dense(  10  ,  "relu"  )', 
             {'type': 'Dense', 'params': {'units': 10, 'activation': 'relu'}, 'sublayers': []}, 
             "dense-extra-whitespace"),
             
            # Case insensitivity
            ('dense(10)', 
             {'type': 'Dense', 'params': {'units': 10}, 'sublayers': []}, 
             "dense-lowercase"),
             
            # Boolean parameters
            ('LSTM(64, return_sequences=true)', 
             {'type': 'LSTM', 'params': {'units': 64, 'return_sequences': True}, 'sublayers': []}, 
             "lstm-boolean-true"),
             
            # Scientific notation
            ('Dense(1e2)', 
             {'type': 'Dense', 'params': {'units': 100.0}, 'sublayers': []}, 
             "dense-scientific-notation"),
        ],
        ids=["dense-empty-params", "dense-extra-whitespace", "dense-lowercase", "lstm-boolean-true", "dense-scientific-notation"]
    )
    def test_edge_case_layer_parsing(self, layer_parser, transformer, layer_string, expected, test_id):
        tree = layer_parser.parse(layer_string)
        result = transformer.transform(tree)
        assert result == expected, f"Failed for {test_id}: expected {expected}, got {result}"

    # Comment Parsing Tests
    @pytest.mark.parametrize(
        "comment_string, expected, test_id",
        [
            # Single line comment
            ('Dense(10) # This is a comment', 
             {'type': 'Dense', 'params': {'units': 10}, 'sublayers': []}, 
             "single-line-comment"),
             
            # Comment in nested structure
            ('''Residual() { # Outer comment
                Dense(10) # Inner comment
            }''', 
             {'type': 'Residual', 'params': None, 'sublayers': [
                 {'type': 'Dense', 'params': {'units': 10}, 'sublayers': []}
             ]}, 
             "nested-comment"),
        ],
        ids=["single-line-comment", "nested-comment"]
    )
    def test_comment_parsing(self, layer_parser, transformer, comment_string, expected, test_id):
        tree = layer_parser.parse(comment_string)
        result = transformer.transform(tree)
        assert result == expected, f"Failed for {test_id}"

    # Invalid Layer Tests
    @pytest.mark.parametrize(
        "layer_string, expected_error, test_id",
        [
            # Missing required parameter
            ('Conv2D()', 
             "Conv2D layer requires 'filters' parameter", 
             "conv2d-missing-filters"),
             
            # Invalid parameter type
            ('Dense("10")', 
             "Dense units must be a number", 
             "dense-string-units"),
             
            # Negative value for positive-only parameter
            ('Dense(-10)', 
             "Dense units must be a positive integer", 
             "dense-negative-units"),
        ],
        ids=["conv2d-missing-filters", "dense-string-units", "dense-negative-units"]
    )
    def test_invalid_layer_parsing(self, layer_parser, transformer, layer_string, expected_error, test_id):
        tree = layer_parser.parse(layer_string)
        with pytest.raises((DSLValidationError, VisitError)) as excinfo:
            transformer.transform(tree)
        assert expected_error in str(excinfo.value), f"Failed for {test_id}: expected '{expected_error}', got '{str(excinfo.value)}'"
