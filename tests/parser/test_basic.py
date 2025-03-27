import os
import sys
import pytest
from lark import Lark, exceptions

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neural.parser.parser import ModelTransformer, create_parser, DSLValidationError, Severity, safe_parse

# Common fixtures
@pytest.fixture
def parser():
    return create_parser()

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

class TestGrammarStructure:
    def test_rule_dependencies(self, parser):
        """Test that grammar rules have correct dependencies."""
        rules = {rule.origin.name: rule for rule in parser.grammar.rules}
        
        # Check essential rule dependencies
        dependencies = {
            'network': ['input_layer', 'layers', 'loss', 'optimizer'],
            'layer': ['conv', 'pooling', 'dropout', 'flatten', 'dense'],
            'conv': ['conv1d', 'conv2d', 'conv3d'],
            'pooling': ['max_pooling', 'average_pooling', 'global_pooling']
        }
        
        for rule_name, required_deps in dependencies.items():
            assert rule_name in rules, f"Missing rule: {rule_name}"
            rule = rules[rule_name]
            for dep in required_deps:
                assert dep in str(rule), f"Rule {rule_name} missing dependency {dep}"

    def test_grammar_ambiguity(self, parser):
        """Test that grammar doesn't have ambiguous rules."""
        test_cases = [
            ('params_order1', 'Dense(10, "relu")'),
            ('params_order2', 'Dense(units=10, activation="relu")'),
            ('mixed_params', 'Conv2D(32, kernel_size=(3,3))'),
            ('nested_params', 'Transformer(num_heads=8) { Dense(10) }')
        ]
        
        for test_id, test_input in test_cases:
            try:
                parser.parse(f"network TestNet {{ input: (1,1) layers: {test_input} }}")
            except exceptions.UnexpectedInput as e:
                pytest.fail(f"Unexpected parse error for {test_id}: {str(e)}")
            except Exception as e:
                pytest.fail(f"Failed to parse {test_id}: {str(e)}")

    def test_rule_precedence(self, parser):
        """Test that grammar rules have correct precedence."""
        test_cases = [
            ('dense_basic', 'Dense(10)'),
            ('dense_params', 'Dense(units=10, activation="relu")'),
            ('conv_basic', 'Conv2D(32, (3,3))'),
            ('conv_params', 'Conv2D(filters=32, kernel_size=(3,3))'),
            ('nested_block', 'Transformer() { Dense(10) }')
        ]
            
        for test_id, test_input in test_cases:
            try:
                result = parser.parse(f"network TestNet {{ input: (1,1) layers: {test_input} }}")
                assert result is not None, f"Failed to parse {test_id}"
            except Exception as e:
                pytest.fail(f"Failed to parse {test_id}: {str(e)}")

    def test_grammar_completeness(self, parser):
        """Test that grammar covers all required language features."""
        features = [
            "network TestNet { input: (1,1) layers: Dense(10) }",
            "network TestNet { input: (1,1) layers: Dense(10) loss: \"mse\" }",
            "network TestNet { input: (1,1) layers: Dense(10) optimizer: Adam() }",
            "network TestNet { input: (1,1) layers: Dense(10) train { epochs: 10 } }",
            "network TestNet { input: (1,1) layers: Dense(10) execute { device: \"cpu\" } }"
        ]
        
        for feature in features:
            try:
                result = parser.parse(feature)
                assert result is not None, f"Failed to parse feature: {feature}"
            except Exception as e:
                pytest.fail(f"Failed to parse feature: {feature}, error: {str(e)}")

class TestTokenPatterns:
    @pytest.mark.parametrize(
        "rule_name, valid_inputs",
        [
            ('STRING', ['"relu"', '"tanh"', '"sigmoid"', '"This is a longer string"']),
            ('CUSTOM_LAYER', ['MyCustomLayer', 'CustomRNN', 'MyTransformer']),
            ('NUMBER', ['10', '3.14', '1e-4', '0.001']),
            ('NAME', ['model_name', 'TestNet', 'SimpleRNN'])
        ]
    )
    def test_token_patterns(self, parser, rule_name, valid_inputs):
        """Test that token patterns match expected inputs."""
        for input_str in valid_inputs:
            try:
                if rule_name == 'STRING':
                    # Test STRING token in a valid context (as activation function)
                    result = parser.parse(f'network TestNet {{ input: (1,1) layers: Dense(10, {input_str}) }}')
                elif rule_name == 'CUSTOM_LAYER':
                    # Test CUSTOM_LAYER token in a valid context (as custom layer name)
                    result = parser.parse(f'network TestNet {{ input: (1,1) layers: {input_str}() }}')
                elif rule_name == 'NUMBER':
                    # Test NUMBER token in a valid context (as numeric value)
                    result = parser.parse(f'network TestNet {{ input: (1,1) layers: Dense({input_str}) }}')
                elif rule_name == 'NAME':
                    # Test NAME token in a valid context (as network name)
                    result = parser.parse(f'network {input_str} {{ input: (1,1) layers: Dense(10) }}')
                
                assert result is not None, f"Failed to parse {rule_name} with input: {input_str}"
            except Exception as e:
                pytest.fail(f"Failed to parse {rule_name} with input: {input_str}, error: {str(e)}")

class TestErrorHandling:
    def test_error_recovery(self, parser):
        """Test parser's error recovery capabilities."""
        test_cases = [
            (
                'incomplete_block',
                '''network Test {
                    input: (1, 1)
                    layers: Dense(10) {''',  # Missing closing brace
                'Unexpected end of input - Check for missing closing braces'
            ),
            (
                'missing_close',
                'network Test { input: (1,1) layers: Dense(10)',
                'Unexpected end of input - Check for missing closing braces'
            )
        ]

        for test_id, test_input, expected_msg in test_cases:
            with pytest.raises(DSLValidationError) as exc_info:
                safe_parse(parser, test_input)
            assert expected_msg in str(exc_info.value), f"Test case {test_id} failed"
