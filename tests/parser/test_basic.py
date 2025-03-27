import os
import sys
import pytest
from lark import Lark, exceptions

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neural.parser.parser import ModelTransformer, create_parser, DSLValidationError, Severity, safe_parse

# Fixtures
@pytest.fixture
def layer_parser():
    return create_parser('layer')

@pytest.fixture
def network_parser():
    return create_parser('network')

@pytest.fixture
def transformer():
    return ModelTransformer()

def test_grammar_ambiguity():
    """Test that grammar doesn't have ambiguous rules."""
    parser = create_parser()
    test_cases = [
        ('params_order1', 'Dense(10, "relu")'),
        ('params_order2', 'Dense(units=10, activation="relu")'),
        ('mixed_params', 'Conv2D(32, kernel_size=(3,3))'),
        ('nested_params', 'Transformer(num_heads=8) { Dense(10) }')
    ]
    
    for test_id, test_input in test_cases:
        try:
            parser.parse(f"network TestNet {{ input: (1,1) layers: {test_input} }}")
        except lark.exceptions.UnexpectedInput as e:
            pytest.fail(f"Unexpected parse error for {test_id}: {str(e)}")
        except Exception as e:
            pytest.fail(f"Failed to parse {test_id}: {str(e)}")