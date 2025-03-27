import os
import sys
import pytest
from lark import exceptions

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neural.parser.parser import ModelTransformer, create_parser, DSLValidationError, Severity

# Fixtures
@pytest.fixture
def network_parser():
    return create_parser('network')

@pytest.fixture
def transformer():
    return ModelTransformer()

# Validation Rules Tests
@pytest.mark.parametrize(
    "network_string, expected_error_msg, test_id",
    [
        (
            """
            network InvalidSplit {
                input: (10,)
                layers: Dense(5)
                loss: "mse"
                optimizer: "sgd"
                train { validation_split: 1.5 }
            }
            """,
            "validation_split must be between 0 and 1, got 1.5",
            "invalid-validation-split"
        ),
        (
            """
            network MissingUnits {
                input: (10,)
                layers: Dense()
                loss: "mse"
                optimizer: "sgd"
            }
            """,
            "Dense layer requires 'units' parameter",
            "missing-units"
        ),
        (
            """
            network NegativeFilters {
                input: (28, 28, 1)
                layers: Conv2D(-32, (3,3))
                loss: "mse"
                optimizer: "sgd"
            }
            """,
            "Conv2D filters must be a positive integer, got -32",
            "negative-filters"
        ),
    ],
    ids=["invalid-validation-split", "missing-units", "negative-filters"]
)
def test_validation_rules(transformer, network_string, expected_error_msg, test_id):
    with pytest.raises(DSLValidationError) as excinfo:
        transformer.parse_network(network_string)
    assert expected_error_msg in str(excinfo.value), f"Failed for {test_id}: expected '{expected_error_msg}', got '{str(excinfo.value)}'"