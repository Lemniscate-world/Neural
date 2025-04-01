import os
import sys
import pytest
from lark import exceptions
from lark.exceptions import VisitError

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neural.parser.parser import ModelTransformer, create_parser, DSLValidationError

class TestWrapperParsing:
    @pytest.fixture
    def layer_parser(self):
        return create_parser('layer')

    @pytest.fixture
    def transformer(self):
        return ModelTransformer()

    @pytest.mark.parametrize(
        "wrapper_string, expected, test_id",
        [
            (
                'TimeDistributed(Dense(128, "relu"), dropout=0.5)',
                {'type': 'TimeDistributed(Dense)', 'params': {'units': 128, 'activation': 'relu'}, 'sublayers': [{'type': 'Dropout', 'params': {'rate': 0.5}, 'sublayers': []}]},
                "timedistributed-dense"
            ),
            (
                'TimeDistributed(Conv2D(32, (3, 3))) { Dropout(0.2) }',
                {
                    'type': 'TimeDistributed(Conv2D)', 'params': {'filters': 32, 'kernel_size': (3, 3)},
                    'sublayers': [{'type': 'Dropout', 'params': {'rate': 0.2}, 'sublayers': []}]
                },
                "timedistributed-conv2d-nested"
            ),
            (
                'TimeDistributed(Dropout("invalid"))',
                None,
                "timedistributed-invalid"
            ),
        ],
        ids=["timedistributed-dense", "timedistributed-conv2d-nested", "timedistributed-invalid"]
    )
    def test_wrapper_parsing(self, layer_parser, transformer, wrapper_string, expected, test_id):
        if expected is None:
            with pytest.raises(VisitError) as exc_info:
                tree = layer_parser.parse(wrapper_string)
                transformer.transform(tree)
            assert isinstance(exc_info.value.__context__, DSLValidationError), f"Expected DSLValidationError, got {type(exc_info.value.__context__)}"
        else:
            tree = layer_parser.parse(wrapper_string)
            result = transformer.transform(tree)
            assert result == expected, f"Failed for {test_id}: expected {expected}, got {result}"