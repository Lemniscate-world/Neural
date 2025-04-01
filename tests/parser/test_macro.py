import os
import sys
import pytest
from lark import exceptions
from lark.exceptions import VisitError

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neural.parser.parser import ModelTransformer, create_parser, DSLValidationError

class TestMacroParsing:
    @pytest.fixture
    def define_parser(self):
        return create_parser('define')

    @pytest.fixture
    def layer_parser(self):
        return create_parser('layer')

    @pytest.fixture
    def transformer(self):
        return ModelTransformer()

    @pytest.mark.parametrize(
        "config, expected_definition, expected_reference, raises_error, test_id",
        [
            (
                """
                define MyDense {
                    Dense(128, "relu")
                }
                """,
                {'type': 'MyDense', 'params': {}, 'sublayers': [{'type': 'Dense', 'params': {'units': 128, 'activation': 'relu'}, 'sublayers': []}]},
                {'type': 'MyDense', 'params': {}, 'sublayers': []},
                False,
                "macro-basic"
            ),
            (
                """
                define ResBlock {
                    Conv2D(64, (3,3))
                    BatchNormalization()
                    ResidualConnection() {
                        Dense(128)
                        Dropout(0.3)
                    }
                }
                """,
                {'type': 'ResBlock', 'params': {}, 'sublayers': 
                [
                    {'type': 'Conv2D', 'params': {'filters': 64, 'kernel_size': (3, 3)}, 'sublayers': []},
                    {'type': 'BatchNormalization', 'params': None, 'sublayers': []},
                    {
                        'type': 'ResidualConnection', 'params': {},
                        'sublayers': [
                            {'type': 'Dense', 'params': {'units': 128}, 'sublayers': []},
                            {'type': 'Dropout', 'params': {'rate': 0.3}, 'sublayers': []}
                        ]
                    }
                ]
                },
                {'type': 'ResBlock', 'params': {}, 'sublayers': []},
                False,
                "macro-nested"
            ),
            (
                "UndefinedMacro()",
                None,
                None,
                True,
                "macro-undefined"
            ),
        ],
        ids=["macro-basic", "macro-nested", "macro-undefined"]
    )
    def test_macro_parsing(self, define_parser, layer_parser, transformer, config, expected_definition, expected_reference, raises_error, test_id):
        if raises_error:
            with pytest.raises(VisitError) as exc_info:
                tree = layer_parser.parse(config)
                transformer.transform(tree)
            # Check that the VisitError contains a DSLValidationError
            assert isinstance(exc_info.value.__context__, DSLValidationError), f"Expected DSLValidationError, got {type(exc_info.value.__context__)}"
            assert "Undefined macro" in str(exc_info.value.__context__), f"Error message mismatch in {test_id}"
        else:
            define_tree = define_parser.parse(config)
            definition_result = transformer.transform(define_tree)
            assert definition_result == expected_definition, f"Definition mismatch in {test_id}"
            
            ref_string = f"{config.split()[1]}()"
            ref_tree = layer_parser.parse(ref_string)
            ref_result = transformer.transform(ref_tree)
            assert ref_result == expected_reference, f"Reference mismatch in {test_id}"