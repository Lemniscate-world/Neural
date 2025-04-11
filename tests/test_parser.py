import os
import sys
from cycler import V
import pytest
import lark
from lark import Lark, exceptions
from lark.exceptions import VisitError

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural.parser.parser import ModelTransformer, create_parser, DSLValidationError, Severity, safe_parse


# Fixtures
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
def define_parser():
    return create_parser('define')

@pytest.fixture
def transformer():
    return ModelTransformer()

# Layer Parsing Tests
@pytest.mark.parametrize(
    "layer_string, expected, test_id",
    [
        # Basic Layers
        ('Dense(128, "relu")', {'type': 'Dense', 'params': {'units': 128, 'activation': 'relu'}, 'sublayers': []}, "dense-relu"),
        ('Dense(units=256, activation="sigmoid")', {'type': 'Dense', 'params': {'units': 256, 'activation': 'sigmoid'}, 'sublayers': []}, "dense-sigmoid"),
        ('Conv2D(32, (3, 3), activation="relu")', {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'}, 'sublayers': []}, "conv2d-relu"),
        ('Conv2D(filters=64, kernel_size=(5, 5), activation="tanh")', {'type': 'Conv2D', 'params': {'filters': 64, 'kernel_size': (5, 5), 'activation': 'tanh'}, 'sublayers': []}, "conv2d-tanh"),
        ('Conv2D(filters=32, kernel_size=3, activation="relu", padding="same")', {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': 3, 'activation': 'relu', 'padding': 'same'}, 'sublayers': []}, "conv2d-padding"),
        ('MaxPooling2D(pool_size=(2, 2))', {'type': 'MaxPooling2D', 'params': {'pool_size': (2, 2)}, 'sublayers': []}, "maxpooling2d"),
        ('MaxPooling2D((3, 3), 2, "valid")', {'type': 'MaxPooling2D', 'params': {'pool_size': (3, 3), 'strides': 2, 'padding': 'valid'}, 'sublayers': []}, "maxpooling2d-strides"),
        ('Flatten()', {'type': 'Flatten', 'params': None, 'sublayers': []}, "flatten"),
        ('Dropout(0.5)', {'type': 'Dropout', 'params': {'rate': 0.5}, 'sublayers': []}, "dropout"),
        ('Dropout(rate=0.25)', {'type': 'Dropout', 'params': {'rate': 0.25}, 'sublayers': []}, "dropout-named"),
        ('BatchNormalization()', {'type': 'BatchNormalization', 'params': None, 'sublayers': []}, "batchnorm"),
        ('LayerNormalization()', {'type': 'LayerNormalization', 'params': None, 'sublayers': []}, "layernorm"),
        ('InstanceNormalization()', {'type': 'InstanceNormalization', 'params': None, 'sublayers': []}, "instancenorm"),
        ('GroupNormalization(groups=32)', {'type': 'GroupNormalization', 'params': {'groups': 32}, 'sublayers': []}, "groupnorm"),

        # Recurrent Layers
        ('LSTM(units=64)', {'type': 'LSTM', 'params': {'units': 64}, 'sublayers': []}, "lstm"),
        ('LSTM(units=128, return_sequences=true)', {'type': 'LSTM', 'params': {'units': 128, 'return_sequences': True}, 'sublayers': []}, "lstm-return"),
        ('GRU(units=32)', {'type': 'GRU', 'params': {'units': 32}, 'sublayers': []}, "gru"),
        ('SimpleRNN(units=16)', {'type': 'SimpleRNN', 'params': {'units': 16}, 'sublayers': []}, "simplernn"),
        ('LSTMCell(units=64)', {'type': 'LSTMCell', 'params': {'units': 64}, 'sublayers': []}, "lstmcell"),
        ('GRUCell(units=128)', {'type': 'GRUCell', 'params': {'units': 128}, 'sublayers': []}, "grucell"),
        ('SimpleRNNDropoutWrapper(units=16, dropout=0.3)', {'type': 'SimpleRNNDropoutWrapper', 'params': {'units': 16, 'dropout': 0.3}, 'sublayers': []}, "simplernn-dropout"),

        # Advanced Layers
        ('Attention()', {'type': 'Attention', 'params': None, 'sublayers': []}, "attention"),
        ('TransformerEncoder(num_heads=8, ff_dim=512)', {'type': 'TransformerEncoder', 'params': {'num_heads': 8, 'ff_dim': 512}, 'sublayers': []}, "transformer-encoder"),
        ('TransformerDecoder(num_heads=4, ff_dim=256)', {'type': 'TransformerDecoder', 'params': {'num_heads': 4, 'ff_dim': 256}, 'sublayers': []}, "transformer-decoder"),
        ('ResidualConnection()', {'type': 'ResidualConnection', 'params': {}, 'sublayers': []}, "residual"),
        ('Inception()', {'type': 'Inception', 'params': {}, 'sublayers': []}, "inception"),
        ('CapsuleLayer()', {'type': 'CapsuleLayer', 'params': {}, 'sublayers': []}, "capsule"),
        ('SqueezeExcitation()', {'type': 'SqueezeExcitation', 'params': {}, 'sublayers': []}, "squeeze"),
        ('GraphAttention(num_heads=4)', {'type': 'GraphAttention', 'params': {'num_heads': 4}, 'sublayers': []}, "graph-attention"),
        ('Embedding(input_dim=1000, output_dim=128)', {'type': 'Embedding', 'params': {'input_dim': 1000, 'output_dim': 128}, 'sublayers': []}, "embedding"),
        ('QuantumLayer()', {'type': 'QuantumLayer', 'params': {}, 'sublayers': []}, "quantum"),

        # Nested Layers
        (
            'TransformerEncoder(num_heads=8, ff_dim=512) { Dense(128, "relu") Dropout(0.3) }',
            {
                'type': 'TransformerEncoder', 'params': {'num_heads': 8, 'ff_dim': 512},
                'sublayers': [
                    {'type': 'Dense', 'params': {'units': 128, 'activation': 'relu'}, 'sublayers': []},
                    {'type': 'Dropout', 'params': {'rate': 0.3}, 'sublayers': []}
                ]
            },
            "transformer-nested"
        ),
        (
            'ResidualConnection() { Conv2D(32, (3,3)) BatchNormalization() }',
            {
                'type': 'ResidualConnection', 'params': {},
                'sublayers': [
                    {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': (3, 3)}, 'sublayers': []},
                    {'type': 'BatchNormalization', 'params': None, 'sublayers': []}
                ]
            },
            "residual-nested"
        ),

        # Merge and Noise Layers
        ('Add()', {'type': 'Add', 'params': None, 'sublayers': []}, "add"),
        ('Concatenate(axis=1)', {'type': 'Concatenate', 'params': {'axis': 1}, 'sublayers': []}, "concatenate"),
        ('GaussianNoise(stddev=0.1)', {'type': 'GaussianNoise', 'params': {'stddev': 0.1}, 'sublayers': []}, "gaussian-noise"),

        # HPO and Device
        (
            'Dense(HPO(choice(128, 256)))',
            {'type': 'Dense', 'params': {'units': {'hpo': {'type': 'categorical', 'values': [128, 256]}}}, 'sublayers': []},
            "dense-hpo-choice"
        ),
        ('Conv2D(64, (3,3)) @ "cuda:0"', {'type': 'Conv2D', 'params': {'filters': 64, 'kernel_size': (3, 3), 'device': 'cuda:0'}, 'sublayers': []}, "conv2d-device"),

        # Error Cases
        ('Dense(units=-1)', None, "dense-negative-units"),
        ('Dropout(1.5)', None, "dropout-high-rate-error"),
        ('TransformerEncoder(num_heads=0)', None, "transformer-zero-heads"),
        ('Conv2D(filters=32, kernel_size=(0, 0))', None, "conv2d-zero-kernel"),
        ('Dense(units="abc")', None, "dense-invalid-units"),
    ],
    ids=[
        "dense-relu", "dense-sigmoid", "conv2d-relu", "conv2d-tanh", "conv2d-padding", "maxpooling2d", "maxpooling2d-strides",
        "flatten", "dropout", "dropout-named", "batchnorm", "layernorm", "instancenorm", "groupnorm", "lstm", "lstm-return",
        "gru", "simplernn", "lstmcell", "grucell", "simplernn-dropout", "attention", "transformer-encoder", "transformer-decoder",
        "residual", "inception", "capsule", "squeeze", "graph-attention", "embedding", "quantum", "transformer-nested", "residual-nested",
        "add", "concatenate", "gaussian-noise", "dense-hpo-choice", "conv2d-device", "dense-negative-units", "dropout-high-rate-error",
        "transformer-zero-heads", "conv2d-zero-kernel", "dense-invalid-units"
    ]
)
def test_layer_parsing(layer_parser, transformer, layer_string, expected, test_id):
    if expected is None:
        with pytest.raises((exceptions.UnexpectedCharacters, exceptions.UnexpectedToken, DSLValidationError, VisitError)):
            tree = layer_parser.parse(layer_string)
            transformer.transform(tree)
    else:
        tree = layer_parser.parse(layer_string)
        result = transformer.transform(tree)
        assert result == expected, f"Failed for {test_id}: expected {expected}, got {result}"

# Network Parsing Tests
@pytest.mark.parametrize(
    "network_string, expected, raises_error, test_id",
    [
        # Complex Network
        (
            """
            network TestModel {
                input: (None, 28, 28, 1)
                layers:
                    Conv2D(32, (3,3), "relu")
                    MaxPooling2D((2, 2))
                    Flatten()
                    Dense(128, "relu")
                    Output(10, "softmax")
                loss: "categorical_crossentropy"
                optimizer: "adam"
                train { epochs: 10 batch_size: 32 }
            }
            """,
            {
                'type': 'model', 'name': 'TestModel',
                'input': {'type': 'Input', 'shape': (None, 28, 28, 1)},
                'layers': [
                    {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'}, 'sublayers': []},
                    {'type': 'MaxPooling2D', 'params': {'pool_size': (2, 2)}, 'sublayers': []},
                    {'type': 'Flatten', 'params': None, 'sublayers': []},
                    {'type': 'Dense', 'params': {'units': 128, 'activation': 'relu'}, 'sublayers': []},
                    {'type': 'Output', 'params': {'units': 10, 'activation': 'softmax'}, 'sublayers': []}
                ],
                'output_layer': {'type': 'Output', 'params': {'units': 10, 'activation': 'softmax'}, 'sublayers': []},
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

        # Nested Network (Vision Transformer-like)
        (
            """
            network ViT {
                input: (224, 224, 3)
                layers:
                    Conv2D(64, (7,7), strides=2) @ "cuda:0"
                    TransformerEncoder(num_heads=8, ff_dim=512) {
                        Conv2D(32, (3,3))
                        Dense(128)
                    } * 2
                    GlobalAveragePooling2D()
                    Dense(1000, "softmax")
                loss: "categorical_crossentropy"
                optimizer: "Adam(learning_rate=1e-4)"
            }
            """,
            {
                'type': 'model', 'name': 'ViT',
                'input': {'type': 'Input', 'shape': (224, 224, 3)},
                'layers': [
                    {'type': 'Conv2D', 'params': {'filters': 64, 'kernel_size': (7, 7), 'strides': 2, 'device': 'cuda:0'}, 'sublayers': []},
                    {
                        'type': 'TransformerEncoder', 'params': {'num_heads': 8, 'ff_dim': 512},
                        'sublayers': [
                            {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': (3, 3)}, 'sublayers': []},
                            {'type': 'Dense', 'params': {'units': 128}, 'sublayers': []}
                        ]
                    },
                    {
                        'type': 'TransformerEncoder', 'params': {'num_heads': 8, 'ff_dim': 512},
                        'sublayers': [
                            {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': (3, 3)}, 'sublayers': []},
                            {'type': 'Dense', 'params': {'units': 128}, 'sublayers': []}
                        ]
                    },
                    {'type': 'GlobalAveragePooling2D', 'params': {}, 'sublayers': []},
                    {'type': 'Dense', 'params': {'units': 1000, 'activation': 'softmax'}, 'sublayers': []}
                ],
                'output_layer': {'type': 'Dense', 'params': {'units': 1000, 'activation': 'softmax'}, 'sublayers': []},
                'output_shape': 1000,
                'loss': 'categorical_crossentropy',
                'optimizer': {'type': 'Adam', 'params': {'learning_rate': 0.0001}},
                'training_config': None,
                'execution_config': {'device': 'auto'},
                'framework': 'tensorflow',
                'shape_info': [],
                'warnings': []
            },
            False,
            "nested-vit"
        ),

        # HPO Network
        (
            """
            network HPOExample {
                input: (10,)
                layers:
                    Dense(HPO(choice(128, 256)))
                    Dropout(HPO(range(0.3, 0.7, step=0.1)))
                loss: "mse"
                optimizer: "Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))"
                train { search_method: "bayesian" }
            }
            """,
            {
                'type': 'model', 'name': 'HPOExample',
                'input': {'type': 'Input', 'shape': (10,)},
                'layers': [
                    {'type': 'Dense', 'params': {'units': {'hpo': {'type': 'categorical', 'values': [128, 256]}}}, 'sublayers': []},
                    {'type': 'Dropout', 'params': {'rate': {'hpo': {'type': 'range', 'start': 0.3, 'end': 0.7, 'step': 0.1}}}, 'sublayers': []}
                ],
                'output_layer': {'type': 'Dropout', 'params': {'rate': {'hpo': {'type': 'range', 'start': 0.3, 'end': 0.7, 'step': 0.1}}}, 'sublayers': []},
                'output_shape': None,
                'loss': 'mse',
                'optimizer': {'type': 'Adam', 'params': {'learning_rate': {'hpo': {'type': 'log_range', 'low': 0.0001, 'high': 0.01}}}},
                'training_config': {'search_method': 'bayesian'},
                'execution_config': {'device': 'auto'},
                'framework': 'tensorflow',
                'shape_info': [],
                'warnings': []
            },
            False,
            "hpo-example"
        ),

        # Error Cases
        (
            """
            network InvalidDevice {
                input: (10,)
                layers:
                    Dense(5) @ "npu"
                loss: "mse"
                optimizer: "Sgd"
            }
            """,
            None,
            True,
            "invalid-device"
        ),
        (
            """
            network MissingLayers {
                input: (10,)
                loss: "mse"
                optimizer: "Sgd"
            }
            """,
            None,
            True,
            "missing-layers"
        ),
    ],
    ids=["complex-model", "nested-vit", "hpo-example", "invalid-device", "missing-layers"]
)
def test_network_parsing(network_parser, transformer, network_string, expected, raises_error, test_id):
    if raises_error:
        with pytest.raises((exceptions.UnexpectedCharacters, exceptions.UnexpectedToken, DSLValidationError, VisitError)):
            transformer.parse_network(network_string)
    else:
        result = transformer.parse_network(network_string)
        assert result == expected, f"Failed for {test_id}: expected {expected}, got {result}"

# Research Parsing Tests
@pytest.mark.parametrize(
    "research_string, expected_name, expected_metrics, expected_references, test_id",
    [
        (
            """
            research ResearchStudy {
                metrics {
                    accuracy: 0.95
                    loss: 0.05
                }
                references {
                    paper: "Paper Title 1"
                    paper: "Another Great Paper"
                }
            }
            """,
            "ResearchStudy", {'accuracy': 0.95, 'loss': 0.05}, ["Paper Title 1", "Another Great Paper"],
            "complete-research"
        ),
        (
            """
            research {
                metrics {
                    precision: 0.8
                    recall: 0.9
                }
            }
            """,
            None, {'precision': 0.8, 'recall': 0.9}, [],
            "no-name-no-ref"
        ),
        (
            """
            research InvalidMetrics {
                metrics {
                    accuracy: "high"
                }
            }
            """,
            None, None, None,
            "invalid-metrics"
        ),
    ],
    ids=["complete-research", "no-name-no-ref", "invalid-metrics"]
)
def test_research_parsing(research_parser, transformer, research_string, expected_name, expected_metrics, expected_references, test_id):
    if expected_metrics is None:
        with pytest.raises((exceptions.UnexpectedCharacters, exceptions.UnexpectedToken, DSLValidationError)):
            tree = research_parser.parse(research_string)
            transformer.transform(tree)
    else:
        tree = research_parser.parse(research_string)
        result = transformer.transform(tree)
        assert result['type'] == 'Research'
        assert result['name'] == expected_name
        assert result.get('params', {}).get('metrics', {}) == expected_metrics
        assert result.get('params', {}).get('references', []) == expected_references

# Macro Parsing Tests
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
def test_macro_parsing(define_parser, layer_parser, transformer, config, expected_definition, expected_reference, raises_error, test_id):
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

# Wrapper Parsing Tests
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
def test_wrapper_parsing(layer_parser, transformer, wrapper_string, expected, test_id):
    if expected is None:
        with pytest.raises(VisitError) as exc_info:
            tree = layer_parser.parse(wrapper_string)
            transformer.transform(tree)
    else:
        tree = layer_parser.parse(wrapper_string)
        result = transformer.transform(tree)
        assert result == expected, f"Failed for {test_id}"

# Lambda Parsing Tests
@pytest.mark.parametrize(
    "lambda_string, expected, test_id",
    [
        (
            'Lambda("x: x * 2")',
            {'type': 'Lambda', 'params': {'function': 'x: x * 2'}, 'sublayers': []},
            "lambda-multiply"
        ),
        (
            'Lambda("lambda x: x + 1")',
            {'type': 'Lambda', 'params': {'function': 'lambda x: x + 1'}, 'sublayers': []},
            "lambda-add"
        ),
        (
            'Lambda(123)',
            None,
            "lambda-invalid"
        ),
    ],
    ids=["lambda-multiply", "lambda-add", "lambda-invalid"]
)
def test_lambda_parsing(layer_parser, transformer, lambda_string, expected, test_id):
    if expected is None:
        with pytest.raises((exceptions.UnexpectedCharacters, exceptions.UnexpectedToken, DSLValidationError)):
            tree = layer_parser.parse(lambda_string)
            transformer.transform(tree)
    else:
        tree = layer_parser.parse(lambda_string)
        result = transformer.transform(tree)
        assert result == expected, f"Failed for {test_id}"

# Custom Shape Parsing Tests
@pytest.mark.parametrize(
    "custom_shape_string, expected, test_id",
    [
        (
            'CustomShape(MyLayer, (32, 32))',
            {"type": "CustomShape", "layer": "MyLayer", "custom_dims": (32, 32)},
            "custom-shape-normal"
        ),
        (
            'CustomShape(MyLayer, (-1, 32))',
            None,
            "custom-shape-negative"
        ),
    ],
    ids=["custom-shape-normal", "custom-shape-negative"]
)
def test_custom_shape_parsing(layer_parser, transformer, custom_shape_string, expected, test_id):
    if expected is None:
        with pytest.raises(VisitError) as exc_info:
            tree = layer_parser.parse(custom_shape_string)
            transformer.transform(tree)
    else:
        tree = layer_parser.parse(custom_shape_string)
        result = transformer.transform(tree)
        assert result == expected, f"Failed for {test_id}"

# Comment Parsing Tests
@pytest.mark.parametrize(
    "comment_string, expected, test_id",
    [
        (
            'Dense(128, "relu")  # Dense layer with ReLU activation',
            {'type': 'Dense', 'params': {'units': 128, 'activation': 'relu'}, 'sublayers': []},
            "dense-with-comment"
        ),
        (
            'Dropout(0.5)  # Dropout layer',
            {'type': 'Dropout', 'params': {'rate': 0.5}, 'sublayers': []},
            "dropout-with-comment"
        ),
        (
            'Conv2D(32, (3, 3)) # Multi-line\n# comment',
            {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': (3, 3)}, 'sublayers': []},
            "conv2d-multi-comment"
        ),
    ],
    ids=["dense-with-comment", "dropout-with-comment", "conv2d-multi-comment"]
)
def test_comment_parsing(layer_parser, transformer, comment_string, expected, test_id):
    tree = layer_parser.parse(comment_string)
    result = transformer.transform(tree)
    assert result == expected, f"Failed for {test_id}"

# Severity Level and Warning Tests
@pytest.mark.parametrize(
    "layer_string, expected_result, expected_warnings, raises_error, test_id",
    [
        (
            'Dropout(1.1)',
            {'type': 'Dropout', 'params': {'rate': 1.1}, 'sublayers': []},
            [{'warning': 'Dropout rate should be between 0 and 1, got 1.1', 'line': 1, 'column': None}],
            False,
            "dropout-high-rate-warning"
        ),
        (
            'Dropout(-0.1)',
            None,
            [],
            True,
            "dropout-negative-rate-error"
        ),
        (
            'Dense(256.0, "relu")',
            {'type': 'Dense', 'params': {'units': 256, 'activation': 'relu'}, 'sublayers': []},
            [{'warning': 'Implicit conversion of 256.0 to integer 256', 'line': 1, 'column': None}],
            False,
            "dense-type-coercion-info"
        ),
        (
            'Conv2D(32, (3,3)) @ "npu"',  # Assuming validation for valid devices: ["cpu", "cuda", "tpu"]
            None,
            [],
            True,
            "invalid-device-critical"
        ),
    ],
    ids=["dropout-high-rate-warning", "dropout-negative-rate-error", "dense-type-coercion-info", "invalid-device-critical"]
)
def test_severity_level_parsing(layer_parser, transformer, layer_string, expected_result, expected_warnings, raises_error, test_id):
    if raises_error:
        with pytest.raises(VisitError) as exc_info:
            tree = layer_parser.parse(layer_string)
            transformer.transform(tree)
    else:
        tree = layer_parser.parse(layer_string)
        result = transformer.transform(tree)
        assert result == expected_result, f"Result mismatch for {test_id}"
        # Note: Warnings are currently logged, not returned. If modified to return, uncomment:
        # assert 'warnings' in result and result['warnings'] == expected_warnings, f"Warnings mismatch for {test_id}"

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
def test_validation_rules(network_parser, transformer, network_string, expected_error_msg, test_id):
    with pytest.raises(DSLValidationError) as exc_info:
        transformer.parse_network(network_string)
    assert expected_error_msg in str(exc_info.value), f"Error message mismatch for {test_id}"

def test_grammar_token_definitions():
    """Test that grammar token definitions are correct and complete."""
    parser = create_parser()
    lexer_conf = parser.parser.lexer_conf

    # Test all expected token patterns
    token_patterns = {
        'TRANSFORMER': r'(?i:transformer)',
        'LSTM': r'(?i:lstm)',
        'GRU': r'(?i:gru)',
        'DENSE': r'(?i:dense)',
        'CONV2D': r'(?i:conv2d)',
        'NAME': r'[a-zA-Z_][a-zA-Z0-9_]*',
        'NUMBER': r'[+-]?([0-9]*[.])?[0-9]+',
        'STRING': r'\"[^"]+\"|\'[^\']+\'',
        'CUSTOM_LAYER': r'[A-Z][a-zA-Z0-9]*Layer'
    }

    for token_name, pattern in token_patterns.items():
        matching_token = next((t for t in lexer_conf.terminals if t.name == token_name), None)
        assert matching_token is not None, f"Token {token_name} not found in grammar"
        # Compare the pattern.value instead of str()
        assert matching_token.pattern.value == pattern, f"Unexpected pattern for {token_name}"

def test_rule_dependencies():
        """Test that grammar rules have correct dependencies."""
        parser = create_parser()
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

@pytest.mark.parametrize("rule_name,valid_inputs", [
    ('NAME', ['valid_name', '_valid_name', 'ValidName123']),
    ('NUMBER', ['123', '-123', '123.456', '-123.456']),
    ('STRING', ['"relu"', "'softmax'"]),  # Strings in valid contexts (as activation functions)
    ('CUSTOM_LAYER', ['CustomLayer', 'MyTestLayer', 'ConvLayer'])
])
def test_token_patterns(rule_name, valid_inputs):
    """Test that token patterns match expected inputs."""
    parser = create_parser()
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
                # Test NAME token in a valid context (as variable name)
                result = parser.parse(f'network TestNet {{ input: (1,1) layers: Dense(10, "{input_str}") }}')
            else:
                result = parser.parse(f'network {input_str} {{ input: (1,1) layers: Dense(10) }}')
            assert result is not None
        except Exception as e:
            pytest.fail(f"Failed to parse {rule_name} with input {input_str}: {str(e)}")

def test_rule_precedence():
    """Test that grammar rules have correct precedence."""
    parser = create_parser()
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
            # Parse normally - LALR parser should resolve any ambiguities
            parser.parse(f"network TestNet {{ input: (1,1) layers: {test_input} }}")
        except lark.exceptions.UnexpectedInput as e:
            pytest.fail(f"Unexpected parse error for {test_id}: {str(e)}")
        except Exception as e:
            pytest.fail(f"Failed to parse {test_id}: {str(e)}")

def test_error_recovery():
    """Test parser's error recovery capabilities."""
    parser = create_parser()
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


@pytest.mark.parametrize(
    "test_input, expected_error",
    [
        # Invalid activation in Dense layer
        (
            "network Test { input: (None, 100) layers: Dense(HPO(choice(64, 128)), activation='invalid_act') }",
            "Invalid activation function 'invalid_act'",
        ),
        # Missing units in LSTM
        (
            "network Test { input: (32, 64) layers: LSTM(return_sequences=True) }",
            "LSTM requires 'units' parameter",
        ),
        # Invalid dropout rate
        (
            "network Test { input: (100,) layers: Dropout(1.5) }",
            "Dropout rate should be between 0 and 1",
        ),
    ],
)
def test_semantic_validation(test_input, expected_error):
    transformer = ModelTransformer()
    with pytest.raises(DSLValidationError) as exc_info:
        transformer.parse_network(test_input)
    assert expected_error in str(exc_info.value)
    assert exc_info.value.severity == Severity.ERROR

def test_grammar_completeness():
        """Test that grammar covers all required language features."""
        parser = create_parser()
        # Additional Layer Parsing Tests
@pytest.mark.parametrize(
            "layer_string, expected, test_id",
            [
                # Extended Basic Layer Tests
                ('Dense(64, activation="tanh")',
                 {'type': 'Dense', 'params': {'units': 64, 'activation': 'tanh'}, 'sublayers': []},
                 "dense-tanh"),

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

                # Layer with Mixed Parameter Styles
                ('LSTM(128, return_sequences=true, dropout=0.2)',
                 {'type': 'LSTM', 'params': {
                     'units': 128,
                     'return_sequences': True,
                     'dropout': 0.2
                 }, 'sublayers': []},
                 "lstm-mixed-params"),

                # Nested Layer with Complex Configuration
                ('''TransformerEncoder(num_heads=8, ff_dim=256) {
                    LayerNormalization()
                    Dense(64, "relu")
                    Dropout(0.1)
                }''',
                 {'type': 'TransformerEncoder',
                  'params': {'num_heads': 8, 'ff_dim': 256},
                  'sublayers': [
                      {'type': 'LayerNormalization', 'params': None, 'sublayers': []},
                      {'type': 'Dense', 'params': {'units': 64, 'activation': 'relu'}, 'sublayers': []},
                      {'type': 'Dropout', 'params': {'rate': 0.1}, 'sublayers': []}
                  ]},
                 "transformer-complex"),

                # Edge Cases
                ('Dense(0)', None, "dense-zero-units"),
                ('Conv2D(32, (0,0))', None, "conv2d-zero-kernel"),
                ('Dropout(2.0)', None, "dropout-invalid-rate"),
                ('LSTM(units=-1)', None, "lstm-negative-units"),

                # Device Specification Tests
                ('Dense(128) @ "cpu"',
                 {'type': 'Dense', 'params': {'units': 128, 'device': 'cpu'}, 'sublayers': []},
                 "dense-cpu-device"),
                ('Dense(128) @ "invalid_device"', None, "dense-invalid-device"),

                # Custom Layer Tests
                ('CustomTestLayer(param1=10, param2="test")',
                 {'type': 'CustomTestLayer', 'params': {'param1': 10, 'param2': 'test'}, 'sublayers': []},
                 "custom-layer-basic"),

                # Activation Layer Tests
                ('Activation("leaky_relu", alpha=0.1)',
                 {'type': 'Activation', 'params': {'function': 'leaky_relu', 'alpha': 0.1}, 'sublayers': []},
                 "activation-with-params"),

                # Layer with HPO Parameters
                ('Dense(HPO(choice(32, 64, 128)), activation=HPO(choice("relu", "tanh")))',
                 {'type': 'Dense',
                  'params': {
                      'units': {'hpo': {'type': 'categorical', 'values': [32, 64, 128]}},
                      'activation': {'hpo': {'type': 'categorical', 'values': ['relu', 'tanh']}}
                  }, 'sublayers': []},
                 "dense-hpo-multiple"),
            ],
            ids=[
                "dense-tanh",
                "conv2d-multiple-params",
                "lstm-mixed-params",
                "transformer-complex",
                "dense-zero-units",
                "conv2d-zero-kernel",
                "dropout-invalid-rate",
                "lstm-negative-units",
                "dense-cpu-device",
                "dense-invalid-device",
                "custom-layer-basic",
                "activation-with-params",
                "dense-hpo-multiple",
            ]
        )
def test_extended_layer_parsing(layer_parser, transformer, layer_string, expected, test_id):
            """Test parsing of various layer configurations with extended test cases."""
            if expected is None:
                with pytest.raises(VisitError) as exc_info:
                    tree = layer_parser.parse(layer_string)
                    transformer.transform(tree)
            else:
                tree = layer_parser.parse(layer_string)
                result = transformer.transform(tree)
                assert result == expected, f"Failed for {test_id}: expected {expected}, got {result}"

@pytest.mark.parametrize(
    "layer_string, validation_error_msg, test_id",
    [
        ('Dense(units="invalid")', "Dense units must be a number", "dense-invalid-units-type"),
        ('Conv2D(filters=-5, kernel_size=(3,3))', "Conv2D filters must be a positive integer", "conv2d-negative-filters"),
        ('LSTM(units=0, return_sequences=true)', "LSTM units must be positive", "lstm-zero-units"),
        ('Dropout(rate=1.5)', "Dropout rate should be between 0 and 1", "dropout-high-rate"),
        ('BatchNormalization(momentum=2.0)', "BatchNormalization momentum must be between 0 and 1", "batchnorm-invalid-momentum"),
        ('Conv2D(32, (-1,-1))', "Conv2D kernel_size should be positive integers", "conv2d-negative-kernel"),
        ('MaxPooling2D(pool_size=(0,0))', "pool_size must be positive", "maxpool-zero-size"),
    ],
    ids=[
        "dense-invalid-units-type",
        "conv2d-negative-filters",
        "lstm-zero-units",
        "dropout-high-rate",
        "batchnorm-invalid-momentum",
        "conv2d-negative-kernel",
        "maxpool-zero-size"
    ]
)

def test_layer_validation_errors(layer_parser, transformer, layer_string, validation_error_msg, test_id):
        """Test validation error messages for invalid layer configurations."""
        with pytest.raises(VisitError) as exc_info:
            tree = layer_parser.parse(layer_string)
            transformer.transform(tree)
        assert validation_error_msg in str(exc_info.value.__context__), f"Error message mismatch for {test_id}"

# Additional Edge Case Tests
@pytest.mark.parametrize(
    "layer_string, expected, test_id",
    [
        # Empty parameter lists
        ('Dense()', None, "dense-empty-params"),
        ('Conv2D()', None, "conv2d-empty-params"),

        # Whitespace handling
        ('Dense ( 128 ,  "relu" )', {'type': 'Dense', 'params': {'units': 128, 'activation': 'relu'}, 'sublayers': []}, "dense-extra-whitespace"),
        ('Conv2D(32,(3,3))', {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': (3, 3)}, 'sublayers': []}, "conv2d-no-whitespace"),

        # Case sensitivity
        ('dense(64, "relu")', None, "dense-lowercase"),
        ('DENSE(64, "relu")', {'type': 'Dense', 'params': {'units': 64, 'activation': 'relu'}, 'sublayers': []}, "dense-uppercase"),

        # Boolean parameters
        ('LSTM(128, return_sequences=true)', {'type': 'LSTM', 'params': {'units': 128, 'return_sequences': True}, 'sublayers': []}, "lstm-boolean-true"),
        ('LSTM(128, return_sequences=false)', {'type': 'LSTM', 'params': {'units': 128, 'return_sequences': False}, 'sublayers': []}, "lstm-boolean-false"),

        # Nested tuples and lists
        ('Conv3D(32, ((3, 3), 3))', {'type': 'Conv3D', 'params': {'filters': 32, 'kernel_size': ((3, 3), 3)}, 'sublayers': []}, "conv3d-nested-tuple"),
        ('CustomLayer(params=[1, 2, [3, 4]])', {'type': 'CustomLayer', 'params': {'params': [1, 2, [3, 4]]}, 'sublayers': []}, "custom-nested-list"),

        # Special characters in strings
        ('Dense(64, activation="relu\\n")', None, "dense-newline-in-string"),
        ('Dense(64, activation="re\\"lu")', None, "dense-quote-in-string"),

        # Mixed positional and named parameters
        ('Conv2D(32, (3, 3), padding="same")', {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': (3, 3), 'padding': 'same'}, 'sublayers': []}, "conv2d-mixed-params"),
        ('Dense(64, activation="relu", use_bias=true)', {'type': 'Dense', 'params': {'units': 64, 'activation': 'relu', 'use_bias': True}, 'sublayers': []}, "dense-mixed-params"),

        # Scientific notation
        ('Dense(1e3)', {'type': 'Dense', 'params': {'units': 1000}, 'sublayers': []}, "dense-scientific-notation"),
        ('Dropout(1e-2)', {'type': 'Dropout', 'params': {'rate': 0.01}, 'sublayers': []}, "dropout-scientific-notation"),

        # Extremely large values
        ('Dense(1000000000)', {'type': 'Dense', 'params': {'units': 1000000000}, 'sublayers': []}, "dense-large-value"),
        ('Conv2D(999999, (9999, 9999))', None, "conv2d-unreasonable-values"),

        # Multiple nested layers with comments
        ('''Residual() {  # Outer comment
            Conv2D(32, (3, 3))  # Inner comment 1
            BatchNormalization()  # Inner comment 2
        }''',
        {'type': 'Residual', 'params': None, 'sublayers': [
            {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': (3, 3)}, 'sublayers': []},
            {'type': 'BatchNormalization', 'params': None, 'sublayers': []}
        ]}, "residual-with-comments"),

        # Complex HPO with multiple nested choices
        ('Dense(HPO(choice(HPO(range(64, 256, 64)), HPO(choice(512, 1024)))))',
         {'type': 'Dense', 'params': {'units': {'hpo': {'type': 'categorical', 'values': [
             {'hpo': {'type': 'range', 'min': 64, 'max': 256, 'step': 64}},
             {'hpo': {'type': 'categorical', 'values': [512, 1024]}}
         ]}}}, 'sublayers': []}, "dense-nested-hpo")
    ],
    ids=[
        "dense-empty-params", "conv2d-empty-params", "dense-extra-whitespace", "conv2d-no-whitespace",
        "dense-lowercase", "dense-uppercase", "lstm-boolean-true", "lstm-boolean-false",
        "conv3d-nested-tuple", "custom-nested-list", "dense-newline-in-string", "dense-quote-in-string",
        "conv2d-mixed-params", "dense-mixed-params", "dense-scientific-notation", "dropout-scientific-notation",
        "dense-large-value", "conv2d-unreasonable-values", "residual-with-comments", "dense-nested-hpo"
    ]
)
def test_edge_case_layer_parsing(layer_parser, transformer, layer_string, expected, test_id):
    """Test parsing of edge cases and unusual syntax patterns."""
    if expected is None:
        with pytest.raises((exceptions.UnexpectedCharacters, exceptions.UnexpectedToken, DSLValidationError, VisitError)):
            tree = layer_parser.parse(layer_string)
            transformer.transform(tree)
    else:
        tree = layer_parser.parse(layer_string)
        result = transformer.transform(tree)
        assert result == expected, f"Failed for {test_id}: expected {expected}, got {result}"

# Network Structure Validation Tests
@pytest.mark.parametrize(
    "network_string, expected_error_msg, test_id",
    [
        # Missing required sections
        (
            """
            network MissingInput {
                layers: Dense(10)
                loss: "mse"
                optimizer: "sgd"
            }
            """,
            "Network must have an input section",
            "missing-input-section"
        ),
        (
            """
            network MissingLayers {
                input: (10,)
                loss: "mse"
                optimizer: "sgd"
            }
            """,
            "Network must have a layers section",
            "missing-layers-section"
        ),

        # Duplicate sections
        (
            """
            network DuplicateInput {
                input: (10,)
                input: (20,)
                layers: Dense(10)
                loss: "mse"
                optimizer: "sgd"
            }
            """,
            "Duplicate input section",
            "duplicate-input-section"
        ),

        # Invalid input shapes
        (
            """
            network InvalidInputShape {
                input: (0, -1)
                layers: Dense(10)
                loss: "mse"
                optimizer: "sgd"
            }
            """,
            "Input dimensions must be positive",
            "negative-input-dimension"
        ),

        # Empty layers section
        (
            """
            network EmptyLayers {
                input: (10,)
                layers:
                loss: "mse"
                optimizer: "sgd"
            }
            """,
            "Layers section cannot be empty",
            "empty-layers-section"
        ),

        # Invalid loss function
        (
            """
            network InvalidLoss {
                input: (10,)
                layers: Dense(10)
                loss: "invalid_loss"
                optimizer: "sgd"
            }
            """,
            "Invalid loss function",
            "invalid-loss-function"
        ),

        # Invalid optimizer
        (
            """
            network InvalidOptimizer {
                input: (10,)
                layers: Dense(10)
                loss: "mse"
                optimizer: "invalid_optimizer"
            }
            """,
            "Invalid optimizer",
            "invalid-optimizer"
        ),

        # Incompatible layer sequence
        (
            """
            network IncompatibleLayers {
                input: (28, 28, 1)
                layers:
                    Dense(128)  # Dense expects flattened input
                    Conv2D(32, (3, 3))  # Conv2D after Dense doesn't make sense
                loss: "mse"
                optimizer: "sgd"
            }
            """,
            "Conv2D cannot follow Dense",
            "incompatible-layer-sequence"
        ),

        # Mismatched input/output dimensions
        (
            """
            network MismatchedDimensions {
                input: (10,)
                layers:
                    Dense(5)
                    Output(20)  # Output size doesn't match problem
                loss: "categorical_crossentropy"  # Categorical loss with single output
                optimizer: "sgd"
            }
            """,
            "Output dimensions don't match loss function",
            "mismatched-dimensions"
        ),

        # Invalid training parameters
        (
            """
            network InvalidTraining {
                input: (10,)
                layers: Dense(5)
                loss: "mse"
                optimizer: "sgd"
                train {
                    epochs: -10
                    batch_size: 0
                }
            }
            """,
            "Training parameters must be positive",
            "invalid-training-params"
        )
    ],
    ids=[
        "missing-input-section", "missing-layers-section", "duplicate-input-section",
        "negative-input-dimension", "empty-layers-section", "invalid-loss-function",
        "invalid-optimizer", "incompatible-layer-sequence", "mismatched-dimensions",
        "invalid-training-params"
    ]
)
def test_network_structure_validation(network_parser, transformer, network_string, expected_error_msg, test_id):
    """Test validation of network structure and configuration."""
    with pytest.raises(DSLValidationError) as exc_info:
        transformer.parse_network(network_string)
    assert expected_error_msg in str(exc_info.value), f"Error message mismatch for {test_id}"

def test_learning_rate_schedule():
    """Test parsing of learning rate schedules."""
    config = """
    network LRScheduleModel {
        input: (28, 28, 1)
        layers:
            Conv2D(32, kernel_size=(3,3), activation="relu")
            MaxPooling2D(pool_size=(2,2))
            Flatten()
            Dense(128, activation="relu")
            Dense(10, activation="softmax")
        optimizer: SGD(
            learning_rate=ExponentialDecay(
                HPO(range(0.05, 0.2, step=0.05)),
                1000,
                HPO(range(0.9, 0.99, step=0.01))
            ),
            momentum=0.9
        )
    }
    """

    expected = {
        'framework': 'tensorflow',
        'input': {'shape': (28, 28, 1), 'type': 'Input'},
        'layers': [
            {'params': {'activation': 'relu', 'filters': 32, 'kernel_size': (3, 3)}, 'sublayers': [], 'type': 'Conv2D'},
            {'params': {'pool_size': (2, 2)}, 'sublayers': [], 'type': 'MaxPooling2D'},
            {'params': None, 'sublayers': [], 'type': 'Flatten'},
            {'params': {'activation': 'relu', 'units': 128}, 'sublayers': [], 'type': 'Dense'},
            {'params': {'activation': 'softmax', 'units': 10}, 'sublayers': [], 'type': 'Dense'}
        ],
        'name': 'LRScheduleModel',
        'optimizer': {
            'type': 'SGD',
            'params': {
                'learning_rate': {
                    'type': 'ExponentialDecay',
                    'params': {
                        'initial_learning_rate': {
                            'hpo': {
                                'type': 'range',
                                'start': 0.05,
                                'end': 0.2,
                                'step': 0.05
                            }
                        },
                        'decay_steps': 1000,
                        'decay_rate': {
                            'hpo': {
                                'type': 'range',
                                'start': 0.9,
                                'end': 0.99,
                                'step': 0.01
                            }
                        }
                    }
                },
                'momentum': 0.9
            }
        },
        'shape_info': [],
        'warnings': []
    }

    result = ModelTransformer().transform(config)
    assert_dict_equal(result, expected)
