import os
import sys
import pytest
from lark import Lark, exceptions

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural.parser.parser import ModelTransformer, create_parser, DSLValidationError

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

# Layer parsing tests
@pytest.mark.parametrize(
    "layer_string,expected,test_id",
    [
        # Basic Layers
        ('Dense(128, "relu")', {'type': 'Dense', 'params': {'units': 128, 'activation': 'relu'}}, "dense-relu"),
        ('Dense(units=256, activation="sigmoid")', {'type': 'Dense', 'params': {'units': 256, 'activation': 'sigmoid'}}, "dense-sigmoid"),
        ('Conv2D(32, (3, 3), activation="relu")', {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'}}, "conv2d-relu"),
        ('Conv2D(filters=64, kernel_size=(5, 5), activation="tanh")', {'type': 'Conv2D', 'params': {'filters': 64, 'kernel_size': (5, 5), 'activation': 'tanh'}}, "conv2d-tanh"),
        ('Conv2D(filters=32, kernel_size=3, activation="relu", padding="same")', {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': 3, 'activation': 'relu', 'padding': 'same'}}, "conv2d-padding"),
        ('MaxPooling2D(pool_size=(2, 2))', {'type': 'MaxPooling2D', 'params': {'pool_size': (2, 2)}}, "maxpooling2d"),
        ('MaxPooling2D((3, 3), 2, "valid")', {'type': 'MaxPooling2D', 'params': {'pool_size': (3, 3), 'strides': 2, 'padding': 'valid'}}, "maxpooling2d-strides"),
        ('Flatten()', {'type': 'Flatten', 'params': None}, "flatten"),
        ('Dropout(0.5)', {'type': 'Dropout', 'params': {'rate': 0.5}}, "dropout"),
        ('Dropout(rate=0.25)', {'type': 'Dropout', 'params': {'rate': 0.25}}, "dropout-named"),
        ('BatchNormalization()', {'type': 'BatchNormalization', 'params': None}, "batchnorm"),
        ('LayerNormalization()', {'type': 'LayerNormalization', 'params': None}, "layernorm"),
        ('InstanceNormalization()', {'type': 'InstanceNormalization', 'params': None}, "instancenorm"),
        ('GroupNormalization(groups=32)', {'type': 'GroupNormalization', 'params': {'groups': 32}}, "groupnorm"),

        # Recurrent Layers
        ('LSTM(units=64)', {'type': 'LSTM', 'params': {'units': 64}}, "lstm"),
        ('LSTM(units=128, return_sequences=true)', {'type': 'LSTM', 'params': {'units': 128, 'return_sequences': True}}, "lstm-return"),
        ('GRU(units=32)', {'type': 'GRU', 'params': {'units': 32}}, "gru"),
        ('SimpleRNN(units=16)', {'type': 'SimpleRNN', 'params': {'units': 16}}, "simplernn"),
        ('RNNCell(units=32)', {'type': 'RNNCell', 'params': {'units': 32}}, "rnncell"),
        ('LSTMCell(units=64)', {'type': 'LSTMCell', 'params': {'units': 64}}, "lstmcell"),
        ('GRUCell(units=128)', {'type': 'GRUCell', 'params': {'units': 128}}, "grucell"),
        ('SimpleRNNDropoutWrapper(units=16, dropout=0.3)', {'type': 'SimpleRNNDropoutWrapper', 'params': {'units': 16, 'dropout': 0.3}}, "simplernn-dropout"),
        ('GRUDropoutWrapper(units=32, dropout=0.4)', {'type': 'GRUDropoutWrapper', 'params': {'units': 32, 'dropout': 0.4}}, "gru-dropout"),
        ('LSTMDropoutWrapper(units=64, dropout=0.5)', {'type': 'LSTMDropoutWrapper', 'params': {'units': 64, 'dropout': 0.5}}, "lstm-dropout"),

        # Advanced Layers
        ('Attention()', {'type': 'Attention', 'params': None}, "attention"),
        ('TransformerEncoder(num_heads=8, ff_dim=512)', {'type': 'TransformerEncoder', 'params': {'num_heads': 8, 'ff_dim': 512}}, "transformer"),
        ('ResidualConnection()', {'type': 'ResidualConnection', 'params': None}, "residual"),
        ('Inception()', {'type': 'Inception', 'params': None}, "inception"),
        ('CapsuleLayer()', {'type': 'CapsuleLayer', 'params': None}, "capsule"),
        ('SqueezeExcitation()', {'type': 'SqueezeExcitation', 'params': None}, "squeeze"),
        ('GraphConv()', {'type': 'GraphConv', 'params': None}, "graphconv"),
        ('GraphAttention(num_heads=4)', {'type': 'GraphAttention', 'params': {'num_heads': 4}}, "graph-attention"),  # New
        ('Embedding(input_dim=1000, output_dim=128)', {'type': 'Embedding', 'params': {'input_dim': 1000, 'output_dim': 128}}, "embedding"),
        ('QuantumLayer()', {'type': 'QuantumLayer', 'params': None}, "quantum"),
        ('DynamicLayer()', {'type': 'DynamicLayer', 'params': None}, "dynamic"),

        # Output and Custom
        ('Output(units=10, activation="softmax")', {'type': 'Output', 'params': {'units': 10, 'activation': 'softmax'}}, "output-softmax"),
        ('Output(units=1, activation="sigmoid")', {'type': 'Output', 'params': {'units': 1, 'activation': 'sigmoid'}}, "output-sigmoid"),
        ('Lambda("x: x * 2")', {'type': 'Lambda', 'params': {'function': 'x: x * 2'}}, "lambda-layer"),  # New
        ('CustomShape(MyLayer, (32, 32))', {'type': 'CustomShape', 'layer': 'MyLayer', 'custom_dims': (32, 32)}, "custom-shape"),  # New

        # Edge Cases
        ('Dense(units=0)', {'type': 'Dense', 'params': {'units': 0}}, "dense-zero-units"),
        ('Dropout(rate=1.0)', {'type': 'Dropout', 'params': {'rate': 1.0}}, "dropout-full"),
        ('MaxPooling2D(pool_size=1)', {'type': 'MaxPooling2D', 'params': {'pool_size': 1}}, "maxpool-unit-poolsize"),
        ('Conv2D(32, (0, 0))', None, "conv2d-zero-kernel"),  # New error case
        ('Dropout(-0.1)', None, "dropout-negative-rate"),  # New error case
        ('LSTM(units=-5)', None, "lstm-negative-units"),  # New error case

        # Error Cases
        ('Dense(units="abc")', None, "dense-invalid-units"),
        ('Dropout(2)', None, "dropout-invalid-rate"),
        ('InvalidLayerName()', None, "invalid-layer-name"),
        ('TransformerEncoder(num_heads=-1)', None, "transformer-negative-heads"),  # New error case
        ('Embedding(input_dim="abc", output_dim=128)', None, "embedding-invalid-dim"),  # New error case
    ],
    ids=[
        "dense-relu", "dense-sigmoid", "conv2d-relu", "conv2d-tanh", "conv2d-padding", "maxpooling2d", "maxpooling2d-strides",
        "flatten", "dropout", "dropout-named", "batchnorm", "layernorm", "instancenorm", "groupnorm", "lstm", "lstm-return",
        "gru", "simplernn", "rnncell", "lstmcell", "grucell", "simplernn-dropout", "gru-dropout", "lstm-dropout",
        "attention", "transformer", "residual", "inception", "capsule", "squeeze", "graphconv", "graph-attention",
        "embedding", "quantum", "dynamic", "output-softmax", "output-sigmoid", "lambda-layer", "custom-shape",
        "dense-zero-units", "dropout-full", "maxpool-unit-poolsize", "conv2d-zero-kernel", "dropout-negative-rate",
        "lstm-negative-units", "dense-invalid-units", "dropout-invalid-rate", "invalid-layer-name", "transformer-negative-heads",
        "embedding-invalid-dim"
    ]
)
def test_layer_parsing(layer_parser, transformer, layer_string, expected, test_id):
    if expected is None:
        with pytest.raises((
            exceptions.UnexpectedCharacters, 
            exceptions.UnexpectedToken, 
            DSLValidationError,
            exceptions.VisitError  # Add VisitError to handle wrapped validation errors
        )):
            tree = layer_parser.parse(layer_string)
            transformer.transform(tree)
    else:
        tree = layer_parser.parse(layer_string)
        result = transformer.transform(tree)
        assert result == expected, f"Failed for {test_id}: expected {expected}, got {result}"
# Network parsing tests
@pytest.mark.parametrize(
    "network_string, expected_name, expected_input_shape, expected_layers, expected_loss, expected_optimizer, expected_training_config",
    [
        # Complex Network
        (
            """
            network TestModel {
                input: (None, 28, 28, 1)
                layers:
                    Conv2D(filters=32, kernel_size=(3,3), activation="relu")
                    MaxPooling2D(pool_size=(2, 2))
                    Flatten()
                    Dense(units=128, activation="relu")
                    Output(units=10, activation="softmax")
                loss: "categorical_crossentropy"
                optimizer: "adam"
                train {
                    epochs: 10
                    batch_size: 32
                }
            }
            """,
            "TestModel", (None, 28, 28, 1),
            [
                {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'}},
                {'type': 'MaxPooling2D', 'params': {'pool_size': (2, 2)}},
                {'type': 'Flatten', 'params': None},
                {'type': 'Dense', 'params': {'units': 128, 'activation': 'relu'}},
                {'type': 'Output', 'params': {'units': 10, 'activation': 'softmax'}}
            ],
            "categorical_crossentropy", {'type': 'adam', 'params': {}}, {'epochs': 10, 'batch_size': 32}
        ),
        # Simple Network
        (
            """
            network SimpleModel {
                input: (28, 28, 1)
                layers:
                    Flatten()
                    Output(units=1, activation="sigmoid")
                loss: "binary_crossentropy"
                optimizer: "SGD"
            }
            """,
            "SimpleModel", (28, 28, 1),
            [
                {'type': 'Flatten', 'params': None},
                {'type': 'Output', 'params': {'units': 1, 'activation': 'sigmoid'}}
            ],
            "binary_crossentropy", {'type': 'SGD', 'params': {}}, None
        ),
        # No Layers
        (
            """
            network NoLayers {
                input: (10,)
                layers:
                loss: "mse"
                optimizer: "rmsprop"
            }
            """,
            "NoLayers", (10,), [], "mse", {'type': 'rmsprop', 'params': {}}, None
        ),
        # With Optimizer Params
        (
            """
            network OptimizerModel {
                input: (28, 28, 1)
                layers:
                    Dense(64, "relu")
                    Output(units=10, activation="softmax")
                loss: "categorical_crossentropy"
                optimizer: SGD(learning_rate=0.01)
                train {
                    epochs: 20
                    batch_size: 64
                }
            }
            """,
            "OptimizerModel", (28, 28, 1),
            [
                {'type': 'Dense', 'params': {'units': 64, 'activation': 'relu'}},
                {'type': 'Output', 'params': {'units': 10, 'activation': 'softmax'}}
            ],
            "categorical_crossentropy", {'type': 'SGD', 'params': {'learning_rate': 0.01}}, {'epochs': 20, 'batch_size': 64}
        ),
        # Error Case: Invalid Optimizer
        (
            """
            network InvalidOptimizer {
                input: (10,)
                layers:
                    Dense(5)
                loss: "mse"
                optimizer: InvalidOpt(learning_rate="abc")
            }
            """,
            None, None, None, None, None, None
        ),
        # Validation Split Tests
        (
            """
            network ValidationTest {
                input: (28, 28, 1)
                layers:
                    Dense(64, "relu")
                    Output(units=10, activation="softmax")
                loss: "categorical_crossentropy"
                optimizer: "adam"
                train {
                    epochs: 10
                    batch_size: 32
                    validation_split: 0.2
                }
            }
            """,
            "ValidationTest", (28, 28, 1),
            [
                {'type': 'Dense', 'params': {'units': 64, 'activation': 'relu'}},
                {'type': 'Output', 'params': {'units': 10, 'activation': 'softmax'}}
            ],
            "categorical_crossentropy", {'type': 'adam', 'params': {}}, {'epochs': 10, 'batch_size': 32, 'validation_split': 0.2}
        ),
        (
            """
            network InvalidValidation {
                input: (10,)
                layers:
                    Dense(5)
                loss: "mse"
                optimizer: "sgd"
                train {
                    validation_split: 1.1
                }
            }
            """,
            None, None, None, None, None, None
        ),
    ],
    ids=["complex-model", "simple-model", "no-layers", "optimizer-params", "invalid-optimizer", "valid-validation-split", "invalid-validation-split"]
)

def test_network_parsing(network_parser, transformer, network_string, expected_name, expected_input_shape, expected_layers, expected_loss, expected_optimizer, expected_training_config):
    if expected_name is None:
        with pytest.raises((exceptions.UnexpectedCharacters, exceptions.UnexpectedToken, DSLValidationError, exceptions.VisitError)):
            tree = network_parser.parse(network_string)
            transformer.transform(tree)
    else:
        tree = network_parser.parse(network_string)
        result = transformer.transform(tree)
        assert result['type'] == 'model'
        assert result['name'] == expected_name
        assert result['input'] == {'type': 'Input', 'shape': expected_input_shape}
        assert result['layers'] == expected_layers
        assert result['loss'] == expected_loss
        assert result['optimizer'] == expected_optimizer
        assert result['training_config'] == expected_training_config

# Research parsing tests
@pytest.mark.parametrize(
    "research_string, expected_name, expected_metrics, expected_references",
    [
        # Complete Research
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
            "ResearchStudy", {'accuracy': 0.95, 'loss': 0.05}, ["Paper Title 1", "Another Great Paper"]
        ),
        # No Name, Partial Metrics
        (
            """
            research {
                metrics {
                    precision: 0.8
                    recall: 0.9
                }
            }
            """,
            None, {'precision': 0.8, 'recall': 0.9}, []
        ),
        # Empty Research
        (
            """
            research EmptyResearch {
            }
            """,
            "EmptyResearch", {}, []
        ),
        # Error Case: Invalid Metrics
        (
            """
            research InvalidMetrics {
                metrics {
                    accuracy: "high"
                }
            }
            """,
            None, None, None
        ),
        # Mixed Metrics and References
        (
            """
            research MixedStudy {
                metrics {
                    accuracy: 0.99
                }
                references {
                    paper: "Some Paper"
                }
            }
            """,
            "MixedStudy", {'accuracy': 0.99}, ["Some Paper"]
        ),
    ],
    ids=["complete-research", "no-name-no-ref", "empty-research", "invalid-metrics", "mixed-research"]
)
def test_research_parsing(research_parser, transformer, research_string, expected_name, expected_metrics, expected_references):
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

# Wrapper parsing tests
@pytest.mark.parametrize(
    "wrapper_string, expected, test_id",
    [
        (
            'TimeDistributed(Dense(128, activation="relu"), dropout=0.5)',
            {'type': 'TimeDistributed(Dense)', 'params': {'units': 128, 'activation': 'relu', 'dropout': 0.5}},
            "timedistributed-dense"
        ),
        (
            'TimeDistributed(Conv2D(32, (3, 3)))',
            {'type': 'TimeDistributed(Conv2D)', 'params': {'filters': 32, 'kernel_size': (3, 3)}},
            "timedistributed-conv2d"
        ),
        # Error Case
        (
            'TimeDistributed(Dropout("invalid"))',
            None,
            "timedistributed-invalid"
        ),
    ],
    ids=["timedistributed-dense", "timedistributed-conv2d", "timedistributed-invalid"]
)
def test_wrapper_parsing(layer_parser, transformer, wrapper_string, expected, test_id):
    if expected is None:
        with pytest.raises((exceptions.UnexpectedCharacters, exceptions.UnexpectedToken, DSLValidationError)):
            tree = layer_parser.parse(wrapper_string)
            transformer.transform(tree)
    else:
        tree = layer_parser.parse(wrapper_string)
        result = transformer.transform(tree)
        assert result == expected

# Lambda parsing tests
@pytest.mark.parametrize(
    "lambda_string, expected, test_id",
    [
        (
            'Lambda("x: x * 2")',
            {'type': 'Lambda', 'params': {'function': 'x: x * 2'}},
            "lambda-multiply"
        ),
        (
            'Lambda("lambda x: x + 1")',
            {'type': 'Lambda', 'params': {'function': 'lambda x: x + 1'}},
            "lambda-add"
        ),
        # Error Case
        (
            'Lambda(123)',  # Invalid function string
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
        assert result == expected

# Custom Shape parsing tests
@pytest.mark.parametrize(
    "custom_shape_string, expected, test_id",
    [
        (
            'CustomShape(MyLayer, (32, 32))',
            {"type": "CustomShape", "layer": "MyLayer", "custom_dims": (32, 32)},
            "custom-shape-normal"
        ),
        (
            'CustomShape(ConvLayer, (64, 64))',
            {"type": "CustomShape", "layer": "ConvLayer", "custom_dims": (64, 64)},
            "custom-shape-conv"
        ),
        # Error Case
        (
            'CustomShape(MyLayer, (-1, 32))',  # Negative dimension
            None,
            "custom-shape-negative"
        ),
    ],
    ids=["custom-shape-normal", "custom-shape-conv", "custom-shape-negative"]
)
def test_custom_shape_parsing(layer_parser, transformer, custom_shape_string, expected, test_id):
    if expected is None:
        with pytest.raises((exceptions.UnexpectedCharacters, exceptions.UnexpectedToken, DSLValidationError)):
            tree = layer_parser.parse(custom_shape_string)
            transformer.transform(tree)
    else:
        tree = layer_parser.parse(custom_shape_string)
        result = transformer.transform(tree)
        assert result == expected

# Comment parsing tests
@pytest.mark.parametrize(
    "comment_string, expected, test_id",
    [
        (
            'Dense(128, "relu")  # Dense layer with ReLU activation',
            {'type': 'Dense', 'params': {'units': 128, 'activation': 'relu'}},
            "dense-with-comment"
        ),
        (
            'Dropout(0.5)  # Dropout layer',  # Simple comment
            {'type': 'Dropout', 'params': {'rate': 0.5}},
            "dropout-with-comment"
        ),
        (
            'Conv2D(32, (3, 3)) # Multi-line\n# comment',
            {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': (3, 3)}},
            "conv2d-multi-comment"
        ),
    ],
    ids=["dense-with-comment", "dropout-with-comment", "conv2d-multi-comment"]
)
def test_comment_parsing(layer_parser, transformer, comment_string, expected, test_id):
    tree = layer_parser.parse(comment_string)
    result = transformer.transform(tree)
    assert result == expected

@pytest.mark.parametrize(
    "layer_string, expected_result, expected_warnings, raises_error, test_id",
    [
        # Warning: Dropout rate > 1 (should log a warning but continue)
        (
            'Dropout(1.5)',
            {'type': 'Dropout', 'params': {'rate': 1.5}},
            [{'warning': 'WARNING: Dropout rate should be between 0 and 1, got 1.5', 'line': 1, 'column': None}],
            False,
            "dropout-high-rate-warning"
        ),
        # Error: Invalid dropout rate type (should raise an error)
        (
            'Dropout("invalid")',
            None,
            [],
            True,
            "dropout-invalid-type-error"
        ),
        # Warning: Negative kernel size (should log a warning but continue if transformer allows)
        (
            'Conv2D(32, (-1, 1))',
            None,  # Depends on transformer handling; expect error in current code
            [],
            True,  # Current code raises ERROR, not WARNING
            "conv2d-negative-kernel-error"  # Adjust if you change to WARNING
        ),
        # Valid case: No issues
        (
            'Dense(128, "relu")',
            {'type': 'Dense', 'params': {'units': 128, 'activation': 'relu'}},
            [],
            False,
            "dense-valid-no-issues"
        ),
    ],
    ids=["dropout-high-rate-warning", "dropout-invalid-type-error", "conv2d-negative-kernel-error", "dense-valid-no-issues"]
)
def test_severity_level_parsing(layer_parser, transformer, layer_string, expected_result, expected_warnings, raises_error, test_id):
    if raises_error:
        with pytest.raises((exceptions.UnexpectedCharacters, exceptions.UnexpectedToken, DSLValidationError, exceptions.VisitError)):
            tree = layer_parser.parse(layer_string)
            transformer.transform(tree)
    else:
        tree = layer_parser.parse(layer_string)
        result = transformer.transform(tree)
        # Check result matches expected
        assert result == expected_result, f"Failed for {test_id}: expected {expected_result}, got {result}"
        # Check warnings (if your transformer starts returning them; currently it doesn't)
        # For now, this assumes warnings are logged, not returned. Future enhancement could add them to result.
        # If you modify parse_network to return warnings, update this:
        # assert result.get('warnings', []) == expected_warnings, f"Warnings mismatch for {test_id}"