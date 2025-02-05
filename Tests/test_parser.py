import os
import sys
import pytest
from lark import Lark, exceptions

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser import ModelTransformer, create_parser

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

@pytest.mark.parametrize("layer_string,expected", [
    (
        'Dense(128, "relu")',
        {'type': 'Dense', 'params': {'units': 128, 'activation': 'relu'}}
    ),
    (
        'Dense(units=256, activation="sigmoid")',
        {'type': 'Dense', 'params': {'units': 256, 'activation': 'sigmoid'}}
    ),
    (
        'Conv2D(32, 3, 3, "relu")',
        {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'}}
    ),
    (
        'Conv2D(filters=64, kernel_size=(5, 5), activation="tanh")',
        {'type': 'Conv2D', 'params': {'filters': 64, 'kernel_size': (5, 5), 'activation': 'tanh'}}
    ),
    (
        'Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")',
        {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu', 'padding': 'same'}}
    ),
    (
        'MaxPooling2D(pool_size=(2, 2))',
        {'type': 'MaxPooling2D', 'params': {'pool_size': (2, 2)}}
    ),
    (
        'MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding="valid")',
         {'type': 'MaxPooling2D', 'params': {'pool_size': (3, 3), 'strides': (2, 2), 'padding': 'valid'}}
    ),
    (
        'Flatten()',
        {'type': 'Flatten', 'params': {}}
    ),
    (
        'Dropout(0.5)',
        {'type': 'Dropout', 'params': {'rate': 0.5}}
    ),
    (
        'Dropout(rate=0.25)',
        {'type': 'Dropout', 'params': {'rate': 0.25}}
    ),
    (
        'BatchNormalization()',
        {'type': 'BatchNormalization', 'params': {}}
    ),
    (
        'LayerNormalization()',
        {'type': 'LayerNormalization', 'params': {}}
    ),
    (
        'InstanceNormalization()',
        {'type': 'InstanceNormalization', 'params': {}}
    ),
    (
        'GroupNormalization(groups=32)',
        {'type': 'GroupNormalization', 'params': {'groups': 32}}
    ),
    (
        'LSTM(units=64)',
        {'type': 'LSTM', 'params': {'units': 64}}
    ),
    (
        'LSTM(units=128, return_sequences=true)',
        {'type': 'LSTM', 'params': {'units': 128, 'return_sequences': True}}
    ),
    (
        'GRU(units=32)',
        {'type': 'GRU', 'params': {'units': 32}}
    ),
    (
        'SimpleRNN(units=16)',
        {'type': 'SimpleRNN', 'params': {'units': 16}}
    ),
    (
        'CuDNNLSTM(units=256)',
        {'type': 'CuDNNLSTM', 'params': {'units': 256}}
    ),
    (
        'CuDNNGRU(units=128)',
        {'type': 'CuDNNGRU', 'params': {'units': 128}}
    ),
    (
        'RNNCell(units=32)',
        {'type': 'RNNCell', 'params': {'units': 32}}
    ),
    (
        'LSTMCell(units=64)',
        {'type': 'LSTMCell', 'params': {'units': 64}}
    ),
    (
        'GRUCell(units=128)',
        {'type': 'GRUCell', 'params': {'units': 128}}
    ),
    (
         'SimpleRNNDropoutWrapper(units=16, dropout=0.3)',
        {'type': 'SimpleRNNDropoutWrapper', 'params': {'units': 16, 'dropout': 0.3}}
    ),
    (
        'GRUDropoutWrapper(units=32, dropout=0.4)',
        {'type': 'GRUDropoutWrapper', 'params': {'units': 32, 'dropout': 0.4}}
    ),
    (
        'LSTMDropoutWrapper(units=64, dropout=0.5)',
        {'type': 'LSTMDropoutWrapper', 'params': {'units': 64, 'dropout': 0.5}}
    ),
    (
        'Attention()',
        {'type': 'Attention', 'params': {}}
    ),
    (
        'TransformerEncoder(num_heads=8, ff_dim=512)',
        {'type': 'TransformerEncoder', 'params': {'num_heads': 8, 'ff_dim': 512}}
    ),
    (
        'ResidualConnection()',
        {'type': 'ResidualConnection', 'params': {}}
    ),
    (
        'InceptionModule()',
        {'type': 'InceptionModule', 'params': {}}
    ),
    (
        'CapsuleLayer()',
        {'type': 'CapsuleLayer', 'params': {}}
    ),
    (
        'SqueezeExcitation()',
        {'type': 'SqueezeExcitation', 'params': {}}
    ),
    (
        'GraphConv()',
        {'type': 'GraphConv', 'params': {}}
    ),
    (
        'Embedding(input_dim=1000, output_dim=128)',
        {'type': 'Embedding', 'params': {'input_dim': 1000, 'output_dim': 128}}
    ),
    (
        'QuantumLayer()',
        {'type': 'QuantumLayer', 'params': {}}
    ),
    (
        'DynamicLayer()',
        {'type': 'DynamicLayer', 'params': {}}
    ),
    (
        'Output(units=10, activation="softmax")',
        {'type': 'Output', 'params': {'units': 10, 'activation': 'softmax'}}
    ),
    (
        'Output(units=1, activation="sigmoid")',
        {'type': 'Output', 'params': {'units': 1, 'activation': 'sigmoid'}}
    ),


])
def test_layer_parsing(layer_parser, transformer, layer_string, expected):
    tree = layer_parser.parse(layer_string)
    result = transformer.transform(tree)
    assert result == expected

def test_network_parsing(network_parser, transformer):
    network_string = """
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
    """
    tree = network_parser.parse(network_string)
    result = transformer.transform(tree)

    assert result['type'] == 'model'
    assert result['name'] == 'TestModel'
    assert result['input'] == {'type': 'Input', 'shape': (None, 28, 28, 1)}
    assert len(result['layers']) == 5

    # Check layers content in more detail
    expected_layers = [
        {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'}},
        {'type': 'MaxPooling2D', 'params': {'pool_size': (2, 2)}},
        {'type': 'Flatten', 'params': {}},
        {'type': 'Dense', 'params': {'units': 128, 'activation': 'relu'}},
        {'type': 'Output', 'params': {'units': 10, 'activation': 'softmax'}}
    ]
    for i, layer in enumerate(result['layers']):
        assert layer['type'] == expected_layers[i]['type']
        assert layer['params'] == expected_layers[i]['params']

    assert result['loss'] == {'type': 'Loss', 'value': 'categorical_crossentropy'}
    assert result['optimizer'] == {'type': 'Optimizer', 'value': 'adam'}
    assert result['training_config'] == {'type': 'TrainingConfig', 'epochs': 10, 'batch_size': 32}

def test_network_parsing_no_train_config(network_parser, transformer):
    network_string = """
    network SimpleModel {
        input: (28, 28, 1)
        layers:
            Flatten()
            Output(units=1, activation="sigmoid")
        loss: "binary_crossentropy"
        optimizer: "SGD"
    }
    """
    tree = network_parser.parse(network_string)
    result = transformer.transform(tree)

    assert result['type'] == 'model'
    assert result['name'] == 'SimpleModel'
    assert result['input'] == {'type': 'Input', 'shape': (28, 28, 1)}
    assert len(result['layers']) == 2
    assert result['training_config'] is None # Training config should be None

def test_research_parsing(research_parser, transformer):
    research_string = """
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
    """
    tree = research_parser.parse(research_string)
    result = transformer.transform(tree)

    assert result['type'] == 'Research'
    assert result['params']['metrics'] == {'accuracy': 0.95, 'loss': 0.05}
    assert result['params']['references'] == ['Paper Title 1', 'Another Great Paper']

def test_research_parsing_no_name(research_parser, transformer):
    research_string = """
    research {
        metrics {
            precision: 0.8
            recall: 0.9
        }
    }
    """
    tree = research_parser.parse(research_string)
    result = transformer.transform(tree)

    assert result['type'] == 'Research'
    assert result['params']['metrics'] == {'precision': 0.8, 'recall': 0.9}
    assert 'references' not in result['params'] # No references defined


def test_invalid_layer(layer_parser):
    with pytest.raises(exceptions.UnexpectedToken): # More specific exception
        layer_parser.parse("InvalidLayer()")

def test_invalid_network(network_parser):
    with pytest.raises(exceptions.UnexpectedToken): # More specific exception
        network_parser.parse("invalid network syntax { }") # Added braces to make it more like network def

def test_invalid_research(research_parser):
    with pytest.raises(exceptions.UnexpectedCharacters): # More specific exception
        research_parser.parse("research { invalid metrics }") # Invalid metrics block syntax

def test_invalid_layer(layer_parser):
    with pytest.raises(exceptions.UnexpectedCharacters):
        layer_parser.parse("InvalidLayer()")