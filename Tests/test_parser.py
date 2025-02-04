import os
import sys
import pytest

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from parser import propagate_shape, parser, ModelTransformer, load_file

@pytest.mark.parametrize("code, expected", [
    # Basic Layers
    ("Dense(units=128, activation='relu')", {'type': 'Dense', 'units': 128, 'activation': 'relu'}),
    ("Conv2D(filters=64, kernel_size=(3,3), activation='sigmoid')",
        {'type': 'Conv2D', 'filters': 64, 'kernel_size': (3,3), 'activation': 'sigmoid'}),
    ("MaxPooling2D(pool_size=(2,2))", {'type': 'MaxPooling2D', 'pool_size': (2,2)}),
    ("Dropout(rate=0.2)", {'type': 'Dropout', 'rate': 0.2}),

    # Normalization
    ("BatchNormalization()", {'type': 'BatchNormalization'}),
    ("LayerNormalization()", {'type': 'LayerNormalization'}),
    ("InstanceNormalization()", {'type': 'InstanceNormalization'}),
    ("GroupNormalization(groups=4)", {'type': 'GroupNormalization', 'groups': 4}),

    # Recurrent
    ("LSTM(units=256)", {'type': 'LSTM', 'units': 256}),
    ("GRU(units=128)", {'type': 'GRU', 'units': 128}),

    # Attention
    ("Attention()", {'type': 'Attention'}),
    ("TransformerEncoder(num_heads=8, ff_dim=512)", {'type': 'TransformerEncoder', 'num_heads': 8, 'ff_dim': 512}),

    # Advanced
    ("ResidualConnection()", {'type': 'ResidualConnection'}),
    ("InceptionModule()", {'type': 'InceptionModule'}),
    ("CapsuleLayer()", {'type': 'CapsuleLayer'}),
    ("GraphConv()", {'type': 'GraphConv'}),
    ("QuantumLayer()", {'type': 'QuantumLayer'}),
    ("DynamicLayer()", {'type': 'DynamicLayer'}),
])
def test_layer_parsing(code, expected):
    tree = parser.parse(f'network TestModel {{ input: (28,28,1) layers: {code} loss: "mse" optimizer: "adam" }}')
    model_data = transformer.transform(tree)
    # The first layer should match expected
    assert model_data['layers'][0] == expected
@pytest.mark.parametrize("filename, expected_type", [
    ("deepseek.neural", "model"),
    ("deepseek.nr", "model"),
    ("deepseek_research.rnr", "research"),
])
def test_file_parsing(filename, expected_type):
    file_type, content = load_file(filename)
    assert file_type == expected_type

    tree = parser.parse(content)
    transformer = ModelTransformer()
    result = transformer.transform(tree)

    assert result["type"] == expected_type
    print(f"Successfully parsed {filename} as {expected_type}!")