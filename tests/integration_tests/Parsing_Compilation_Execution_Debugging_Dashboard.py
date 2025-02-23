import pytest
import json
from parser import create_parser, ModelTransformer
from dashboard import app
from tensor_flow import create_animated_network

# Sample Neural Network Definition
SAMPLE_NETWORK = """
network MyModel {
    input: (28, 28, 1)
    layers:
        Conv2D(filters=32, kernel_size=(3,3), activation='relu')
        MaxPooling2D(pool_size=(2,2))
        Flatten()
        Dense(units=128, activation='relu')
        Output(units=10)
    loss: "categorical_crossentropy"
    optimizer: "adam"
    train {
        epochs: 10
        batch_size: 32
    }
    execution {
        device: "cpu"
    }
}
"""

@pytest.fixture
def parser():
    return create_parser("network")

@pytest.fixture
def transformer():
    return ModelTransformer()

@pytest.fixture
def dash_client():
    return app.test_client()

# --- PARSING TESTS ---
def test_parsing(parser):
    """Test if neural network definition is parsed correctly."""
    tree = parser.parse(SAMPLE_NETWORK)
    assert tree is not None

def test_model_transformation(parser, transformer):
    """Test if parsed tree transforms correctly into a model dictionary."""
    tree = parser.parse(SAMPLE_NETWORK)
    model = transformer.transform(tree)
    assert isinstance(model, dict)
    assert "layers" in model
    assert model["optimizer"] == "adam"

# --- EXECUTION TESTS ---
def test_dashboard_execution(dash_client):
    """Test if dashboard updates execution trace correctly."""
    response = dash_client.get("/trace")
    assert response.status_code == 200
    trace_data = json.loads(response.data)
    assert isinstance(trace_data, list)

# --- DEBUGGING TESTS ---
def test_tensor_flow_viz():
    """Ensure the tensor flow visualization works."""
    layer_data = [
        {"layer": "Conv2D", "output_shape": "(26, 26, 32)"},
        {"layer": "MaxPooling2D", "output_shape": "(13, 13, 32)"},
        {"layer": "Flatten", "output_shape": "(5408,)"},
        {"layer": "Dense", "output_shape": "(128,)"},
        {"layer": "Output", "output_shape": "(10,)"}
    ]
    fig = create_animated_network(layer_data)
    assert fig is not None
