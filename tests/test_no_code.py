# tests/test_no_code.py
import pytest
from unittest.mock import Mock, patch
from dash.testing.application_runners import import_app
from neural.no_code.no_code import app
from neural.code_generation.code_generator import generate_code
from neural.shape_propagation.shape_propagator import ShapePropagator

@pytest.fixture
def dash_duo():
    dash_test = DashTest(server=app)
    yield dash_test
    dash_test.finalize()

@patch('neural.no_code.generate_code')
@patch('neural.no_code.ShapePropagator')
def test_no_code_interface(mock_propagator, mock_generate_code, dash_duo):
    """Test basic no-code interface functionality."""
    # Mock ShapePropagator
    mock_instance = Mock()
    mock_propagator.return_value = mock_instance
    mock_instance.propagate.return_value = (1, 14, 14, 32)  # Mock Conv2D output shape
    mock_instance.get_trace.return_value = [
        {"layer": "Conv2D", "execution_time": 0.001, "compute_time": 0.0007, "transfer_time": 0.0003}
    ]

    # Mock generate_code
    mock_generate_code.return_value = "Mock TensorFlow code"

    # Start the Dash server
    dash_duo.start_server()

    # Simulate selecting a layer type (Conv2D)
    dash_duo.select_dcc_dropdown("#layer-type", "Conv2D")
    dash_duo.click("#add-layer")

    # Check if parameters are populated
    params = dash_duo.find_element("#layer-params").text
    assert "filters" in params and "32" in params, "Conv2D parameters not displayed"

    # Check architecture preview
    graph = dash_duo.find_element("#architecture-preview")
    assert graph, "Architecture preview graph not rendered"

    # Simulate compile button click
    dash_duo.click("#compile-btn")
    output = dash_duo.find_element("#output").text
    assert "Mock TensorFlow code" in output, "Compile output not displayed"

def test_invalid_layer_type(dash_duo):
    """Test handling of invalid layer type selection."""
    dash_duo.start_server()
    dash_duo.select_dcc_dropdown("#layer-type", None)  # No selection
    dash_duo.click("#add-layer")
    params = dash_duo.find_element("#layer-params").text
    assert not params, "Parameters should be empty for invalid layer type"

@pytest.mark.parametrize("layer_type, expected_params", [
    ("Conv2D", {"filters": 32, "kernel_size": "(3, 3)", "activation": "relu"}),
    ("Dense", {"units": 128, "activation": "relu"}),
    ("Dropout", {"rate": "0.5"})
])
def test_layer_param_population(layer_type, expected_params, dash_duo):
    """Test parameter population for different layer types."""
    dash_duo.start_server()
    dash_duo.select_dcc_dropdown("#layer-type", layer_type)
    dash_duo.click("#add-layer")
    params = dash_duo.find_element("#layer-params").text
    for key, value in expected_params.items():
        assert f"{key}: {value}" in params, f"{key} not found in parameters for {layer_type}"