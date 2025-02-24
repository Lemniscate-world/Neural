import base64
import json
import pytest
from neural.shape_propagation.data_to_dashboard import server, app, update_graph, propagator
import plotly.graph_objects as go

# Fixture to provide a test client for the Flask server.
@pytest.fixture
def client():
    with server.test_client() as client:
        yield client

def get_auth_headers(username="admin", password="default"):
    """
    Returns a dictionary with the correct Authorization header for HTTP Basic Auth.
    """
    auth_str = f"{username}:{password}"
    encoded_auth = base64.b64encode(auth_str.encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {encoded_auth}"}

def test_trace_endpoint(client):
    """
    Tests that the /trace endpoint returns a JSON response containing a list
    of trace entries when provided with correct authentication.
    """
    headers = get_auth_headers()
    response = client.get("/trace", headers=headers)
    assert response.status_code == 200, "Expected 200 OK with valid credentials"
    data = response.get_json()
    assert isinstance(data, list), "Expected JSON response to be a list"
    
    # If trace data is present, verify that required keys are available.
    if data:
        required_keys = {
            "layer", "input_shape", "output_shape",
            "flops", "memory", "execution_time",
            "compute_time", "transfer_time"
        }
        for entry in data:
            assert required_keys.issubset(entry.keys()), "Missing keys in trace entry"

def test_trace_endpoint_auth_failure(client):
    """
    Tests that accessing the /trace endpoint without proper authentication fails.
    """
    # No Authorization header provided.
    response = client.get("/trace")
    assert response.status_code == 401, "Expected 401 Unauthorized without credentials"

def test_update_graph_callback():
    """
    Tests the update_graph callback function to ensure it returns a valid Plotly figure.
    """
    # Call the callback (simulating n_intervals = 0)
    figure = update_graph(0)
    # update_graph should return a plotly.graph_objects.Figure instance.
    assert isinstance(figure, go.Figure), "update_graph did not return a valid Figure object"
    
    # Check that the figure layout has the expected title.
    layout_title = figure.layout.title.text if figure.layout.title else ""
    assert "Shape Propagation" in layout_title, "Figure title does not mention 'Shape Propagation'"

def test_propagator_shape_history():
    """
    Tests that the propagator (instantiated on startup) has a non-empty shape history.
    """
    # data_to_dashboard.py already propagates a series of layers on startup.
    assert len(propagator.shape_history) > 0, "Expected non-empty shape history after propagation"
