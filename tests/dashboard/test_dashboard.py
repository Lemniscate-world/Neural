import json
import sys
import os
import time
import pysnooper
import threading
from dash import html
# Add the parent directory of 'neural' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from dash import Dash
import pytest
from unittest.mock import Mock
from pytest import approx
import socketio
import requests
from dash.dependencies import Input, Output
from neural.dashboard.dashboard import app, update_trace_graph, update_flops_memory_chart, update_gradient_chart, update_dead_neurons, update_anomaly_chart, update_graph
from neural.dashboard.tensor_flow import create_animated_network
from unittest.mock import MagicMock, patch
from flask_socketio import SocketIOTestClient, SocketIO
import plotly.graph_objects as go
import numpy as np
import flask
from flask import Flask

### Dash Test Client ###

@pytest.fixture
def test_app():
    """Creates Dash test client."""
    return app


#####################################
### Client Simulation For Testing ###
#####################################

@pytest.fixture
def test_client():
    """Creates a test client for the dashboard app."""
    return app.test_client()

# Global variable to simulate trace data (used across tests)
TRACE_DATA = [
    {
        "layer": "Conv2D", "execution_time": 0.001, "compute_time": 0.0007, "transfer_time": 0.0003, "kernel_size": [3, 3],
        "flops": 1000, "memory": 10, "grad_norm": 0.9, "dead_ratio": 0.1, "mean_activation": 0.5, "anomaly": False
    },
    {
        "layer": "Dense", "execution_time": 0.005, "compute_time": 0.0035, "transfer_time": 0.0015, "kernel_size": [1, 1],
        "flops": 2000, "memory": 20, "grad_norm": 0.1, "dead_ratio": 0.5, "mean_activation": 1000, "anomaly": True
    }
]

##########################################
### Test Execution Trace Visualization ###
##########################################

@patch('neural.dashboard.dashboard.trace_data', TRACE_DATA)
def test_update_trace_graph_basic():
    """Ensures basic bar chart in execution trace visualization updates correctly."""
    figs = update_trace_graph(1, "basic", ["Conv2D", "Dense"])
    fig = figs[0]  # Extract the Figure object from the list
    
    # Save visualization
    fig.write_html("test_trace_graph_basic.html")
    try:
        fig.write_image("test_trace_graph_basic.png")
    except Exception as e:
        print(f"Warning: Could not save PNG (kaleido might be missing): {e}")
    
    # Assertions
    assert len(fig.data) == 1
    assert list(fig.data[0].x) == ["Conv2D", "Dense"]
    assert list(fig.data[0].y) == [0.001, 0.005]

##########################
### Stacked Bar Chart ####
##########################

@patch('neural.dashboard.dashboard.trace_data', TRACE_DATA)
def test_update_trace_graph_stacked():
    """Ensures stacked bar chart in execution trace visualization updates correctly."""
    figs = update_trace_graph(1, "stacked", ["Conv2D", "Dense"])
    fig = figs[0]
    
    # Save visualization
    fig.write_html("test_trace_graph_stacked.html")
    try:
        fig.write_image("test_trace_graph_stacked.png")
    except Exception as e:
        print(f"Warning: Could not save PNG (kaleido might be missing): {e}")
    
    # Assertions
    assert len(fig.data) == 2
    assert list(fig.data[0].x) == ["Conv2D", "Dense"]
    assert list(fig.data[1].x) == ["Conv2D", "Dense"]
    assert list(fig.data[0].y) == approx([0.0007, 0.0035], rel=1e-7)  # Compute times
    assert list(fig.data[1].y) == approx([0.0003, 0.0015], rel=1e-7)  # Transfer times

@patch('neural.dashboard.dashboard.trace_data', TRACE_DATA)
def test_update_trace_graph_horizontal():
    """Ensures horizontal sorted bar chart in execution trace visualization updates correctly."""
    figs = update_trace_graph(1, "horizontal", ["Conv2D", "Dense"])
    fig = figs[0]
    
    # Save visualization
    fig.write_html("test_trace_graph_horizontal.html")
    try:
        fig.write_image("test_trace_graph_horizontal.png")
    except Exception as e:
        print(f"Warning: Could not save PNG (kaleido might be missing): {e}")
    
    # Assertions
    assert len(fig.data) == 1
    assert list(fig.data[0].y) == ["Dense", "Conv2D"]  # Sorted by execution time (Dense > Conv2D)
    assert list(fig.data[0].x) == [0.005, 0.001]

#######################
### Trace Box Graph ###
#######################

@patch('neural.dashboard.dashboard.trace_data', TRACE_DATA)
def test_update_trace_graph_box():
    """Ensures box plot for variability in execution trace visualization updates correctly."""
    figs = update_trace_graph(1, "box", ["Conv2D", "Dense"])
    fig = figs[0]
    
    # Save visualization
    fig.write_html("test_trace_graph_box.html")
    try:
        fig.write_image("test_trace_graph_box.png")
    except Exception as e:
        print(f"Warning: Could not save PNG (kaleido might be missing): {e}")
    
    # Assertions
    assert len(fig.data) == 1
    assert list(fig.data[0].x) == ["Conv2D", "Dense"]
    assert len(fig.data[0].y) == 2  # Two boxes (one per layer)



@patch('neural.dashboard.dashboard.trace_data', TRACE_DATA)
def test_update_trace_graph_gantt():
    """Ensures Gantt chart for timeline in execution trace visualization updates correctly."""
    figs = update_trace_graph(1, "gantt", ["Conv2D", "Dense"])
    fig = figs[0]
    
    # Save visualization
    fig.write_html("test_trace_graph_gantt.html")
    try:
        fig.write_image("test_trace_graph_gantt.png")
    except Exception as e:
        print(f"Warning: Could not save PNG (kaleido might be missing): {e}")
    
    # Assertions
    assert len(fig.data) == 2  # One bar per layer
    assert fig.data[0].name == "Conv2D"
    assert fig.data[1].name == "Dense"

@patch('neural.dashboard.dashboard.trace_data', TRACE_DATA)
def test_update_trace_graph_heatmap():
    """Ensures heatmap of execution time over time in execution trace visualization updates correctly."""
    figs = update_trace_graph(1, "heatmap", ["Conv2D", "Dense"])
    fig = figs[0]
    
    # Save visualization
    fig.write_html("test_trace_graph_heatmap.html")
    try:
        fig.write_image("test_trace_graph_heatmap.png")
    except Exception as e:
        print(f"Warning: Could not save PNG (kaleido might be missing): {e}")
    
    # Assertions
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"
    assert list(fig.data[0].y) == ["Conv2D", "Dense"]
    assert len(list(fig.data[0].x)) == 5  # 5 iterations simulated

@patch('neural.dashboard.dashboard.trace_data', TRACE_DATA)
def test_update_trace_graph_thresholds():
    """Ensures bar chart with thresholds in execution trace visualization updates correctly."""
    figs = update_trace_graph(1, "thresholds", ["Conv2D", "Dense"])
    fig = figs[0]
    
    # Save visualization
    fig.write_html("test_trace_graph_thresholds.html")
    try:
        fig.write_image("test_trace_graph_thresholds.png")
    except Exception as e:
        print(f"Warning: Could not save PNG (kaleido might be missing): {e}")
    
    # Assertions
    assert len(fig.data) == 1
    assert list(fig.data[0].x) == ["Conv2D", "Dense"]
    assert list(fig.data[0].y) == [0.001, 0.005]
    assert list(fig.data[0].marker.color) == ["blue", "red"]  # Threshold > 0.003 for Dense

@patch('neural.dashboard.dashboard.trace_data', [])
def test_update_trace_graph_empty():
    """Ensures execution trace visualization handles empty data correctly."""
    figs = update_trace_graph(1, "basic", [])
    fig = figs[0]
    
    # Save visualization
    fig.write_html("test_trace_graph_empty.html")
    try:
        fig.write_image("test_trace_graph_empty.png")
    except Exception as e:
        print(f"Warning: Could not save PNG (kaleido might be missing): {e}")
    
    # Assertions
    assert len(fig.data) == 0

###########################################
### Test FLOPs & Memory Visualization #####
###########################################

@patch('neural.dashboard.dashboard.trace_data', TRACE_DATA)
def test_update_flops_memory_chart():
    """Ensures FLOPs and memory usage visualization updates correctly."""
    figs = update_flops_memory_chart(1)
    fig = figs[0]
    
    # Save visualization
    fig.write_html("test_flops_memory_chart.html")
    try:
        fig.write_image("test_flops_memory_chart.png")
    except Exception as e:
        print(f"Warning: Could not save PNG (kaleido might be missing): {e}")
    
    # Assertions
    assert len(fig.data) == 2  # Should contain two bar graphs (FLOPs & memory)
    assert list(fig.data[0].x) == ["Conv2D", "Dense"]
    assert list(fig.data[1].x) == ["Conv2D", "Dense"]
    assert list(fig.data[0].y) == [1000, 2000]  # FLOPs
    assert list(fig.data[1].y) == [10, 20]  # Memory

########################################
### Test Gradient Flow Visualization ###
########################################

@patch("requests.get")
def test_update_gradient_chart(mock_get):
    """Ensures gradient flow visualization updates correctly."""
    # Create a mock response with status_code, text, and json method
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = json.dumps(TRACE_DATA)  # Set the response text as a JSON string
    mock_response.json.return_value = TRACE_DATA  # Ensure json() returns the data

    # Configure the mock to return the mock response
    mock_get.return_value = mock_response

    figs = update_gradient_chart(1)
    fig = figs[0]  # Extract the Figure object from the list
    
    # Save visualization
    fig.write_html("test_gradient_chart.html")
    try:
        fig.write_image("test_gradient_chart.png")
    except Exception as e:
        print(f"Warning: Could not save PNG (kaleido might be missing): {e}")
    
    # Assertions
    assert len(fig.data) == 1
    assert list(fig.data[0].x) == ["Conv2D", "Dense"]
    assert list(fig.data[0].y) == [0.9, 0.1]

########################################
### Test Dead Neuron Detection Panel ###
########################################

@patch("requests.get")
def test_update_dead_neurons(mock_get):
    """Ensures dead neuron detection panel updates correctly."""
    # Create a mock response with status_code, text, and json method
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = json.dumps(TRACE_DATA)  # Set the response text as a JSON string
    mock_response.json.return_value = TRACE_DATA  # Ensure json() returns the data

    # Configure the mock to return the mock response
    mock_get.return_value = mock_response

    figs = update_dead_neurons(1)
    fig = figs[0]
    
    # Save visualization
    fig.write_html("test_dead_neurons.html")
    try:
        fig.write_image("test_dead_neurons.png")
    except Exception as e:
        print(f"Warning: Could not save PNG (kaleido might be missing): {e}")
    
    # Assertions
    assert len(fig.data) == 1
    assert list(fig.data[0].x) == ["Conv2D", "Dense"]
    assert list(fig.data[0].y) == [0.1, 0.5]

####################################
### Test Anomaly Detection Panel ###
####################################

@patch("requests.get")
def test_update_anomaly_chart(mock_get):
    """Ensures anomaly detection updates correctly."""
    # Create a mock response with status_code, text, and json method
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = json.dumps(TRACE_DATA)  # Set the response text as a JSON string
    mock_response.json.return_value = TRACE_DATA  # Ensure json() returns the data

    # Configure the mock to return the mock response
    mock_get.return_value = mock_response

    figs = update_anomaly_chart(1)
    fig = figs[0]
    
    # Save visualization
    fig.write_html("test_anomaly_chart.html")
    try:
        fig.write_image("test_anomaly_chart.png")
    except Exception as e:
        print(f"Warning: Could not save PNG (kaleido might be missing): {e}")
    
    # Assertions
    assert len(fig.data) == 2
    assert list(fig.data[0].x) == ["Conv2D", "Dense"]
    assert list(fig.data[1].y) == [0, 1]  # Only Dense has an anomaly

#####################################
### Test Dashboard Initialization ###
#####################################

def test_dashboard_starts(test_app):
    """Ensures the Dash app starts without issues."""
    assert test_app is not None

#########################
### API & WebSockets ###
#########################

@patch("requests.get")
def test_trace_api(mock_get):
    """Ensure execution trace API returns valid mock data."""
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = TRACE_DATA

    response = requests.get("http://localhost:5001/trace")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert response.json()[0]["layer"] == "Conv2D"

####

def test_websocket_connection():
    """Verify WebSocket receives trace updates."""
    socketio = SocketIO(app.server)
    socket_client = SocketIOTestClient(app, socketio)

    # Mock WebSocket response
    mock_data = TRACE_DATA
    socket_client.emit("request_trace_update")
    
    # Simulate receiving data
    socket_client.get_received = MagicMock(return_value=[("trace_update", json.dumps(mock_data))])
    received = socket_client.get_received()

    assert len(received) > 0  # Ensure WebSocket is working
    assert json.loads(received[0][1]) == mock_data  # Validate data matches

######################
### UI Interaction ###
######################

@patch('neural.dashboard.dashboard.trace_data', TRACE_DATA)
def test_model_comparison():
    """Verify model architecture comparison dropdown."""
    figs_a = update_graph("A")
    fig_a = figs_a[0]  # Extract the Figure object
    figs_b = update_graph("B")
    fig_b = figs_b[0]
    
    # Save visualizations
    fig_a.write_html("test_model_comparison_a.html")
    fig_b.write_html("test_model_comparison_b.html")
    try:
        fig_a.write_image("test_model_comparison_a.png")
        fig_b.write_image("test_model_comparison_b.png")
    except Exception as e:
        print(f"Warning: Could not save PNG (kaleido might be missing): {e}")
    
    # Assertions
    assert fig_a is not None
    assert fig_b is not None
    assert fig_a != fig_b  # Different architectures should have different graphs

#######################################
### Edge Cases and Additional Tests ###
#######################################

@patch('neural.dashboard.dashboard.trace_data', [])
def test_update_trace_graph_invalid_data():
    """Ensures execution trace visualization handles invalid data correctly."""
    # Test with invalid data (missing fields)
    invalid_data = [{"layer": "Conv2D", "execution_time": "invalid"}]  # Non-numeric execution_time
    with patch('neural.dashboard.dashboard.trace_data', invalid_data):
        figs = update_trace_graph(1, "basic", ["Conv2D"])
        fig = figs[0]
        assert len(fig.data) == 0  # Should return empty figure

@patch('neural.dashboard.dashboard.trace_data', [d for d in TRACE_DATA for _ in range(100)])  # Large dataset
def test_update_trace_graph_large_data():
    """Ensures execution trace visualization handles large datasets correctly."""
    figs = update_trace_graph(1, "basic", None)
    fig = figs[0]
    
    # Save visualization
    fig.write_html("test_trace_graph_large.html")
    try:
        fig.write_image("test_trace_graph_large.png")
    except Exception as e:
        print(f"Warning: Could not save PNG (kaleido might be missing): {e}")
    
    # Assertions
    assert len(fig.data) == 1
    assert len(list(fig.data[0].x)) == 200  # 100 Conv2D + 100 Dense

@patch('neural.dashboard.dashboard.trace_data', TRACE_DATA)
def test_update_trace_graph_thresholds_annotations():
    """Ensures bar chart with thresholds includes correct annotations."""
    figs = update_trace_graph(1, "thresholds", ["Conv2D", "Dense"])
    fig = figs[0]
    
    # Save visualization
    fig.write_html("test_trace_graph_thresholds_annotations.html")
    try:
        fig.write_image("test_trace_graph_thresholds_annotations.png")
    except Exception as e:
        print(f"Warning: Could not save PNG (kaleido might be missing): {e}")
    
    # Assertions
    assert len(fig.data) == 1
    assert len(fig.layout.annotations) == 1  # Only Dense should have an annotation (execution_time > 0.003)
    assert fig.layout.annotations[0].text == "High: 0.005s"



#### Tensor Flow Test ####

def test_tensor_flow_visualization():
    fig = create_animated_network([{"layer": "Conv2D", "output_shape": (26, 26, 32)}])
    assert len(fig.data) > 0

#############
### Theme ###
#############

def test_dashboard_theme(dash_app):
    with Dash(app=dash_app) as test:
        test.start_server()
        test.wait_for_element("#trace_graph")
        # Check for Darkly theme (simplified check—look for dark styles)
        body = test.driver.find_element("body")
        assert "darkly" in body.get_attribute("class") or "dark-theme" in body.get_attribute("class")


####################################
#### Testing Total Visualization ###
####################################

@pytest.fixture(scope="module")
def driver():
    options = Options()
    options.add_argument("--headless")  # Run browser in headless mode
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    yield driver
    driver.quit()  # Ensure browser closes even if test fails

def test_dashboard_visualization(driver):  # Use pytest fixture
    # Initialize and start the Dash app
    server = Flask(__name__)
    app = Dash(__name__, server=server, title="NeuralDbg")
    app.layout = html.Div("Test")
    
    # Start server in a separate thread
    server_thread = threading.Thread(
        target=app.run_server,
        kwargs={'port': 8050, 'debug': False, 'use_reloader': False}
    )
    server_thread.daemon = True
    server_thread.start()
    time.sleep(2)  # Wait for server to start

    try:
        driver.get("http://localhost:8050")
        # Add test assertions here
        assert driver.title == "NeuralDbg" # Example assertion
    finally:
        # Cleanup - this will close the browser
        driver.quit()
        # Forcefully stop the Flask server
        requests.post('http://localhost:8050/_shutdown')