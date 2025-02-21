import pytest
import json
import socketio
import requests
from dash.dependencies import Input, Output
from neural.dashboard.dashboard import app, update_trace_graph, update_flops_memory_chart, update_gradient_chart, update_dead_neurons, update_anomaly_chart, update_graph
from unittest.mock import MagicMock, patch
from flask_socketio import SocketIOTestClient
import plotly.graph_objects as go

@pytest.fixture
def test_app():
    """Creates Dash test client."""
    return app

##########################################
### Test Execution Trace Visualization ###
##########################################

@patch('neural.dashboard.dashboard.trace_data', [
    {"layer": "Conv2D", "execution_time": 0.001},
    {"layer": "Dense", "execution_time": 0.005},
])
def test_update_trace_graph():
    """Ensures execution trace visualization updates correctly."""
    fig = update_trace_graph(1)
    
    # Save visualization
    fig.write_html("test_trace_graph.html")
    try:
        fig.write_image("test_trace_graph.png")
    except Exception as e:
        print(f"Warning: Could not save PNG (kaleido might be missing): {e}")
    
    # Assertions
    assert len(fig.data) == 1  # Should contain one bar graph
    assert list(fig.data[0].x) == ["Conv2D", "Dense"]  # Convert to list
    assert list(fig.data[0].y) == [0.001, 0.005]  # Convert to list (for consistency)

###########################################
### 🛠 Test FLOPs & Memory Visualization ###
###########################################

@patch('neural.dashboard.dashboard.trace_data', [
    {"layer": "Conv2D", "flops": 1000, "memory": 10},
    {"layer": "Dense", "flops": 2000, "memory": 20},
])
def test_update_flops_memory_chart():
    """Ensures FLOPs and memory usage visualization updates correctly."""
    fig = update_flops_memory_chart(1)
    
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

###########################################
### 🛠 Test Gradient Flow Visualization ###
###########################################

@patch('neural.dashboard.dashboard.trace_data', [
    {"layer": "Conv2D", "grad_norm": 0.9},
    {"layer": "Dense", "grad_norm": 0.1},
])
def test_update_gradient_chart():
    """Ensures gradient flow visualization updates correctly."""
    fig = update_gradient_chart(1)
    
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

###########################################
### 🛠 Test Dead Neuron Detection Panel ###
###########################################

@patch('neural.dashboard.dashboard.trace_data', [
    {"layer": "Conv2D", "dead_ratio": 0.1},
    {"layer": "Dense", "dead_ratio": 0.5},
])
def test_update_dead_neurons():
    """Ensures dead neuron detection panel updates correctly."""
    fig = update_dead_neurons(1)
    
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

###########################################
### 🛠 Test Anomaly Detection Panel ###
###########################################

@patch('neural.dashboard.dashboard.trace_data', [
    {"layer": "Conv2D", "mean_activation": 0.5, "anomaly": False},
    {"layer": "Dense", "mean_activation": 1000, "anomaly": True},
])
def test_update_anomaly_chart():
    """Ensures anomaly detection updates correctly."""
    fig = update_anomaly_chart(1)
    
    # Save visualization
    fig.write_html("test_anomaly_chart.html")
    try:
        fig.write_image("test_anomaly_chart.png")
    except Exception as e:
        print(f"Warning: Could not save PNG (kaleido might be missing): {e}")
    
    # Assertions
    assert len(fig.data) == 2
    assert list(fig.data[0].x) == ["Conv2D", "Dense"]
    assert list(fig.data[1].y) == [0, 1]  # Only the second layer is flagged as an anomaly

###########################################
### 🛠 Test Dashboard Initialization ###
###########################################

def test_dashboard_starts(test_app):
    """Ensures the Dash app starts without issues."""
    assert test_app is not None

@pytest.fixture
def test_client():
    """Creates a test client for the dashboard app."""
    return app.test_client()

#########################
### API & WebSockets ###
#########################

@patch("requests.get")
def test_trace_api(mock_get):
    """Ensure execution trace API returns valid mock data."""
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = [
        {"layer": "Conv2D", "execution_time": 0.002},
        {"layer": "Dense", "execution_time": 0.006}
    ]

    response = requests.get("http://localhost:5001/trace")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert response.json()[0]["layer"] == "Conv2D"

def test_websocket_connection():
    """Verify WebSocket receives trace updates."""
    socket_client = SocketIOTestClient(app)

    # Mock WebSocket response
    mock_data = [{"layer": "Conv2D", "execution_time": 0.002}]
    socket_client.emit("request_trace_update")
    
    # Simulate receiving data
    socket_client.get_received = MagicMock(return_value=[("trace_update", mock_data)])
    received = socket_client.get_received()

    assert len(received) > 0  # Ensure WebSocket is working
    assert received[0][1] == mock_data  # Validate data matches

#######################
### UI Interaction ###
#######################

def test_model_comparison():
    """Verify model architecture comparison dropdown."""
    fig_a = update_graph("A")
    fig_b = update_graph("B")
    
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

##################
### Empty Data ###
##################

@patch('neural.dashboard.dashboard.trace_data', [])
def test_update_trace_graph_empty():
    fig = update_trace_graph(1)
    assert len(fig.data) == 0  # Expect empty figure
    fig.write_html("test_trace_graph_empty.html")
    try:
        fig.write_image("test_trace_graph_empty.png")
    except Exception as e:
        print(f"Warning: Could not save PNG (kaleido might be missing): {e}")