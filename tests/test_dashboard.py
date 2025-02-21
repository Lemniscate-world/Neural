import pytest
from dash.dependencies import Input, Output
from dashboard import app, update_trace_graph, update_flops_memory_chart, update_gradient_chart, update_dead_neurons, update_anomaly_chart

@pytest.fixture
def test_app():
    """Creates Dash test client."""
    return app

###########################################
### 🛠 Test Execution Trace Visualization ###
###########################################

def test_update_trace_graph():
    """Ensures execution trace visualization updates correctly."""
    test_data = [
        {"layer": "Conv2D", "execution_time": 0.001},
        {"layer": "Dense", "execution_time": 0.005},
    ]
    fig = update_trace_graph(1)
    
    assert len(fig.data) == 1  # Should contain one bar graph
    assert fig.data[0].x == ["Conv2D", "Dense"]  # Layers must match
    assert fig.data[0].y == [0.001, 0.005]  # Execution times must match

###########################################
### 🛠 Test FLOPs & Memory Visualization ###
###########################################

def test_update_flops_memory_chart():
    """Ensures FLOPs and memory usage visualization updates correctly."""
    test_data = [
        {"layer": "Conv2D", "flops": 1000, "memory": 10},
        {"layer": "Dense", "flops": 2000, "memory": 20},
    ]
    fig = update_flops_memory_chart(1)
    
    assert len(fig.data) == 2  # Should contain two bar graphs (FLOPs & memory)
    assert fig.data[0].x == ["Conv2D", "Dense"]
    assert fig.data[1].x == ["Conv2D", "Dense"]

###########################################
### 🛠 Test Gradient Flow Visualization ###
###########################################

def test_update_gradient_chart():
    """Ensures gradient flow visualization updates correctly."""
    test_data = [
        {"layer": "Conv2D", "grad_norm": 0.9},
        {"layer": "Dense", "grad_norm": 0.1},
    ]
    fig = update_gradient_chart(1)
    
    assert len(fig.data) == 1
    assert fig.data[0].x == ["Conv2D", "Dense"]
    assert fig.data[0].y == [0.9, 0.1]

###########################################
### 🛠 Test Dead Neuron Detection Panel ###
###########################################

def test_update_dead_neurons():
    """Ensures dead neuron detection panel updates correctly."""
    test_data = [
        {"layer": "Conv2D", "dead_ratio": 0.1},
        {"layer": "Dense", "dead_ratio": 0.5},
    ]
    fig = update_dead_neurons(1)
    
    assert len(fig.data) == 1
    assert fig.data[0].x == ["Conv2D", "Dense"]
    assert fig.data[0].y == [0.1, 0.5]

###########################################
### 🛠 Test Anomaly Detection Panel ###
###########################################

def test_update_anomaly_chart():
    """Ensures anomaly detection updates correctly."""
    test_data = [
        {"layer": "Conv2D", "mean_activation": 0.5, "anomaly": False},
        {"layer": "Dense", "mean_activation": 1000, "anomaly": True},
    ]
    fig = update_anomaly_chart(1)
    
    assert len(fig.data) == 2
    assert fig.data[0].x == ["Conv2D", "Dense"]
    assert fig.data[1].y == [0, 1]  # Only the second layer is flagged as an anomaly

###########################################
### 🛠 Test Dashboard Initialization ###
###########################################

def test_dashboard_starts(test_app):
    """Ensures the Dash app starts without issues."""
    assert test_app is not None
