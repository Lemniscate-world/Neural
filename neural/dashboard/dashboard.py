import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from flask import Flask
from numpy import random
import json
import requests
from flask_socketio import SocketIO
import threading

from neural.shape_propagation.shape_propagator import ShapePropagator
from neural.dashboard.tensor_flow import create_animated_network



# Flask app for WebSocket integration (if needed later)
server = Flask(__name__)

# Dash app
app = dash.Dash(__name__, server=server)

# Initialize WebSocket Connection
socketio = SocketIO(cors_allowed_origins="*")

##############################################################
### Layout: Real-time execution trace, FLOPs, Memory Usage ###
##############################################################
app.layout = html.Div([
    html.H1("NeuralDbg: Real-Time Execution Monitoring"),
    
    # Execution Trace Visualization
    dcc.Graph(id="trace_graph"),
    
    # FLOPs & Memory Usage Bar Chart
    dcc.Graph(id="flops_memory_chart"),
    
    # Live Update Interval
    dcc.Interval(id="interval_component", interval=1000, n_intervals=0)
])

# Store Execution Trace Data
trace_data = []

# WebSocket Listener for Live Updates
@socketio.on("trace_update")
def update_trace_data(data):
    global trace_data
    trace_data = json.loads(data)

# Fetch Initial Data from API
def fetch_trace_data():
    global trace_data
    response = requests.get("http://localhost:5001/trace")  # Ensure `data_to_dashboard.py` is running
    if response.status_code == 200:
        return go.Figure()

# Start WebSocket in a Separate Thread
threading.Thread(target=socketio.run, args=("localhost", 5001), daemon=True).start()

@app.callback(
    Output("trace_graph", "figure"),
    Input("interval_component", "n_intervals")
)

def update_trace_graph(n):
    """Update execution trace graph with new data."""
    if not trace_data:
        return go.Figure()

    # Extract Layer Data
    layers = [entry["layer"] for entry in trace_data]
    execution_time = [entry["execution_time"] for entry in trace_data]

    # Generate Bar Graph (Execution Time per Layer)
    fig = go.Figure([go.Bar(x=layers, y=execution_time, name="Execution Time (s)")])
    fig.update_layout(title="Layer Execution Time", xaxis_title="Layers", yaxis_title="Time (s)")
    
    # Stacked Bar Chart
    fig = go.Figure([
    go.Bar(x=layers, y=[entry["compute_time"] for entry in trace_data], name="Compute Time"),
    go.Bar(x=layers, y=[entry["transfer_time"] for entry in trace_data], name="Data Transfer"),])
    fig.update_layout(barmode="stack", title="Layer Execution Time Breakdown", xaxis_title="Layers", yaxis_title="Time (s)")

    return fig

@app.callback(
    Output("flops_memory_chart", "figure"),
    Input("interval_component", "n_intervals")
)
def update_flops_memory_chart(n):
    """Update FLOPs and memory usage visualization."""
    if not trace_data:
        return go.Figure()

    layers = [entry["layer"] for entry in trace_data]
    flops = [entry["flops"] for entry in trace_data]
    memory = [entry["memory"] for entry in trace_data]

    # Create Dual Bar Graph (FLOPs & Memory)
    fig = go.Figure([
        go.Bar(x=layers, y=flops, name="FLOPs"),
        go.Bar(x=layers, y=memory, name="Memory Usage (MB)")
    ])
    fig.update_layout(title="FLOPs & Memory Usage", xaxis_title="Layers", yaxis_title="Values", barmode="group")
    
    return fig

##########################################################
### Layout: Real-time Shape Propagation & Loss/metrics ###
##########################################################
app.layout = html.Div([
    html.H1("Neural Shape Propagation Dashboard"),
    
    # Shape propagation graph
    dcc.Graph(id="shape_graph"),

    # Live-updating loss/metrics
    dcc.Interval(id="interval_component", interval=1000, n_intervals=0)  # Updates every 1s
])

app.layout = html.Div([
    html.H1("Training Metrics"),
    dcc.Graph(id="loss_graph"),
    dcc.Graph(id="accuracy_graph"),
    dcc.Interval(id="interval_component", interval=1000, n_intervals=0)  # Update every second
])

@app.callback(
    Output("loss_graph", "figure"),
    Input("interval_component", "n_intervals")
)

def update_loss(n):
    loss_data = [random.uniform(0.1, 1.0) for _ in range(n)]  # Simulated loss data
    fig = go.Figure(data=[go.Scatter(y=loss_data, mode="lines+markers")])
    fig.update_layout(title="Loss Over Time")
    return fig

app.layout = html.Div([
    html.H1("Compare Architectures"),
    dcc.Dropdown(id="architecture_selector", options=[
        {"label": "Model A", "value": "A"},
        {"label": "Model B", "value": "B"},
    ], value="A"),
    dcc.Graph(id="architecture_graph"),
])

@app.callback(
    Output("architecture_graph", "figure"),
    Input("architecture_selector", "value")
)
def update_graph(selected_model):
    if selected_model == "A":
        layers = [{"type": "Conv2D", "params": {"filters": 32}}, {"type": "Dense", "params": {"units": 128}}]
    else:
        layers = [{"type": "Dense", "params": {"units": 256}}]

    propagator = ShapePropagator()
    for layer in layers:
        input_shape = propagator.propagate(input_shape, layer, framework='tensorflow')
    return create_animated_network(input_shape)


###########################
### Gradient Flow Panel ###
###########################
@app.callback(
    Output("gradient_flow_chart", "figure"),
    Input("interval_component", "n_intervals")
)
def update_gradient_chart(n):
    """Visualizes gradient flow per layer."""
    response = requests.get("http://localhost:5001/trace")
    trace_data = response.json()
    
    layers = [entry["layer"] for entry in trace_data]
    grad_norms = [entry.get("grad_norm", 0) for entry in trace_data]

    fig = go.Figure([go.Bar(x=layers, y=grad_norms, name="Gradient Magnitude")])
    fig.update_layout(title="Gradient Flow", xaxis_title="Layers", yaxis_title="Gradient Magnitude")
    
    return fig

#########################
### Dead Neuron Panel ###
#########################
@app.callback(
    Output("dead_neuron_chart", "figure"),
    Input("interval_component", "n_intervals")
)
def update_dead_neurons(n):
    """Displays percentage of dead neurons per layer."""
    response = requests.get("http://localhost:5001/trace")
    trace_data = response.json()

    layers = [entry["layer"] for entry in trace_data]
    dead_ratios = [entry.get("dead_ratio", 0) for entry in trace_data]

    fig = go.Figure([go.Bar(x=layers, y=dead_ratios, name="Dead Neurons (%)")])
    fig.update_layout(title="Dead Neuron Detection", xaxis_title="Layers", yaxis_title="Dead Ratio", yaxis_range=[0, 1])
    
    return fig

##############################
### Anomaly Detection Panel###
##############################
@app.callback(
    Output("anomaly_chart", "figure"),
    Input("interval_component", "n_intervals")
)
def update_anomaly_chart(n):
    """Visualizes unusual activations per layer."""
    response = requests.get("http://localhost:5001/trace")
    trace_data = response.json()

    layers = [entry["layer"] for entry in trace_data]
    activations = [entry.get("mean_activation", 0) for entry in trace_data]
    anomalies = [1 if entry.get("anomaly", False) else 0 for entry in trace_data]

    fig = go.Figure([
        go.Bar(x=layers, y=activations, name="Mean Activation"),
        go.Bar(x=layers, y=anomalies, name="Anomaly Detected", marker_color="red")
    ])
    fig.update_layout(title="Activation Anomalies", xaxis_title="Layers", yaxis_title="Activation Magnitude")
    
    return fig

###########################
### Step Debugger Button###
###########################
@app.callback(
    Output("step_debug_output", "children"),
    Input("step_debug_button", "n_clicks")
)
def trigger_step_debug(n):
    """Manually pauses execution at a layer."""
    if n:
        requests.get("http://localhost:5001/trigger_step_debug")
        return "Paused. Check terminal for tensor inspection."
    return "Click to pause execution."


if __name__ == "__main__":
    app.run_server(debug=True)