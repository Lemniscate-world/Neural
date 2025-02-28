import dash
from dash import Dash, dcc, html
import sys
import os
import numpy as np
import pysnooper
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from flask import Flask
from numpy import random
import json
import requests
import time
from flask_socketio import SocketIO
import threading
from dash_bootstrap_components import themes

# Add the parent directory of 'neural' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from neural.shape_propagation.shape_propagator import ShapePropagator
from neural.dashboard.tensor_flow import create_animated_network



# Flask app for WebSocket integration (if needed later)
server = Flask(__name__)

# Dash app
app = dash.Dash(
    __name__,
    server=server,
    title="NeuralDbg: Real-Time Execution Monitoring",
    external_stylesheets=[themes.DARKLY]
)

# Initialize WebSocket Connection
socketio = SocketIO(server, cors_allowed_origins=["http://localhost:8050"])

# Configuration (load from config.yaml or set defaults)
try:
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except Exception as e:
    config = {}

UPDATE_INTERVAL = config.get("websocket_interval", 1000) # Use default if config file not found


# Store Execution Trace Data
trace_data = []

########################################################################
#### WebSocket Listener (Sending trace & Intervals) for Live Updates ###
########################################################################

@socketio.on("request_trace_update")
def send_trace_update():
    global trace_data
    while True:
        trace_data = propagator.get_trace()
        # Double-check kernel_size is a tuple
        trace_data = [
            {k: tuple(v) if k == "kernel_size" and isinstance(v, list) else v for k, v in entry.items()}
            for entry in trace_data
        ]
        print(f"DEBUG: Emitting trace_data with kernel_size: {trace_data[0]['kernel_size']}, type: {type(trace_data[0]['kernel_size'])}")
        socketio.emit("trace_update", json.dumps(trace_data))
        time.sleep(UPDATE_INTERVAL / 1000)

### Interval Updates ####
@app.callback(
    [Output("interval_component", "interval")],
    [Input("update_interval", "value")]
)

def update_interval(new_interval):
    """Update the interval dynamically based on slider value."""
    return [new_interval]

# Start WebSocket in a Separate Thread
propagator = ShapePropagator()
threading.Thread(target=socketio.run, args=("localhost", 5001), daemon=True).start()

####################################################
#### Layers Execution Trace Graph & Its Subplots ###
####################################################

@app.callback(
    [Output("trace_graph", "figure")],
    [Input("interval_component", "n_intervals"), Input("viz_type", "value"), Input("layer_filter", "value")]
)
def update_trace_graph(n, viz_type, selected_layers=None):
    """Update execution trace graph with various visualization types."""
    global trace_data
    

    ### ***Errors Handling*** ###
    if not trace_data or any(not isinstance(entry["execution_time"], (int, float)) for entry in trace_data):
        return [go.Figure()]  # Return empty figure for invalid data

    # Filter data based on selected layers (if any)
    if selected_layers:
        filtered_data = [entry for entry in trace_data if entry["layer"] in selected_layers]
    else:
        filtered_data = trace_data

    if not filtered_data:
        return [go.Figure()]

    layers = [entry["layer"] for entry in filtered_data]
    execution_times = [entry["execution_time"] for entry in filtered_data]

    # Simulate compute_time and transfer_time for stacked bar (you can extend ShapePropagator to include these)
    compute_times = [t * 0.7 for t in execution_times]  # 70% of execution time for compute
    transfer_times = [t * 0.3 for t in execution_times]  # 30% for data transfer

    fig = go.Figure()

    if viz_type == "basic":
        ### Basic Bar Chart ###
        fig = go.Figure([go.Bar(x=layers, y=execution_times, name="Execution Time (s)")])
        fig.update_layout(
            title="Layer Execution Time",
            xaxis_title="Layers",
            yaxis_title="Time (s)",
            template="plotly_white"
        )

    elif viz_type == "stacked":
        # Stacked Bar Chart
        fig = go.Figure([
            go.Bar(x=layers, y=compute_times, name="Compute Time"),
            go.Bar(x=layers, y=transfer_times, name="Data Transfer"),
        ])
        fig.update_layout(
            barmode="stack",
            title="Layer Execution Time Breakdown",
            xaxis_title="Layers",
            yaxis_title="Time (s)",
            template="plotly_white"
        )

    elif viz_type == "horizontal":
        # Horizontal Bar Chart with Sorting
        sorted_data = sorted(filtered_data, key=lambda x: x["execution_time"], reverse=True)
        sorted_layers = [entry["layer"] for entry in sorted_data]
        sorted_times = [entry["execution_time"] for entry in sorted_data]
        fig = go.Figure([go.Bar(x=sorted_times, y=sorted_layers, orientation="h", name="Execution Time")])
        fig.update_layout(
            title="Layer Execution Time (Sorted)",
            xaxis_title="Time (s)",
            yaxis_title="Layers",
            template="plotly_white"
        )

    elif viz_type == "box":
        ### Box Plots for Variability ###
        # Use unique layers from filtered_data, maintaining original order
        unique_layers = list(dict.fromkeys(entry["layer"] for entry in filtered_data))  # Preserves order, removes duplicates
        times_by_layer = {layer: [entry["execution_time"] for entry in filtered_data if entry["layer"] == layer] for layer in unique_layers}
        fig = go.Figure([go.Box(x=unique_layers, y=[times_by_layer[layer] for layer in unique_layers], name="Execution Variability")])
        fig.update_layout(
            title="Layer Execution Time Variability",
            xaxis_title="Layers",
            yaxis_title="Time (s)",
            template="plotly_white"
        )

    elif viz_type == "gantt":
        # Gantt Chart for Timeline
        for i, entry in enumerate(filtered_data):
            fig.add_trace(go.Bar(x=[i, i], y=[0, entry["execution_time"]], orientation="v", name=entry["layer"]))
        fig.update_layout(
            title="Layer Execution Timeline",
            xaxis_title="Layers",
            yaxis_title="Time (s)",
            showlegend=True,
            template="plotly_white"
        )

    elif viz_type == "heatmap":
        # Ensure TRACE_DATA has multiple iterations or simulate them
        iterations = 5
        heatmap_data = np.random.rand(len(layers), iterations)
        fig = go.Figure(data=go.Heatmap(z=heatmap_data, x=[f"Iteration {i+1}" for i in range(iterations)], y=layers))
        fig.update_layout(title="Execution Time Heatmap", xaxis_title="Iterations", yaxis_title="Layers")
            
        

    elif viz_type == "thresholds":
        # Bar Chart with Annotations and Thresholds
        fig = go.Figure([go.Bar(x=layers, y=execution_times, name="Execution Time", 
                               marker_color=["red" if t > 0.003 else "blue" for t in execution_times])])
        for i, t in enumerate(execution_times):
            if t > 0.003:
                fig.add_annotation(
                    x=layers[i], y=t, text=f"High: {t}s", showarrow=True, arrowhead=2, 
                    font=dict(size=10), align="center"
                )
        fig.update_layout(
            title="Layer Execution Time with Thresholds",
            xaxis_title="Layers",
            yaxis_title="Time (s)",
            template="plotly_white"
        )

    # Add common layout enhancements
    fig.update_layout(
        showlegend=True,
        hovermode="x unified",
        template="plotly_white",
        height=600,
        width=1000
    )

    return [fig]

############################
#### FLOPS Memory Chart ####
############################

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
    
    return [fig]

##################
### Loss Graph ###
##################

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

##########################
### Architecture Graph ###
##########################


@app.callback(
    Output("architecture_graph", "figure"),
    Input("architecture_selector", "value")
)
def update_graph(arch):
    # Initialize input shape (e.g., for a 28x28 RGB image)
    input_shape = (1, 28, 28, 3)  # Batch, height, width, channels
    
    if arch == "A":
        layers = ["Conv2D", "Dense"]  # Example
        params = [{"kernel_size": (3, 3), "units": 128} for _ in layers]

    propagator = ShapePropagator()
    for layer in layers:
        input_shape = propagator.propagate(input_shape, layer, framework='tensorflow')  # Update input shape
    
    return create_animated_network(propagator.shape_history)  # Pass shape history


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
    
    return [fig]

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
    
    return [fig]

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
    
    return [fig]

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

####################################
### Resource Monitoring Callback ###
####################################

@app.callback(
    [Output("resource_graph", "figure")],
    [Input("interval_component", "n_intervals")]
)
def update_resource_graph(n):
    """Visualize CPU/GPU usage, memory, and I/O bottlenecks."""
    import psutil
    import torch

    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    gpu_memory = 0
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # GB

    fig = go.Figure([
        go.Bar(x=["CPU", "Memory", "GPU"], y=[cpu_usage, memory_usage, gpu_memory], name="Resource Usage (%)"),
    ])
    fig.update_layout(
        title="Resource Monitoring",
        xaxis_title="Resource",
        yaxis_title="Usage (%)",
        template="plotly_dark",
        height=400
    )
    return [fig]

#################################
### Tensor Flow Visualization ###
#################################
@app.callback(
    Output("tensor_flow_graph", "figure"),
    Input("interval_component", "n_intervals")
)
def update_tensor_flow(n):
    from neural.tensor_flow import create_animated_network
    return create_animated_network(propagator.shape_history)


# Custom Theme
app = Dash(__name__, external_stylesheets=[themes.DARKLY])  # Darkly theme for Dash Bootstrap

# Custom CSS for additional styling
app.css.append_css({
    "external_url": "https://custom-theme.com/neural.css"  # Create this file or use inline CSS
})


########################
### Principal Layout ###
########################

app.layout = html.Div([
    html.H1("NeuralDbg: Real-Time Execution Monitoring"),

    # Visualization Selector
    dcc.Dropdown(
        id="viz_type",
        options=[
            {"label": "Basic Bar Chart", "value": "basic"},
            {"label": "Stacked Bar Chart", "value": "stacked"},
            {"label": "Sorted Horizontal Bar", "value": "horizontal"},
            {"label": "Box Plot (Variability)", "value": "box"},
            {"label": "Gantt Chart (Timeline)", "value": "gantt"},
            {"label": "Heatmap (Over Time)", "value": "heatmap"},
            {"label": "Bar with Thresholds", "value": "thresholds"},
        ],
        value="basic",  # Default visualization
        multi=False

    ),
    dcc.Slider(
        id="update_interval",
        min=500, max=5000, step=500, value=UPDATE_INTERVAL,
        marks={i: f"{i}ms" for i in range(500, 5500, 500)},
        tooltip={"placement": "bottom", "always_visible": True}
    ),
    dcc.Dropdown(
        id="layer_filter",
        options=[{"label": l, "value": l} for l in ["Conv2D", "Dense"]],
        multi=True,
        value=["Conv2D", "Dense"]
    ),
    
    # Execution Trace Visualization
    dcc.Graph(id="trace_graph"),
    
    # FLOPs & Memory Usage
    dcc.Graph(id="flops_memory_chart"),
    
    # Shape Propagation
    html.H1("Neural Shape Propagation Dashboard"),
    dcc.Graph(id="shape_graph"),
    
    # Training Metrics
    html.H1("Training Metrics"),
    dcc.Graph(id="loss_graph"),
    dcc.Graph(id="accuracy_graph"),
    
    # Model Comparison
    html.H1("Compare Architectures"),
    dcc.Dropdown(
        id="architecture_selector",
        options=[
            {"label": "Model A", "value": "A"},
            {"label": "Model B", "value": "B"},
        ],
        value="A"
    ),
    dcc.Graph(id="architecture_graph"),
    
    # Resource Monitoring
    html.H1("Resource Monitoring"),
    dcc.Graph(id="resource_graph"),
    
    # Interval for updates (initial value, updated dynamically)
    dcc.Interval(id="interval_component", interval=UPDATE_INTERVAL, n_intervals=0)
])

if __name__ == "__main__":
    app.run_server(debug=False, use_reloader=False)