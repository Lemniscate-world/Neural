import dash
from dash import dcc, html
import plotly.graph_objects as go
from flask import Flask
from numpy import random

# Flask app for WebSocket integration (if needed later)
server = Flask(__name__)

# Dash app
app = dash.Dash(__name__, server=server)

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
        layers = [{"type": "Dense", "params": {"units": 256"}}]

    shape_data = propagate_shapes(layers)
    return create_animated_network(shape_data)


if __name__ == "__main__":
    app.run_server(debug=True)