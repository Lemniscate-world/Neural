import dash
import time
from dash import dcc, html
from dash.dependencies import Input, Output
import json
import plotly.graph_objects as go
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
import threading
import yaml

from neural.shape_propagation.shape_propagator import ShapePropagator


# Backend shape propagator instance
propagator = ShapePropagator()

# Example shape propagation (to simulate training updates)
input_shape = (1, 28, 28, 1)
layers = [
    {"type": "Conv2D", "params": {"filters": 32, "kernel_size": (3, 3), "padding": "same"}},
    {"type": "MaxPooling2D", "params": {"pool_size": (2, 2)}},
    {"type": "Flatten", "params": {}},
    {"type": "Dense", "params": {"units": 128}},
    {"type": "Output", "params": {"units": 10}}
]
for layer in layers:
    input_shape = propagator.propagate(input_shape, layer)

# Flask server
server = Flask(__name__)
app = dash.Dash(__name__, server=server)
socketio = SocketIO(server, cors_allowed_origins=["http://localhost:8050"], logger=True)

app.layout = html.Div([
    html.H1("Neural Shape Propagation Dashboard"),
    dcc.Graph(id="shape_graph"),
    dcc.Interval(id="interval_component", interval=1000, n_intervals=0)
])

@app.callback(
    Output("shape_graph", "figure"),
    Input("interval_component", "n_intervals")
)
def update_graph(n):
    shape_data = json.loads(propagator.get_shape_data())

    # Build network graph for visualization
    edge_x, edge_y, node_x, node_y, node_labels = [], [], [], [], []
    positions = {}
    for i, shape in enumerate(shape_data):
        positions[shape["layer"]] = (i, -1)
        node_x.append(i)
        node_y.append(-1)
        node_labels.append(f"{shape['layer']}\n{shape['output_shape']}")

        if i > 0:
            edge_x.extend([i-1, i])
            edge_y.extend([-1, -1])

    fig = go.Figure()

    # Draw edges
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=2)))

    # Draw nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=30, color="lightblue"),
        text=node_labels, textposition="top center"
    ))

    fig.update_layout(title="Shape Propagation", showlegend=False)
    return fig

############################
### NNTRACE DATA SENDING ###
############################

# Load configuration (e.g., from config.yaml)
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        users = {config["auth"]["username"]: generate_password_hash(config["auth"]["password"])}
except:
    config = {"username": "admin", "password": "default"}

users = {
    config["username"]: generate_password_hash(config["password"])
}
auth = HTTPBasicAuth()

@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users[username], password):
        return username

@server.route("/trace", methods=["GET"])
@auth.login_required
def get_nntrace():
    """API endpoint to fetch the latest execution trace for dashboard."""
    return jsonify(propagator.get_trace())

@socketio.on("request_trace_update")
@auth.login_required
def send_trace_update():
    """Streams real-time execution traces to the dashboard via WebSockets with authentication."""
    username = request.headers.get('Authorization').split(' ')[1]  # Basic auth token
    join_room(username)  # Create a room for the user
    while True:
        trace_data = propagator.get_trace()
        emit("trace_update", json.dumps(trace_data), room=username)
        time.sleep(UPDATE_INTERVAL / 1000)

if __name__ == "__main__":
    threading.Thread(target=socketio.run, args=(server, "localhost", 5001), daemon=True).start()
