import dash
from dash import dcc, html
import plotly.graph_objects as go
from flask import Flask

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

if __name__ == "__main__":
    app.run_server(debug=True)
