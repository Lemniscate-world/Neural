from dash import Dash, dcc, html, Input, Output, dash_table
from dash_bootstrap_components import themes, Navbar, NavItem, NavLink, Container, Row, Col
import json
from neural.parser.parser import create_parser, ModelTransformer
from neural.shape_propagation.shape_propagator import ShapePropagator
from neural.code_generation.code_generator import generate_code

app = Dash(__name__, external_stylesheets=[themes.DARKLY])  # Custom theme (dark mode)

app.layout = html.Div([
    Navbar([
        NavItem(NavLink("Neural No-Code", href="#")),
    ]),
    Row([
        Col(dcc.Dropdown(
            id="layer-type",
            options=[
                {"label": "Conv2D", "value": "Conv2D"},
                {"label": "Dense", "value": "Dense"},
                {"label": "Dropout", "value": "Dropout"},
                # Add more layer types
            ],
            value=None
        )),
        Col(dash_table.DataTable(
            id="layer-params",
            columns=[
                {"name": "Parameter", "id": "param"},
                {"name": "Value", "id": "value"}
            ],
            data=[],
            style_table={'overflowX': 'auto'}
        )),
    ]),
    html.Button("Add Layer", id="add-layer"),
    dcc.Graph(id="architecture-preview"),
    html.Button("Compile", id="compile-btn"),
    html.Div(id="output")
])

@app.callback(
    [Output("layer-params", "data"), Output("architecture-preview", "figure")],
    [Input("layer-type", "value"), Input("add-layer", "n_clicks")]
)
def update_layer_params(layer_type, n_clicks):
    if layer_type and n_clicks:
        params = {"type": layer_type, "params": {}}
        if layer_type == "Conv2D":
            params["params"] = {"filters": 32, "kernel_size": (3, 3), "activation": "relu"}
        elif layer_type == "Dense":
            params["params"] = {"units": 128, "activation": "relu"}
        elif layer_type == "Dropout":
            params["params"] = {"rate": 0.5}
        
        data = [{"param": k, "value": str(v)} for k, v in params["params"].items()]
        propagator = ShapePropagator()
        input_shape = (1, 28, 28, 1)  # Default input shape
        shape_history = []
        for layer in [params]:
            input_shape = propagator.propagate(input_shape, layer, "tensorflow")
            shape_history.append({"layer": layer["type"], "output_shape": input_shape})
        
        fig = go.Figure(data=[go.Scatter(x=[i for i in range(len(shape_history))], y=[np.prod(s["output_shape"]) for s in shape_history], mode="lines+markers")])
        fig.update_layout(title="Shape Propagation Preview", xaxis_title="Layer", yaxis_title="Parameters")
        return data, fig
    return [], go.Figure()

@app.callback(
    Output("output", "children"),
    [Input("compile-btn", "n_clicks")],
    [State("layer-params", "data")]
)
def compile_model(n_clicks, params_data):
    if n_clicks and params_data:
        layers = []
        for param in params_data:
            layer_type = None
            layer_params = {}
            for row in params_data:
                if row["param"] == "type":
                    layer_type = row["value"]
                else:
                    try:
                        layer_params[row["param"]] = eval(row["value"]) if row["value"].replace(".", "").isdigit() else row["value"]
                    except:
                        layer_params[row["param"]] = row["value"]
            layers.append({"type": layer_type, "params": layer_params})
        
        model_data = {
            "type": "model",
            "input": {"type": "Input", "shape": (1, 28, 28, 1)},
            "layers": layers,
            "loss": {"value": '"categorical_crossentropy"'},
            "optimizer": {"type": "Adam", "params": {"learning_rate": 0.001}}
        }
        code_tf = generate_code(model_data, "tensorflow")
        code_torch = generate_code(model_data, "pytorch")
        return html.Div([
            html.H3("Generated Code"),
            html.Pre(code_tf, style={"whiteSpace": "pre-wrap"}),
            html.H3("PyTorch Code"),
            html.Pre(code_torch, style={"whiteSpace": "pre-wrap"})
        ])
    return "Click 'Compile' to generate code."

if __name__ == "__main__":
    app.run_server(debug=True, port=8051)