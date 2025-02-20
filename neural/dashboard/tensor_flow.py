import networkx as nx
import plotly.graph_objects as go

def create_animated_network(layer_data):
    G = nx.DiGraph()
    positions = {}
    
    for i, layer in enumerate(layer_data):
        G.add_node(layer["layer"], pos=(i, -1))
        if i > 0:
            G.add_edge(layer_data[i-1]["layer"], layer["layer"])
        positions[layer["layer"]] = (i, -1)
    
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = positions[edge[0]]
        x1, y1 = positions[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines"))
    
    node_x, node_y = zip(*positions.values())
    node_labels = [f"{layer['layer']} - {layer['output_shape']}" for layer in layer_data]
    
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                             text=node_labels, textposition="top center",
                             marker=dict(size=20, color="lightblue")))

    fig.update_layout(title="Tensor Flow Visualization")
    return fig
