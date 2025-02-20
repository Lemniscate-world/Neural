import networkx as nx
import plotly.graph_objects as go
from networkx.drawing.nx_agraph import graphviz_layout

def create_animated_network(layer_data):
    if not layer_data:
        return go.Figure()
    G = nx.DiGraph()
    
    # Add nodes with layer metadata
    for layer in layer_data:
        G.add_node(layer["layer"], output_shape=layer["output_shape"])
    
    # Add edges (sequential for now; customize for branched architectures)
    for i in range(1, len(layer_data)):
        G.add_edge(layer_data[i-1]["layer"], layer_data[i]["layer"])
    
    # Use Graphviz tree layout to prevent overlaps
    pos = graphviz_layout(G, prog="dot", args="-Grankdir=TB")  # Top-to-bottom hierarchy
    
    # Extract edge coordinates
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Plot nodes and edges with Plotly
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_labels = [f"{node}\n{G.nodes[node]['output_shape']}" for node in G.nodes()]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="gray")))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=node_labels, textposition="middle center",
        marker=dict(size=30, color="lightblue", line=dict(width=2, color="darkblue"))
    ))
    fig.update_layout(title="Tensor Flow Visualization", showlegend=False)
    return fig