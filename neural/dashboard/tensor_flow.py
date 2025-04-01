import networkx as nx
import plotly.graph_objects as go
from networkx.drawing.nx_agraph import graphviz_layout
import time
from tqdm import tqdm
import numpy as np
from dash import html
import plotly.express as px

def create_animated_network(layer_data, show_progress=True):
    """Create an animated network visualization with detailed progress tracking.
    
    Args:
        layer_data: List of layer dictionaries with 'layer' and 'output_shape'
        show_progress: Whether to display progress information
        
    Returns:
        Plotly Figure object with the network visualization
    """
    if not layer_data:
        return go.Figure()
    
    # Initialize progress tracking
    progress_steps = 4
    progress_data = {
        "steps": ["Building graph", "Creating connections", "Calculating layout", "Rendering visualization"],
        "current_step": 0,
        "progress": 0,
        "details": "",
        "start_time": time.time()
    }
    
    # Create graph
    G = nx.DiGraph()
    
    # Add nodes with layer metadata
    for i, layer in enumerate(layer_data):
        if show_progress:
            progress_data["details"] = f"Adding node {i+1}/{len(layer_data)}: {layer['layer']}"
            progress_data["progress"] = (i+1) / len(layer_data) * 25  # 25% for this step
            _update_progress(progress_data)
            
        G.add_node(layer["layer"], output_shape=layer["output_shape"])
        time.sleep(0.05)  # Small delay to show progress
    
    if show_progress:
        progress_data["current_step"] = 1
        progress_data["details"] = "Creating layer connections"
        _update_progress(progress_data)
    
    # Add edges (sequential for now; customize for branched architectures)
    for i in range(1, len(layer_data)):
        if show_progress:
            progress_data["details"] = f"Connecting {layer_data[i-1]['layer']} → {layer_data[i]['layer']}"
            progress_data["progress"] = 25 + (i) / (len(layer_data)-1) * 25  # 25-50% for this step
            _update_progress(progress_data)
            
        G.add_edge(layer_data[i-1]["layer"], layer_data[i]["layer"])
        time.sleep(0.05)  # Small delay to show progress
    
    if show_progress:
        progress_data["current_step"] = 2
        progress_data["details"] = "Calculating optimal layout"
        progress_data["progress"] = 50  # 50% complete
        _update_progress(progress_data)
    
    # Use Graphviz tree layout to prevent overlaps
    pos = graphviz_layout(G, prog="dot", args="-Grankdir=TB")  # Top-to-bottom hierarchy
    
    if show_progress:
        progress_data["progress"] = 75  # 75% complete
        progress_data["details"] = "Layout calculation complete"
        _update_progress(progress_data)
    
    # Extract edge coordinates
    edge_x, edge_y = [], []
    for i, edge in enumerate(G.edges()):
        if show_progress:
            progress_data["details"] = f"Processing edge {i+1}/{len(G.edges())}"
            progress_data["progress"] = 75 + (i+1) / len(G.edges()) * 12.5  # 75-87.5% for this step
            _update_progress(progress_data)
            
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Extract node coordinates
    node_x, node_y = [], []
    node_labels = []
    node_sizes = []
    node_colors = []
    node_hover_texts = []
    
    for i, node in enumerate(G.nodes()):
        if show_progress:
            progress_data["details"] = f"Processing node {i+1}/{len(G.nodes())}"
            progress_data["progress"] = 87.5 + (i+1) / len(G.nodes()) * 12.5  # 87.5-100% for this step
            _update_progress(progress_data)
            
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_labels.append(node)
        
        # Size nodes based on output shape complexity
        output_shape = G.nodes[node].get('output_shape', (1,))
        tensor_size = np.prod(output_shape)
        node_sizes.append(min(50, max(20, 10 * np.log10(tensor_size + 1))))
        
        # Color nodes by layer type
        if "input" in node.lower():
            node_colors.append("green")
        elif "output" in node.lower():
            node_colors.append("red")
        elif "conv" in node.lower():
            node_colors.append("blue")
        elif "pool" in node.lower():
            node_colors.append("purple")
        elif "dense" in node.lower():
            node_colors.append("orange")
        else:
            node_colors.append("lightblue")
            
        # Create hover text with shape information
        hover_text = f"Layer: {node}<br>Shape: {output_shape}<br>Elements: {tensor_size:,}"
        node_hover_texts.append(hover_text)
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1, color="gray"),
        hoverinfo="none"
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=node_labels, textposition="middle center",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color="darkblue")
        ),
        hovertext=node_hover_texts,
        hoverinfo="text"
    ))
    
    # Update layout
    fig.update_layout(
        title="Neural Network Architecture Visualization",
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    if show_progress:
        progress_data["progress"] = 100
        progress_data["details"] = "Visualization complete"
        _update_progress(progress_data)
    
    return fig

def _update_progress(progress_data):
    """Update and display progress information."""
    elapsed = time.time() - progress_data["start_time"]
    step = progress_data["steps"][progress_data["current_step"]]
    
    # Print progress to console
    print(f"[{progress_data['progress']:.1f}%] Step {progress_data['current_step']+1}/4: {step}")
    print(f"  → {progress_data['details']}")
    print(f"  → Elapsed: {elapsed:.2f}s")
    
    # You could also emit this via websockets for dashboard updates
    # This would be implemented if using with Dash/Flask

def create_progress_component():
    """Create a Dash component for displaying progress."""
    return html.Div([
        html.H4("Visualization Progress"),
        html.Div(id="progress-bar-container", children=[
            html.Div(id="progress-bar", 
                     style={"width": "0%", "backgroundColor": "#4CAF50", "height": "30px"})
        ], style={"width": "100%", "backgroundColor": "#ddd"}),
        html.Div(id="progress-text", children="0%"),
        html.Div(id="progress-details", children="Initializing...")
    ])

def create_layer_computation_timeline(layer_data):
    """Create a Gantt chart showing computation time for each layer."""
    if not layer_data:
        return go.Figure()
    
    # Extract execution times if available
    layers = []
    start_times = []
    end_times = []
    colors = []
    
    current_time = 0
    for layer in layer_data:
        layer_name = layer["layer"]
        # Use actual execution time if available, otherwise simulate
        exec_time = layer.get("execution_time", 0.1 + np.random.random() * 0.5)
        
        layers.append(layer_name)
        start_times.append(current_time)
        current_time += exec_time
        end_times.append(current_time)
        
        # Color based on layer type
        if "conv" in layer_name.lower():
            colors.append("blue")
        elif "pool" in layer_name.lower():
            colors.append("purple")
        elif "dense" in layer_name.lower():
            colors.append("orange")
        else:
            colors.append("lightblue")
    
    # Create Gantt chart
    fig = px.timeline(
        x_start=start_times,
        x_end=end_times,
        y=layers,
        color=colors,
        labels={"x_start": "Start Time", "x_end": "End Time", "y": "Layer"}
    )
    
    fig.update_layout(
        title="Layer Computation Timeline",
        xaxis_title="Time (seconds)",
        showlegend=False
    )
    
    return fig
