"""
Backend of the Dynamic Visualizer
It provides the data transformation logic that converts Neural DSL model structures into the JSON format that D3.js expects
The model_to_d3_json method specifically creates the nodes and links structure that D3.js uses
"""

import tensorflow as tf
import keras
from matplotlib import pyplot as plt
from graphviz import Digraph
import plotly.graph_objects as go
import numpy as np
import os
import sys
import json


# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from neural.parser.parser import ModelTransformer, create_parser


class NeuralVisualizer:
    def __init__(self, model_data):
        self.model_data = model_data
        self.figures = [] # Store generated visualizations

    ### Converting layers data to json for D3 visualization ########
    # Converts the model structure to a JSON format that's compatible with D3.js (with nodes and links)
    # JSON data is used with Matplotlib to create static images
    # The D3-compatible format is just used as a convenient intermediate representation
    def model_to_d3_json(self):  # Remove model_data parameter since we have it in self
        """Convert parsed model data to D3 visualization format"""
        nodes = []
        links = []

        # Input Layer
        input_data = self.model_data.get('input', {})
        nodes.append({
            "id": "input",
            "type": "Input",
            "shape": input_data.get('shape', None)
        })

        # Hidden Layers
        layers = self.model_data.get('layers', [])
        for idx, layer in enumerate(layers):
            node_id = f"layer{idx+1}"
            nodes.append({
                "id": node_id,
                "type": layer.get('type', 'Unknown'),
                "params": layer.get('params', {})
            })

            # Create connections
            prev_node = "input" if idx == 0 else f"layer{idx}"
            links.append({
                "source": prev_node,
                "target": node_id
            })

        # Output Layer
        output_layer = self.model_data.get('output_layer', {})
        nodes.append({
            "id": "output",
            "type": output_layer.get('type', 'Output'),
            "params": output_layer.get('params', {})
        })

        if layers:  # Only add final link if there are layers
            links.append({
                "source": f"layer{len(layers)}",
                "target": "output"
            })

        return {"nodes": nodes, "links": links}

    # Creates a 3D visualization of the shape propagation through the network
    def create_3d_visualization(self, shape_history):
        fig = go.Figure()

        for i, (name, shape) in enumerate(shape_history):
            # Uses Plotly's Scatter3d to create a 3D scatter plot
            # Each point represents a dimension in a tensor shape
            fig.add_trace(go.Scatter3d(
                x=[i]*len(shape),
                y=list(range(len(shape))),
                z=shape,
                mode='markers+text',
                text=[str(d) for d in shape],
                name=name
            ))

        fig.update_layout(
            scene=dict(
                xaxis_title='Layer Depth',
                yaxis_title='Dimension Index',
                zaxis_title='Dimension Size'
            )
        )
        return fig # Plotly figue object

    # JSON data is used with Matplotlib to create static images
    def save_architecture_diagram(self, filename):
        """Save the architecture diagram to a file.

        Args:
            filename: The name of the file to save the diagram to.
        """
        import matplotlib.pyplot as plt

        # Create a simple architecture diagram using matplotlib
        fig, ax = plt.subplots(figsize=(10, 8))

        # Get nodes and links from D3 format
        d3_data = self.model_to_d3_json()
        nodes = d3_data['nodes']
        links = d3_data['links']

        # Plot nodes as boxes
        for i, node in enumerate(nodes):
            y_pos = len(nodes) - i - 1  # Reverse order for top-to-bottom layout
            ax.add_patch(plt.Rectangle((0.2, y_pos - 0.4), 0.6, 0.8, fill=True,
                                      color='lightblue', alpha=0.7))
            ax.text(0.5, y_pos, f"{node['type']}", ha='center', va='center', fontweight='bold')

            # Add parameters text if available
            if 'params' in node and node['params']:
                param_text = ', '.join(f"{k}={v}" for k, v in node['params'].items())
                ax.text(0.5, y_pos - 0.2, param_text, ha='center', va='center', fontsize=8)

        # Plot links as arrows
        for link in links:
            source_idx = next(i for i, n in enumerate(nodes) if n['id'] == link['source'])
            target_idx = next(i for i, n in enumerate(nodes) if n['id'] == link['target'])
            source_y = len(nodes) - source_idx - 1
            target_y = len(nodes) - target_idx - 1
            ax.arrow(0.5, source_y - 0.4, 0, target_y - source_y + 0.4,
                    head_width=0.05, head_length=0.1, fc='black', ec='black')

        # Set plot limits and remove axes
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(nodes) - 0.5)
        ax.axis('off')

        # Save the figure
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def save_shape_visualization(self, fig, filename):
        """Save the shape visualization to an HTML file.

        Args:
            fig: The plotly figure to save.
            filename: The name of the file to save the visualization to.
        """
        import plotly.io
        plotly.io.write_html(fig, filename)


##### EXAMPLE ########


if __name__ == '__main__':
    # Example usage
    nr_content = """
    network TestNet {
        input: (None, 28, 28, 1)
        layers:
            Conv2D(filters=32, kernel_size=(3,3), activation="relu")
            MaxPooling2D(pool_size=(2,2))
            Flatten()
            Dense(128, "relu")
            Output(10, "softmax")
        loss: "categorical_crossentropy"
        optimizer: "adam"
    }
    """

    parser = create_parser('network')
    parsed = parser.parse(nr_content)
    model_data = ModelTransformer().transform(parsed)

    visualizer = NeuralVisualizer(model_data)
    print(visualizer.model_to_d3_json())
