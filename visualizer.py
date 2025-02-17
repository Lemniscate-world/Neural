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


from parser import ModelTransformer, create_parser

class NeuralVisualizer:
    def __init__(self, model_data):
        self.model_data = model_data
        self.figures = []
    
    ### Converting layers data to json for D3 visualization ########

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
        
    def create_3d_visualization(self, shape_history):
        fig = go.Figure()
        
        for i, (name, shape) in enumerate(shape_history):
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
        return fig
  


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