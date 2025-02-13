import tensorflow as tf
import keras
from matplotlib import pyplot as plt
from graphviz import Digraph
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
    
    ### Converting layers data to json for D3 visualization ########

    def model_to_d3_json(self, model_data):
        """Convert parsed model data to D3 visualization format"""
        nodes = []
        links = []
        
        # Input Layer
        nodes.append({
            "id": "input",
            "type": "Input",
            "shape": model_data['input']['shape']
        })
        
        # Hidden Layers
        for idx, layer in enumerate(model_data['layers']):
            node_id = f"layer{idx+1}"
            nodes.append({
                "id": node_id,
                "type": layer['type'],
                "params": layer.get('params', {})
            })
            
            # Create connections
            prev_node = "input" if idx == 0 else f"layer{idx}"
            links.append({
                "source": prev_node,
                "target": node_id
            })
        
        # Output Layer
        nodes.append({
            "id": "output",
            "type": model_data['output_layer']['type'],
            "params": model_data['output_layer'].get('params', {})
        })
        links.append({
            "source": f"layer{len(model_data['layers'])}",
            "target": "output"
        })

        print(nodes)
        print(links)
        
        return  {"nodes": nodes, "links": links}
  

if __name__ == '__main__':

    model_data = create_parser('network')
    Visualizer = NeuralVisualizer(model_data)
    print(Visualizer)
    print(Visualizer.model_to_d3_json(model_data))