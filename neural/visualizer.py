import tensorflow as tf
import keras
from matplotlib import pyplot as plt
from graphviz import Digraph
import numpy as np
import os
import sys


# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from shape_propagator import calculate_shape_propagation
from code_generator import generate_code
from parser import ModelTransformer, create_parser

class NeuralVisualizer:
    def __init__(self, model_data):
        self.model_data = create_parser('network')
        self.shape_history = calculate_shape_propagation(model_data)
    
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

        ### Nodes Shape Info ########
        shape_history = calculate_shape_propagation(model_data)
        for node, shape in zip(nodes[1:-1], shape_history[1:-1]):
            node['shape'] = shape[1]
        
        return {"nodes": nodes, "links": links}


    def export_for_tensorboard(self, model_data, backend='tensorflow'):
        # Generate backend-specific code
        model_code = generate_code(model_data, backend)
        
        # Save temporary model
        with open("temp_model.py", "w") as f:
            f.write(model_code)
        
        # For TensorFlow
        if backend == 'tensorflow':
            from tensorflow.keras.models import load_model
            model = load_model("temp_model.h5")
            tf.keras.utils.plot_model(model, to_file='tensorboard_model.png')


    def visualize_conv_filters(self, layer_data):
        if layer_data['type'] != 'Conv2D':
            return
        
        filters = layer_data['params'].get('filters', 32)
        kernel_size = layer_data['params'].get('kernel_size', (3,3))
        
        # Generate sample filter visualizations
        fig, axes = plt.subplots(4, 8, figsize=(10,5))
        for ax in axes.flatten():
            img = np.random.randn(*kernel_size)
            ax.imshow(img, cmap='viridis')
            ax.axis('off')
        
        plt.suptitle(f"Conv2D Layer Visualization\n{filters} Filters ({kernel_size[0]}x{kernel_size[1]})")
        plt.savefig("conv_filters.png")
        plt.close()


    def visualize_model_architecture(self, model_data, filename="model_architecture"):
        """Generates a Graphviz diagram of the model architecture."""
        dot = Digraph(comment='Neural Network Architecture', format='png')
        
        # Input Layer
        dot.node('input', f"Input\nShape: {model_data['input']['shape']}", shape='box')
        
        # Hidden Layers
        for idx, layer in enumerate(model_data['layers']):
            layer_label = (
                f"{layer['type']}\n"
                f"Params: {', '.join([f'{k}={v}' for k,v in layer.get('params', {}).items()])}"
            )
            dot.node(f'layer{idx}', layer_label, shape='ellipse')
        
        # Connections
        prev_node = 'input'
        for idx in range(len(model_data['layers'])):
            dot.edge(prev_node, f'layer{idx}')
            prev_node = f'layer{idx}'
        
        # Output Layer
        dot.node('output', f"Output\nShape: {model_data['output_shape']}", shape='box')
        dot.edge(prev_node, 'output')
        
        dot.render(filename, cleanup=True)
        print(f"Saved architecture diagram to {filename}.png")