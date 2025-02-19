import logging
import numpy as np
import plotly.graph_objects as go
from graphviz import Digraph
from typing import Dict, Tuple, Optional, Any, List

from parser.parser import ModelTransformer

class ShapePropagator:
    
    def __init__(self, debug=False):
        self.debug = debug
        self.shape_history = []
        self.layer_connections = []
        self.current_layer = 0
        
        # Framework compatibility mappings
        self.param_aliases = {
            'Conv2D': {'filters': 'out_channels', 'kernel_size': 'kernel_size'},
            'BatchNormalization': {'axis': 'dim'},
            'Dense': {'units': 'out_features'}
        }
        
        # Initialize visualization
        self.dot = Digraph(comment='Neural Network Architecture')
        self.dot.attr('node', shape='record', style='filled', fillcolor='lightgrey')

    def propagate(self, input_shape: Tuple[Optional[int], ...], 
                layer: Dict[str, Any], 
                framework: str = 'tensorflow') -> Tuple[Optional[int], ...]:
        
        # Store initial shape
        self._log_shape(input_shape, 'input')
        self._visualize_layer('input', input_shape)
        
        output_shape = self._process_layer(input_shape, layer, framework)
        
        self._log_shape(output_shape, 'output')
        self._create_connection('input', layer['type'])
        return output_shape

    def _process_layer(self, input_shape, layer, framework):
        layer_type = layer['type']
        params = self._standardize_params(layer['params'], layer_type, framework)
        
        # Unified parameter handling
        handler_name = f"_handle_{layer_type.lower()}"
        if hasattr(self, handler_name):
            output_shape = getattr(self, handler_name)(input_shape, params)
        else:
            output_shape = self._handle_default(input_shape, params)
        
        # Visualization
        self._visualize_layer(layer_type, output_shape)
        self._create_connection(self.current_layer-1, self.current_layer)
        return output_shape

    def _standardize_params(self, params, layer_type, framework):
        # Convert framework-specific parameters to canonical form
        standardized = {}
        aliases = self.param_aliases.get(layer_type, {})
        
        for k, v in params.items():
            if framework == 'pytorch' and k in aliases.values():
                standardized[aliases[k]] = v
            else:
                standardized[k] = v
                
        # Add framework-specific defaults
        if framework == 'pytorch':
            standardized.setdefault('data_format', 'channels_first')
        else:
            standardized.setdefault('data_format', 'channels_last')
            
        return standardized

    def _handle_conv2d(self, input_shape, params):
        # Unified convolution handler for TF/PyTorch
        data_format = params['data_format']
        channels_dim = 1 if data_format == 'channels_first' else 3
        
        # Calculate output shape using framework-agnostic formula
        kernel = params['kernel_size']
        stride = params.get('stride', 1)
        padding = self._calculate_padding(params, input_shape[channels_dim])
        
        output_shape = [
            (dim + 2*padding - kernel) // stride + 1 
            for dim in input_shape[2:]  # Spatial dimensions
        ]
        
        # Reconstruct full shape
        if data_format == 'channels_first':
            return (input_shape[0], params['out_channels'], *output_shape)
        return (input_shape[0], *output_shape, params['out_channels'])


    # Handle default helper
    def _handle_default(self, input_shape, params):
        # Default handler for unsupported layers
        return input_shape
    
    # Padding calculation
    def _calculate_padding(self, params, input_dim):
        if isinstance(params.get('padding'), int):
            return params['padding']
        elif isinstance(params.get('padding'), list):
            return tuple(params['padding'])
        else:
            return [params['padding']]*(input_dim-2)

    def _visualize_layer(self, layer_name, shape):
        label = f"{layer_name}\n{shape}"
        self.dot.node(str(self.current_layer), label)
        self.shape_history.append((layer_name, shape))
        self.current_layer += 1

    def _create_connection(self, from_id, to_id):
        self.layer_connections.append((from_id, to_id))
        self.dot.edge(str(from_id), str(to_id))

    def generate_report(self):
        """Generate interactive visualization and shape report"""
        # Plotly visualization
        fig = go.Figure()
        
        # Add shape dimensions as bar chart
        shapes = [str(s[1]) for s in self.shape_history]
        fig.add_trace(go.Bar(
            x=[s[0] for s in self.shape_history],
            y=[np.prod(s[1]) for s in self.shape_history],
            text=shapes,
            name='Parameter Count'
        ))
        
        fig.update_layout(
            title='Network Shape Propagation',
            xaxis_title='Layer',
            yaxis_title='Parameters',
            template='plotly_white'
        )
        
        return {
            'dot_graph': self.dot,
            'plotly_chart': fig,
            'shape_history': self.shape_history
        }

    def _log_shape(self, shape, stage):
        if self.debug:
            logging.info(f"{stage.upper()} SHAPE: {shape}")
            logging.debug(f"Shape details: {self._shape_analysis(shape)}")

    def _shape_analysis(self, shape):
        return {
            'total_parameters': np.prod([d for d in shape if d]),
            'spatial_dims': shape[2:-1] if len(shape) > 2 else None,
            'channel_dim': shape[1] if len(shape) > 1 else None
        }
    

### Shape Validation for Error Handling ###

class ShapeValidator:
    @staticmethod
    def validate_layer(layer_type, input_shape, params):
        validators = {
            'Conv2D': lambda: ShapeValidator._validate_conv(input_shape, params),
            'Dense': lambda: ShapeValidator._validate_dense(input_shape, params)
        }
        
        if validator := validators.get(layer_type):
            validator()
            
    @staticmethod
    def _validate_conv(input_shape, params):
        if len(input_shape) != 4:
            raise ValueError(f"Conv layers need 4D input. Got {len(input_shape)}D")
        if params['kernel_size'] > input_shape[2]:
            raise ValueError(f"Kernel size {params['kernel_size']} "
                           f"exceeds input dimension {input_shape[2]}")
        
    @staticmethod
    def _validate_dense(input_shape, params):
        if len(input_shape) > 2:
            raise ValueError(f"Dense layers only accept 2D input. Got {len(input_shape)}D")

# Unified parameter handling for TF/PyTorch
FRAMEWORK_DEFAULTS = {
    'tensorflow': {
        'data_format': 'channels_last',
        'padding': 'same'
    },
    'pytorch': {
        'data_format': 'channels_first',
        'padding': 0
    }
}

def get_framework_params(framework):
    return FRAMEWORK_DEFAULTS.get(framework.lower(), FRAMEWORK_DEFAULTS['tensorflow'])

