import logging
import json
import time
import torch
import numpy as np
import psutil
import plotly.graph_objects as go
from graphviz import Digraph
from typing import Dict, Tuple, Optional, Any, List

import sys
import os

# Add the parent directory of 'neural' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from parser.parser import ModelTransformer
from pretrained_models.pretrained import PretrainedModelHub

class PerformanceMonitor:
    def __init__(self):
        self.resource_history = []

    def monitor_resources(self):
        """Monitor CPU, memory, and GPU usage."""
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        io_counters = psutil.disk_io_counters()
        io_usage = (io_counters.read_bytes + io_counters.write_bytes) / (1024 ** 2)  # MB

        self.resource_history.append({
            "timestamp": time.time(),
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "gpu_memory": gpu_memory,
            "io_usage": io_usage
        })
        return self.resource_history[-1]


class ShapePropagator:
    
    def __init__(self, debug=False):
        self.debug = debug
        self.shape_history = []
        self.layer_connections = []
        self.current_layer = 0
        self.execution_trace = []  # Stores nntrace logs
        self.performance_monitor = PerformanceMonitor()
        self.hub = PretrainedModelHub()
        
        # Framework compatibility mappings
        self.param_aliases = {
            'Conv2D': {'filters': 'out_channels', 'kernel_size': 'kernel_size'},
            'BatchNormalization': {'axis': 'dim'},
            'Dense': {'units': 'out_features'},
            'LSTM': {'units': 'hidden_size'},
            'BatchNormalization': {'momentum': 'decay'}
        }
        
        # Initialize visualization
        self.dot = Digraph(comment='Neural Network Architecture')
        self.dot.attr('node', shape='record', style='filled', fillcolor='lightgrey')

    def propagate(self, input_shape: Tuple[Optional[int], ...], 
              layer: Dict[str, Any], 
              framework: str = 'tensorflow') -> Tuple[Optional[int], ...]:
        """Processes a layer and logs shape changes for nntrace."""
        layer_type = layer["type"]
        params = layer.get("params", {})
        kernel_size = tuple(params.get("kernel_size", [3, 3]) if isinstance(params.get("kernel_size"), list) else (3, 3))
        params["kernel_size"] = kernel_size  # Ensure tuple in params

        if layer['type'] == 'TransformerEncoder':
            if framework == 'tensorflow':
                return input_shape  # Shape preserved through self-attention
            elif framework == 'pytorch':
                return (input_shape[0], input_shape[1])  # (seq_len, d_model)

        start_time = time.time()  # Measure execution time

        output_shape = self._process_layer(input_shape, layer, framework)
        prev_layer = self.current_layer - 1 if self.current_layer > 0 else None

        # Compute FLOPs, memory, compute_time, and transfer_time
        flops, mem_usage, compute_time, transfer_time = self._compute_performance(layer, input_shape, output_shape)

        # Capture nntrace log with additional timing details
        trace_entry = {
            "layer": layer_type,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "flops": flops,
            "memory": mem_usage,
            "execution_time": time.time() - start_time,
            "compute_time": compute_time,
            "transfer_time": transfer_time,
        }
        self.execution_trace.append(trace_entry)

        resources = self.performance_monitor.monitor_resources()
        trace_entry.update({
            "cpu_usage": resources["cpu_usage"],
            "memory_usage": resources["memory_usage"],
            "gpu_memory": resources["gpu_memory"],
            "io_usage": resources["io_usage"]
        })

        if self.debug:
            print(f"TRACE: {trace_entry}")  # Debugging output

        self._visualize_layer(layer['type'], output_shape)  # Creates node and increments self.current_layer
        if prev_layer is not None:
            self._create_connection(prev_layer, self.current_layer - 1)  # Connect previous to current
        return output_shape

###############################
### Performance Computation ###
###############################

    def _compute_performance(self, layer: dict, input_shape: tuple, output_shape: tuple) -> tuple:
        """Compute performance metrics (FLOPs, memory usage, etc.)."""
        # Replace None with 1 to avoid NoneType math errors
        input_shape = tuple(1 if dim is None else dim for dim in input_shape)
        output_shape = tuple(1 if dim is None else dim for dim in output_shape)

        # FLOPs calculation (example for Conv2D)
        if layer['type'] == 'Conv2D':
            kernel_size = layer['params']['kernel_size']
            filters = layer['params']['filters']
            flops = np.prod(kernel_size) * np.prod(output_shape) * input_shape[-1]
        else:
            flops = 0  # Default for other layers

        # Memory usage (output tensor size in MB)
        memory_usage = np.prod(output_shape) * 4 / (1024 ** 2)  # 4 bytes per float

        # Simplified timing estimates
        compute_time = flops / 1e9  # 1 GFLOP/s
        transfer_time = memory_usage * 1e3 / 1e9  # 1 GB/s

        return flops, memory_usage, compute_time, transfer_time

##################################################
### Send execution trace data to the dashboard ###
##################################################
    def get_trace(self):
        trace = []
        for layer_type, exec_time, comp_time, trans_time, params, flops, memory, grad_norm, dead_ratio, mean_act, anomaly in self.execution_trace:
            kernel_size = params.get("kernel_size", (1, 1))
            if isinstance(kernel_size, list):
                print(f"WARNING: Converting list kernel_size {kernel_size} to tuple for {layer_type}")
                kernel_size = tuple(kernel_size)
            elif not isinstance(kernel_size, tuple):
                print(f"WARNING: Unexpected kernel_size type {type(kernel_size)} for {layer_type}, defaulting to (1, 1)")
                kernel_size = (1, 1)
            trace.append({
                "layer": layer_type, "execution_time": exec_time, "compute_time": comp_time,
                "transfer_time": trans_time, "kernel_size": kernel_size,
                "flops": flops, "memory": memory, "grad_norm": grad_norm, "dead_ratio": dead_ratio,
                "mean_activation": mean_act, "anomaly": anomaly
            })
            
        return trace

    def _process_layer(self, input_shape, layer, framework):
        layer_type = layer['type']
        params = self._standardize_params(layer.get('params', {}), layer_type, framework)
        # Unified parameter handling
        handler_name = f"_handle_{layer_type.lower()}"
        if hasattr(self, handler_name):
            output_shape = getattr(self, handler_name)(input_shape, params)
        else:
            output_shape = self._handle_default(input_shape, params)
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

####################################################################
### Shape propagation through 2 Dimensional Convolutional Layers ###
####################################################################

    def _handle_conv2d(self, input_shape, params):
        """Calculates the output shape for a Conv2D layer.

        This method computes the output shape based on the input shape,
        kernel size, stride, padding, and data format.

        Args:
            input_shape (tuple): Input tensor shape.
            params (dict): Layer parameters.

        Returns:
            tuple: Output tensor shape.
        """    
        data_format = params['data_format']
        if data_format == 'channels_first':
            spatial_dims = input_shape[2:]  # (height, width)
        else:  # channels_last
            spatial_dims = input_shape[1:3]  # (height, width)

        # Handle kernel_size as tuple or integer
        kernel = params['kernel_size']
        if isinstance(kernel, int):
            kernel = (kernel,) * len(spatial_dims)
        elif isinstance(kernel, tuple):
            if len(kernel) != len(spatial_dims):
                raise ValueError("Kernel size must match spatial dimensions")

        kernel_size = tuple(params.get("kernel_size", [3, 3]) if isinstance(params.get("kernel_size"), list) else (3, 3))

        stride = params.get('stride', 1)
        padding = self._calculate_padding(params, input_shape[1] if data_format=='channels_last' else input_shape[2])

        # Ensure padding is a tuple matching spatial dimensions
        if isinstance(padding, int):
            padding = (padding,) * len(kernel)
        elif isinstance(padding, (list, tuple)):
            if len(padding) != len(kernel):
                raise ValueError(f"Padding length {len(padding)} != kernel dimensions {len(kernel)}")
            padding = tuple(padding)
        else:
            raise TypeError(f"Invalid padding type: {type(padding)}")

        # Calculate output shape per spatial dimension
        output_spatial = [
            (dim + 2*pad - k) // stride + 1
            for dim, k, pad in zip(spatial_dims, kernel, padding)
        ]

        # Reconstruct full shape
        if data_format == 'channels_first':
            return (input_shape[0], params['filters'], *output_spatial)
        else:
            return (input_shape[0], *output_spatial, params['filters'])
        
    def _handle_maxpooling2d(self, input_shape, params):
        data_format = params.get('data_format', 'channels_last')
        pool_size = params['pool_size']
        stride = params.get('stride', pool_size)

        # Handle stride as tuple or integer
        if isinstance(stride, (tuple, list)):
            stride_h, stride_w = stride
        else:
            stride_h = stride_w = stride

        # Calculate spatial dimensions based on data format
        if data_format == 'channels_last':
            # TensorFlow: input_shape = (batch, height, width, channels)
            new_height = input_shape[1] // stride_h
            new_width = input_shape[2] // stride_w
            return (input_shape[0], new_height, new_width, input_shape[3])
        else:
            # PyTorch: input_shape = (batch, channels, height, width)
            new_height = input_shape[2] // stride_h
            new_width = input_shape[3] // stride_w
            return (input_shape[0], input_shape[1], new_height, new_width)
    
    def _handle_flatten(self, input_shape, params):
        # If there is a batch dimension, keep it.
        if len(input_shape) >= 1:
            batch = input_shape[0]
            # Multiply all dimensions after the batch dimension
            flattened = np.prod(input_shape[1:])
            return (batch, flattened)
        else:
            return (np.prod(input_shape),)


    def _handle_dense(self, input_shape, params):
        # If input_shape has two or more dimensions, preserve the batch dimension.
        if len(input_shape) >= 2:
            return (input_shape[0], params['units'])
        else:
            return (params['units'],)

    def _handle_output(self, input_shape, params):
        # Preserves the batch dimension and converts the feature dimension to the number of output units.
        if len(input_shape) >= 2:
            return (input_shape[0], params['units'])
        else:
            return (params['units'],)


    # Handle default helper
    def _handle_default(self, input_shape, params):
        # Default handler for unsupported layers
        return input_shape
    
    ### Padding detection, extraction and calculation ###
    def _calculate_padding(self, params, input_dim):
        """Calculates padding based on provided parameters and input dimension.

        This method handles different padding types: integer, list, or string.
        It returns the appropriate padding value based on the input.

        Args:
            params (dict): Layer parameters containing padding information.
            input_dim (int): Input dimension.

        Returns:
            int or tuple or list: Calculated padding value.
        """
        padding = params.get('padding', 0)
    
        if isinstance(padding, int):
            return padding
        elif isinstance(padding, (list, tuple)):
            return tuple(padding)
        elif padding == 'same':
            # Handle kernel_size as tuple or integer
            kernel = params['kernel_size']
            if isinstance(kernel, int):
                return (kernel - 1) // 2
            elif isinstance(kernel, tuple):
                return tuple((k - 1) // 2 for k in kernel)  # Process each dimension
        elif padding == 'valid':
            return 0
        else:
            return [padding] * (input_dim - 2)
    
    ### Layers Shape Propagation Visualization ###
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
    
    ### Loading Pretrained Models ####

    def load_pretrained(self, model_name, pretrained=True):
        model = self.hub.load(model_name, pretrained)
        # Propagate shapes through pretrained model
        input_shape = (1, 3, 224, 224)  # Default for ResNet50
        for layer in model.layers:
            input_shape = self.propagate(input_shape, layer, "pytorch")

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
            raise ValueError(
                f"Dense layer expects 2D input (batch, features). "
                f"Got {len(input_shape)}D: {input_shape}"
            )
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

### Real-Time Shape Visualization ###
def get_shape_data(self):
        """Returns shape history as JSON."""
        return json.dumps([
        {"layer": layer[0], "output_shape": layer[1]} 
        for layer in self.shape_history
    ])

def _calculate_shape(self, input_shape, layer):
    if layer["type"] == "Dense":
        return (input_shape[0], layer["params"]["units"])
    elif layer["type"] == "Conv2D":
        return (input_shape[0], input_shape[1], input_shape[2], layer["params"]["filters"])
    elif layer["type"] == "Flatten":
        return (input_shape[0], np.prod(input_shape[1:]))
    return input_shape

### Compute FLOPs and memory usage for visualization ###
def compute_flops_params(layer, input_shape):
    """Estimate FLOPs and parameter counts for a given layer."""
    if layer["type"] == "Dense":
        units = layer["params"]["units"]
        params = input_shape[1] * units + units  # Weights + biases
        flops = 2 * params  # Two operations per weight (multiply + add)
    
    elif layer["type"] == "Conv2D":
        filters = layer["params"]["filters"]
        kernel_size = layer["params"]["kernel_size"]
        stride = layer["params"].get("stride", 1)
        params = (kernel_size[0] * kernel_size[1] * input_shape[-1] + 1) * filters
        output_height = (input_shape[1] - kernel_size[0]) // stride + 1
        output_width = (input_shape[2] - kernel_size[1]) // stride + 1
        flops = params * output_height * output_width
    
    return params, flops

#######################################
### Gradient Flow Visualization #######
#######################################
def register_gradient_hooks(model):
    """Attaches hooks to capture gradient magnitudes during backprop."""
    gradient_trace = []

    def hook(module, grad_input, grad_output):
        if grad_output[0] is not None:
            grad_norm = grad_output[0].detach().abs().mean().item()
            gradient_trace.append({"layer": module.__class__.__name__, "grad_norm": grad_norm})

    for layer in model.children():
        layer.register_backward_hook(hook)

    return gradient_trace

#####################################
### Dead Neurons Detection ##########
#####################################
def detect_dead_neurons(layer, input, output):
    """Detects inactive neurons (dead neurons)."""
    dead_neurons = (output.detach() == 0).sum().item()
    total_neurons = output.numel()
    dead_ratio = dead_neurons / total_neurons

    return {"layer": layer.__class__.__name__, "dead_ratio": dead_ratio}

######################################
### Activation Anomalies Detection ###
######################################
def detect_activation_anomalies(layer, input, output):
    """Flags NaNs, extremely high activations, or overflows."""
    mean_activation = output.detach().abs().mean().item()
    has_nan = torch.isnan(output).sum().item() > 0
    is_exploding = mean_activation > 1000  # Arbitrary threshold for huge activations

    return {
        "layer": layer.__class__.__name__,
        "mean_activation": mean_activation,
        "anomaly": has_nan or is_exploding
    }


######################
### Step Debugging ###
######################
def step_debug_hook(module, input, output):
    """Pauses execution at this layer for manual debugging."""
    print(f"Paused at layer: {module.__class__.__name__}")
    print(f"Input shape: {input[0].shape}, Output shape: {output.shape}")

    # Wait for user input before continuing
    input("Press Enter to continue...")
