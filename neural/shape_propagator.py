import os
import numpy as np
import lark
from numbers import Number
import plotly.graph_objects as go
from typing import Any, Dict, List, Tuple, Union, Optional, Callable
from neural.parser import create_parser

def NUMBER(x):
    try:
        return int(x)
    except ValueError:
        return float(x)

def propagate_shape(input_shape: Tuple[Optional[int], ...], layer: Dict[str, Any]) -> Tuple[Optional[int], ...]:
    """
    Propagates the input shape through a neural network layer to calculate the output shape.
    Supports various layer types and handles shape transformations accordingly.

    Parameters:
    input_shape (tuple): The shape of the input to the layer.
    layer (dict): A dictionary containing the layer configuration, including 'type' and 'params'.

    Returns:
    tuple: The output shape of the layer.

    Raises:
    TypeError: If layer is not a dictionary or input_shape is not a tuple.
    ValueError: If layer type is unsupported, or layer parameters are invalid, or input shape is incompatible.
    """
    if not isinstance(layer, dict):
        raise TypeError(f"Layer must be a dictionary, got {type(layer)}")
    if not isinstance(input_shape, tuple):
        raise TypeError(f"Input shape must be a tuple, got {type(input_shape)}")

    layer_type = layer.get('type')
    if not layer_type:
        raise ValueError("Layer dictionary must contain a 'type' key.")

    layer_type = str(layer_type) # Ensure layer_type is string


    def validate_input_dims(expected_dims: NUMBER, layer_name: str):
        """Helper function to validate input dimensions."""
        if len(input_shape) != expected_dims:
            raise ValueError(f"{layer_name} layer expects {expected_dims}D input (including batch), got {len(input_shape)}D input with shape {input_shape}")

    if layer_type == 'Input':
        return layer.get('shape', input_shape) # Input layer can optionally override shape

    elif layer_type == 'Conv2D':
        validate_input_dims(4, 'Conv2D') # Expecting (batch, height, width, channels)
        params = layer.get('params', layer) # Check both 'params' and direct layer for config

        filters = params.get('filters')
        kernel_size = params.get('kernel_size', (1, 1))
        strides = params.get('strides', (1, 1)) # Default strides to 1x1
        padding_mode = params.get('padding', 'valid').lower() # Default to 'valid' padding
        dilation_rate = params.get('dilation_rate', (1, 1)) # Default dilation rate to 1x1

        if not filters:
            raise ValueError("Conv2D layer requires 'filters' parameter.")
        try:
            filters = int(filters)
            if isinstance(kernel_size, (tuple, list)):
                kernel_h, kernel_w = map(int, kernel_size)
            elif isinstance(kernel_size, int):
                kernel_h = kernel_w = kernel_size
            else:
                raise ValueError("kernel_size must be an int, tuple, or list.")
            kernel_h, kernel_w = map(NUMBER, kernel_size)
            stride_h, stride_w = map(NUMBER, strides)
            dilation_h, dilation_w = map(NUMBER, dilation_rate)
        except ValueError as e:
            raise ValueError(f"Invalid Conv2D parameter format: {e}") from e

        batch_size, in_h, in_w, in_channels = input_shape

        # Calculate output dimensions based on padding mode
        if padding_mode == 'valid':
            out_h = ((in_h - dilation_h * (kernel_h - 1) - 1) // stride_h) + 1
            out_w = ((in_w - dilation_w * (kernel_w - 1) - 1) // stride_w) + 1
        elif padding_mode == 'same':
            out_h = in_h # Output height is same as input height
            out_w = in_w # Output width is same as input width
        else:
            raise ValueError(f"Unsupported padding mode for Conv2D: {padding_mode}. Valid modes are 'valid' or 'same'.")

        return (batch_size, out_h, out_w, filters)

    elif layer_type == 'MaxPooling2D':
        validate_input_dims(4, 'MaxPooling2D') # Expecting (batch, height, width, channels)
        params = layer.get('params', layer)

        pool_size = params.get('pool_size', (2, 2))
        strides = params.get('strides', pool_size) # Default stride to pool_size if not provided
        padding_mode = params.get('padding', 'valid').lower() # Default to 'valid' padding

        if not isinstance(pool_size, (tuple, list, int)):
            raise ValueError("pool_size must be an int, tuple, or list.")
        try:
            if isinstance(pool_size, (tuple, list)):
                pool_h, pool_w = map(int, pool_size)
            elif isinstance(pool_size, int):
                pool_h = pool_w = pool_size
            else:
                raise ValueError("pool_size must be an int, tuple, or list.")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid MaxPooling2D parameter format: {e}") from e


        batch_size, in_h, in_w, in_channels = input_shape

        # Calculate output dimensions based on padding mode
        if padding_mode == 'valid':
            out_h = ((in_h - pool_h) // stride_h) + 1
            out_w = ((in_w - pool_w) // stride_w) + 1
        elif padding_mode == 'same':
            out_h = in_h // stride_h if in_h % stride_h == 0 else (in_h // stride_h) + 1 # Ceil division for 'same'
            out_w = in_w // stride_w if in_w % stride_w == 0 else (in_w // stride_w) + 1 # Ceil division for 'same'
        else:
            raise ValueError(f"Unsupported padding mode for MaxPooling2D: {padding_mode}. Valid modes are 'valid' or 'same'.")


        return (batch_size, out_h, out_w, in_channels) # Channels remain the same

    elif layer_type == 'Flatten':
        validate_input_dims(4, 'Flatten') # Expecting (batch, height, width, channels)
        batch_size = input_shape[0]
        return (batch_size, np.prod(input_shape[1:]),) # Flatten spatial dimensions, keep batch

    elif layer_type == 'Dense':
        validate_input_dims(2, 'Dense') # Expecting (batch, features_in)
        params = layer.get('params', layer)
        units = params.get('units')

        if not units:
            raise ValueError("Dense layer requires 'units' parameter.")
        try:
            units = NUMBER(units)
        except ValueError as e:
            raise ValueError(f"Invalid Dense units parameter: {e}") from e

        batch_size = input_shape[0]
        return (batch_size, units,)

    elif layer_type == 'Dropout':
        return input_shape # Dropout does not change shape

    elif layer_type == 'Output':
        validate_input_dims(2, 'Output') # Assuming output layer also takes 2D input (batch, features_in)
        params = layer.get('params', layer)
        units = params.get('units')
        if not units:
            raise ValueError("Output layer requires 'units' parameter.")
        try:
            units = NUMBER(units)
        except ValueError as e:
            raise ValueError(f"Invalid Output units parameter: {e}") from e
        batch_size = input_shape[0]
        return (batch_size, units,)

    elif layer_type in ['BatchNormalization', 'LayerNormalization', 'InstanceNormalization', 'GroupNormalization']:
        return input_shape # Normalization layers typically preserve shape

    # Recurrent Layers - Assuming input is (batch, time_steps, features) for simplicity
    elif layer_type in ['LSTM', 'GRU', 'SimpleRNN', 'CuDNNLSTM', 'CuDNNGRU', 'RNNCell', 'LSTMCell', 'GRUCell',
                        'SimpleRNNDropoutWrapper', 'GRUDropoutWrapper', 'LSTMDropoutWrapper']:
        validate_input_dims(3, layer_type) # Expecting (batch, time_steps, features)
        params = layer.get('params', layer)
        units = params.get('units')
        return_sequences = params.get('return_sequences', False)

        if not units:
            raise ValueError(f"{layer_type} layer requires 'units' parameter.")
        try:
            units = NUMBER(units)
        except ValueError as e:
            raise ValueError(f"Invalid {layer_type} units parameter: {e}") from e

        batch_size, time_steps, _ = input_shape
        if return_sequences:
            return (batch_size, time_steps, units) # Shape is (batch, time_steps, units) if return_sequences=True
        else:
            return (batch_size, units) # Shape is (batch, units) if return_sequences=False


    elif layer_type in ['Attention', 'TransformerEncoder', 'Residual', 'InceptionModule',
                        'CapsuleLayer', 'SqueezeExcitation', 'GraphConv', 'Embedding', 'QuantumLayer', 'DynamicLayer']:
        return input_shape # Placeholder for advanced layers, needs specific shape logic

    elif layer_type == 'MyCustomLayer':
        # Example: custom logic for a custom layer
        params = layer.get('params', {})
        factor = params.get('scale_factor', 1)
        # Let’s say the custom layer scales spatial dimensions by 'scale_factor'
        batch_size, h, w, channels = input_shape
        new_h = int(h * factor)
        new_w = int(w * factor)
        return (batch_size, new_h, new_w, channels)


    else:
        raise ValueError(f"Unsupported layer type: {layer_type}")
