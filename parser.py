import lark
from lark import Tree, Transformer, Token
import numpy as np
import os
import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp
from plugins import LAYER_PLUGINS


grammar = r"""

    start: network | research  # Allow parsing either networks or research files


    # Top-level network definition - defines the structure of an entire neural network
    network: "network" NAME "{" input_layer layers loss optimizer config* "}"

    # Configuration can be either training-related or execution-related
    config: training_config | execution_config
    
    # Input layer definition - specifies the shape of input data
    input_layer: "input:" "(" shape ")"
    # Shape can contain multiple dimensions, each being a number or None
    shape: number_or_none ("," number_or_none)*
    # Dimensions can be specific integers or None (for variable dimensions)
    number_or_none: INT | "None"

    # Layers section - contains all layer definitions separated by newlines
    layers: "layers:" _NL* layer (_NL+ layer)* _NL*

    # All possible layer types that can be used in the network
    layer: conv2d_layer              # Convolutional layers
         | max_pooling2d_layer       # Pooling layers
         | dropout_layer             # Regularization
         | flatten_layer             # Reshaping
         | dense_layer              # Fully connected
         | output_layer             # Network output
         | batch_norm_layer         # Normalization layers
         | layer_norm_layer
         | instance_norm_layer
         | group_norm_layer
         | recurrent_layer          # RNN-based layers
         | attention_layer          # Attention mechanisms
         | transformer_layer        # Transformer architecture
         | residual_layer          # Skip connections
         | inception_layer         # Inception modules
         | capsule_layer          # Capsule networks
         | squeeze_excitation_layer # SE blocks
         | graph_conv_layer       # Graph convolutions
         | embedding_layer        # Embeddings
         | quantum_layer         # Quantum computing layers
         | dynamic_layer         # Dynamic/adaptive layers

    # Output layer parameters
    output_layer: "Output(" units_param "," activation_param ")"
    units_param: "units=" INT
    activation_param: "activation=" ESCAPED_STRING
         
    # Convolutional layer parameters
    conv2d_layer: "Conv2D(" filters_param "," kernel_param "," activation_param ")"
    filters_param: "filters=" INT
    kernel_param: "kernel_size=" "(" INT "," INT ")"
    
    # Pooling layer parameters
    max_pooling2d_layer: "MaxPooling2D(" pool_param ")"
    pool_param: "pool_size=" "(" INT "," INT ")"
    
    # Dropout layer for regularization
    dropout_layer: "Dropout(" rate_param ")"
    rate_param: "rate=" FLOAT
    
    # Normalization layers
    batch_norm_layer: "BatchNormalization(" ")"
    layer_norm_layer: "LayerNormalization(" ")"
    instance_norm_layer: "InstanceNormalization(" ")"
    group_norm_layer: "GroupNormalization(" groups_param ")"
    groups_param: "groups=" INT

    # Basic layer types
    dense_layer: "Dense(" units_param "," activation_param ")"
    flatten_layer: "Flatten(" ")"
    
    # Recurrent layers section - includes all RNN variants
    recurrent_layer: lstm_layer
                   | gru_layer
                   | simple_rnn_layer
                   | cudnn_lstm_layer
                   | cudnn_gru_layer
                   | rnn_cell_layer
                   | lstm_cell_layer
                   | gru_cell_layer
                   | dropout_wrapper_layer

    # Different types of RNN layers with optional return sequences parameter
    lstm_layer: "LSTM(" "units=" INT ["," "return_sequences=" return_sequences] ")"
    gru_layer: "GRU(" "units=" INT ["," "return_sequences=" return_sequences] ")"
    simple_rnn_layer: "SimpleRNN(" units_param ["," return_sequences_param] ")"
    cudnn_lstm_layer: "CuDNNLSTM(" units_param ["," return_sequences_param] ")"
    cudnn_gru_layer: "CuDNNGRU(" units_param ["," return_sequences_param] ")"

    # RNN cell variants - single-step RNN computations
    rnn_cell_layer: "RNNCell(" INT ")"
    lstm_cell_layer: "LSTMCell(" INT ")"
    gru_cell_layer: "GRUCell(" INT ")"

    # Dropout wrapper layers for RNNs
    dropout_wrapper_layer: simple_rnn_dropout | gru_dropout | lstm_dropout
    simple_rnn_dropout: "SimpleRNNDropoutWrapper" "(" INT "," dropout_param ")"
    gru_dropout: "GRUDropoutWrapper" "(" INT "," dropout_param ")"
    lstm_dropout: "LSTMDropoutWrapper" "(" INT "," dropout_param ")"
    dropout_param: "dropout=" FLOAT
    
    # Return sequences parameter for RNN layers
    return_sequences_param: "return_sequences=" return_sequences
    return_sequences: "true" -> true
                   | "false" -> false

    # Attention and transformer mechanisms
    attention_layer: "Attention" "(" ")"
    transformer_layer: "TransformerEncoder" "(" heads_param "," dim_param ")"
    heads_param: "num_heads=" INT
    dim_param: "ff_dim=" INT

    # Advanced architecture layers
    residual_layer: "ResidualConnection" "(" ")"
    inception_layer: "InceptionModule" "(" ")"
    capsule_layer: "CapsuleLayer" "(" ")"
    squeeze_excitation_layer: "SqueezeExcitation" "(" ")"
    graph_conv_layer: "GraphConv" "(" ")"
    embedding_layer: "Embedding" "(" input_dim_param "," output_dim_param ")"
    input_dim_param: "input_dim=" INT
    output_dim_param: "output_dim=" INT
    
    # Special purpose layers
    quantum_layer: "QuantumLayer" "(" ")"
    dynamic_layer: "DynamicLayer" "(" ")"

    # Training configuration block
    training_config: "train" "{" [epochs_param] [batch_size_param] "}"
    epochs_param: "epochs:" INT
    batch_size_param: "batch_size:" INT
    
    # Loss and optimizer specifications
    loss: "loss:" ESCAPED_STRING
    optimizer: "optimizer:" ESCAPED_STRING

    # Execution environment configuration
    execution_config: "execution" "{" device_param "}"
    device_param: "device:" ESCAPED_STRING

    # Research-specific configurations
    research: "research" NAME "{" metrics references? "}"
    # Metrics tracking
    metrics: "metrics" "{" [accuracy_param] [loss_param] [precision_param] [recall_param] "}"
    accuracy_param: "accuracy:" FLOAT
    loss_param: "loss:" FLOAT
    precision_param: "precision:" FLOAT
    recall_param: "recall:" FLOAT
    # Paper references
    references: "references" "{" paper_param+ "}"
    paper_param: "paper:" ESCAPED_STRING

    # Import common tokens from Lark
    %import common.NEWLINE -> _NL        # Newline characters
    %import common.CNAME -> TRANSFORMERENCODER
    %import common.WS_INLINE             # Inline whitespace
    %import common.BOOL                  # Boolean values
    %import common.CNAME -> NAME         # Names/identifiers
    %import common.INT                   # Integer numbers
    %import common.FLOAT                 # Floating point numbers
    %import common.ESCAPED_STRING        # Quoted strings
    %import common.WS                    # All whitespace
    %ignore WS                          # Ignore whitespace in processing
"""


parser = lark.Lark(grammar, start='network', parser='lalr')
# Creating a separate parser instance for research files
research_parser = lark.Lark(grammar, start='research', parser='lalr')


class ModelTransformer(lark.Transformer):
    def layer(self, items):
        """
                Process a layer in the neural network model.

                This method extracts information about a single layer from the parsed items
                and returns a dictionary containing the layer's type and parameters.

                Parameters:
                items (list): A list containing the parsed information for the layer.
                            The first item is expected to be a dictionary with layer information.

                Returns:
                dict: A dictionary containing two keys:
                    'type': The type of the layer (e.g., 'Conv2D', 'Dense', etc.)
                    'params': A dictionary containing all the parameters for the layer
        """
        layer_info = items[0]  # Extract first item (which should be the layer type)

        # Ensure `layer_info` is a dictionary and contains 'type'
        if isinstance(layer_info, dict):
            layer_type = layer_info.get("type")  # Extract the layer type (string)
        elif isinstance(layer_info, lark.Tree):
            layer_type = layer_info.data  # Extract type from Tree structure
        else:
            layer_type = str(layer_info)  # Fallback: Convert to string if it's neither

        # Check if the extracted layer type exists in the plugin registry
        if layer_type in LAYER_PLUGINS:
            return LAYER_PLUGINS[layer_type](items[1:])  # Pass remaining parameters

        # If it's a dictionary with a "type" key, return as is
        if isinstance(layer_info, dict) and "type" in layer_info:
            return layer_info

        # If it's a lark Tree, extract parameters
        if isinstance(layer_info, lark.Tree):
            return {
                'type': layer_type,
                'params': layer_info.children[0] if layer_info.children else {}
            }

        # If it's neither a dictionary nor a Tree, assume it's a simple type
        return {
            'type': layer_type,
            'params': items[1] if len(items) > 1 else {}
        }

    
    def input_layer(self, items):
    # items[0] is the result from the 'shape' rule, a tuple.
        return {
            'type': 'Input',
            'shape': items[0]
        }

    def conv2d_layer(self, items):
        return {
            "type": "Conv2D",
            "filters": int(self.extract_value(items[0])),
            "kernel_size": (
                int(self.extract_value(items[1])),
                int(self.extract_value(items[2]))
            ),
            "activation": self.extract_value(items[3]).strip('"')
        }
    

    
    def extract_value(self, item):
        """Recursively extract a value from a Token or Tree."""
        if isinstance(item, Token):
            return item.value
        elif isinstance(item, Tree):
            # If the tree has children, return the value from the first child.
            if item.children:
                return self.extract_value(item.children[0])
            else:
                return ""
        else:
            return str(item)



    def dense_layer(self, items):
        return {
            "type": "Dense",
            "units": int(self.extract_value(items[0])),
            "activation": self.extract_value(items[1]).strip('"')
        }
    
    def output_layer(self, items):
        """
        Parses and processes the configuration of an output layer in a neural network.

        Parameters:
        items (list): A list containing the parsed information for the output layer.
                      The list should contain two elements:
                      - The number of units for the output layer.
                      - The activation function for the output layer as a string.

        Returns:
        dict: A dictionary containing the following keys:
              'type': The type of the layer, which is 'Output'.
              'shape': A tuple representing the shape of the output data, containing the number of units.
              'units': The number of units for the output layer.
              'activation': The activation function for the output layer.
        """
        units = int(items[0])
        return {
                'type': 'Output',
                'shape': (units,),  # Output shape is a tuple with the number of units
                'units': int(items[0]),
                'activation': items[1].strip('"')
            }
    
    def loss(self, items):
        return {
            'type': 'Loss',
            'value': items[0]
        }
    
    def optimizer(self, items):
        return {'type': 'Optimizer',
                'value': items[0]
            }
    
    def layers(self, items):
        parsed_layers = []
        for item in items:
            if isinstance(item, lark.Tree):
                layer_data = self.layer(item)
                parsed_layers.append(layer_data)
            else:
                parsed_layers.append(item)
        return {
            'type': 'Layers',
            'layers': parsed_layers
        }
    
    def dropout_layer(self, items):
        return {
            "type": "Dropout",
            "rate": float(self.extract_value(items[0]))
        }

    
    def flatten_layer(self, items):
        return {'type': 'Flatten'}
    
    def training_config(self, items):
        return {
            'type': 'TrainingConfig',
            'epochs': int(items[0]['epochs']) if 'epochs' in items[0] else None,
            'batch_size': int(items[0]['batch_size']) if 'batch_size' in items[0] else None
        }
    
    def shape(self, items):
        """
        Convert the list of tokens (or strings) into a tuple.
        If a token equals "None", convert it to Python's None.
        Otherwise, convert to integer.
        """
        result = []
        for item in items:
            # item is a Token, so convert it to a string.
            val = str(item)
            if val == "None":
                result.append(None)
            elif val.isdigit() or (val.startswith("-") and val[1:].isdigit()):
                result.append(int(val))
            elif val.isdigit():
                result.append(int(val))
            elif val.startswith("[") and val.endswith("]"):
                result.append(self.shape(val[1:-1].split(", ")))  # Recursive call for nested lists
            elif val.startswith("<") and val.endswith(">"):
                result.append(self.shape(val[1:-1].split(", ")))  # Recursive call for nested sets
            elif val.startswith("(") and val.endswith(")"):
                result.append(self.shape(val[1:-1].split(", ")))  # Recursive call for nested tuples
            elif val.startswith("{") and val.endswith("}"):
                result.append(self.shape(val[1:-1].split(", ")))  # Recursive call for nested dictionaries
                # Note: This assumes that the items in the dictionary are always simple types (not nested dictionaries)
            else:
                result.append(val)
        return tuple(result)


    def max_pooling2d_layer(self, items):
        return {
            "type": "MaxPooling2D",
            "pool_size": (
                int(self.extract_value(items[0])),
                int(self.extract_value(items[1]))
            )
        }
    

    def batch_norm_layer(self, items):
        return {'type': 'BatchNormalization'}

    def layer_norm_layer(self, items):
        return {'type': 'LayerNormalization'}

    def instance_norm_layer(self, items):
        return {'type': 'InstanceNormalization'}

    def group_norm_layer(self, items):
        return {
            "type": "GroupNormalization",
            "groups": int(self.extract_value(items[0]))
        }

    #### Recurrent Layers ###################################
    
    
    def recurrent_layer(self, items):
        """
        Process a recurrent layer definition and return a dictionary containing its parameters.
        
        Expected minimal signature:
        [<TYPE token>, <INT token>]
        For additional parameters, they appear at fixed positions.
        """



        def _get_value(items, idx, conv=lambda x: x, default=None):
            """Safely extract and convert the value at index idx from items.
            
            If the item has a .value attribute (as tokens do), use it; otherwise, use str(item).
            """
            try:
                token = items[idx]
                val = token.value if hasattr(token, "value") else str(token)
                return conv(val)
            except (IndexError, ValueError):
                return default
        
        # Extract the number of units from the items.
        units = _get_value(items, 1, int, None)

        # Extract the layer type as a string.
        if hasattr(items[0], "value"):
            layer_type = str(items[0].value)
        elif isinstance(items[0], lark.Tree):
            # Map tree data to uppercase; adjust as needed.
            mapping = {
                "lstm_layer": "LSTM",
                "gru_layer": "GRU",
                "simplernn_layer": "SimpleRNN",
                "simplernncell_layer": "SimpleRNNCell",
                "grucell_layer": "GRUCell",
                "lstmcell_layer": "LSTMCell",
                "cudnnlstm_layer": "CuDNNLSTM",
                "cudnngru_layer": "CuDNNGRU",
                "cudnnsimplernn_layer": "CuDNNSimpleRNN",
                "rnn_layer": "RNN",
            }
            layer_type = mapping.get(items[0].data, items[0].data)
        else:
            layer_type = str(items[0])
        
        # For basic units, extract from index 1.
        units = _get_value(items, 1, int, None)
        
        # Group similar layer types:
        basic = {"LSTM", "GRU", "SimpleRNN", "CuDNNLSTM", "CuDNNGRU", "CuDNNSimpleRNN", "RNN"}
        cell_variants = {"SimpleRNNCell", "GRUCell", "LSTMCell"}
        dropout_wrappers = {"SimpleRNNDropoutWrapper", "GRUDropoutWrapper", "LSTMDropoutWrapper",
                            "GRUCellDropoutWrapper", "LSTMCellDropoutWrapper"}
        projection = {"BasicLSTMCellWithProjection", "BasicGRUCellWithProjection",
                    "BasicRNNCellWithProjection", "BasicRNNCellWithInputProjection",
                    "GRUCellWithInputProjection"}
        complex_variants = {
            "BasicRNNCellWithInputProjectionAndPeephole",
            "BasicRNNCellWithInputProjectionAndPeepholeAndProjection",
            "BasicRNNCellWithInputProjectionAndPeepholeAndProjectionAndHiddenSize",
            "BasicRNNCellWithInputProjectionAndPeepholeAndProjectionAndHiddenSizeAndNumLayers",
            "BasicRNNCellWithInputProjectionAndPeepholeAndProjectionAndHiddenSizeAndNumLayersAndDropout",
            "BasicRNNCellWithInputProjectionAndPeepholeAndProjectionAndHiddenSizeAndNumLayersAndDropoutAndkernel_initializer"
        }
        basic_cells = {"BasicLSTMCell", "BasicGRUCell", "BasicRNNCell", "BasicLSTMCellWithPeephole", "BasicRNNCellWithPeephole"}
        
        # Helper: if minimal (only two items), then return basic result.
        if len(items) == 2:
            return {"type": layer_type, "units": units}
        
        # For layers in 'basic' group: expect an optional return_sequences flag at index 2.
        if layer_type in basic:
            # Only add return_sequences if provided.
            if len(items) >= 3:
                return_seq = _get_value(items, 2, lambda x: x, False)
                return {"type": layer_type, "units": units, "return_sequences": return_seq}
            else:
                return {"type": layer_type, "units": units}
        
        # For cell variants: expect return_sequences at index 3.
        elif layer_type in cell_variants:
            if len(items) >= 4:
                return_seq = _get_value(items, 3, lambda x: x, False)
                return {"type": layer_type, "units": units, "return_sequences": return_seq}
            else:
                return {"type": layer_type, "units": units}
        
        # For dropout wrappers: expect return_sequences at index 2 and dropout (float) at index 3.
        elif layer_type in dropout_wrappers:
            if len(items) >= 4:
                return_seq = _get_value(items, 2, lambda x: x, False)
                dropout = _get_value(items, 3, float, None)
                return {"type": layer_type, "units": units, "return_sequences": return_seq, "dropout": dropout}
            else:
                return {"type": layer_type, "units": units}
        
        # For projection variants: expect projection_size at index 2.
        elif layer_type in projection:
            projection_size = _get_value(items, 2, int, None)
            return {"type": layer_type, "units": units, "projection_size": projection_size}
        
        # For complex variants: expect projection_size (idx2), hidden_size (idx3), num_layers (idx4),
        # dropout (idx5), kernel_initializer (idx6)
        elif layer_type in complex_variants:
            result = {"type": layer_type, "units": units}
            result["projection_size"] = _get_value(items, 2, int, None)
            result["hidden_size"] = _get_value(items, 3, int, None)
            result["num_layers"] = _get_value(items, 4, int, None)
            result["dropout"] = _get_value(items, 5, float, None)
            result["kernel_initializer"] = _get_value(items, 6, lambda x: x.value, None)
            return result
        
        # For Attention wrappers (special-case):
        elif layer_type.startswith("AttentionWrapper"):
            # For simplicity, assume:
            # items[1]: attention_layer (processed via self.layer_norm_layer)
            # items[2]: attention_layer_size (INT)
            # items[3]: attention_type (optional; if not present, default to "bahdanau")
            attn_type = _get_value(items, 3, lambda x: x.value if hasattr(x, "value") else x, "bahdanau")
            return {
                "type": layer_type,
                "attention_layer": self.layer_norm_layer(items[1]),
                "attention_layer_size": _get_value(items, 2, int, None),
                "attention_type": attn_type
            }
        
        # For basic cells:
        elif layer_type in basic_cells:
            return {"type": layer_type, "units": units}
        
        # Fallback: return basic info.
        else:
            return {"type": layer_type, "units": units}

    #### Recurrent Layers #################################


    def attention_layer(self, items):
        return {'type': 'Attention'}

    def transformer_layer(self, items):
        return {'type': 'TransformerEncoder', 'num_heads': int(items[0]), 'ff_dim': int(items[1])}

    def residual_layer(self, items):
        return {'type': 'ResidualConnection'}

    def inception_layer(self, items):
        return {'type': 'InceptionModule'}

    def capsule_layer(self, items):
        return {'type': 'CapsuleLayer'}

    def squeeze_excitation_layer(self, items):
        return {'type': 'SqueezeExcitation'}

    def graph_conv_layer(self, items):
        return {'type': 'GraphConv'}

    def embedding_layer(self, items):
        return {'type': 'Embedding', 'input_dim': int(items[0]), 'output_dim': int(items[1])}

    def quantum_layer(self, items):
        return {
            'type': 'QuantumLayer'
            }

    def dynamic_layer(self, items):
        return {'type': 'DynamicLayer'}
    
    def lstm_layer(self, items):
        # items[0] is the INT token for units.
        # Optional items[1] is the return_sequences token.
        units = int(self.extract_value(items[0]))
        if len(items) > 1:
            rs = self.extract_value(items[1]).lower()
            return_sequences = (rs == "true")
        else:
            return_sequences = False
        return {"type": "LSTM", "units": units, "return_sequences": return_sequences}

    def gru_layer(self, items):
        units = int(self.extract_value(items[0]))
        if len(items) > 1:
            rs = self.extract_value(items[1]).lower()
            return_sequences = (rs == "true")
        else:
            return_sequences = False
        return {"type": "GRU", "units": units, "return_sequences": return_sequences}
    def transformer_layer(self, items):
        return {
            "type": "TransformerEncoder",
            "num_heads": int(self.extract_value(items[0])),
            "ff_dim": int(self.extract_value(items[1]))
        }

    def network(self, items):
        name = str(items[0])
        input_shape = items[1]['shape']  # Input layer configuration
        layers = items[2]['layers']  # Layers configuration
        # Use the separate output layer if provided
        output_layer = next((item for item in items if isinstance(item, dict) and item.get('type') == 'Output'), None)
        
        # If no explicit output layer was found, check if there's an output layer in the layers
        if output_layer is None:
            output_layer = next((layer for layer in reversed(layers) if layer['type'] == 'Output'), None)
        
        # If still no output layer, use a default
        if output_layer is None:
            output_layer = {
                'type': 'Output', 
                'units': 1, 
                'activation': 'linear'
            }
        
        # Determine output shape
        if 'shape' in output_layer:
            output_shape = output_layer['shape']
        elif 'units' in output_layer:
            output_shape = (output_layer['units'],)
        else:
            output_shape = (1,)
        
        # Find loss and optimizer
        loss = next((item for item in items if isinstance(item, dict) and item.get('type') == 'Loss'), None)
        optimizer = next((item for item in items if isinstance(item, dict) and item.get('type') == 'Optimizer'), None)
        
        # The rest are configuration blocks (could be training_config and/or execution_config)
        training_config = None
        execution_config = {"device": "auto"}  # default
        # items[5:] contains all config blocks
        for conf in items[5:]:
            # We'll assume training_config returns a dict with type "TrainingConfig"
            # and execution_config returns a dict with type "ExecutionConfig"
            if conf == "TrainingConfig":
                training_config = conf
            elif conf == "ExecutionConfig":
                execution_config = conf

        # Construct the neural network configuration dictionary
        # This includes the name, input shape, layers, output layer, output shape, loss, optimizer, and training configuration
        # If execution configuration is provided, it will be included as well.
        # If no explicit output layer was found, it will be added to the layers list.
        # If no loss or optimizer is found, default values will be used.
        # If no training configuration is found, default values will be used.

        return {
            'type': 'model',
            'name': name,
            'input_shape': input_shape,
            'layers': layers,
            'output_layer': output_layer,
            'output_shape': output_shape,
            'loss': loss,
            'optimizer': optimizer,
            'training_config': training_config,
            'execution_config': execution_config
        }

    def research(self,items):
        return{
            "type": "research",
            "name": str(items[0]),
            "metrics": items[1],
            "references": items[2] if len(items) > 2 else None
            
        }

def propagate_shape(input_shape, layer):
    """
    Propagates the input shape through a neural network layer to calculate the output shape.

    This function determines the output shape of a layer given its input shape and layer configuration.
    It supports Conv2D, Flatten, Dense, and Dropout layers.

    Parameters:
    input_shape (tuple): The shape of the input to the layer. For Conv2D, it should be a 3-tuple (height, width, channels).
    layer (dict): A dictionary containing the layer configuration. Must include a 'type' key specifying the layer type.

    Returns:
    tuple: The output shape of the layer after processing the input.

    Raises:
    TypeError: If the layer parameter is not a dictionary.
    KeyError: If the layer dictionary is missing required keys.
    ValueError: If the layer type is unsupported or if the input shape is invalid for the given layer type.
    """
    # Defensive type checking
    if not isinstance(layer, dict):
        raise TypeError(f"Layer must be a dictionary, got {type(layer)}")

    # Extract actual layer type, handling nested dictionary case
    layer_type = layer['type'] if isinstance(layer['type'], str) else layer['type']['type']

    print("Layer Type:", layer_type)
    print("Input Shape:", input_shape)
    print("Full Layer:", layer)

    # Helper to check input shape is 3D for convolution and pooling layers
    def check_3d_input(layer_type):
        if len(input_shape) != 3:
            raise ValueError(f"{layer_type} requires 3D input shape (h,w,c), got {input_shape}")
        return input_shape[0], input_shape[1], input_shape[2]

    if layer_type == 'Conv2D':
        # Handle both direct and nested dictionary cases for layer parameters
        if isinstance(layer['type'], dict):
            filters = layer['type']['filters']
            kernel_size = layer['type']['kernel_size']
        else:
            filters = layer.get('filters')
            kernel_size = layer.get('kernel_size')

        h, w, c = check_3d_input('Conv2D')
        
        # Validate required parameters
        if filters is None or kernel_size is None:
            raise KeyError("Conv2D layer missing required parameters")

        kernel_h, kernel_w = kernel_size

        # Simple shape calculation without padding (assuming valid padding)
        return (h - kernel_h + 1, w - kernel_w + 1, filters)

    elif layer_type == 'MaxPooling2D':
        h, w, c = check_3d_input('MaxPooling2D')
        
        # Handle both direct and nested dictionary cases for pool_size
        if isinstance(layer['type'], dict):
            pool_size = layer['type']['pool_size']
        else:
            pool_size = layer.get('pool_size')
        
        # Validate required parameters
        if pool_size is None:
            raise KeyError("MaxPooling2D layer missing pool_size parameter")
        
        pool_h, pool_w = pool_size
        return (h // pool_h, w // pool_w, c)

    elif layer_type == 'Flatten':
        # Flatten converts multi-dimensional input to 1D
        return (np.prod(input_shape),)

    elif layer_type == 'Dense':
        # Handle both direct and nested dictionary cases for units
        if isinstance(layer['type'], dict):
            units = layer['type'].get('units')
        else:
            units = layer.get('units')

        # Validate required parameters
        if units is None:
            raise KeyError("Dense layer missing units parameter")
        return (units,)

    elif layer_type == 'Dropout':
        # Dropout doesn't change the shape
        return input_shape

    elif layer_type == 'Output':
        # Handle both direct and nested dictionary cases for units
        if isinstance(layer['type'], dict):
            units = layer['type'].get('units')
        else:
            units = layer.get('units')

        if units is None:
            raise KeyError("Output layer missing units parameter")
        return (units,)

    else:
        raise ValueError(f"Unsupported layer type: {layer_type}") 


class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers, n_features):
        """
        n_qubits: Number of quantum bits to use.
        n_layers: Number of variational layers in the quantum circuit.
        n_features: Dimension of the input features per sample.
        """
        super(QuantumLayer, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Define a quantum device on the CPU or GPU via PennyLane's default.qubit device.
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # The total number of parameters in the variational circuit.
        self.weight_shapes = {"weights": (n_layers, n_qubits, 3)}

        # Create a QNode that wraps our quantum circuit.
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            # Encode classical data into the quantum state.
            # For simplicity, we embed each feature onto a qubit using an RY gate.
            for i in range(self.n_qubits):
                if i < len(inputs):
                    qml.RY(inputs[i], wires=i)
                else:
                    qml.RY(0.0, wires=i)

            # Apply a variational circuit (multiple layers of rotations).
            for layer in range(self.n_layers):
                for qubit in range(self.n_qubits):
                    qml.RX(weights[layer, qubit, 0], wires=qubit)
                    qml.RY(weights[layer, qubit, 1], wires=qubit)
                    qml.RZ(weights[layer, qubit, 2], wires=qubit)
                # Entangle adjacent qubits with CNOTs
                for qubit in range(self.n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
                # Optionally, add a CNOT from the last to the first qubit to create a ring entanglement.
                qml.CNOT(wires=[self.n_qubits - 1, 0])

            # Measure expectation values on each qubit.
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

        # Register the quantum circuit as a torch parameter.
        self.weights = nn.Parameter(torch.randn(self.weight_shapes["weights"]))

        # Save the QNode function
        self.circuit = circuit

        # A final linear layer to map the quantum outputs to the desired feature size.
        self.fc = nn.Linear(n_qubits, n_features)

    def forward(self, x):
        # x is expected to be of shape (batch_size, n_features_in)
        # We will process each sample individually.
        outputs = []
        for sample in x:
            # For each sample, convert it to a 1D tensor (if not already).
            sample = sample.flatten()
            # Run the quantum circuit on the sample with current weights.
            q_out = self.circuit(sample, self.weights)
            q_out = torch.stack(q_out)  # Convert list to tensor.
            # Map the quantum outputs to desired features.
            outputs.append(self.fc(q_out))
        # Stack all outputs into a batch tensor.
        return torch.stack(outputs)

class DynamicLayer(nn.Module):
    def __init__(self, threshold=1.0, branch1_units=64, branch2_units=32):
        """
        threshold: A value used to decide which branch to activate.
        branch1_units: Number of output features for branch 1.
        branch2_units: Number of output features for branch 2.
        """
        super(DynamicLayer, self).__init__()
        self.threshold = threshold

        # Define two branches as simple feedforward layers.
        self.branch1 = nn.Sequential(
            nn.Linear(128, branch1_units),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Linear(128, branch2_units),
            nn.ReLU()
        )
        # Final layer to map the branch outputs to a common dimension.
        # Here we choose the maximum of branch outputs as the final dimension.
        self.final_dim = max(branch1_units, branch2_units)
        self.final_layer = nn.Linear(self.final_dim, 128)

    def forward(self, x):
        """
        x: Expected shape (batch_size, 128)
        Decision logic: if the mean absolute value of the sample is greater than threshold,
        use branch1; otherwise, use branch2.
        """
        # Compute a simple statistic per sample
        branch_outputs = []
        for sample in x:
            if sample.abs().mean() > self.threshold:
                out = self.branch1(sample)
            else:
                out = self.branch2(sample)
            branch_outputs.append(out)
        # Stack outputs into a batch tensor
        branch_outputs = torch.stack(branch_outputs)
        # If branch outputs have different dimensions, we pad the smaller ones (for demonstration)
        if branch_outputs.shape[1] != self.final_dim:
            # A simple padding example (in practice, better design the branches to output the same dims)
            padded = torch.zeros(x.size(0), self.final_dim, device=x.device)
            padded[:, :branch_outputs.shape[1]] = branch_outputs
            branch_outputs = padded
        # Pass through a final mapping
        return self.final_layer(branch_outputs)


def generate_code(model_data, backend="tensorflow"):
    """
    This function generates code for creating and training a neural network model based on the given model data and backend.

    Parameters:
    model_data (dict): A dictionary containing the model configuration and data. It should have the following keys:
                       - 'input': A dictionary containing the input shape of the model. It should have a 'shape' key.
                       - 'layers': A list of dictionaries, each representing a layer in the model. Each layer dictionary should have a 'type' key.
                       - 'loss': A dictionary containing the loss function for the model. It should have a 'value' key.
                       - 'optimizer': A dictionary containing the optimizer for the model. It should have a 'value' key.
                       - 'training_config' (optional): A dictionary containing the training configuration for the model. It should have 'epochs' and 'batch_size' keys.

    backend (str): The backend framework to use for generating the code. It can be either 'tensorflow' or 'pytorch'.

    Returns:
    str: The generated code for creating and training the neural network model based on the given model data and backend.

    Raises:
    ValueError: If the specified backend is not supported.
    """
    if backend == "tensorflow":
        code = "import tensorflow as tf\n"
        code += "model = tf.keras.Sequential([\n"
        for layer in model_data['layers']:
            if layer['type'] == 'Conv2D':
                code += f"    tf.keras.layers.Conv2D({layer['filters']}, {layer['kernel_size']}, activation='{layer['activation']}'),\n"
            elif layer['type'] == 'Dense':
                code += f"    tf.keras.layers.Dense({layer['units']}, activation='{layer['activation']}'),\n"
            elif layer['type'] == 'Flatten':
                code += "    tf.keras.layers.Flatten(),\n"
            elif layer['type'] == 'Dropout':
                code += f"    tf.keras.layers.Dropout({layer['rate']}),\n"
        code += "])\n"

        # Remove quotes from optimizer and loss values
        optimizer = str(model_data['optimizer']['value']).strip('"')
        loss = str(model_data['loss']['value']).strip('"')
        code += f"model.compile(loss='{loss}', optimizer='{optimizer}')\n"

        # Extract training configuration
        if 'training' in model_data and model_data['training_config']:
            epochs = model_data['training_config'].get('epochs', 10) # Default to 10 epochs
            batch_size = model_data['training_config'].get('batch_size', 32) # Default to batch size of 32
            code += f"model.fit(data, epochs={epochs}, batch_size={batch_size})\n"
        return code

    elif backend == "pytorch":
        # PyTorch code generation logic 
        code = "import torch\n"
        code += "class Model(torch.nn.Module):\n"
        code += "    def __init__(self):\n"
        code += "        super(Model, self).__init__()\n"
        input_shape = model_data['input']['shape']
        model_layers = []
        for layer in model_data['layers']:
            output_shape = propagate_shape(input_shape, layer)
            if layer['type'] == 'Conv2D':
                code += f"        self.conv = torch.nn.Conv2d({input_shape[2]}, {layer['filters']}, {layer['kernel_size']}, padding=1)\n"
                code += f"        self.relu = torch.nn.ReLU()\n"
                input_shape = output_shape
                code += f"        self.pool = torch.nn.MaxPool2d({layer['kernel_size']})\n"
            elif layer['type'] == 'Flatten':
                code += "        self.flatten = torch.nn.Flatten()\n"
                input_shape = output_shape
                code += f"        self.fc = torch.nn.Linear({np.prod(input_shape)}, {layer['units']})\n"
                code += f"        self.relu = torch.nn.ReLU()\n"
                input_shape = output_shape
                code += f"        self.softmax = torch.nn.Softmax(dim=1)\n"
                code += f"    def forward(self, x):\n"
                code += f"        x = self.conv(x)\n"
                code += f"        x = self.relu(x)\n"
                code += f"        x = self.pool(x)\n"
                code += f"        x = self.flatten(x)\n"
                code += f"        x = self.fc(x)\n"
                code += f"        x = self.relu(x)\n"
                code += f"        x = self.softmax(x)\n"
                code += f"        return x\n"
            elif layer["type"] == "QuantumLayer":
                # Instantiate the QuantumLayer with chosen hyperparameters.
                model_layers.append("QuantumLayer(n_qubits=4, n_layers=2, n_features=128)")
            elif layer["type"] == "DynamicLayer":
                model_layers.append("DynamicLayer(threshold=0.5, branch1_units=64, branch2_units=32)")
        code += "model = Model().to(device)\n"
        code += f"model.to('{backend}')')\n"
        code += f"loss_fn = torch.nn.CrossEntropyLoss()\n"
        code += f"optimizer = torch.optim.{model_data['optimizer']['value']}()\n"


        if 'training_config' in model_data:
            code += f"for epoch in range({model_data['training_config']['epochs']}):\n"
            code += "    for batch_idx, (data, target) in enumerate(data):\n"
            code += "        optimizer.zero_grad()\n"
            code += "        output = model(data)\n"
            code += "        loss = loss_fn(output, target)\n"
            code += "        loss.backward()\n"
            code += "        optimizer.step()\n"
            code += "print('Finished Training')"
        return code
    else:
        raise ValueError("Unsupported backend")
    return code

def load_file(filename):
    """
    Reads a file and categorizes it as a model definition file or a research file based on its extension.
    
    Returns:
        tuple: (file_type, parsed_tree)
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found.")
    
    file_ext = os.path.splitext(filename)[-1].lower()
    with open(filename, "r") as f:
        content = f.read()
    
    if file_ext in [".neural", ".nr"]:
        print(f"✅ Loading Model Definition from {filename}...")
        # Parse with network start rule
        return "model", parser.parse(content, start="network")
    elif file_ext == ".rnr":
        print(f"✅ Loading Research Report from {filename}...")
        # Parse with research start rule
        return "research", research_parser.parse(content, start="research")
    else:
        raise ValueError(f"Unsupported file extension: {file_ext}")
