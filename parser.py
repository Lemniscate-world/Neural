import lark
from lark import Tree, Transformer, Token
import numpy as np
import os
import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp
from plugins import LAYER_PLUGINS
from typing import Any, Dict, List, Tuple, Union, Optional

def create_parser(start_rule: str = 'network') -> lark.Lark:
    """
    Creates a Lark parser for neural network configuration files.

    Args:
        start_rule (str): The starting rule for the parser. Defaults to 'network'.

    Returns:
        lark.Lark: A Lark parser object configured with the defined grammar.
    """
    grammar = r"""

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
        %ignore WS   
        ?start: network | layer | research  # Allow parsing either networks or research files

        # File type rules
        neural_file: network
        nr_file: network
        rnr_file: research

        # Parameter styles
        named_params: named_param ("," named_param)*
        ordered_params: value ("," value)* 
        named_rate: "rate=" FLOAT
        value: INT | FLOAT | ESCAPED_STRING | tuple
        tuple: "(" WS_INLINE* INT WS_INLINE* "," WS_INLINE* INT WS_INLINE* ")"

        # name_param rules
        named_data_format: "data_format" "=" value 
        named_units: "units" "=" INT
        named_activation: "activation" "=" value
        named_filters: "filters" "=" INT
        named_kernel_size: "kernel_size" "=" value
        named_strides: "strides" "=" value
        named_padding: "padding" "=" value
        named_dilation_rate: "dilation_rate" "=" value
        named_groups: "groups" "=" INT
        named_channels: "channels" "=" INT
        named_pool_size: "pool_size" "=" value
        named_return_sequences: "return_sequences" "=" number_or_none
        named_num_heads: "num_heads" "=" INT
        named_ff_dim: "ff_dim" "=" INT
        named_input_dim: "input_dim" "=" INT
        named_output_dim: "output_dim" "=" INT  
        named_param:
                    | named_units
                    | named_activation
                    | named_filters
                    | named_kernel_size
                    | named_strides
                    | named_padding
                    | named_dilation_rate
                    | named_groups
                    | named_data_format
                    | named_channels 
                    | named_rate  // for Dropout, etc.
                    | named_pool_size // for MaxPooling
                    | named_return_sequences // for LSTM, etc.
                    | named_num_heads // for TransformerEncoder
                    | named_ff_dim // for TransformerEncoder
                    | named_input_dim // for Embedding
                    | named_output_dim // for Embedding

        # Layer parameter styles
        ?param_style1: named_params    // Dense(units=128, activation="relu")
                    | ordered_params   // Dense(128, "relu")

        # Top-level network definition - defines the structure of an entire neural network
        network: "network" NAME "{" input_layer layers loss optimizer [training_config] "}"

        # Configuration can be either training-related or execution-related
        config: training_config | execution_config

        # Input layer definition - specifies the shape of input data
        input_layer: "input:" "(" shape ")"
        # Shape can contain multiple dimensions, each being a number or None
        shape: number_or_none ("," number_or_none)*
        # Dimensions can be specific integers or None (for variable dimensions)
        number_or_none: INT | "None"

        # Layers section - contains all layer definitions separated by newlines
        layers: "layers:" _NL* layer (_NL* layer)* _NL*

        # All possible layer types that can be used in the network
        ?layer: basic_layer  | recurrent_layer | advanced_layer


        # Basic layers group
        ?basic_layer: conv2d_layer
                    | max_pooling2d_layer
                    | dropout_layer
                    | flatten_layer
                    | dense_layer
                    | output_layer
                    | norm_layer


        # Output layer parameters
        output_layer: "Output(" units_param "," activation_param ")"
        units_param: "units=" INT
        activation_param: "activation=" ESCAPED_STRING

        # Convolutional layer parameters
        conv2d_layer: "Conv2D(" (named_params | ordered_params) ")"

        # Pooling layer parameters
        max_pooling2d_layer: "MaxPooling2D(" named_params ")" # Changed to named params, was pool_param
        pool_param: "pool_size=" tuple # Removed, using named_params directly in max_pooling2d_layer

        # Dropout layer for regularization
        dropout_layer: "Dropout(" (named_rate | FLOAT) ")"

        # Normalization layers
        ?norm_layer: batch_norm_layer
                    | layer_norm_layer
                    | instance_norm_layer
                    | group_norm_layer

        batch_norm_layer: "BatchNormalization(" ")"
        layer_norm_layer: "LayerNormalization(" ")"
        instance_norm_layer: "InstanceNormalization(" ")"
        group_norm_layer: "GroupNormalization(" groups_param ")"
        groups_param: "groups=" INT

        # Basic layer types
        dense_layer: "Dense(" (named_params | ordered_params) ")"
        flatten_layer: "Flatten(" ")"

        # Recurrent layers section - includes all RNN variants
        ?recurrent_layer: lstm_layer
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
        rnn_cell_layer: "RNNCell(" units_param ")"
        lstm_cell_layer: "LSTMCell(" units_param ")"
        gru_cell_layer: "GRUCell(" units_param ")"

        # Dropout wrapper layers for RNNs
        dropout_wrapper_layer: simple_rnn_dropout | gru_dropout | lstm_dropout
        simple_rnn_dropout: "SimpleRNNDropoutWrapper" "(" units_param "," dropout_param ")"
        gru_dropout: "GRUDropoutWrapper" "(" units_param "," dropout_param ")"
        lstm_dropout: "LSTMDropoutWrapper" "(" units_param "," dropout_param ")"
        dropout_param: "dropout=" FLOAT

        # Return sequences parameter for RNN layers
        return_sequences_param: "return_sequences=" return_sequences
        return_sequences: "true" -> true
                    | "false" -> false

        # Advanced layers group
        ?advanced_layer: attention_layer
                        | transformer_layer
                        | residual_layer
                        | inception_layer
                        | capsule_layer
                        | squeeze_excitation_layer
                        | graph_conv_layer
                        | embedding_layer
                        | quantum_layer
                        | dynamic_layer

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
        research: "research" NAME? "{" [research_params] "}"
        research_params: (metrics | references)*

        # Metrics tracking
        metrics: "metrics" "{" [accuracy_param] [loss_param] [precision_param] [recall_param] "}"
        accuracy_param: "accuracy:" FLOAT
        loss_param: "loss:" FLOAT
        precision_param: "precision:" FLOAT
        recall_param: "recall:" FLOAT

        # Paper references
        references: "references" "{" paper_param+ "}"
        paper_param: "paper:" ESCAPED_STRING

    """
    return lark.Lark(grammar, start=[start_rule], parser='lalr')

network_parser = create_parser('network')
layer_parser = create_parser('layer')
research_parser = create_parser('research')

class ModelTransformer(lark.Transformer):
    """
    Transforms the parsed tree into a structured dictionary representing the neural network model.
    """
    def layer(self, items: List[Any]) -> Dict[str, Any]:
        """Process a layer in the neural network model."""
        layer_info = items[0].data # Get layer type from Tree data
        params = {}
        if items[0].children: # Get params from Tree children
            params_tree = items[0].children[0] # Get the param_style1 (or similar) Tree
            params = self.transform(params_tree) # **TRANSFORM the params_tree!**

        layer_type = str(layer_info)
        return {'type': layer_type, 'params': params}
        
    def visit_layer(self, tree: Tree) -> Dict[str, Any]:
        """Visits a layer tree node and processes its children."""
        layer_type_token = tree.children[0]
        if not isinstance(layer_type_token, Token):
            raise TypeError(f"Expected Token for layer type, got {type(layer_type_token)}")
        layer_type = layer_type_token.value
        params: Dict[str, Any] = {}
        for param_tree in tree.children[1:]:
            if not isinstance(param_tree, Tree) or len(param_tree.children) != 2:
                continue # or raise error, depending on how strict you want to be
            key_token, value_tree = param_tree.children
            if not isinstance(key_token, Token):
                continue # or raise error
            key = key_token.value
            params[key] = self.visit(value_tree)

        if layer_type in ['LSTM', 'GRU'] and 'units' in params:
            try:
                params['units'] = int(params['units'])
            except ValueError:
                print(f"Warning: Could not convert 'units' to int for {layer_type} layer.")

        return {'type': layer_type, **params}

    def input_layer(self, items):
        shape = tuple(items[0])
        return {'type': 'Input', 'shape': shape}

    def conv2d_layer(self, items):
        if isinstance(items[0], list):
            # Ordered params: filters, kernel_size (h, w), activation
            params = {
                'filters': items[0][0],
                'kernel_size': (items[0][1], items[0][2]),
                'activation': items[0][3].strip('"') if len(items[0]) > 3 else None
            }
        else:
            params = items[0]
        return {'type': 'Conv2D', 'params': params}
    

    def extract_value(self, token: Union[Token, List[Token]]) -> str:
        """Safely extracts the value from a token or list of tokens."""
        if isinstance(token, list) and len(token) == 1:
            token = token[0]
        if isinstance(token, Token):
            return token.value.strip('"')
        raise TypeError(f"Expected Token, got {type(token)}")

    def dense_layer(self, items: List[Any]) -> Dict[str, Any]: # items can be List or Dict now
        """Processes the dense layer definition."""
        params = {}
        if items:
            if isinstance(items[0], dict): # Named params (still handle named params)
                params = items[0]
            elif isinstance(items[0], list): # Ordered params - HANDLE LIST CASE!
                ordered_vals = items[0]
                if len(ordered_vals) >= 1: # At least units is expected
                    params['units'] = ordered_vals[0]
                if len(ordered_vals) >= 2: # Activation is optional, if provided
                    params['activation'] = ordered_vals[1]
        return {'type': 'Dense', 'params': params}

    def loss(self, items: List[Token]) -> Dict[str, str]:
        """Processes the loss function definition."""
        if not items:
            raise ValueError("Loss definition is missing.")
        return {
            'type': 'Loss',
            'value': self.extract_value(items[0])
        }

    def optimizer(self, items: List[Token]) -> Dict[str, str]:
        """Processes the optimizer definition."""
        if not items:
            raise ValueError("Optimizer definition is missing.")
        return {'type': 'Optimizer',
                'value': self.extract_value(items[0])
            }

    def layers(self, items: List[Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Processes the layers section."""
        parsed_layers: List[Dict[str, Any]] = []
        for item in items:
            if isinstance(item, lark.Tree):
                layer_data = self.layer(item.children) # Correctly process Tree items
                parsed_layers.append(layer_data)
            elif isinstance(item, dict):
                parsed_layers.append(item) # Already processed layer
            else:
                continue # or handle unexpected item type, maybe raise an error

        return {
            'type': 'Layers',
            'layers': parsed_layers
        }

    def flatten_layer(self, items: List[Any]) -> Dict[str, str]: # No params, still needs 'params': {}
        """Processes the flatten layer definition."""
        return {'type': 'Flatten', 'params': {}} # Ensure 'params' is always present


    def training_config(self, items: List[Dict[str, int]]) -> Dict[str, Any]:
        """Processes the training configuration block."""
        config: Dict[str, Any] = {'type': 'TrainingConfig'}
        if items:
            params = items[0]
            if 'epochs' in params:
                try:
                    config['epochs'] = int(params['epochs'])
                except ValueError:
                    print("Warning: Invalid epoch value, defaulting to None.")
            if 'batch_size' in params:
                try:
                    config['batch_size'] = int(params['batch_size'])
                except ValueError:
                    print("Warning: Invalid batch_size value, defaulting to None.")
        return config


    def shape(self, items: List[Token]) -> Tuple[Optional[int], ...]:
        """
        Convert the list of tokens into a tuple representing shape.
        Handles 'None' for variable dimensions and converts integers.
        """
        shape_list: List[Optional[int]] = []
        for item in items:
            val_str = str(item)
            if val_str == "None":
                shape_list.append(None)
            else:
                try:
                    shape_list.append(int(val_str))
                except ValueError:
                    raise ValueError(f"Invalid shape dimension: {val_str}. Must be an integer or 'None'.")
        return tuple(shape_list)


    def max_pooling2d_layer(self, items: List[Dict[str, Any]]) -> Dict[str, Union[str, Tuple[int, int]]]: # Expecting List[Dict]
        """Processes the max pooling 2D layer definition."""
        params = items[0] if items else {} # items[0] is now the params dict
        return {"type": "MaxPooling2D", "params": params}


    def batch_norm_layer(self, items: List[Any]) -> Dict[str, str]: # No params, still needs 'params': {}
        """Processes the batch normalization layer definition."""
        return {'type': 'BatchNormalization', 'params': {}} # Ensure 'params' is always present

    def layer_norm_layer(self, items: List[Any]) -> Dict[str, str]: # No params, still needs 'params': {}
        """Processes the layer normalization layer definition."""
        return {'type': 'LayerNormalization', 'params': {}} # Ensure 'params' is always present


    def instance_norm_layer(self, items: List[Any]) -> Dict[str, str]: # No params, still needs 'params': {}
        """Processes the instance normalization layer definition."""
        return {'type': 'InstanceNormalization', 'params': {}} # Ensure 'params' is always present

    def group_norm_layer(self, items):
        params = items[0]
        return {'type': 'GroupNormalization', 'params': params}

    def lstm_layer(self, items):
        params = items[0]
        return {'type': 'LSTM', 'params': params}

    def gru_layer(self, items):
        params = items[0]
        return {'type': 'GRU', 'params': params}

    def simple_rnn_layer(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Processes SimpleRNN layer definition."""
        params = items[0] if items else {}
        return {'type': 'SimpleRNN', 'params': params}

    def cudnn_lstm_layer(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Processes CuDNNLSTM layer definition."""
        params = items[0] if items else {}
        return {'type': 'CuDNNLSTM', 'params': params}

    def cudnn_gru_layer(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Processes CuDNNGRU layer definition."""
        params = items[0] if items else {}
        return {'type': 'CuDNNGRU', 'params': params}

    def rnn_cell_layer(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Processes RNNCell layer definition."""
        params = items[0] if items else {}
        return {'type': 'RNNCell', 'params': params}

    def lstm_cell_layer(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Processes LSTMCell layer definition."""
        params = items[0] if items else {}
        return {'type': 'LSTMCell', 'params': params}

    def gru_cell_layer(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Processes GRUCell layer definition."""
        params = items[0] if items else {}
        return {'type': 'GRUCell', 'params': params}

    def simple_rnn_dropout(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Processes SimpleRNNDropoutWrapper layer definition."""
        params = items[0] if items else {} # Expecting transformed params dict now
        return {"type": "SimpleRNNDropoutWrapper", 'params': params}

    def gru_dropout(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Processes GRUDropoutWrapper layer definition."""
        params = items[0] if items else {} # Expecting transformed params dict now
        return {"type": "GRUDropoutWrapper", 'params': params}

    def lstm_dropout(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Processes LSTMDropoutWrapper layer definition."""
        params = items[0] if items else {} # Expecting transformed params dict now
        return {"type": "LSTMDropoutWrapper", 'params': params}
    

    #### Advanced Layers #################################

    def attention_layer(self, items: List[Any]) -> Dict[str, str]: # No params, still needs 'params': {}
        """Processes Attention layer definition."""
        return {'type': 'Attention', 'params': {}} # Ensure 'params' is always present


    def residual_layer(self, items: List[Any]) -> Dict[str, str]: # No params, still needs 'params': {}
        """Processes ResidualConnection layer definition."""
        return {'type': 'ResidualConnection', 'params': {}} # Ensure 'params' is always present

    def inception_layer(self, items: List[Any]) -> Dict[str, str]: # No params, still needs 'params': {}
        """Processes InceptionModule layer definition."""
        return {'type': 'InceptionModule', 'params': {}} # Ensure 'params' is always present

    def dynamic_layer(self, items: List[Any]) -> Dict[str, str]: # No params, still needs 'params': {}
        """Processes DynamicLayer definition."""
        return {'type': 'DynamicLayer', 'params': {}} # Ensure 'params' is always present


    def research(self, items: List[Any]) -> Dict[str, Any]:
        """Processes research file definition."""
        if not items:
            return {'type': 'Research', 'params': {}} # Allow empty research block
        if len(items) > 1:
          return {
              'type': 'Research',
              'name': self.extract_value(items[0]) if items[0] else None, # Research name might be optional now
              'params': items[1] if len(items) > 1 else {}
          }
        return {
            'type': 'Research',
            'params': items[0] if items else {}
        }

    def network(self, items: List[Any]) -> Dict[str, Any]:
        """Processes network file definition."""
        if len(items) < 6: # Minimum components: network name, input, layers, loss, optimizer, and potentially training config block
            raise ValueError("Network definition is incomplete. Requires at least name, input, layers, loss, optimizer, and enclosing braces.")

        name_token = items[0]
        if not isinstance(name_token, Token):
            raise TypeError(f"Expected Token for network name, got {type(name_token)}")
        name = str(name_token.value)
        input_layer_config = items[1] # Already processed input layer
        layers_config = items[2]['layers'] # Layers are nested under 'layers' key
        loss_config = items[3] # Already processed loss
        optimizer_config = items[4] # Already processed optimizer
        training_config = next((item for item in items[5:] if item.get('type') == 'TrainingConfig'), None) # Optional training config, default to None
        execution_config = next((item for item in items[5:] if item.get('type') == 'ExecutionConfig'), {"device": "auto"}) # Default execution config

        output_layer = next((layer for layer in reversed(layers_config) if layer['type'] == 'Output'), None)

        if output_layer is None:
            output_layer = { 'type': 'Output', 'units': 1, 'activation': 'linear' } # Default output layer

        output_shape = output_layer.get('shape')
        if output_shape is None and 'units' in output_layer:
            output_shape = (output_layer['units'],)
        elif output_shape is None:
            output_shape = (1,) # Fallback if still no output shape

        return {
            'type': 'model',
            'name': name,
            'input': input_layer_config, # Keep input config as nested dict
            'layers': layers_config, # Layers is already a list of layer configs
            'output_layer': output_layer,
            'output_shape': output_shape,
            'loss': loss_config,
            'optimizer': optimizer_config,
            'training_config': training_config,
            'execution_config': execution_config # Ensure execution config is included, default or parsed
        }


    def research_params(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Processes research parameters, merging metrics and references."""
        params: Dict[str, Any] = {}
        for item in items:
            if isinstance(item, dict):
                params.update(item) # Merge dictionaries directly
        return params

    def metrics(self, items: List[Dict[str, float]]) -> Dict[str, float]:
        """Processes metrics parameters."""
        metrics_dict: Dict[str, float] = {}
        for item in items:
            if isinstance(item, dict) and 'type' in item and 'value' in item:
                metrics_dict[item['type']] = item['value']
        return {'metrics': metrics_dict}


    def accuracy_param(self, items: List[Token]) -> Dict[str, Union[str, float]]:
        """Processes accuracy parameter."""
        if not items:
            raise ValueError("Accuracy parameter missing value.")
        try:
            return {'type': 'accuracy', 'value': float(self.extract_value(items[0]))}
        except ValueError as e:
            raise ValueError(f"Invalid accuracy value: {e}") from e

    def loss_param(self, items: List[Token]) -> Dict[str, Union[str, float]]:
        """Processes loss metric parameter."""
        if not items:
            raise ValueError("Loss metric parameter missing value.")
        try:
            return {'type': 'loss', 'value': float(self.extract_value(items[0]))}
        except ValueError as e:
            raise ValueError(f"Invalid loss metric value: {e}") from e

    def precision_param(self, items: List[Token]) -> Dict[str, Union[str, float]]:
        """Processes precision parameter."""
        if not items:
            raise ValueError("Precision parameter missing value.")
        try:
            return {'type': 'precision', 'value': float(self.extract_value(items[0]))}
        except ValueError as e:
            raise ValueError(f"Invalid precision value: {e}") from e

    def recall_param(self, items: List[Token]) -> Dict[str, Union[str, float]]:
        """Processes recall parameter."""
        if not items:
            raise ValueError("Recall parameter missing value.")
        try:
            return {'type': 'recall', 'value': float(self.extract_value(items[0]))}
        except ValueError as e:
            raise ValueError(f"Invalid recall value: {e}") from e


    def references(self, items: List[Dict[str, str]]) -> Dict[str, List[str]]:
        """Processes references block."""
        references_list: List[str] = []
        for item in items:
            if isinstance(item, dict) and 'paper' in item:
                references_list.append(item['paper'])
        return {'references': references_list}

    def paper_param(self, items: List[Token]) -> Dict[str, str]:
        """Processes paper parameter."""
        if not items:
            raise ValueError("Paper reference missing value.")
        return {'paper': self.extract_value(items[0])}


    def named_params(self, items: List[Any]) -> Dict[str, Any]: # items can be List of Trees or mixed
        """Processes named parameters, returning as a dictionary."""
        print("\n--- named_params ---")
        print(f"Items received in named_params: {items}")

        param_dicts = [] # Initialize an empty list to collect parameter dictionaries
        for item in items: # Iterate through the items received (which are children of 'named_params' tree)
            if isinstance(item, Tree): # IMPORTANT: Check if item is a Tree (named_param subtree)
                param_dict = self.visit(item) # VISIT each child TREE to transform named_param
                param_dicts.append(param_dict) # Add the transformed dictionary to the list
            # else: What to do if it's not a Tree? (Shouldn't happen based on grammar, but for robustness, can add error handling or ignore)

        print(f"Collected param dicts in named_params: {param_dicts}") # Print the collected list of dictionaries
        params = {}
        for param_dict in param_dicts: # Now iterate through the *list of dictionaries*
            if isinstance(param_dict, dict): # Double check it's a dict before updating
                params.update(param_dict) # MERGE dictionaries into a single dictionary
            else:
                print(f"Warning: Unexpected non-dictionary item in param_dicts: {param_dict}") # Warning if not a dict

        print(f"Merged params in named_params: {params}")
        return params

    def ordered_params(self, items: List[Any]) -> List[Any]:
        """Processes ordered parameters, returning as a list."""
        print("\n--- ordered_params ---") # Added print statement
        print(f"Items received in ordered_params: {items}") # Added print statement
        # visited_items = [self.visit(item) for item in items] # REMOVE this line
        # print(f"Visited items in ordered_params: {visited_items}") # REMOVE this line
        return items # Just return the items as they are already transformed in 'value'

    def value(self, items):
        if isinstance(items[0], Tree) and items[0].data == 'tuple':
            return self.visit(items[0])
        return super().value(items)

    def tuple(self, items):
        return (int(items[1].value), int(items[3].value))

    def named_return_sequences(self, items: List[Any]) -> Dict[str, bool]:
        return {'return_sequences': self.visit(items[2])}

    def named_num_heads(self, items: List[Any]) -> Dict[str, int]:
        return {'num_heads': self.visit(items[2])}

    def named_ff_dim(self, items: List[Any]) -> Dict[str, int]:
        return {'ff_dim': self.visit(items[2])}

    def named_input_dim(self, items: List[Any]) -> Dict[str, int]:
        return {'input_dim': self.visit(items[2])}

    def named_output_dim(self, items: List[Any]) -> Dict[str, int]:
        return {'output_dim': self.visit(items[2])}

    # Correct number_or_none method
    def number_or_none(self, items):
        token = items[0]
        if token.value == 'None':
            return None
        return int(token.value)
    
    def named_filters(self, items):
        return {'filters': int(items[2].value)}
    
    def named_activation(self, items):
        return {'activation': items[2].value.strip('"')}
    
    def named_kernel_size(self, items):
        return {'kernel_size': self.visit(items[2])}
    
    def named_padding(self, items):
        return {'padding': items[2].value.strip('"')}
    
    def named_strides(self, items):
        return {'strides': self.visit(items[2])}
    
    def named_rate(self, items):
        return {'rate': float(items[2].value)}


    def dropout_layer(self, items):
        if isinstance(items[0], (int, float)):
            return {'type': 'Dropout', 'params': {'rate': items[0]}}
        return {'type': 'Dropout', 'params': items[0]}

    # Add methods for Output layer parameters
    def units_param(self, items):
        return {'units': int(items[2].value)}

    def activation_param(self, items: List[Any]) -> Dict[str, str]:
        return {'activation': items[2].value.strip('"')}

    def output_layer(self, items):
        params = {}
        for item in items:
            if isinstance(item, dict):
                params.update(item)
        return {'type': 'Output', 'params': params}

    # Add methods for advanced layers
    def capsule_layer(self, items):
        return {'type': 'CapsuleLayer', 'params': {}}

    def squeeze_excitation_layer(self, items):
        return {'type': 'SqueezeExcitation', 'params': {}}

    def graph_conv_layer(self, items):
        return {'type': 'GraphConv', 'params': {}}

    def quantum_layer(self, items):
        return {'type': 'QuantumLayer', 'params': {}}
    
    def transformer_layer(self, items):
        params = items[0]
        return {'type': 'TransformerEncoder', 'params': params}
    
    def embedding_layer(self, items):
        params = items[0]
        return {'type': 'Embedding', 'params': params}

    def param_style1(self, items: List[Union[Dict[str, Any], List[Any]]]) -> Union[Dict[str, Any], List[Any]]:
        """Handles both named and ordered parameters styles."""
        if isinstance(items[0], dict):
            return items[0] # Named parameters dictionary
        else:
            return items # Ordered parameters list

    def epochs_param(self, items: List[Token]) -> Dict[str, int]:
        """Processes epochs parameter in training config."""
        if len(items) == 1:
            try:
                return {'epochs': int(items[0])}
            except ValueError as e:
                raise ValueError(f"Invalid epochs value: {e}") from e
        raise ValueError("Invalid epochs parameter format.")

    def batch_size_param(self, items: List[Token]) -> Dict[str, int]:
        """Processes batch_size parameter in training config."""
        if len(items) == 1:
            try:
                return {'batch_size': int(items[0])}
            except ValueError as e:
                raise ValueError(f"Invalid batch_size value: {e}") from e
        raise ValueError("Invalid batch_size parameter format.")

    def device_param(self, items: List[Token]) -> Dict[str, str]:
        """Processes device parameter in execution config."""
        if len(items) == 1:
            return {'device': self.extract_value(items[0])}
        raise ValueError("Invalid device parameter format.")

    def dropout_param(self, items: List[Tree]) -> Dict[str, float]: # dropout_param already returns a dict
        """Processes dropout parameter for DropoutWrapper layers."""
        if items and items[0].data == 'dropout_param':
            dropout_token = items[0].children[0] # Access the FLOAT token
            if isinstance(dropout_token, Token) and dropout_token.type == 'FLOAT':
                try:
                    return {'dropout': float(dropout_token.value)}
                except ValueError:
                    raise ValueError(f"Invalid float value for dropout: {dropout_token.value}")
        return {} # Return empty dict if parsing fails or no dropout found

    def named_param(self, items: List[Any]) -> Dict[str, Any]: # Is this method even being called?
        """Processes a single named parameter."""
        print("\n--- named_param ---") # Added print statement - IS THIS PRINTING?
        print(f"Items received in named_param: {items}") # Added print statement
        # It should just return the dictionary created by named_units, named_activation etc.
        if items:
            return items[0] if isinstance(items[0], dict) else {} # Return first item if it's a dict
        return {}

    def named_dilation_rate(self, items: List[Any]) -> Dict[str, Tuple[int, int]]:
        return {'dilation_rate': self.visit(items[2])}

    def named_groups(self, items: List[Any]) -> Dict[str, int]:
        return {'groups': self.visit(items[2])}

    def named_pool_size(self, items: List[Any]) -> Dict[str, Tuple[int, int]]:
        return {'pool_size': self.visit(items[2])}

    def execution_config(self, items: List[Tree]) -> Dict[str, Any]:
        """Processes execution configuration block."""
        config: Dict[str, Any] = {'type': 'ExecutionConfig'}
        if items:
            for item in items:
                if item.data == 'device_param':
                    config.update(self.device_param(item.children))
        return config
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


    def validate_input_dims(expected_dims: int, layer_name: str):
        """Helper function to validate input dimensions."""
        if len(input_shape) != expected_dims:
            raise ValueError(f"{layer_name} layer expects {expected_dims}D input (including batch), got {len(input_shape)}D input with shape {input_shape}")

    if layer_type == 'Input':
        return layer.get('shape', input_shape) # Input layer can optionally override shape

    elif layer_type == 'Conv2D':
        validate_input_dims(4, 'Conv2D') # Expecting (batch, height, width, channels)
        params = layer.get('params', layer) # Check both 'params' and direct layer for config

        filters = params.get('filters')
        kernel_size = params.get('kernel_size')
        strides = params.get('strides', (1, 1)) # Default strides to 1x1
        padding_mode = params.get('padding', 'valid').lower() # Default to 'valid' padding
        dilation_rate = params.get('dilation_rate', (1, 1)) # Default dilation rate to 1x1

        if not filters or not kernel_size:
            raise ValueError("Conv2D layer requires 'filters' and 'kernel_size' parameters.")

        try:
            filters = int(filters)
            kernel_h, kernel_w = map(int, kernel_size)
            stride_h, stride_w = map(int, strides)
            dilation_h, dilation_w = map(int, dilation_rate)
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

        pool_size = params.get('pool_size')
        strides = params.get('strides', pool_size) # Default stride to pool_size if not provided
        padding_mode = params.get('padding', 'valid').lower() # Default to 'valid' padding

        if not pool_size:
            raise ValueError("MaxPooling2D layer requires 'pool_size' parameter.")

        try:
            pool_h, pool_w = map(int, pool_size)
            stride_h, stride_w = map(int, strides) if strides else map(int, pool_size) # If strides is None, use pool_size
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
            units = int(units)
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
            units = int(units)
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
            units = int(units)
        except ValueError as e:
            raise ValueError(f"Invalid {layer_type} units parameter: {e}") from e

        batch_size, time_steps, _ = input_shape
        if return_sequences:
            return (batch_size, time_steps, units) # Shape is (batch, time_steps, units) if return_sequences=True
        else:
            return (batch_size, units) # Shape is (batch, units) if return_sequences=False


    elif layer_type in ['Attention', 'TransformerEncoder', 'ResidualConnection', 'InceptionModule',
                        'CapsuleLayer', 'SqueezeExcitation', 'GraphConv', 'Embedding', 'QuantumLayer', 'DynamicLayer']:
        return input_shape # Placeholder for advanced layers, needs specific shape logic


    else:
        raise ValueError(f"Unsupported layer type: {layer_type}")

def generate_code(model_data: Dict[str, Any], backend: str = "tensorflow") -> str:
    """
    Generates code for creating and training a neural network model based on model_data and backend.
    Supports TensorFlow and PyTorch backends.

    Parameters:
    model_data (dict): Parsed model configuration dictionary.
    backend (str): Backend framework ('tensorflow' or 'pytorch'). Default: 'tensorflow'.

    Returns:
    str: Generated code as a string.

    Raises:
    ValueError: If the backend is unsupported or model_data is invalid.
    """
    if not isinstance(model_data, dict) or 'layers' not in model_data or 'input' not in model_data or 'loss' not in model_data or 'optimizer' not in model_data:
        raise ValueError("Invalid model_data format. Ensure it contains 'layers', 'input', 'loss', and 'optimizer'.")

    indent = "    " # Define consistent indentation

    if backend == "tensorflow":
        code = "import tensorflow as tf\n\n"
        code += "model = tf.keras.Sequential([\n"

        input_shape = model_data['input']['shape']
        if not input_shape:
            raise ValueError("Input layer shape is not defined in model_data.")
        input_layer_code = f"{indent}tf.keras.layers.Input(shape={input_shape}),\n" # Explicit input layer
        code += input_layer_code


        current_input_shape = input_shape # Start shape propagation from input
        for layer_config in model_data['layers']:
            layer_type = layer_config['type']
            params = layer_config.get('params', layer_config) # Check both 'params' and direct config

            if layer_type == 'Conv2D':
                filters = params.get('filters')
                kernel_size = params.get('kernel_size')
                activation = params.get('activation', 'relu') # Default activation
                if not filters or not kernel_size:
                    raise ValueError("Conv2D layer config missing 'filters' or 'kernel_size'.")
                code += f"{indent}tf.keras.layers.Conv2D(filters={filters}, kernel_size={kernel_size}, activation='{activation}'),\n"
                current_input_shape = propagate_shape(current_input_shape, layer_config) # Update shape

            elif layer_type == 'MaxPooling2D':
                pool_size = params.get('pool_size')
                if not pool_size:
                    raise ValueError("MaxPooling2D layer config missing 'pool_size'.")
                code += f"{indent}tf.keras.layers.MaxPool2D(pool_size={pool_size}),\n"
                current_input_shape = propagate_shape(current_input_shape, layer_config) # Update shape

            elif layer_type == 'Flatten':
                code += f"{indent}tf.keras.layers.Flatten(),\n"
                current_input_shape = propagate_shape(current_input_shape, layer_config) # Update shape

            elif layer_type == 'Dense':
                units = params.get('units')
                activation = params.get('activation', 'relu') # Default activation
                if not units:
                    raise ValueError("Dense layer config missing 'units'.")
                code += f"{indent}tf.keras.layers.Dense(units={units}, activation='{activation}'),\n"
                current_input_shape = propagate_shape(current_input_shape, layer_config) # Update shape

            elif layer_type == 'Dropout':
                rate = params.get('rate')
                if rate is None:
                    raise ValueError("Dropout layer config missing 'rate'.")
                code += f"{indent}tf.keras.layers.Dropout(rate={rate}),\n"
                current_input_shape = propagate_shape(current_input_shape, layer_config) # Update shape

            elif layer_type == 'Output':
                units = params.get('units')
                activation = params.get('activation', 'linear') # Default linear output
                if not units:
                    raise ValueError("Output layer config missing 'units'.")
                code += f"{indent}tf.keras.layers.Dense(units={units}, activation='{activation}'),\n"
                current_input_shape = propagate_shape(current_input_shape, layer_config) # Update shape

            elif layer_type == 'BatchNormalization':
                code += f"{indent}tf.keras.layers.BatchNormalization(),\n"
                current_input_shape = propagate_shape(current_input_shape, layer_config) # Update shape
            elif layer_type == 'LayerNormalization':
                code += f"{indent}tf.keras.layers.LayerNormalization(),\n"
                current_input_shape = propagate_shape(current_input_shape, layer_config) # Update shape
            elif layer_type == 'InstanceNormalization':
                code += f"{indent}tf.keras.layers.InstanceNormalization(),\n"
                current_input_shape = propagate_shape(current_input_shape, layer_config) # Update shape
            elif layer_type == 'GroupNormalization':
                groups = params.get('groups')
                if groups is None:
                    raise ValueError("GroupNormalization layer config missing 'groups'.")
                code += f"{indent}tf.keras.layers.GroupNormalization(groups={groups}),\n"
                current_input_shape = propagate_shape(current_input_shape, layer_config) # Update shape

            elif layer_type in ['LSTM', 'GRU', 'SimpleRNN', 'Bidirectional', 'CuDNNLSTM', 'CuDNNGRU']: # Recurrent layers
                units = params.get('units')
                return_sequences = params.get('return_sequences', False) # Default to False
                if units is None:
                    raise ValueError(f"{layer_type} layer config missing 'units'.")
                # Map layer_type to TensorFlow name if different
                tf_layer_name = layer_type
                if layer_type == 'SimpleRNN':
                    tf_layer_name = 'SimpleRNN' # Already correct
                elif layer_type == 'CuDNNLSTM':
                    tf_layer_name = 'CuDNNLSTM'
                elif layer_type == 'CuDNNGRU':
                    tf_layer_name = 'CuDNNGRU'

                code += f"{indent}tf.keras.layers.{tf_layer_name}(units={units}, return_sequences={str(return_sequences).lower()}),\n" # Ensure boolean is lowercase for TF
                current_input_shape = propagate_shape(current_input_shape, layer_config) # Update shape

            elif layer_type == 'Embedding':
                input_dim = params.get('input_dim')
                output_dim = params.get('output_dim')
                if not input_dim or not output_dim:
                    raise ValueError("Embedding layer config missing 'input_dim' or 'output_dim'.")
                code += f"{indent}tf.keras.layers.Embedding(input_dim={input_dim}, output_dim={output_dim}),\n"
                current_input_shape = propagate_shape(current_input_shape, layer_config) # Update shape

            elif layer_type == 'Attention': # Basic Attention (you might need to adjust based on specific AttentionLayer)
                code += f"{indent}tf.keras.layers.Attention(),\n"
                current_input_shape = propagate_shape(current_input_shape, layer_config) # Shape might not change drastically

            elif layer_type == 'TransformerEncoder': # Simple TransformerEncoder (adjust parameters as needed)
                num_heads = params.get('num_heads', 4) # Default num_heads
                ff_dim = params.get('ff_dim', 32)      # Default ff_dim
                code += f"{indent}tf.keras.layers.TransformerEncoder(num_heads={num_heads}, ffn_units={ff_dim}),\n" # ffn_units for ff_dim in TF
                current_input_shape = propagate_shape(current_input_shape, layer_config) # Shape transformation depends on Transformer

            elif layer_type in ['ResidualConnection', 'InceptionModule', 'CapsuleLayer', 'SqueezeExcitation', 'GraphConv', 'QuantumLayer', 'DynamicLayer']:
                print(f"Warning: {layer_type} is an advanced or custom layer type. Code generation for TensorFlow might require manual implementation. Skipping layer code generation for now.")
                continue # Skip code generation for unsupported/complex layers, warn user

            else:
                raise ValueError(f"Unsupported layer type: {layer_type} for TensorFlow backend.")


        code += "])\n\n" # End Sequential model definition

        # Loss and Optimizer
        loss_value = model_data['loss']['value'].strip('"') # Remove quotes
        optimizer_value = model_data['optimizer']['value'].strip('"') # Remove quotes

        code += f"optimizer = tf.keras.optimizers.{optimizer_value}()\n" # Instantiate optimizer
        code += f"loss_fn = tf.keras.losses.{loss_value}\n" # Define loss function (string lookup)


        # Training configuration - basic training loop for demonstration
        training_config = model_data.get('training_config')
        if training_config:
            epochs = training_config.get('epochs', 10) # Default epochs
            batch_size = training_config.get('batch_size', 32) # Default batch size

            code += "\n# Example training loop (requires data loading and handling)\n"
            code += "epochs = {}\n".format(epochs)
            code += "batch_size = {}\n".format(batch_size)
            code += "for epoch in range(epochs):\n"
            code += f"{indent}for batch_idx, (data, labels) in enumerate(dataset):\n" # Assumes 'dataset' is defined
            code += f"{indent}{indent}with tf.GradientTape() as tape:\n"
            code += f"{indent}{indent}{indent}predictions = model(data)\n"
            code += f"{indent}{indent}{indent}loss = loss_fn(labels, predictions)\n"
            code += f"{indent}{indent}gradients = tape.gradient(loss, model.trainable_variables)\n"
            code += f"{indent}{indent}optimizer.apply_gradients(zip(gradients, model.trainable_variables))\n"
            code += f"{indent}print(f'Epoch {{epoch+1}}, Batch {{batch_idx}}, Loss: {{loss.numpy()}}')\n"
        else:
            code += "\n# No training configuration provided in model definition.\n"
            code += "# Training loop needs to be implemented manually.\n"


        return code

    elif backend == "pytorch":
        code = "import torch\n"
        code += "import torch.nn as nn\n"
        code += "import torch.optim as optim\n\n"

        code += "class NeuralNetworkModel(nn.Module):\n"
        code += indent + "def __init__(self):\n"
        code += indent + indent + "super(NeuralNetworkModel, self).__init__()\n"

        input_shape = model_data['input']['shape']
        if not input_shape:
            raise ValueError("Input layer shape is not defined in model_data.")
        current_input_shape = input_shape # Track shape for PyTorch layers

        layers_code = [] # Store layer instantiations for sequential definition
        forward_code_body = [] # Store forward pass operations

        for i, layer_config in enumerate(model_data['layers']):
            layer_type = layer_config['type']
            params = layer_config.get('params', layer_config) # Check both places for params
            layer_name = f"self.layer{i+1}" # Unique layer name

            if layer_type == 'Conv2D':
                filters = params.get('filters')
                kernel_size = params.get('kernel_size')
                activation_name = params.get('activation', 'relu') # Default activation
                padding = params.get('padding', 'same').lower() # Default padding to 'same'

                if not filters or not kernel_size:
                    raise ValueError("Conv2D layer config missing 'filters' or 'kernel_size'.")

                layers_code.append(f"{layer_name}_conv = nn.Conv2d(in_channels={current_input_shape[-1]}, out_channels={filters}, kernel_size={kernel_size}, padding='{padding}')") # Assuming channels_last
                layers_code.append(f"{layer_name}_activation = nn.ReLU() if '{activation_name}' == 'relu' else nn.Identity()") # Add other activations as needed
                forward_code_body.append(f"x = self.layer{i+1}_conv(x)")
                forward_code_body.append(f"x = self.layer{i+1}_activation(x)")
                current_input_shape = propagate_shape(current_input_shape, layer_config) # Update shape


            elif layer_type == 'MaxPooling2D':
                pool_size = params.get('pool_size')
                if not pool_size:
                    raise ValueError("MaxPooling2D layer config missing 'pool_size'.")
                layers_code.append(f"{layer_name}_pool = nn.MaxPool2d(kernel_size={pool_size})")
                forward_code_body.append(f"x = self.layer{i+1}_pool(x)")
                current_input_shape = propagate_shape(current_input_shape, layer_config) # Update shape

            elif layer_type == 'Flatten':
                layers_code.append(f"{layer_name}_flatten = nn.Flatten()")
                forward_code_body.append(f"x = self.layer{i+1}_flatten(x)")
                current_input_shape = propagate_shape(current_input_shape, layer_config) # Update shape

            elif layer_type == 'Dense':
                units = params.get('units')
                activation_name = params.get('activation', 'relu') # Default activation
                if not units:
                    raise ValueError("Dense layer config missing 'units'.")
                layers_code.append(f"{layer_name}_dense = nn.Linear(in_features={np.prod(current_input_shape[1:]) if len(current_input_shape)>1 else current_input_shape[1]}, out_features={units})") # Handle flattened input
                layers_code.append(f"{layer_name}_activation = nn.ReLU() if '{activation_name}' == 'relu' else nn.Identity()") # Add other activations
                forward_code_body.append(f"x = self.layer{i+1}_dense(x)")
                forward_code_body.append(f"x = self.layer{i+1}_activation(x)")
                current_input_shape = propagate_shape(current_input_shape, layer_config) # Update shape


            elif layer_type == 'Dropout':
                rate = params.get('rate')
                if rate is None:
                    raise ValueError("Dropout layer config missing 'rate'.")
                layers_code.append(f"{layer_name}_dropout = nn.Dropout(p={rate})")
                forward_code_body.append(f"x = self.layer{i+1}_dropout(x)")
                current_input_shape = propagate_shape(current_input_shape, layer_config) # Shape remains same

            elif layer_type == 'Output':
                units = params.get('units')
                activation_name = params.get('activation', 'linear') # Default linear output
                if not units:
                    raise ValueError("Output layer config missing 'units'.")
                layers_code.append(f"{layer_name}_output = nn.Linear(in_features={np.prod(current_input_shape[1:]) if len(current_input_shape)>1 else current_input_shape[1]}, out_features={units})") # Handle flattened input for output
                layers_code.append(f"{layer_name}_activation = nn.Sigmoid() if '{activation_name}' == 'sigmoid' else (nn.Softmax(dim=1) if '{activation_name}' == 'softmax' else nn.Identity())") # Example activations - expand as needed
                forward_code_body.append(f"x = self.layer{i+1}_output(x)")
                forward_code_body.append(f"x = self.layer{i+1}_activation(x)")
                current_input_shape = propagate_shape(current_input_shape, layer_config) # Update shape

            elif layer_type == 'BatchNormalization':
                layers_code.append(f"{layer_name}_bn = nn.BatchNorm2d(num_features={current_input_shape[-1]})") # Assuming channels_last, adjust if channels_first
                forward_code_body.append(f"x = self.layer{i+1}_bn(x)")
                current_input_shape = propagate_shape(current_input_shape, layer_config) # Shape same

            elif layer_type in ['LSTM', 'GRU', 'SimpleRNN', 'CuDNNLSTM', 'CuDNNGRU']: # Recurrent Layers
                units = params.get('units')
                return_sequences = params.get('return_sequences', False)
                if units is None:
                    raise ValueError(f"{layer_type} layer config missing 'units'.")
                # Map layer_type to PyTorch name
                torch_layer_name = layer_type
                if layer_type == 'SimpleRNN':
                    torch_layer_name = 'RNN' # Correct PyTorch name is RNN
                elif layer_type == 'CuDNNLSTM':
                    torch_layer_name = 'LSTM' # CuDNN is implementation detail, use basic LSTM in code
                elif layer_type == 'CuDNNGRU':
                    torch_layer_name = 'GRU' # Same for GRU

                layers_code.append(f"{layer_name}_rnn = nn.{torch_layer_name}(input_size={current_input_shape[-1]}, hidden_size={units}, batch_first=True, bidirectional=False)") # Assuming batch_first=True
                forward_code_body.append(f"x, _ = self.layer{i+1}_rnn(x)") # RNN returns output and hidden state
                current_input_shape = propagate_shape(current_input_shape, layer_config) # Update shape

            elif layer_type == 'Flatten': # Flatten again if needed between RNN and Dense
                layers_code.append(f"{layer_name}_flatten = nn.Flatten(start_dim=1)") # Flatten from dimension 1 onwards (keep batch dim)
                forward_code_body.append(f"x = self.layer{i+1}_flatten(x)")
                current_input_shape = propagate_shape(current_input_shape, layer_config) # Update shape


            elif layer_type in ['Attention', 'TransformerEncoder', 'ResidualConnection', 'InceptionModule', 'CapsuleLayer', 'SqueezeExcitation', 'GraphConv', 'Embedding', 'QuantumLayer', 'DynamicLayer']:
                print(f"Warning: {layer_type} is an advanced or custom layer type. Code generation for PyTorch might require manual implementation. Skipping layer code generation for now.")
                continue # Skip advanced layers for now, warn user

            else:
                raise ValueError(f"Unsupported layer type: {layer_type} for PyTorch backend.")


        # __init__ method layer instantiation
        code += indent + indent + "# Layer Definitions\n"
        for layer_init_code in layers_code:
            code += indent + indent + layer_init_code + "\n"
        code += "\n" # Add newline before forward method

        # forward method
        code += indent + "def forward(self, x):\n"
        code += indent + indent + "# Forward Pass\n"
        code += indent + indent + "batch_size, h, w, c = x.size()\n" # Example input shape assumption, adjust if needed
        for forward_op in forward_code_body:
            code += indent + indent + forward_op + "\n"
        code += indent + indent + "return x\n" # Assuming last 'x' is the output

        code += "model = NeuralNetworkModel()\n"
        code += "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
        code += "model.to(device)\n\n"

        # Loss and Optimizer
        loss_value = model_data['loss']['value'].strip('"')
        optimizer_value = model_data['optimizer']['value'].strip('"')

        # Map loss names to PyTorch conventions if needed (e.g., "categorical_crossentropy" to "CrossEntropyLoss")
        if loss_value.lower() == 'categorical_crossentropy' or loss_value.lower() == 'sparse_categorical_crossentropy':
            loss_fn_code = "loss_fn = nn.CrossEntropyLoss()" # Common for classification
        elif loss_value.lower() == 'mean_squared_error':
            loss_fn_code = "loss_fn = nn.MSELoss()"
        else:
            loss_fn_code = f"loss_fn = nn.{loss_value}()" # Try direct name if mapping not defined
            print(f"Warning: Loss function '{loss_value}' might not be directly supported in PyTorch. Verify the name and compatibility.")


        code += loss_fn_code + "\n"
        code += f"optimizer = optim.{optimizer_value}(model.parameters(), lr=0.001)\n\n" # Example LR - make configurable


        # Training loop - basic example
        training_config = model_data.get('training_config')
        if training_config:
            epochs = training_config.get('epochs', 10)
            batch_size = training_config.get('batch_size', 32)

            code += "# Example training loop (requires data loader setup)\n"
            code += f"epochs = {epochs}\n"
            code += f"batch_size = {batch_size}\n"
            code += "for epoch in range(epochs):\n"
            code += indent + "for batch_idx, (data, target) in enumerate(train_loader):\n" # Assume train_loader is defined
            code += indent + indent + "data, target = data.to(device), target.to(device)\n"
            code += indent + indent + "optimizer.zero_grad()\n"
            code += indent + indent + "output = model(data)\n"
            code += indent + indent + "loss = loss_fn(output, target)\n"
            code += indent + indent + "loss.backward()\n"
            code += indent + indent + "optimizer.step()\n"
            code += indent + indent + "if batch_idx % 100 == 0:\n"
            code += indent + indent + indent + f"print('Epoch: {{epoch+1}} [{{batch_idx*len(data)}}/{{len(train_loader.dataset)}} ({{100.*batch_idx/len(train_loader):.0f}}%)]\\tLoss: {{loss.item():.6f}}')\n"
            code += "print('Finished Training')\n"

        else:
            code += "# No training configuration provided. Training loop needs manual implementation.\n"


        return code


    else:
        raise ValueError(f"Unsupported backend: {backend}. Choose 'tensorflow' or 'pytorch'.")



def load_file(filename: str) -> Tree:
    """
    Loads and parses a neural network or research file based on its extension.

    Args:
        filename (str): The path to the file to load.

    Returns:
        Tree: A Lark parse tree representing the file content.

    Raises:
        ValueError: If the file type is unsupported or parsing fails.
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    try:
        with open(filename, 'r') as f:
            content = f.read()
    except Exception as e:
        raise IOError(f"Error reading file: {filename}. {e}") from e

    if filename.endswith(('.neural', '.nr')):
        parser = network_parser # Use network parser for .neural and .nr files
    elif filename.endswith('.rnr'):
        parser = research_parser # Use research parser for .rnr files
    elif filename.endswith('.layer'):
        parser = layer_parser # Use layer parser for individual layer files
    else:
        raise ValueError(f"Unsupported file type: {filename}. Supported types are .neural, .nr, .rnr, .layer.")

    try:
        parse_tree = parser.parse(content)
        return parse_tree
    except lark.exceptions.LarkError as e: # Catch parsing errors specifically
        raise ValueError(f"Parsing error in file: {filename}. {e}") from e

