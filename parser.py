import lark
from lark import Tree, Transformer, Token
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
        %import common.CNAME -> NAME
        %import common.SIGNED_NUMBER -> NUMBER
        %import common.WS_INLINE             # Inline whitespace
        %import common.INT
        %import common.BOOL                  # Boolean values
        %import common.CNAME -> NAME         # Names/identifiers
        %import common.FLOAT                 # Floating poNUMBER numbers
        %import common.ESCAPED_STRING        # Quoted strings
        %import common.WS                    # All whitespace
        %ignore WS   

        ?start: network | layer | research  # Allow parsing either networks or research files

        # File type rules
        neural_file: network
        nr_file: network
        rnr_file: research

        # Parameter & Properties
        named_params: named_param ("," named_param)*
        activation_param: "activation" "=" ESCAPED_STRING
        ordered_params: value ("," value)* 
        named_rate: "rate" "=" FLOAT
        value: NUMBER | ESCAPED_STRING | tuple_ | BOOL
        tuple_: "(" WS_INLINE* NUMBER WS_INLINE* "," WS_INLINE* NUMBER WS_INLINE* ")"
        BOOL: "true" | "false"

        # name_param rules
        named_data_format: "data_format" "=" value 
        named_units: "units" "=" NUMBER
        named_activation: "activation" "=" ESCAPED_STRING
        named_filters: "filters" "=" NUMBER
        named_kernel_size: "kernel_size" "=" value
        named_strides: "strides" "=" value
        named_padding: "padding" "=" value
        named_dilation_rate: "dilation_rate" "=" value
        named_groups: "groups" "=" NUMBER
        named_channels: "channels" "=" NUMBER
        named_pool_size: "pool_size" "=" value
        named_return_sequences: "return_sequences" "=" bool_value
        bool_value: BOOL -> bool
        named_num_heads: "num_heads" "=" NUMBER
        named_ff_dim: "ff_dim" "=" NUMBER
        named_input_dim: "input_dim" "=" NUMBER
        named_output_dim: "output_dim" "=" NUMBER  
        named_dropout: "dropout" "=" FLOAT
        ?named_param: named_units | named_activation | named_filters | named_kernel_size | named_strides | named_padding | named_dilation_rate | named_groups | named_data_format | named_channels | named_rate | named_pool_size | named_return_sequences | named_num_heads | named_ff_dim | named_input_dim | named_output_dim | named_dropout

        # Layer parameter styles
        ?param_style1: named_params    // Dense(units=128, activation="relu")
                    | ordered_params   // Dense(128, "relu")

        # Top-level network definition - defines the structure of an entire neural network
        network: "network" NAME "{" input_layer layers loss optimizer [training_config] [execution_config] "}" _NL*

        # Configuration can be either training-related or execution-related
        config: training_config | execution_config

        # Input layer definition - specifies the shape of input data
        input_layer: "input:" "(" shape ")"
        # Shape can contain multiple dimensions, each being a number or None
        shape: number_or_none ("," number_or_none)*
        # Dimensions can be specific NUMBERegers or None (for variable dimensions)
        number_or_none: NUMBER | "None"

        # Layers section - contains all layer definitions separated by newlines
        layers: "layers:" _NL* layer+ _NL*

        # All possible layer types that can be used in the network
        ?layer: basic_layer  | recurrent_layer | advanced_layer | activation_layer


        # Basic layers group
        ?basic_layer: conv_layer
                    | pooling_layer
                    | dropout_layer
                    | flatten_layer
                    | dense_layer
                    | output_layer
                    | norm_layer


        # Output layer 
        output_layer: "Output(" named_params ")"

        # Convolutional layers 
        conv_layer: conv1d_layer | conv2d_layer | conv3d_layer
        conv1d_layer: "Conv1D(" named_params ")"
        conv2d_layer: "Conv2D(" named_params ")"
        conv3d_layer: "Conv3D(" named_params ")"

        # Pooling layer parameters
        pooling_layer: max_pooling_layer | average_pooling_layer | global_pooling_layer | adaptive_pooling_layer
        max_pooling_layer: max_pooling1d_layer | max_pooling2d_layer | max_pooling3d_layer
        max_pooling1d_layer: "MaxPooling1D(" named_params ")"
        max_pooling2d_layer: "MaxPooling2D(" named_params ")"
        max_pooling3d_layer: "MaxPooling3D(" named_params ")"
        average_pooling_layer: average_pooling1d_layer | average_pooling2d_layer | average_pooling3d_layer
        average_pooling1d_layer: "AveragePooling1D(" named_params ")"
        average_pooling2d_layer: "AveragePooling2D(" named_params ")"
        average_pooling3d_layer: "AveragePooling3D(" named_params ")"
        global_pooling_layer: global_max_pooling_layer | global_average_pooling_layer
        global_max_pooling_layer: global_max_pooling1d_layer | global_max_pooling2d_layer | global_max_pooling3d_layer
        global_max_pooling1d_layer: "GlobalMaxPooling1D(" named_params ")"
        global_max_pooling2d_layer: "GlobalMaxPooling2D(" named_params ")"
        global_max_pooling3d_layer: "GlobalMaxPooling3D(" named_params ")"
        global_average_pooling_layer: global_average_pooling1d_layer | global_average_pooling2d_layer | global_average_pooling3d_layer
        global_average_pooling1d_layer: "GlobalAveragePooling1D(" named_params ")"
        global_average_pooling2d_layer: "GlobalAveragePooling2D(" named_params ")"
        global_average_pooling3d_layer: "GlobalAveragePooling3D(" named_params ")"
        adaptive_pooling_layer: adaptive_max_pooling_layer | adaptive_average_pooling_layer
        adaptive_max_pooling_layer: adaptive_max_pooling1d_layer | adaptive_max_pooling2d_layer | adaptive_max_pooling3d_layer
        adaptive_max_pooling1d_layer: "AdaptiveMaxPooling1D(" named_params ")"
        adaptive_max_pooling2d_layer: "AdaptiveMaxPooling2D(" named_params ")"
        adaptive_max_pooling3d_layer: "AdaptiveMaxPooling3D(" named_params ")"
        adaptive_average_pooling_layer: adaptive_average_pooling1d_layer | adaptive_average_pooling2d_layer | adaptive_average_pooling3d_layer
        adaptive_average_pooling1d_layer: "AdaptiveAveragePooling1D(" named_params ")"
        adaptive_average_pooling2d_layer: "AdaptiveAveragePooling2D(" named_params ")"
        adaptive_average_pooling3d_layer: "AdaptiveAveragePooling3D(" named_params ")"

        # Dropout layer for regularization
        dropout_layer: "Dropout(" named_params ")"

        # Normalization layers
        ?norm_layer: batch_norm_layer
                    | layer_norm_layer
                    | instance_norm_layer
                    | group_norm_layer

        batch_norm_layer: "BatchNormalization(" named_params ")"
        layer_norm_layer: "LayerNormalization(" named_params ")"
        instance_norm_layer: "InstanceNormalization(" named_params ")"
        group_norm_layer: "GroupNormalization(" named_params ")"

        # Basic layer types
        dense_layer: "Dense(" named_params ")"
        flatten_layer: "Flatten(" named_params ")"

        # Recurrent layers section - includes all RNN variants
        ?recurrent_layer: rnn_layer | bidirectional_rnn_layer | conv_rnn_layer | rnn_cell_layer
        rnn_layer: simple_rnn_layer | lstm_layer | gru_layer
        simple_rnn_layer: "SimpleRNN(" named_params ")"
        lstm_layer: "LSTM(" named_params ")"
        gru_layer: "GRU(" named_params ")"

        

        # Dropout wrapper layers for RNNs
        dropout_wrapper_layer: simple_rnn_dropout | gru_dropout | lstm_dropout
        simple_rnn_dropout: "SimpleRNNDropoutWrapper" "(" named_params ")"
        gru_dropout: "GRUDropoutWrapper" "(" named_params ")"
        lstm_dropout: "LSTMDropoutWrapper" "(" named_params ")"
        bidirectional_rnn_layer: bidirectional_simple_rnn_layer | bidirectional_lstm_layer | bidirectional_gru_layer
        bidirectional_simple_rnn_layer: "Bidirectional(SimpleRNN(" named_params "))"
        bidirectional_lstm_layer: "Bidirectional(LSTM(" named_params "))"
        bidirectional_gru_layer: "Bidirectional(GRU(" named_params "))"
        conv_rnn_layer: conv_lstm_layer | conv_gru_layer
        conv_lstm_layer: "ConvLSTM2D(" named_params ")"  # Add 2D for clarity
        conv_gru_layer: "ConvGRU2D(" named_params ")"  # Add 2D for clarity
        rnn_cell_layer: simple_rnn_cell_layer | lstm_cell_layer | gru_cell_layer
        simple_rnn_cell_layer: "SimpleRNNCell(" named_params ")"
        lstm_cell_layer: "LSTMCell(" named_params ")"
        gru_cell_layer: "GRUCell(" named_params ")"

        # Advanced layers group
        ?advanced_layer: attention_layer
                        | transformer_layer
                        | residual_layer
                        | inception_layer
                        | capsule_layer
                        | squeeze_excitation_layer
                        | graph_layer
                        | embedding_layer
                        | quantum_layer
                        | dynamic_layer
                        | noise_layer
                        | normalization_layer
                        | regularization_layer
                        | custom_layer

        # Attention and transformer mechanisms
        attention_layer: "Attention(" named_params ")"
        transformer_layer: "Transformer(" named_params ")" | "TransformerEncoder(" named_params ")" | "TransformerDecoder(" named_params ")"

        # Advanced architecture layers
        residual_layer: "Residual(" named_params ")"  # Shortened for consistency
        inception_layer: "Inception(" named_params ")"  # Shortened for consistency
        capsule_layer: "Capsule(" named_params ")"  # Shortened for consistency
        squeeze_excitation_layer: "SqueezeExcitation(" named_params ")"
        graph_layer: "GraphConv(" named_params ")" | "GraphAttention(" named_params ")"  # Added GraphAttention
        embedding_layer: "Embedding(" named_params ")"
        quantum_layer: "QuantumLayer(" named_params ")"
        dynamic_layer: "DynamicLayer(" named_params ")"
        noise_layer: "GaussianNoise(" named_params ")" | "GaussianDropout(" named_params ")" | "AlphaDropout(" named_params ")"
        normalization_layer: "BatchNormalization(" named_params ")" | "LayerNormalization(" named_params ")" | "InstanceNormalization(" named_params ")" | "GroupNormalization(" named_params ")"
        regularization_layer: "Dropout(" named_params ")" | "SpatialDropout1D(" named_params ")" | "SpatialDropout2D(" named_params ")" | "SpatialDropout3D(" named_params ")" | "ActivityRegularization(" named_params ")" | "L1L2(" named_params ")"

        custom_layer: NAME "(" named_params ")"  # Allow any custom layer name

        activation_layer: activation_with_params | activation_without_params
        activation_with_params: "Activation(" ESCAPED_STRING "," named_params ")" # e.g., Activation("leaky_relu", alpha=0.3)
        activation_without_params: "Activation(" ESCAPED_STRING ")" # e.g., Activation("relu")

        # Training configuration block
        training_config: "train" "{" training_params "}"
        training_params: (epochs_param | batch_size_param)*
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
    Transforms the parsed tree NUMBERo a structured dictionary representing the neural network model.
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
        

    # Basic Layers & Properties ###################

    def input_layer(self, items):
        shape = tuple(items[0])
        return {'type': 'Input', 'shape': shape}

    def output_layer(self, items):
        return {'type': 'Output', 'params': items[0]}

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
    

    def dense_layer(self, items):
        return {'type': 'Dense', 'params': items[0]}

    def loss(self, items):
        return items[0].value.strip('"')

    def optimizer(self, items):
        return items[0].value.strip('"')

    def layers(self, items):
        return items

    def flatten_layer(self, items):
        return {'type': 'Flatten', 'params': items[0]}

    def dropout_layer(self, items):
        return {'type': 'Dropout', 'params': items[0]}

    ### Training Configurations ############################""

    def training_config(self, items):
        config = {}
        for item in items:
            config.update(item)
        return config

    def shape(self, items):
        return tuple(items)

    def max_pooling2d_layer(self, items):
        return {"type": "MaxPooling2D", "params": items[0]}

    # End Basic Layers & Properties #########################

    def batch_norm_layer(self, items):
        return {'type': 'BatchNormalization', 'params': items[0]}

    def layer_norm_layer(self, items):
        return {'type': 'LayerNormalization', 'params': items[0]}

    def instance_norm_layer(self, items):
        return {'type': 'InstanceNormalization', 'params': items[0]}

    def group_norm_layer(self, items):
        return {'type': 'GroupNormalization', 'params': items[0]}

    def lstm_layer(self, items):
        return {'type': 'LSTM', 'params': items[0]}

    def gru_layer(self, items):
        return {'type': 'GRU', 'params': items[0]}

    def simple_rnn_layer(self, items):
        return {'type': 'SimpleRNN', 'params': items[0]}

    def cudnn_lstm_layer(self, items):
        return {'type': 'CuDNNLSTM', 'params': items[0]}

    def cudnn_gru_layer(self, items):
        return {'type': 'CuDNNGRU', 'params': items[0]}

    def rnn_cell_layer(self, items):
        return {'type': 'RNNCell', 'params': items[0]}

    def lstm_cell_layer(self, items):
        return {'type': 'LSTMCell', 'params': items[0]}

    def gru_cell_layer(self, items):
        return {'type': 'GRUCell', 'params': items[0]}

    def simple_rnn_dropout(self, items):
        return {"type": "SimpleRNNDropoutWrapper", 'params': items[0]}

    def gru_dropout(self, items):
        return {"type": "GRUDropoutWrapper", 'params': items[0]}

    def lstm_dropout(self, items):
        return {"type": "LSTMDropoutWrapper", 'params': items[0]}

    #### Advanced Layers #################################

    def attention_layer(self, items):
        return {'type': 'Attention', 'params': items[0]}

    def residual_layer(self, items):
        return {'type': 'ResidualConnection', 'params': items[0]}

    def inception_layer(self, items):
        return {'type': 'InceptionModule', 'params': items[0]}

    def dynamic_layer(self, items):
        return {'type': 'DynamicLayer', 'params': items[0]}

    ### Everything Research ##################

    def research(self, items):
        name = items[0].value if items and isinstance(items[0], Token) else None
        params = items[1] if len(items) > 1 else {}
        return {'type': 'Research', 'name': name, 'params': params}


    # NETWORK ACTIVATION ##############################

    def network(self, items):
        name = str(items[0].value)
        input_layer_config = items[1]
        layers_config = items[2]
        loss_config = items[3]
        optimizer_config = items[4]
        training_config = next((item for item in items[5:] if isinstance(item, dict)), {})
        execution_config = next((item for item in items[5:] if 'device' in item), {'device': 'auto'})

        output_layer = next((layer for layer in reversed(layers_config) if layer['type'] == 'Output'), None)
        if output_layer is None:
            output_layer = {'type': 'Output', 'params': {'units': 1, 'activation': 'linear'}}

        output_shape = output_layer.get('params', {}).get('units')
        if output_shape is not None:
            output_shape = (output_shape,)

        return {
            'type': 'model',
            'name': name,
            'input': input_layer_config,
            'layers': layers_config,
            'output_layer': output_layer,
            'output_shape': output_shape,
            'loss': loss_config,
            'optimizer': optimizer_config,
            'training_config': training_config,
            'execution_config': execution_config
        }


    # Parameters  #######################################################

    def named_param(self, items):
        name = str(items[0])
        value = items[2]
        if isinstance(value, Tree) and value.data == 'tuple_':
            value = self.tuple_(value.children)
        elif isinstance(value, Token) and value.type == 'BOOL':
            value = value.value == 'true'
        return {name: value}

    def named_params(self, items):
        params = {}
        for item in items:
            params.update(item)
        return params

    def value(self, items):
        item = items[0]
        if isinstance(item, Tree) and item.data == 'tuple_':
            return self.tuple_(item.children)
        elif isinstance(item, Token):
            if item.type == 'ESCAPED_STRING':
                return item.value.strip('"')
            elif item.type in ('NUMBER', 'FLOAT', 'BOOL'):
                return eval(item.value)
        return item

    def tuple_(self, items):
        return tuple(eval(item.value) for item in items if isinstance(item, Token) and item.type == 'NUMBER')
    
    def groups_param(self, items):
        return {'groups': items[2]}


    def research_params(self, items):
        params = {}
        for item in items:
            params.update(item)
        return params

    def metrics(self, items):
        return {'metrics': items}


    def accuracy_param(self, items):
        return {'accuracy': float(items[0].value)}

    def loss_param(self, items):
        return {'loss': float(items[0].value)}

    def precision_param(self, items):
        return {'precision': float(items[0].value)}

    def recall_param(self, items):
        return {'recall': float(items[0].value)}


    def references(self, items):
        return {'references': items}

    def paper_param(self, items):
        return items[0].value.strip('"')

    def epochs_param(self, items):
        return {'epochs': int(items[0].value)}

    def batch_size_param(self, items):
        return {'batch_size': int(items[0].value)}

    def device_param(self, items):
        return {'device': items[0].value.strip('"')}
    
    # Named_params & Their Properties ##################################################

    def bool_value(self, items):
        return items[0].value == 'true'

    def named_return_sequences(self, items):
        return {'return_sequences': items[2]}

    def named_num_heads(self, items):
        return {'num_heads': items[2]}

    def named_ff_dim(self, items):
        return {'ff_dim': items[2]}

    def named_input_dim(self, items):
        return {'input_dim': items[2]}

    def named_output_dim(self, items):
        return {'output_dim': items[2]}

    def number_or_none(self, items):
        if items[0].value.lower() == 'none':  # Case-insensitive check
            return None
        return eval(items[0].value)    # Evaluate the number

    def named_filters(self, items):
        return {'filters': items[2]}

    def named_activation(self, items):
        return {'activation': items[2]}

    def named_kernel_size(self, items):
        return {'kernel_size': items[2]}

    def named_padding(self, items):
        return {'padding': items[2]}

    def named_strides(self, items):
        return {'strides': items[2]}

    def named_rate(self, items):
        return {'rate': items[2]}

    def named_dilation_rate(self, items):
        return {'dilation_rate': items[2]}

    def named_groups(self, items):
        return {'groups': items[2]}

    def named_pool_size(self, items):
        return {'pool_size': items[2]}

    def named_dropout(self, items):
        return {"dropout": items[2]}


    #End named_params ################################################

    ### Advanced layers ###############################
    def capsule_layer(self, items):
        return {'type': 'CapsuleLayer', 'params': items[0]}

    def squeeze_excitation_layer(self, items):
        return {'type': 'SqueezeExcitation', 'params': items[0]}

    def graph_conv_layer(self, items):
        return {'type': 'GraphConv', 'params': items[0]}

    def quantum_layer(self, items):
        return {'type': 'QuantumLayer', 'params': items[0]}

    def transformer_layer(self, items):
        return {'type': 'TransformerEncoder', 'params': items[0]}

    def embedding_layer(self, items):
        return {'type': 'Embedding', 'params': items[0]}

    def execution_config(self, items: List[Tree]) -> Dict[str, Any]:
        """Processes execution configuration block."""
        config: Dict[str, Any] = {'type': 'ExecutionConfig'}
        if items:
            for item in items:
                if item.data == 'device_param':
                    config.update(self.device_param(item.children))
        return config

# Shape Propagation ##########################

def propagate_shape(input_shape: Tuple[Optional[NUMBER], ...], layer: Dict[str, Any]) -> Tuple[Optional[NUMBER], ...]:
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
        kernel_size = params.get('kernel_size')
        strides = params.get('strides', (1, 1)) # Default strides to 1x1
        padding_mode = params.get('padding', 'valid').lower() # Default to 'valid' padding
        dilation_rate = params.get('dilation_rate', (1, 1)) # Default dilation rate to 1x1

        if not filters or not kernel_size:
            raise ValueError("Conv2D layer requires 'filters' and 'kernel_size' parameters.")

        try:
            filters = NUMBER(filters)
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

        pool_size = params.get('pool_size')
        strides = params.get('strides', pool_size) # Default stride to pool_size if not provided
        padding_mode = params.get('padding', 'valid').lower() # Default to 'valid' padding

        if not pool_size:
            raise ValueError("MaxPooling2D layer requires 'pool_size' parameter.")

        try:
            pool_h, pool_w = map(NUMBER, pool_size)
            stride_h, stride_w = map(NUMBER, strides) if strides else map(NUMBER, pool_size) # If strides is None, use pool_size
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


    elif layer_type in ['Attention', 'TransformerEncoder', 'ResidualConnection', 'InceptionModule',
                        'CapsuleLayer', 'SqueezeExcitation', 'GraphConv', 'Embedding', 'QuantumLayer', 'DynamicLayer']:
        return input_shape # Placeholder for advanced layers, needs specific shape logic


    else:
        raise ValueError(f"Unsupported layer type: {layer_type}")

def generate_code(model_data,backend):
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

    indent = "    "

    if backend == "tensorflow":
        code = "import tensorflow as tf\n\n"
        code += "model = tf.keras.Sequential([\n"

        input_shape = model_data['input']['shape']
        if not input_shape:
            raise ValueError("Input layer shape is not defined in model_data.")
        input_layer_code = f"{indent}tf.keras.layers.Input(shape={input_shape}),\n"
        code += input_layer_code

        current_input_shape = input_shape
        for layer_config in model_data['layers']:
            layer_type = layer_config['type']
            params = layer_config.get('params', layer_config)

            if layer_type == 'Conv2D':
                filters = params.get('filters')
                kernel_size = params.get('kernel_size')
                activation = params.get('activation', 'relu')
                if not filters or not kernel_size:
                    raise ValueError("Conv2D layer config missing 'filters' or 'kernel_size'.")
                code += f"{indent}tf.keras.layers.Conv2D(filters={filters}, kernel_size={kernel_size}, activation='{activation}'),\n"
                current_input_shape = propagate_shape(current_input_shape, layer_config)

            elif layer_type == 'MaxPooling2D':
                pool_size = params.get('pool_size')
                if not pool_size:
                    raise ValueError("MaxPooling2D layer config missing 'pool_size'.")
                code += f"{indent}tf.keras.layers.MaxPool2D(pool_size={pool_size}),\n"
                current_input_shape = propagate_shape(current_input_shape, layer_config)

            elif layer_type == 'Flatten':
                code += f"{indent}tf.keras.layers.Flatten(),\n"
                current_input_shape = propagate_shape(current_input_shape, layer_config)

            elif layer_type == 'Dense':
                units = params.get('units')
                activation = params.get('activation', 'relu')
                if not units:
                    raise ValueError("Dense layer config missing 'units'.")
                code += f"{indent}tf.keras.layers.Dense(units={units}, activation='{activation}'),\n"
                current_input_shape = propagate_shape(current_input_shape, layer_config)

            elif layer_type == 'Dropout':
                rate = params.get('rate')
                if rate is None:
                    raise ValueError("Dropout layer config missing 'rate'.")
                code += f"{indent}tf.keras.layers.Dropout(rate={rate}),\n"
                current_input_shape = propagate_shape(current_input_shape, layer_config)

            elif layer_type == 'Output':
                units = params.get('units')
                activation = params.get('activation', 'linear')
                if not units:
                    raise ValueError("Output layer config missing 'units'.")
                code += f"{indent}tf.keras.layers.Dense(units={units}, activation='{activation}'),\n"
                current_input_shape = propagate_shape(current_input_shape, layer_config)

            elif layer_type == 'BatchNormalization':
                code += f"{indent}tf.keras.layers.BatchNormalization(),\n"
                current_input_shape = propagate_shape(current_input_shape, layer_config)

            elif layer_type == 'LayerNormalization':
                code += f"{indent}tf.keras.layers.LayerNormalization(),\n"
                current_input_shape = propagate_shape(current_input_shape, layer_config)

            elif layer_type == 'InstanceNormalization':
                code += f"{indent}tf.keras.layers.InstanceNormalization(),\n"
                current_input_shape = propagate_shape(current_input_shape, layer_config)

            elif layer_type == 'GroupNormalization':
                groups = params.get('groups')
                if groups is None:
                    raise ValueError("GroupNormalization layer config missing 'groups'.")
                code += f"{indent}tf.keras.layers.GroupNormalization(groups={groups}),\n"
                current_input_shape = propagate_shape(current_input_shape, layer_config)

            elif layer_type in ['LSTM', 'GRU', 'SimpleRNN', 'Bidirectional', 'CuDNNLSTM', 'CuDNNGRU']:
                units = params.get('units')
                return_sequences = params.get('return_sequences', False)
                if units is None:
                    raise ValueError(f"{layer_type} layer config missing 'units'.")
                tf_layer_name = layer_type
                if layer_type == 'SimpleRNN':
                    tf_layer_name = 'SimpleRNN'
                elif layer_type == 'CuDNNLSTM':
                    tf_layer_name = 'CuDNNLSTM'
                elif layer_type == 'CuDNNGRU':
                    tf_layer_name = 'CuDNNGRU'

                code += f"{indent}tf.keras.layers.{tf_layer_name}(units={units}, return_sequences={str(return_sequences).lower()}),\n"
                current_input_shape = propagate_shape(current_input_shape, layer_config)

            elif layer_type == 'Embedding':
                input_dim = params.get('input_dim')
                output_dim = params.get('output_dim')
                if not input_dim or not output_dim:
                    raise ValueError("Embedding layer config missing 'input_dim' or 'output_dim'.")
                code += f"{indent}tf.keras.layers.Embedding(input_dim={input_dim}, output_dim={output_dim}),\n"
                current_input_shape = propagate_shape(current_input_shape, layer_config)

            elif layer_type == 'Attention':
                code += f"{indent}tf.keras.layers.Attention(),\n"
                current_input_shape = propagate_shape(current_input_shape, layer_config)

            elif layer_type == 'TransformerEncoder':
                num_heads = params.get('num_heads', 4)
                ff_dim = params.get('ff_dim', 32)
                code += f"{indent}tf.keras.layers.TransformerEncoder(num_heads={num_heads}, ffn_units={ff_dim}),\n"
                current_input_shape = propagate_shape(current_input_shape, layer_config)

            elif layer_type in ['ResidualConnection', 'InceptionModule', 'CapsuleLayer', 'SqueezeExcitation', 'GraphConv', 'QuantumLayer', 'DynamicLayer']:
                prNUMBER(f"Warning: {layer_type} is an advanced or custom layer type. Code generation for TensorFlow might require manual implementation. Skipping layer code generation for now.")
                continue

            else:
                raise ValueError(f"Unsupported layer type: {layer_type} for TensorFlow backend.")

        code += "])\n\n"

        loss_value = model_data['loss']['value'].strip('"')
        optimizer_value = model_data['optimizer']['value'].strip('"')

        code += f"optimizer = tf.keras.optimizers.{optimizer_value}()\n"
        code += f"loss_fn = tf.keras.losses.{loss_value}\n"

        training_config = model_data.get('training_config')
        if training_config:
            epochs = training_config.get('epochs', 10)
            batch_size = training_config.get('batch_size', 32)

            code += "\n# Example training loop (requires data loading and handling)\n"
            code += "epochs = {}\n".format(epochs)
            code += "batch_size = {}\n".format(batch_size)
            code += "for epoch in range(epochs):\n"
            code += f"{indent}for batch_idx, (data, labels) in enumerate(dataset):\n"
            code += f"{indent}{indent}with tf.GradientTape() as tape:\n"
            code += f"{indent}{indent}{indent}predictions = model(data)\n"
            code += f"{indent}{indent}{indent}loss = loss_fn(labels, predictions)\n"
            code += f"{indent}{indent}gradients = tape.gradient(loss, model.trainable_variables)\n"
            code += f"{indent}{indent}optimizer.apply_gradients(zip(gradients, model.trainable_variables))\n"
            code += f"{indent}prNUMBER(f'Epoch {{epoch+1}}, Batch {{batch_idx}}, Loss: {{loss.numpy()}}')\n"
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
        current_input_shape = input_shape

        layers_code = []
        forward_code_body = []

        for i, layer_config in enumerate(model_data['layers']):
            layer_type = layer_config['type']
            params = layer_config.get('params', layer_config)
            layer_name = f"self.layer{i+1}"

            if layer_type == 'Conv2D':
                filters = params.get('filters')
                kernel_size = params.get('kernel_size')
                activation_name = params.get('activation', 'relu')
                padding = params.get('padding', 'same').lower()

                if not filters or not kernel_size:
                    raise ValueError("Conv2D layer config missing 'filters' or 'kernel_size'.")

                layers_code.append(f"{layer_name}_conv = nn.Conv2d(in_channels={current_input_shape[-1]}, out_channels={filters}, kernel_size={kernel_size}, padding='{padding}')")
                layers_code.append(f"{layer_name}_activation = nn.ReLU() if '{activation_name}' == 'relu' else nn.Identity()")
                forward_code_body.append(f"x = self.layer{i+1}_conv(x)")
                forward_code_body.append(f"x = self.layer{i+1}_activation(x)")
                current_input_shape = propagate_shape(current_input_shape, layer_config)

            elif layer_type == 'MaxPooling2D':
                pool_size = params.get('pool_size')
                if not pool_size:
                    raise ValueError("MaxPooling2D layer config missing 'pool_size'.")
                layers_code.append(f"{layer_name}_pool = nn.MaxPool2d(kernel_size={pool_size})")
                forward_code_body.append(f"x = self.layer{i+1}_pool(x)")
                current_input_shape = propagate_shape(current_input_shape, layer_config)

            elif layer_type == 'Flatten':
                layers_code.append(f"{layer_name}_flatten = nn.Flatten()")
                forward_code_body.append(f"x = self.layer{i+1}_flatten(x)")
                current_input_shape = propagate_shape(current_input_shape, layer_config)

            elif layer_type == 'Dense':
                units = params.get('units')
                activation_name = params.get('activation', 'relu')
                if not units:
                    raise ValueError("Dense layer config missing 'units'.")
                layers_code.append(f"{layer_name}_dense = nn.Linear(in_features={np.prod(current_input_shape[1:]) if len(current_input_shape)>1 else current_input_shape[1]}, out_features={units})")
                layers_code.append(f"{layer_name}_activation = nn.ReLU() if '{activation_name}' == 'relu' else nn.Identity()")
                forward_code_body.append(f"x = self.layer{i+1}_dense(x)")
                forward_code_body.append(f"x = self.layer{i+1}_activation(x)")
                current_input_shape = propagate_shape(current_input_shape, layer_config)

            elif layer_type == 'Dropout':
                rate = params.get('rate')
                if rate is None:
                    raise ValueError("Dropout layer config missing 'rate'.")
                layers_code.append(f"{layer_name}_dropout = nn.Dropout(p={rate})")
                forward_code_body.append(f"x = self.layer{i+1}_dropout(x)")
                current_input_shape = propagate_shape(current_input_shape, layer_config)

            elif layer_type == 'Output':
                units = params.get('units')
                activation_name = params.get('activation', 'linear')
                if not units:
                    raise ValueError("Output layer config missing 'units'.")
                layers_code.append(f"{layer_name}_output = nn.Linear(in_features={np.prod(current_input_shape[1:]) if len(current_input_shape)>1 else current_input_shape[1]}, out_features={units})")
                layers_code.append(f"{layer_name}_activation = nn.Sigmoid() if '{activation_name}' == 'sigmoid' else (nn.Softmax(dim=1) if '{activation_name}' == 'softmax' else nn.Identity())")
                forward_code_body.append(f"x = self.layer{i+1}_output(x)")
                forward_code_body.append(f"x = self.layer{i+1}_activation(x)")
                current_input_shape = propagate_shape(current_input_shape, layer_config)

            elif layer_type == 'BatchNormalization':
                layers_code.append(f"{layer_name}_bn = nn.BatchNorm2d(num_features={current_input_shape[-1]})")
                forward_code_body.append(f"x = self.layer{i+1}_bn(x)")
                current_input_shape = propagate_shape(current_input_shape, layer_config)

            elif layer_type in ['LSTM', 'GRU', 'SimpleRNN', 'CuDNNLSTM', 'CuDNNGRU']:
                units = params.get('units')
                return_sequences = params.get('return_sequences', False)
                if units is None:
                    raise ValueError(f"{layer_type} layer config missing 'units'.")
                torch_layer_name = layer_type
                if layer_type == 'SimpleRNN':
                    torch_layer_name = 'RNN'
                elif layer_type == 'CuDNNLSTM':
                    torch_layer_name = 'LSTM'
                elif layer_type == 'CuDNNGRU':
                    torch_layer_name = 'GRU'

                layers_code.append(f"{layer_name}_rnn = nn.{torch_layer_name}(input_size={current_input_shape[-1]}, hidden_size={units}, batch_first=True, bidirectional=False)")
                forward_code_body.append(f"x, _ = self.layer{i+1}_rnn(x)")
                current_input_shape = propagate_shape(current_input_shape, layer_config)

            elif layer_type == 'Flatten':
                layers_code.append(f"{layer_name}_flatten = nn.Flatten(start_dim=1)")
                forward_code_body.append(f"x = self.layer{i+1}_flatten(x)")
                current_input_shape = propagate_shape(current_input_shape, layer_config)

            elif layer_type in ['Attention', 'TransformerEncoder', 'ResidualConnection', 'InceptionModule', 'CapsuleLayer', 'SqueezeExcitation', 'GraphConv', 'Embedding', 'QuantumLayer', 'DynamicLayer']:
                prNUMBER(f"Warning: {layer_type} is an advanced or custom layer type. Code generation for PyTorch might require manual implementation. Skipping layer code generation for now.")
            else:
                raise ValueError(f"Unsupported layer type: {layer_type} for PyTorch backend.")

        code += indent + indent + "# Layer Definitions\n"
        for layer_init_code in layers_code:
            code += indent + indent + layer_init_code + "\n"
        code += "\n"

        code += indent + "def forward(self, x):\n"
        code += indent + indent + "# Forward Pass\n"
        code += indent + indent + "batch_size, h, w, c = x.size()\n"
        for forward_op in forward_code_body:
            code += indent + indent + forward_op + "\n"
        code += indent + indent + "return x\n"

        code += "model = NeuralNetworkModel()\n"
        code += "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
        code += "model.to(device)\n\n"

        loss_value = model_data['loss']['value'].strip('"')
        optimizer_value = model_data['optimizer']['value'].strip('"')

        if loss_value.lower() == 'categorical_crossentropy' or loss_value.lower() == 'sparse_categorical_crossentropy':
            loss_fn_code = "loss_fn = nn.CrossEntropyLoss()"
        elif loss_value.lower() == 'mean_squared_error':
            loss_fn_code = "loss_fn = nn.MSELoss()"
        else:
            loss_fn_code = f"loss_fn = nn.{loss_value}()"
            prNUMBER(f"Warning: Loss function '{loss_value}' might not be directly supported in PyTorch. Verify the name and compatibility.")

        code += loss_fn_code + "\n"
        code += f"optimizer = optim.{optimizer_value}(model.parameters(), lr=0.001)\n\n"

        training_config = model_data.get('training_config')
        if training_config:
            epochs = training_config.get('epochs', 10)
            batch_size = training_config.get('batch_size', 32)

            code += "# Example training loop (requires data loader setup)\n"
            code += f"epochs = {epochs}\n"
            code += f"batch_size = {batch_size}\n"
            code += "for epoch in range(epochs):\n"
            code += indent + "for batch_idx, (data, target) in enumerate(train_loader):\n"
            code += indent + indent + "data, target = data.to(device), target.to(device)\n"
            code += indent + indent + "optimizer.zero_grad()\n"
            code += indent + indent + "output = model(data)\n"
            code += indent + indent + "loss = loss_fn(output, target)\n"
            code += indent + indent + "loss.backward()\n"
            code += indent + indent + "optimizer.step()\n"
            code += indent + indent + "if batch_idx % 100 == 0:\n"
            code += indent + indent + indent + f"prNUMBER('Epoch: {{epoch+1}} [{{batch_idx*len(data)}}/{{len(train_loader.dataset)}} ({{100.*batch_idx/len(train_loader):.0f}}%)]\\tLoss: {{loss.item():.6f}}')\n"
            code += "prNUMBER('Finished Training')\n"

        else:
            code += "# No training configuration provided. Training loop needs manual implementation.\n"

        return code

    else:
        raise ValueError(f"Unsupported backend: {backend}. Choose 'tensorflow' or 'pytorch'.")

def save_file(filename: str, content: str) -> None:
    """
    Saves the provided content to a file.

    Args:
        filename (str): The path to the file to save.
        content (str): The content to save.
    """
    try:
        with open(filename, 'w') as f:
            f.write(content)
    except Exception as e:
        raise IOError(f"Error writing file: {filename}. {e}") from e
    prNUMBER(f"Successfully saved file: {filename}")
    return None


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

