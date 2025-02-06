import lark
from lark import Tree, Transformer, Token
from typing import Any, Dict, List, Tuple, Union, Optional, Callable

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
        value: ESCAPED_STRING | tuple_ | number | BOOL  
        activation_param: "activation" "=" ESCAPED_STRING
        ordered_params: value ("," value)* 
        tuple_: "(" WS_INLINE* number WS_INLINE* "," WS_INLINE* number WS_INLINE* ")"  
        number: NUMBER  
        BOOL: "true" | "false"

        # name_param rules
        bool_value: BOOL  // Example: true or false
        named_return_sequences: "return_sequences" "=" bool_value
        named_units: "units" "=" NUMBER  // Example: units=64
        named_activation: "activation" "=" ESCAPED_STRING  
        named_filters: "filters" "=" NUMBER  // Example: filters=32
        named_kernel_size: "kernel_size" "=" value  
        named_strides: "strides" "=" value  // Example: strides=(1, 1) or strides=1
        named_padding: "padding" "=" value  // Example: padding="same" or padding="valid"
        named_dilation_rate: "dilation_rate" "=" value  // Example: dilation_rate=(2, 2) or dilation_rate=2
        named_groups: "groups" "=" NUMBER  // Example: groups=32
        named_channels: "channels" "=" NUMBER  // Example: channels=3
        named_pool_size: "pool_size" "=" value  // Example: pool_size=(2, 2) or pool_size=2
        named_num_heads: "num_heads" "=" NUMBER  // Example: num_heads=8
        named_ff_dim: "ff_dim" "=" NUMBER  // Example: ff_dim=512
        named_input_dim: "input_dim" "=" NUMBER  // Example: input_dim=1000
        named_output_dim: "output_dim" "=" NUMBER  // Example: output_dim=128
        named_rate: "rate" "=" FLOAT  // Example: rate=0.5
        named_dropout: "dropout" "=" FLOAT  // Example: dropout=0.2
        named_axis: "axis" "=" NUMBER  // Example: axis=1
        named_momentum: "momentum" "=" FLOAT  // Example: momentum=0.9
        named_epsilon: "epsilon" "=" FLOAT  // Example: epsilon=1e-05
        named_center: "center" "=" BOOL  // Example: center=true
        named_scale: "scale" "=" BOOL  // Example: scale=true
        named_beta_initializer: "beta_initializer" "=" ESCAPED_STRING  // Example: beta_initializer="zeros"
        named_gamma_initializer: "gamma_initializer" "=" ESCAPED_STRING  // Example: gamma_initializer="ones"
        named_moving_mean_initializer: "moving_mean_initializer" "=" ESCAPED_STRING  // Example: moving_mean_initializer="zeros"
        named_moving_variance_initializer: "moving_variance_initializer" "=" ESCAPED_STRING  // Example: moving_variance_initializer="ones"
        named_training: "training" "=" BOOL  // Example: training=true
        named_trainable: "trainable" "=" BOOL  // Example: trainable=false
        named_use_bias: "use_bias" "=" BOOL  // Example: use_bias=true
        named_kernel_initializer: "kernel_initializer" "=" ESCAPED_STRING  // Example: kernel_initializer="glorot_uniform"
        named_bias_initializer: "bias_initializer" "=" ESCAPED_STRING  // Example: bias_initializer="zeros"
        named_kernel_regularizer: "kernel_regularizer" "=" ESCAPED_STRING  // Example: kernel_regularizer="l2"
        named_bias_regularizer: "bias_regularizer" "=" ESCAPED_STRING  // Example: bias_regularizer="l1"
        named_activity_regularizer: "activity_regularizer" "=" ESCAPED_STRING  // Example: activity_regularizer="l1_l2"
        named_kernel_constraint: "kernel_constraint" "=" ESCAPED_STRING  // Example: kernel_constraint="max_norm"
        named_bias_constraint: "bias_constraint" "=" ESCAPED_STRING  // Example: bias_constraint="min_max_norm"
        named_return_state: "return_state" "=" BOOL  // Example: return_state=true
        named_go_backwards: "go_backwards" "=" BOOL  // Example: go_backwards=false
        named_stateful: "stateful" "=" BOOL  // Example: stateful=true
        named_time_major: "time_major" "=" BOOL  // Example: time_major=false
        named_unroll: "unroll" "=" BOOL  // Example: unroll=true
        named_input_shape: "input_shape" "=" value  // Example: input_shape=(28, 28, 1)
        named_batch_input_shape: "batch_input_shape" "=" value  // Example: batch_input_shape=(None, 32, 32, 3)
        named_dtype: "dtype" "=" ESCAPED_STRING  // Example: dtype="float32"
        named_name: "name" "=" ESCAPED_STRING  // Example: name="my_layer"
        named_weights: "weights" "=" value  // Example: weights=[...]
        named_embeddings_initializer: "embeddings_initializer" "=" ESCAPED_STRING  // Example: embeddings_initializer="uniform"
        named_mask_zero: "mask_zero" "=" BOOL  // Example: mask_zero=true
        named_input_length: "input_length" "=" NUMBER  // Example: input_length=100
        named_embeddings_regularizer: "embeddings_regularizer" "=" ESCAPED_STRING  // Example: embeddings_regularizer="l1"
        named_embeddings_constraint: "embeddings_constraint" "=" value // Example: embeddings_constraint="non_neg"
        named_num_layers: "num_layers" "=" NUMBER // Example: num_layers=2
        named_bidirectional: "bidirectional" "=" BOOL // Example: bidirectional=true
        named_merge_mode: "merge_mode" "=" ESCAPED_STRING // Example: merge_mode="concat"
        named_recurrent_dropout: "recurrent_dropout" "=" FLOAT // Example: recurrent_dropout=0.1
        named_noise_shape: "noise_shape" "=" value // Example: noise_shape=(3,)
        named_seed: "seed" "=" NUMBER // Example: seed=42
        named_target_shape: "target_shape" "=" value // Example: target_shape=(64, 64)
        named_data_format: "data_format" "=" ESCAPED_STRING // Example: data_format="channels_first"
        named_interpolation: "interpolation" "=" ESCAPED_STRING // Example: interpolation="nearest"
        named_crop_to_aspect_ratio: "crop_to_aspect_ratio" "=" BOOL // Example: crop_to_aspect_ratio=true
        named_mask_value: "mask_value" "=" NUMBER // Example: mask_value=0
        named_return_attention_scores: "return_attention_scores" "=" BOOL // Example: return_attention_scores=true
        named_causal: "causal" "=" BOOL // Example: causal=false
        named_use_scale: "use_scale" "=" BOOL // Example: use_scale=true
        named_key_dim: "key_dim" "=" NUMBER // Example: key_dim=64
        named_value_dim: "value_dim" "=" NUMBER 
        named_output_shape: "output_shape" "=" value 
        named_arguments: "arguments" "=" value 
        named_initializer: "initializer" "=" ESCAPED_STRING 
        named_regularizer: "regularizer" "=" ESCAPED_STRING 
        named_constraint: "constraint" "=" ESCAPED_STRING
        named_l1: "l1" "=" FLOAT  // Example: l1=0.01
        named_l2: "l2" "=" FLOAT  // Example: l2=0.001
        named_l1_l2: "l1_l2" "=" tuple_  // Example: l1_l2=(0.01, 0.001)
        ?named_param: (named_units | named_activation | named_filters | named_kernel_size | kernel_size_tuple | named_strides | named_padding | named_dilation_rate | named_groups | named_data_format | named_channels | named_pool_size | named_return_sequences | named_num_heads | named_ff_dim | named_input_dim | named_output_dim | named_rate | named_dropout | named_axis | named_momentum | named_epsilon | named_center | named_scale | named_beta_initializer | named_gamma_initializer | named_moving_mean_initializer | named_moving_variance_initializer | named_training | named_trainable | named_use_bias | named_kernel_initializer | named_bias_initializer | named_kernel_regularizer | named_bias_regularizer | named_activity_regularizer | named_kernel_constraint | named_bias_constraint | named_return_state | named_go_backwards | named_stateful | named_time_major | named_unroll | named_input_shape | named_batch_input_shape | named_dtype | named_name | named_weights | named_embeddings_initializer | named_mask_zero | named_input_length | named_embeddings_regularizer | named_embeddings_constraint | named_num_layers | named_bidirectional | named_merge_mode | named_recurrent_dropout | named_noise_shape | named_seed | named_target_shape | named_interpolation | named_crop_to_aspect_ratio | named_mask_value | named_return_attention_scores | named_causal | named_use_scale | named_key_dim | named_value_dim | named_output_shape | named_arguments | named_initializer | named_regularizer | named_constraint | named_l1 | named_l2 | named_l1_l2 | named_int | named_float | named_number | number_param | string_param)  // Added string_param
        number_param: NUMBER
        string_param: ESCAPED_STRING
        named_int: NAME "=" INT
        named_float: NAME "=" FLOAT
        named_number: NAME "=" NUMBER
        kernel_size_tuple: tuple_

        layer_param: named_param | number_param | string_param | kernel_size_tuple | bool_value

        # Layer parameter styles
        ?param_style1: named_params    // Dense(units=128, activation="relu")
                    | ordered_params 

        # Top-level network definition - defines the structure of an entire neural network
        network: "network" NAME "{" input_layer layers loss optimizer [training_config] [execution_config] "}" 

        # Configuration can be either training-related or execution-related
        config: training_config | execution_config

        # Input layer definition - specifies the shape of input data
        input_layer: "input" ":" "(" shape ")"
        # Shape can contain multiple dimensions, each being a number or None
        shape: number_or_none ("," number_or_none)*
        # Dimensions can be specific NUMBERegers or None (for variable dimensions)
        number_or_none: number | "None"

        # Layers section - contains all layer definitions separated by newlines
        layers: "layers" ":" _NL* layer+ _NL*

        # All possible layer types that can be used in the network
        ?layer: (basic | recurrent | advanced | activation | merge | noise | normalization | regularization | custom | wrapper | lambda_ )  
        lambda_: "Lambda(" ESCAPED_STRING ")"
        wrapper: wrapper_layer_type "(" layer "," named_params ")"  
        wrapper_layer_type: "TimeDistributed" 

        # Basic layers group
        ?basic: conv | pooling | dropout | flatten | dense | output
        dropout: "Dropout(" named_params ")"

        regularization: spatial_dropout1d | spatial_dropout2d | spatial_dropout3d | activity_regularization | l1 | l2 | l1_l2 
        l1: "L1(" named_params ")"
        l2: "L2(" named_params ")"
        l1_l2: "L1L2(" named_params ")" 


        # Output layer 
        output: "Output(" named_params ")"

        # Convolutional layers 
        conv: conv1d | conv2d | conv3d | conv_transpose | depthwise_conv2d | separable_conv2d
        conv1d: "Conv1D(" named_params ")"
        conv2d: "Conv2D(" named_params ")"
        conv3d: "Conv3D(" named_params ")"
        conv_transpose: conv1d_transpose | conv2d_transpose | conv3d_transpose
        conv1d_transpose: "Conv1DTranspose(" named_params ")"
        conv2d_transpose: "Conv2DTranspose(" named_params ")"
        conv3d_transpose: "Conv3DTranspose(" named_params ")"
        depthwise_conv2d: "DepthwiseConv2D(" named_params ")"
        separable_conv2d: "SeparableConv2D(" named_params ")"

        # Pooling layer parameters
        pooling: max_pooling | average_pooling | global_pooling | adaptive_pooling
        max_pooling: max_pooling1d | max_pooling2d | max_pooling3d
        max_pooling1d: "MaxPooling1D(" named_params ")"
        max_pooling2d: "MaxPooling2D(" named_params ")"
        max_pooling3d: "MaxPooling3D(" named_params ")"
        average_pooling: average_pooling1d | average_pooling2d | average_pooling3d
        average_pooling1d: "AveragePooling1D(" named_params ")"
        average_pooling2d: "AveragePooling2D(" named_params ")"
        average_pooling3d: "AveragePooling3D(" named_params ")"
        global_pooling: global_max_pooling | global_average_pooling
        global_max_pooling: global_max_pooling1d | global_max_pooling2d | global_max_pooling3d
        global_max_pooling1d: "GlobalMaxPooling1D(" named_params ")"
        global_max_pooling2d: "GlobalMaxPooling2D(" named_params ")"
        global_max_pooling3d: "GlobalMaxPooling3D(" named_params ")"
        global_average_pooling: global_average_pooling1d | global_average_pooling2d | global_average_pooling3d
        global_average_pooling1d: "GlobalAveragePooling1D(" named_params ")"
        global_average_pooling2d: "GlobalAveragePooling2D(" named_params ")"
        global_average_pooling3d: "GlobalAveragePooling3D(" named_params ")"
        adaptive_pooling: adaptive_max_pooling | adaptive_average_pooling
        adaptive_max_pooling: adaptive_max_pooling1d | adaptive_max_pooling2d | adaptive_max_pooling3d
        adaptive_max_pooling1d: "AdaptiveMaxPooling1D(" named_params ")"
        adaptive_max_pooling2d: "AdaptiveMaxPooling2D(" named_params ")"
        adaptive_max_pooling3d: "AdaptiveMaxPooling3D(" named_params ")"
        adaptive_average_pooling: adaptive_average_pooling1d | adaptive_average_pooling2d | adaptive_average_pooling3d
        adaptive_average_pooling1d: "AdaptiveAveragePooling1D(" named_params ")"
        adaptive_average_pooling2d: "AdaptiveAveragePooling2D(" named_params ")"
        adaptive_average_pooling3d: "AdaptiveAveragePooling3D(" named_params ")"

        # Normalization layers
        ?norm_layer: batch_norm | layer_norm | instance_norm | group_norm
        batch_norm: "BatchNormalization(" named_params ")"
        layer_norm: "LayerNormalization(" named_params ")"
        instance_norm: "InstanceNormalization(" named_params ")"
        group_norm: "GroupNormalization(" named_params ")"
        

        # Basic layer types
        dense: "Dense(" named_params ")"
        flatten: "Flatten(" named_params ")"

        # Recurrent layers section - includes all RNN variants
        ?recurrent: rnn | bidirectional_rnn | conv_rnn | rnn_cell  
        bidirectional_rnn: "Bidirectional(" rnn "," named_params ")" 
        rnn: simple_rnn | lstm | gru
        simple_rnn: "SimpleRNN(" named_params ")"
        lstm: "LSTM(" named_params ")"
        gru: "GRU(" named_params ")"


        conv_rnn: conv_lstm | conv_gru
        conv_lstm: "ConvLSTM2D(" named_params ")"
        conv_gru: "ConvGRU2D(" named_params ")"

        rnn_cell: simple_rnn_cell | lstm_cell | gru_cell
        simple_rnn_cell: "SimpleRNNCell(" named_params ")"
        lstm_cell: "LSTMCell(" named_params ")"
        gru_cell: "GRUCell(" named_params ")"

        

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
        ?advanced: attention | transformer | residual | inception | capsule | squeeze_excitation | graph | embedding | quantum | dynamic
        attention: "Attention(" named_params ")"
        transformer: "Transformer(" named_params ")" | "TransformerEncoder(" named_params ")" | "TransformerDecoder(" named_params ")"
        residual: "Residual(" named_params ")"
        inception: "Inception(" named_params ")"
        capsule: "Capsule(" named_params ")"
        squeeze_excitation: "SqueezeExcitation(" named_params ")"
        graph: graph_conv | graph_attention
        graph_conv: "GraphConv(" named_params ")"
        graph_attention: "GraphAttention(" named_params ")"
        embedding: "Embedding(" named_params ")"
        quantum: "QuantumLayer(" named_params ")"
        dynamic: "DynamicLayer(" named_params ")"

        merge: add | subtract | multiply | average | maximum | concatenate | dot
        add: "Add(" named_params ")"
        subtract: "Subtract(" named_params ")"
        multiply: "Multiply(" named_params ")"
        average: "Average(" named_params ")"
        maximum: "Maximum(" named_params ")"
        concatenate: "Concatenate(" named_params ")"
        dot: "Dot(" named_params ")"

        noise: gaussian_noise | gaussian_dropout | alpha_dropout
        gaussian_noise: "GaussianNoise(" named_params ")"
        gaussian_dropout: "GaussianDropout(" named_params ")"
        alpha_dropout: "AlphaDropout(" named_params ")"

        normalization: batch_normalization | layer_normalization | instance_normalization | group_normalization
        batch_normalization: "BatchNormalization(" named_params ")"
        layer_normalization: "LayerNormalization(" named_params ")"
        instance_normalization: "InstanceNormalization(" named_params ")"
        group_normalization: "GroupNormalization(" named_params ")"

        spatial_dropout1d: "SpatialDropout1D(" named_params ")"
        spatial_dropout2d: "SpatialDropout2D(" named_params ")"
        spatial_dropout3d: "SpatialDropout3D(" named_params ")"
        activity_regularization: "ActivityRegularization(" named_params ")"

        custom: NAME "(" named_params ")"

        activation: activation_with_params | activation_without_params
        activation_with_params: "Activation(" ESCAPED_STRING "," named_params ")"
        activation_without_params: "Activation(" ESCAPED_STRING ")"

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
    def layer(self, items):
        layer_type = items[0].data
        params = self.visit(items[0].children[0]) if items[0].children else {}
        return {'type': layer_type, 'params': params}
        
    def wrapper(self, items):
        wrapper_type = items[0]
        layer = items[1]
        params = items[2]
        # Merge layer params with wrapper params
        layer['params'].update(params)
        return {'type': f"{wrapper_type}({layer['type']})", 'params': layer['params']}

    # Basic Layers & Properties ###################

    def input_layer(self, items):
        shape = tuple(items[0])
        return {'type': 'Input', 'shape': shape}

    def output(self, items):
        return {'type': 'Output', 'params': items[0]}
    
    def regularization(self, items):  # Added method to handle regularization layers
        return {'type': items[0].data.capitalize(), 'params': items[0].children[0]}

    ### Convolutional Layers ####################
    def conv1d(self, items):
        return {'type': 'Conv1D', 'params': items[0]}

    def conv2d(self, items):
        return {'type': 'Conv2D', 'params': items[0]}

    def conv3d(self, items):
        return {'type': 'Conv3D', 'params': items[0]}
    
    def conv1d_transpose(self, items):
        return {'type': 'Conv1DTranspose', 'params': items[0]}

    def conv2d_transpose(self, items):
        return {'type': 'Conv2DTranspose', 'params': items[0]}

    def conv3d_transpose(self, items):
        return {'type': 'Conv3DTranspose', 'params': items[0]}

    def depthwise_conv2d(self, items):
        return {'type': 'DepthwiseConv2D', 'params': items[0]}

    def separable_conv2d(self, items):
        return {'type': 'SeparableConv2D', 'params': items[0]}
    
    def dense(self, items):
        return {
            'type': 'Dense',
            'params': items[0]
        }

    def loss(self, items):
        return items[0].value.strip('"')

    def optimizer(self, items):
        return items[0].value.strip('"')

    def layers(self, items):
        return items

    def flatten(self, items):
        return {'type': 'Flatten', 'params': items[0]}

    def dropout(self, items):
        return {'type': 'Dropout', 'params': items[0]}

    ### Training Configurations ############################""

    def training_config(self, items):
        config = {}
        for item in items:
            config.update(item)
        return config

    def shape(self, items):
        return tuple(items)
    
    ### Pooling Layers #############################

    def max_pooling1d(self, items):
        return {'type': 'MaxPooling1D', 'params': items[0]}

    def max_pooling2d(self, items):
        return {'type': 'MaxPooling2D', 'params': items[0]}

    def max_pooling3d(self, items):
        return {'type': 'MaxPooling3D', 'params': items[0]}
    
    def average_pooling1d(self, items):
        return {'type': 'AveragePooling1D', 'params': items[0]}

    def average_pooling2d(self, items):
        return {'type': 'AveragePooling2D', 'params': items[0]}

    def average_pooling3d(self, items):
        return {'type': 'AveragePooling3D', 'params': items[0]}

    def global_max_pooling1d(self, items):
        return {'type': 'GlobalMaxPooling1D', 'params': items[0]}

    def global_max_pooling2d(self, items):
        return {'type': 'GlobalMaxPooling2D', 'params': items[0]}

    def global_max_pooling3d(self, items):
        return {'type': 'GlobalMaxPooling3D', 'params': items[0]} 

    def global_average_pooling1d(self, items):
        return {'type': 'GlobalAveragePooling1D', 'params': items[0]}

    def global_average_pooling2d(self, items):
        return {'type': 'GlobalAveragePooling2D', 'params': items[0]}

    def global_average_pooling3d(self, items):
        return {'type': 'GlobalAveragePooling3D', 'params': items[0]}
    
    def adaptive_max_pooling1d(self, items):
        return {'type': 'AdaptiveMaxPooling1D', 'params': items[0]}

    def adaptive_max_pooling2d(self, items):
        return {'type': 'AdaptiveMaxPooling2D', 'params': items[0]}

    def adaptive_max_pooling3d(self, items):
        return {'type': 'AdaptiveMaxPooling3D', 'params': items[0]}
    
    def adaptive_average_pooling1d(self, items):
        return {'type': 'AdaptiveAveragePooling1D', 'params': items[0]}

    def adaptive_average_pooling2d(self, items):
        return {'type': 'AdaptiveAveragePooling2D', 'params': items[0]}

    def adaptive_average_pooling3d(self, items):
        return {'type': 'AdaptiveAveragePooling3D', 'params': items[0]}

    # End Basic Layers & Properties #########################

    def batch_norm(self, items):
        return {'type': 'BatchNormalization', 'params': items[0]}

    def layer_norm(self, items):
        return {'type': 'LayerNormalization', 'params': items[0]}

    def instance_norm(self, items):
        return {'type': 'InstanceNormalization', 'params': items[0]}

    def group_norm(self, items):
        return {'type': 'GroupNormalization', 'params': items[0]}

    def lstm(self, items):
        return {'type': 'LSTM', 'params': items[0]}

    def gru(self, items):
        return {'type': 'GRU', 'params': items[0]}
    
    ### Recurrent Layers ############

    def simple_rnn(self, items):
        return {'type': 'SimpleRNN', 'params': items[0]}
    

    
    def conv_lstm(self, items):
        return {'type': 'ConvLSTM2D', 'params': items[0]}

    def conv_gru(self, items):
        return {'type': 'ConvGRU2D', 'params': items[0]}

    def bidirectional_rnn(self, items):
        rnn_layer = items[0]
        bidirectional_params = items[1]
        rnn_layer['params'].update(bidirectional_params)  # Merge params
        return {'type': f"Bidirectional({rnn_layer['type']})", 'params': rnn_layer['params']}

    def cudnn_gru_layer(self, items):  # No such thing as CuDNNGRU in PyTorch
        return {'type': 'GRU', 'params': items[0]}

    def bidirectional_simple_rnn_layer(self, items):
        return {'type': 'Bidirectional(SimpleRNN)', 'params': items[0]}

    def bidirectional_lstm_layer(self, items):
        return {'type': 'Bidirectional(LSTM)', 'params': items[0]}

    def bidirectional_gru_layer(self, items):
        return {'type': 'Bidirectional(GRU)', 'params': items[0]}

    def conv_lstm_layer(self, items):
        return {'type': 'ConvLSTM2D', 'params': items[0]}

    def conv_gru_layer(self, items):
        return {'type': 'ConvGRU2D', 'params': items[0]}

    def rnn_cell_layer(self, items):
        return {'type': 'RNNCell', 'params': items[0]}
    
    def simple_rnn_cell(self, items):
        return {'type': 'SimpleRNNCell', 'params': items[0]}

    def lstm_cell(self, items):
        return {'type': 'LSTMCell', 'params': items[0]}

    def gru_cell(self, items):
        return {'type': 'GRUCell', 'params': items[0]}

    def simple_rnn_dropout(self, items):
        return {"type": "SimpleRNNDropoutWrapper", 'params': items[0]}

    def gru_dropout(self, items):
        return {"type": "GRUDropoutWrapper", 'params': items[0]}

    def lstm_dropout(self, items):
        return {"type": "LSTMDropoutWrapper", 'params': items[0]}

    #### Advanced Layers #################################

    def attention(self, items):
        return {'type': 'Attention', 'params': items[0]}

    def residual(self, items):
        return {'type': 'Residual', 'params': items[0]}

    def inception(self, items):
        return {'type': 'Inception', 'params': items[0]}


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


    ### Parameters  #######################################################

    def _extract_value(self, item):  # Helper function to extract values from tokens and tuples
        if isinstance(item, Token):
            if item.type in ('INT', 'FLOAT', 'NUMBER', 'SIGNED_NUMBER'):
                try:
                    return int(item)
                except ValueError:
                    return float(item)
            elif item.type == 'BOOL':
                return item.value.lower() == 'true'
            elif item.type == 'ESCAPED_STRING':
                return item.value.strip('"')
        elif isinstance(item, list):  # Handles nested lists
            return [self._extract_value(elem) for elem in item]
        elif isinstance(item, dict):  # Handles nested dictionaries
            return {k: self._extract_value(v) for k, v in item.items()}
        elif isinstance(item, Tree):  # Handles all Tree types, not just tuple_
            if item.data == 'tuple_':
                return tuple(self._extract_value(child) for child in item.children)
            else:  # Extract values from other tree types
                return {k: self._extract_value(v) for k, v in zip(item.children[::2], item.children[1::2])}

        return item
    
    def named_param(self, items):  # Corrected to use _extract_value and return a dictionary
        name = str(items[0])
        value = self._extract_value(items[2])  # Extract the value using the helper function
        return {name: self._extract_value(value)}
    
    def number_param(self, items):
        return {"units": self._extract_value(items[0])}

    def string_param(self, items):
        return {"activation": self._extract_value(items[0])}

    def number(self, items):
        """Handles standalone numbers."""
        return eval(items[0].value)  # Evaluate the number

    def string(self, items):
        """Handles standalone strings."""
        return items[0].value.strip('"')
    
    def number_or_none(self, items):
        if items[0].value.lower() == 'none':
            return None
        return int(items[0])  # Convert to int after checking for 'none'

    def named_params(self, items):
        """Handles named parameters, extracting values and converting to a dictionary."""
        params = {}
        for item in items:
            if isinstance(item, Tree):
                item = self.visit(item)
            if isinstance(item, tuple):  # Handle tuple values directly assigned to kernel_size
                params['kernel_size'] = item
            elif isinstance(item, dict):
                params.update(item)
            elif isinstance(item, (int, float, str)):
                params['filters'] = item  # Assign unnamed number to 'filters' for Conv layers
        return params

    def value(self, items):
        """Extracts the value from different token types."""
        item = items[0]
        if isinstance(item, Tree) and item.data == 'tuple_':
            return self.tuple_(item.children)
        elif isinstance(item, Token):
            if item.type == 'ESCAPED_STRING':
                return item.value.strip('"')
            elif item.type in ('NUMBER', 'FLOAT', 'BOOL'):
                return eval(item.value)  # Evaluate numeric and boolean literals
        return item

    def tuple_(self, items):
        """Extracts tuple values."""
        return tuple(eval(item.value) for item in items if isinstance(item, Token) and item.type in ('NUMBER', 'FLOAT'))
    
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
        return self._extract_value(items[0])
    
    def kernel_size_tuple(self, items):
        """Handles kernel size specified as a tuple directly."""
        return self.tuple_(items)  # Directly return the tuple



    def named_filters(self, items):
        return {"filters": self._extract_value(items[2])}

    def named_units(self, items):  
        return {"units": self._extract_value(items[0])}

    def named_activation(self, items): 
        return {"activation": self._extract_value(items[0])}

    def named_kernel_size(self, items):
        return {"kernel_size": self._extract_value(items[2])}

    def named_padding(self, items):
        return {"padding": self._extract_value(items[2])}

    def named_strides(self, items):
        return {"strides": self._extract_value(items[2])}

    def named_rate(self, items):
        return {"rate": self._extract_value(items[2])}

    def named_dilation_rate(self, items):
        return {"dilation_rate": self._extract_value(items[2])}

    def named_groups(self, items):
        return {"groups": self._extract_value(items[2])}

    def named_pool_size(self, items):
        return {"pool_size": self._extract_value(items[2])}

    def named_dropout(self, items):
        return {"dropout": self._extract_value(items[2])}

    def named_return_sequences(self, items):
        return {"return_sequences": self._extract_value(items[2])}

    def named_num_heads(self, items):
        return {"num_heads": self._extract_value(items[2])}

    def named_ff_dim(self, items):
        return {"ff_dim": self._extract_value(items[2])}

    def named_input_dim(self, items):
        return {"input_dim": self._extract_value(items[2])}

    def named_output_dim(self, items):
        return {"output_dim": self._extract_value(items[2])}

    def groups_param(self, items):
        return {'groups': self._extract_value(items[2])}

    def epochs_param(self, items):
        return {'epochs': self._extract_value(items[0])}

    def batch_size_param(self, items):
        return {'batch_size': self._extract_value(items[0])}

    def device_param(self, items):
        return {'device': self._extract_value(items[0])}

    def paper_param(self, items):
        return self._extract_value(items[0])

    def accuracy_param(self, items):
        return {'accuracy': self._extract_value(items[0])}

    def loss_param(self, items):
        return {'loss': self._extract_value(items[0])}

    def precision_param(self, items):
        return {'precision': self._extract_value(items[0])}

    def recall_param(self, items):
        return {'recall': self._extract_value(items[0])}


    ### End named_params ################################################

    ### Advanced layers ###############################
    
    def graph_layer(self, items):
        if items[0].data == 'graph_conv':
            return {'type': 'GraphConv', 'params': items[0].children[0]}
        elif items[0].data == 'graph_attention':
            return {'type': 'GraphAttention', 'params': items[0].children[0]}

    def dynamic(self, items):
        return {'type': 'DynamicLayer', 'params': items[0]}
    
    def noise_layer(self, items):
        return {'type': items[0].data.capitalize(), 'params': items[0].children[0]}

    def normalization_layer(self, items):
        return {'type': items[0].data.capitalize(), 'params': items[0].children[0]}

    def regularization_layer(self, items):
        return {'type': items[0].data.capitalize(), 'params': items[0].children[0]}

    def custom_layer(self, items):
        return {'type': items[0], 'params': items[1]}

    def activation_layer(self, items):
        activation_type = items[0].value.strip('"')
        params = items[1] if len(items) > 1 else {}
        return {'type': 'Activation', 'activation': activation_type, 'params': params}
    
    def capsule(self, items):
        return {'type': 'Capsule', 'params': items[0]}

    def squeeze_excitation(self, items):
        return {'type': 'SqueezeExcitation', 'params': items[0]}

    def graph_conv(self, items):
        return {'type': 'GraphConv', 'params': items[0]}

    def quantum(self, items):
        return {'type': 'QuantumLayer', 'params': items[0]}

    def transformer_layer(self, items):
        if items[0].data == 'transformer_encoder':
            return {'type': 'TransformerEncoder', 'params': items[0].children[0]}
        elif items[0].data == 'transformer_decoder':
            return {'type': 'TransformerDecoder', 'params': items[0].children[0]}
        else:  # Handle the base 'Transformer' case
            return {'type': 'Transformer', 'params': items[0].children[0]}

    def embedding(self, items):
        return {'type': 'Embedding', 'params': items[0]}
    
    def graph_attention(self, items):
        return {'type': 'GraphAttention', 'params': items[0]}

    def execution_config(self, items: List[Tree]) -> Dict[str, Any]:
        """Processes execution configuration block."""
        config: Dict[str, Any] = {'type': 'ExecutionConfig'}
        if items:
            for item in items:
                if item.data == 'device_param':
                    config.update(self.device_param(item.children))
        return config
    
    def lambda_(self, items):
        return {'type': 'Lambda', 'params': {'function': items[0].value.strip('"')}}

    
    def add(self, items):
        return {'type': 'Add', 'params': items[0]}

    def subtract(self, items):
        return {'type': 'Subtract', 'params': items[0]}

    def multiply(self, items):
        return {'type': 'Multiply', 'params': items[0]}

    def average(self, items):
        return {'type': 'Average', 'params': items[0]}

    def maximum(self, items):
        return {'type': 'Maximum', 'params': items[0]}

    def concatenate(self, items):
        return {'type': 'Concatenate', 'params': items[0]}

    def dot(self, items):
        return {'type': 'Dot', 'params': items[0]}

    def gaussian_noise(self, items):
        return {'type': 'GaussianNoise', 'params': items[0]}

    def gaussian_dropout(self, items):
        return {'type': 'GaussianDropout', 'params': items[0]}

    def alpha_dropout(self, items):
        return {'type': 'AlphaDropout', 'params': items[0]}

    def batch_normalization(self, items):
        return {'type': 'BatchNormalization', 'params': items[0]}

    def layer_normalization(self, items):
        return {'type': 'LayerNormalization', 'params': items[0]}

    def instance_normalization(self, items):
        return {'type': 'InstanceNormalization', 'params': items[0]}

    def group_normalization(self, items):
        return {'type': 'GroupNormalization', 'params': items[0]}

    def spatial_dropout1d(self, items):
        return {'type': 'SpatialDropout1D', 'params': items[0]}

    def spatial_dropout2d(self, items):
        return {'type': 'SpatialDropout2D', 'params': items[0]}

    def spatial_dropout3d(self, items):
        return {'type': 'SpatialDropout3D', 'params': items[0]}

    def activity_regularization(self, items):
        return {'type': 'ActivityRegularization', 'params': items[0]}

    def l1_l2(self, items):
        return {'type': 'L1L2', 'params': items[0]}

    def custom(self, items):
        return {'type': items[0], 'params': items[1]}

    def activation(self, items):
        activation_type = items[0].value.strip('"') if isinstance(items[0], Token) else None
        params = items[1] if len(items) > 1 else {}
        return {'type': 'Activation', 'activation': activation_type, 'params': params}

### Shape Propagation ##########################

import os
import numpy as np
import lark
from numbers import Number

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
                elif layer_type == 'CuDnnLSTM':
                    tf_layer_name = 'CuDnnLSTM'
                elif layer_type == 'CuDnnGRU':
                    tf_layer_name = 'CuDnnGRU'
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

                print(f"Warning: {layer_type} is an advanced or custom layer type. Code generation for TensorFlow might require manual implementation. Skipping layer code generation for now.")
                code += f"{indent}tf.keras.layers.Attention(),\n"
                current_input_shape = propagate_shape(current_input_shape, layer_config)

            elif layer_type == 'TransformerEncoder':
                num_heads = params.get('num_heads', 4)
                ff_dim = params.get('ff_dim', 32)
                code += f"{indent}tf.keras.layers.TransformerEncoder(num_heads={num_heads}, ffn_units={ff_dim}),\n"
                current_input_shape = propagate_shape(current_input_shape, layer_config)

            elif layer_type in ['ResidualConnection', 'InceptionModule', 'CapsuleLayer', 'SqueezeExcitation', 'GraphConv', 'QuantumLayer', 'DynamicLayer']:
                NUMBER(f"Warning: {layer_type} is an advanced or custom layer type. Code generation for TensorFlow might require manual implementation. Skipping layer code generation for now.")
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
            elif layer_type in ['CuDNNLSTM', 'CuDNNGRU', 'SimpleRNN']:
                if layer_type == 'CuDNNLSTM':
                    torch_layer_name = 'LSTM'
                elif layer_type == 'CuDNNGRU':
                    torch_layer_name = 'GRU'
                elif layer_type == 'SimpleRNN':
                    torch_layer_name = 'RNN'
                units = params.get('units')
                return_sequences = params.get('return_sequences', False)
                if units is None:
                    raise ValueError(f"{layer_type} layer config missing 'units'.")
                print(f"Warning: {layer_type} is an advanced or custom layer type. Code generation for PyTorch might require manual implementation. Skipping layer code generation for now.")
                layers_code.append(f"{layer_name}_rnn = nn.{torch_layer_name}(input_size={current_input_shape[-1]}, hidden_size={units}, batch_first=True, bidirectional=False)")
                forward_code_body.append(f"x, _ = self.layer{i+1}_rnn(x)")
                current_input_shape = propagate_shape(current_input_shape, layer_config)
            elif layer_type == 'Flatten':
                layers_code.append(f"{layer_name}_flatten = nn.Flatten(start_dim=1)")
                forward_code_body.append(f"x = self.layer{i+1}_flatten(x)")
                current_input_shape = propagate_shape(current_input_shape, layer_config)
            elif layer_type in ['Attention', 'TransformerEncoder', 'ResidualConnection', 'InceptionModule', 'CapsuleLayer', 'SqueezeExcitation', 'GraphConv', 'Embedding', 'QuantumLayer', 'DynamicLayer']:
                NUMBER(f"Warning: {layer_type} is an advanced or custom layer type. Code generation for PyTorch might require manual implementation. Skipping layer code generation for now.")
            else:
                raise ValueError(f"Unsupported layer type: {layer_type} for PyTorch backend.")
                code += indent + indent + "# Layer Definitions\n"
        for layer_init_code in layers_code:
            code += indent + indent + layer_init_code + "\n"
        code += "\n"

        code += indent + "def forward(self, x):\n"
        code += indent + indent + "# Forward Pass\n"
        code += indent + indent + "batch_size, h, w, c = x.size()\n"
        if loss_value.lower() in {'categorical_crossentropy', 'sparse_categorical_crossentropy'}:
            loss_fn_code = "loss_fn = nn.CrossEntropyLoss()"
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
            NUMBER(f"Warning: Loss function '{loss_value}' might not be directly supported in PyTorch. Verify the name and compatibility.")

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
    print(f"Successfully saved file: {filename}")

    Args:
        filename (str): The path to the file to save.
        content (str): The content to save.
    """
    try:
        with open(filename, 'w') as f:
            f.write(content)
    except Exception as e:
        raise IOError(f"Error writing file: {filename}. {e}") from e
    NUMBER(f"Successfully saved file: {filename}")
    return None


def load_file(filename: str) -> Tree:
    """
    Loads and parses a neural network or research file based on its extension.
    if not os.path.exists(filename):
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
            return parser.parse(content)
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

