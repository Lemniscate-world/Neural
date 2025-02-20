from fastapi import params
import lark
from lark import Tree, Transformer, Token
from typing import Any, Dict, List, Tuple, Union, Optional, Callable
import json
import plotly.graph_objects as go


def create_parser(start_rule: str = 'network') -> lark.Lark:
    """
    Creates a Lark parser for neural network configuration files.

    Args:
        start_rule (str): The starting rule for the parser. Defaults to 'network'.

    Returns:
        lark.Lark: A Lark parser object configured with the defined grammar.
    """

    grammar = r"""

        VARIABLE: /[a-zA-Z_][a-zA-Z0-9_]*/
        STRING: "\"" /[^"]+/ "\"" | "\'" /[^']+/ "\'"
        %ignore /\#[^\n]*/  // Ignore line comments              
        // Import common tokens from Lark
        %import common.NEWLINE -> _NL        // Newline characters
        %import common.CNAME -> NAME
        %import common.SIGNED_NUMBER -> NUMBER
        %import common.WS_INLINE             // Inline whitespace
        %import common.INT
        %import common.CNAME -> NAME         // Names/identifiers
        %import common.FLOAT  
        %import common.TUPLE                                                        // Floating poNUMBER numbers
        %import common.WS                    // All whitespace
        %ignore WS   

        ?start: network | layer | research   // Allow parsing either networks or research files

        // File type rules
        neural_file: network
        nr_file: network
        rnr_file: research

        // Parameter & Properties
        named_params: named_param ("," named_param)*
        activation_param: "activation" "=" STRING
        ordered_params: value ("," value)* 
        ?value: STRING -> string_value | number | tuple_ | BOOL  
        tuple_: "("  number  ","  number  ")"  
        number: NUMBER  
        number1: INT
        BOOL: "true" | "false"
        explicit_tuple: "(" value ("," value)+ ")"

        // name_param rules
        bool_value: BOOL  // Example: true or false
        named_return_sequences: "return_sequences" "=" bool_value
        named_units: "units" "=" number  // Example: units=64
        named_activation: "activation" "=" STRING
        named_size: NAME ":" explicit_tuple  
        named_filters: "filters" "=" NUMBER  // Example: filters=32
        named_strides: "strides" "=" value  // Example: strides=(1, 1) or strides=1
        named_padding: "padding" "=" STRING  // Example: padding="same" or padding="valid"
        named_dilation_rate: "dilation_rate" "=" value  // Example: dilation_rate=(2, 2) or dilation_rate=2
        named_groups: "groups" "=" NUMBER  // Example: groups=32
        named_channels: "channels" "=" NUMBER  // Example: channels=3
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
        named_beta_initializer: "beta_initializer" "=" STRING  // Example: beta_initializer="zeros"
        named_gamma_initializer: "gamma_initializer" "=" STRING  // Example: gamma_initializer="ones"
        named_moving_mean_initializer: "moving_mean_initializer" "=" STRING  // Example: moving_mean_initializer="zeros"
        named_moving_variance_initializer: "moving_variance_initializer" "=" STRING  // Example: moving_variance_initializer="ones"
        named_training: "training" "=" BOOL  // Example: training=true
        named_trainable: "trainable" "=" BOOL  // Example: trainable=false
        named_use_bias: "use_bias" "=" BOOL  // Example: use_bias=true
        named_kernel_initializer: "kernel_initializer" "=" STRING  // Example: kernel_initializer="glorot_uniform"
        named_bias_initializer: "bias_initializer" "=" STRING  // Example: bias_initializer="zeros"
        named_kernel_regularizer: "kernel_regularizer" "=" STRING  // Example: kernel_regularizer="l2"
        named_bias_regularizer: "bias_regularizer" "=" STRING  // Example: bias_regularizer="l1"
        named_activity_regularizer: "activity_regularizer" "=" STRING  // Example: activity_regularizer="l1_l2"
        named_kernel_constraint: "kernel_constraint" "=" STRING  // Example: kernel_constraint="max_norm"
        named_kernel_size: "kernel_size" "=" value // Example:
        named_bias_constraint: "bias_constraint" "=" STRING  // Example: bias_constraint="min_max_norm"
        named_return_state: "return_state" "=" BOOL  // Example: return_state=true
        named_go_backwards: "go_backwards" "=" BOOL  // Example: go_backwards=false
        named_stateful: "stateful" "=" BOOL  // Example: stateful=true
        named_time_major: "time_major" "=" BOOL  // Example: time_major=false
        named_unroll: "unroll" "=" BOOL  // Example: unroll=true
        named_input_shape: "input_shape" "=" value  // Example: input_shape=(28, 28, 1)
        named_batch_input_shape: "batch_input_shape" "=" value  // Example: batch_input_shape=(None, 32, 32, 3)
        named_dtype: "dtype" "=" STRING  // Example: dtype="float32"
        named_name: "name" "=" STRING  // Example: name="my_layer"
        named_weights: "weights" "=" value  // Example: weights=[...]
        named_embeddings_initializer: "embeddings_initializer" "=" STRING  // Example: embeddings_initializer="uniform"
        named_mask_zero: "mask_zero" "=" BOOL  // Example: mask_zero=true
        named_input_length: "input_length" "=" NUMBER  // Example: input_length=100
        named_embeddings_regularizer: "embeddings_regularizer" "=" STRING  // Example: embeddings_regularizer="l1"
        named_embeddings_constraint: "embeddings_constraint" "=" value // Example: embeddings_constraint="non_neg"
        named_num_layers: "num_layers" "=" NUMBER // Example: num_layers=2
        named_bidirectional: "bidirectional" "=" BOOL // Example: bidirectional=true
        named_merge_mode: "merge_mode" "=" STRING // Example: merge_mode="concat"
        named_recurrent_dropout: "recurrent_dropout" "=" FLOAT // Example: recurrent_dropout=0.1
        named_noise_shape: "noise_shape" "=" value // Example: noise_shape=(3,)
        named_seed: "seed" "=" NUMBER // Example: seed=42
        named_target_shape: "target_shape" "=" value // Example: target_shape=(64, 64)
        named_data_format: "data_format" "=" STRING // Example: data_format="channels_first"
        named_interpolation: "interpolation" "=" STRING // Example: interpolation="nearest"
        named_crop_to_aspect_ratio: "crop_to_aspect_ratio" "=" BOOL // Example: crop_to_aspect_ratio=true
        named_mask_value: "mask_value" "=" NUMBER // Example: mask_value=0
        named_return_attention_scores: "return_attention_scores" "=" BOOL // Example: return_attention_scores=true
        named_causal: "causal" "=" BOOL // Example: causal=false
        named_use_scale: "use_scale" "=" BOOL // Example: use_scale=true
        named_key_dim: "key_dim" "=" NUMBER // Example: key_dim=64
        named_value_dim: "value_dim" "=" NUMBER 
        named_output_shape: "output_shape" "=" value 
        named_arguments: "arguments" "=" value 
        named_initializer: "initializer" "=" STRING 
        named_regularizer: "regularizer" "=" STRING 
        named_constraint: "constraint" "=" STRING
        named_l1: "l1" "=" FLOAT  // Example: l1=0.01
        named_l2: "l2" "=" FLOAT  // Example: l2=0.001
        named_l1_l2: "l1_l2" "=" tuple_  // Example: l1_l2=(0.01, 0.001)
        named_int: NAME "=" INT | NAME ":" INT
        named_string: NAME "=" STRING | NAME ":" STRING
        named_float: NAME "=" FLOAT | NAME ":" FLOAT
        simple_number: number1
        rate: "rate" ":" FLOAT
        simple_float: FLOAT
        named_clipvalue: "clipvalue=" FLOAT
        named_clipnorm: "clipnorm=" FLOAT
        ?named_param: ( rate | named_clipvalue | named_clipnorm | simple_float | explicit_tuple | simple_number| named_units | pool_size | named_kernel_size | named_size | named_activation | named_filters | named_strides | named_padding | named_dilation_rate | named_groups | named_data_format | named_channels | named_return_sequences | named_num_heads | named_ff_dim | named_input_dim | named_output_dim | named_rate | named_dropout | named_axis | named_momentum | named_epsilon | named_center | named_scale | named_beta_initializer | named_gamma_initializer | named_moving_mean_initializer | named_moving_variance_initializer | named_training | named_trainable | named_use_bias | named_kernel_initializer | named_bias_initializer | named_kernel_regularizer | named_bias_regularizer | named_activity_regularizer | named_kernel_constraint | named_bias_constraint | named_return_state | named_go_backwards | named_stateful | named_time_major | named_unroll | named_input_shape | named_batch_input_shape | named_dtype | named_name | named_weights | named_embeddings_initializer | named_mask_zero | named_input_length | named_embeddings_regularizer | named_embeddings_constraint | named_num_layers | named_bidirectional | named_merge_mode | named_recurrent_dropout | named_noise_shape | named_seed | named_target_shape | named_interpolation | named_crop_to_aspect_ratio | named_mask_value | named_return_attention_scores | named_causal | named_use_scale | named_key_dim | named_value_dim | named_output_shape | named_arguments | named_initializer | named_regularizer | named_constraint | named_l1 | named_l2 | named_l1_l2 | named_int | named_float | NAME "=" value )
        
        // Layer parameter styles
        ?param_style1: named_params | (dense_ordered_params ["," named_params])

        // Top-level network definition - defines the structure of an entire neural network
        network: "network" NAME "{" input_layer layers loss optimizer [training_config] [execution_config] "}" 

        // Configuration can be either training-related or execution-related
        config: training_config | execution_config

        // Input layer definition - specifies the shape of input data
        input_layer: "input" ":" shape ("," shape)*
        shape: "(" [number_or_none ("," number_or_none)* [","]] ")"
        number_or_none: number | "NONE" | "None"

        // Layers section - contains all layer definitions separated by newlines
        layers: "layers" ":" _NL* (layer _NL*)*

        // All possible layer types that can be used in the network
        ?layer: (basic | recurrent | advanced | activation | merge | noise | norm_layer | regularization | custom | wrapper | lambda_ )  
        
        // Lambda functions
        lambda_: "Lambda("  STRING  ")"

        // Wrapper Layer Functions 
        wrapper: "TimeDistributed" "(" layer ["," named_params] ")"

        
        // Basic layer types & group
        ?basic: conv | pooling | dropout | flatten | dense | output
        dropout: "Dropout(" named_params ")" 
        dense: "Dense" "(" dense_params ")"
        dense_params: (NUMBER ("," (NUMBER | STRING | named_param))* ) | named_params
        dense_ordered_params: value ("," value)*  // Allow any valid 'value'
        flatten: "Flatten" "(" [named_params] ")"
        // Recurrent layers section - includes all RNN variants
        ?recurrent: rnn | bidirectional_rnn | conv_rnn | rnn_cell  
        bidirectional_rnn: "Bidirectional(" rnn "," named_params ")" 
        rnn: simple_rnn | lstm | gru
        simple_rnn: "SimpleRNN(" named_params ")"
        lstm: "LSTM(" named_params ")"
        gru: "GRU(" named_params ")"
        

        // Regularization layers group
        regularization: spatial_dropout1d | spatial_dropout2d | spatial_dropout3d | activity_regularization | l1 | l2 | l1_l2 
        l1: "L1(" named_params ")"
        l2: "L2(" named_params ")"
        l1_l2: "L1L2(" named_params ")" 


        // Output layer 
        output: "Output(" named_params ")"

        // Convolutional layers 
        conv: conv1d | conv2d | conv3d | conv_transpose | depthwise_conv2d | separable_conv2d
        conv1d: "Conv1D(" param_style1 ")"
        conv2d: "Conv2D(" param_style1 ")"
        conv3d: "Conv3D(" param_style1 ")"
        conv_transpose: conv1d_transpose | conv2d_transpose | conv3d_transpose
        conv1d_transpose: "Conv1DTranspose(" param_style1 ")"
        conv2d_transpose: "Conv2DTranspose(" param_style1 ")"
        conv3d_transpose: "Conv3DTranspose(" param_style1 ")"
        depthwise_conv2d: "DepthwiseConv2D(" param_style1 ")"
        separable_conv2d: "SeparableConv2D(" param_style1 ")"

        // Pooling layer parameters
        pooling: max_pooling | average_pooling | global_pooling | adaptive_pooling
        max_pooling: max_pooling1d | max_pooling2d | max_pooling3d
        max_pooling1d: "MaxPooling1D" "(" param_style1 ")"
        max_pooling2d: "MaxPooling2D" "(" param_style1 ")"
        max_pooling3d: "MaxPooling3D" "(" param_style1 ")"
        pool_size : "pool_size" "=" value
        average_pooling: average_pooling1d | average_pooling2d | average_pooling3d
        average_pooling1d: "AveragePooling1D(" param_style1 ")"
        average_pooling2d: "AveragePooling2D(" param_style1 ")"
        average_pooling3d: "AveragePooling3D(" param_style1 ")"
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

        // Normalization layers
        ?norm_layer: batch_norm | layer_norm | instance_norm | group_norm
        batch_norm: "BatchNormalization" "(" [named_params] ")"
        layer_norm: "LayerNormalization" "(" [named_params] ")"
        instance_norm: "InstanceNormalization" "(" [named_params] ")"
        group_norm: "GroupNormalization" "(" [named_params] ")"
        
        conv_rnn: conv_lstm | conv_gru
        conv_lstm: "ConvLSTM2D(" named_params ")"
        conv_gru: "ConvGRU2D(" named_params ")"

        rnn_cell: simple_rnn_cell | lstm_cell | gru_cell
        simple_rnn_cell: "SimpleRNNCell(" named_params ")"
        lstm_cell: "LSTMCell(" named_params ")"
        gru_cell: "GRUCell(" named_params ")"

        
        // Dropout wrapper layers for RNNs
        dropout_wrapper_layer: simple_rnn_dropout | gru_dropout | lstm_dropout
        simple_rnn_dropout: "SimpleRNNDropoutWrapper" "(" named_params ")"
        gru_dropout: "GRUDropoutWrapper" "(" named_params ")"
        lstm_dropout: "LSTMDropoutWrapper" "(" named_params ")"
        bidirectional_rnn_layer: bidirectional_simple_rnn_layer | bidirectional_lstm_layer | bidirectional_gru_layer
        bidirectional_simple_rnn_layer: "Bidirectional(SimpleRNN(" named_params "))"
        bidirectional_lstm_layer: "Bidirectional(LSTM(" named_params "))"
        bidirectional_gru_layer: "Bidirectional(GRU(" named_params "))"
        conv_rnn_layer: conv_lstm_layer | conv_gru_layer
        conv_lstm_layer: "ConvLSTM2D(" named_params ")"  // Add 2D for clarity
        conv_gru_layer: "ConvGRU2D(" named_params ")"  // Add 2D for clarity
        rnn_cell_layer: simple_rnn_cell_layer | lstm_cell_layer | gru_cell_layer
        simple_rnn_cell_layer: "SimpleRNNCell(" named_params ")"
        lstm_cell_layer: "LSTMCell(" named_params ")"
        gru_cell_layer: "GRUCell(" named_params ")"

        // Advanced layers group
        ?advanced: ( attention | transformer | residual | inception | capsule | squeeze_excitation | graph | embedding | quantum | dynamic )
        attention: "Attention" "(" [named_params] ")"

        // Transformers
        transformer: "Transformer" "(" [named_params] ")" | "TransformerEncoder" "(" [named_params] ")" | "TransformerDecoder" "(" [named_params] ")"
        
        residual: "ResidualConnection" "(" [named_params] ")"
        inception: "Inception" "(" [named_params] ")"
        capsule: "CapsuleLayer" "(" [named_params] ")"
        squeeze_excitation: "SqueezeExcitation" "(" [named_params] ")"
        graph: graph_conv | graph_attention
        graph_conv: "GraphConv" "(" [named_params] ")"
        graph_attention: "GraphAttention" "(" [named_params] ")"
        embedding: "Embedding" "(" [named_params] ")"
        quantum: "QuantumLayer" "(" [named_params] ")"
        dynamic: "DynamicLayer" "(" [named_params] ")"

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

        spatial_dropout1d: "SpatialDropout1D(" named_params ")"
        spatial_dropout2d: "SpatialDropout2D(" named_params ")"
        spatial_dropout3d: "SpatialDropout3D(" named_params ")"
        activity_regularization: "ActivityRegularization(" named_params ")"

        custom: NAME "(" named_params ")"

        // Activation functions
        activation: activation_with_params | activation_without_params
        activation_with_params: "Activation(" STRING "," named_params ")"  
        activation_without_params: "Activation(" STRING ")" 

        // Training configuration block
        training_config: "train" "{" training_params "}"
        training_params: (epochs_param | batch_size_param | optimizer_param | search_method_param)*
        epochs_param: "epochs:" INT
        batch_size_param: "batch_size:" values_list
        values_list: "[" value ("," value)* "]" | value ("," value)*
        optimizer_param: "optimizer:" named_optimizer 
        named_optimizer : "named_optimizer(" learning_rate_param ")"
        learning_rate_param: "learning_rate=" FLOAT
        search_method_param: "search_method:" STRING


        // Loss and optimizer specifications
        loss: "loss" ":" (NAME | STRING) ["(" named_params ")"]
        optimizer: "optimizer:" (NAME | STRING) ["(" named_params ")"] 
        schedule: NAME "(" params ")"
        params: [value ("," value)*]

        // Execution environment configuration
        execution_config: "execution" "{" device_param "}"
        device_param: "device:" STRING
        // Research-specific configurations
        research: "research" NAME? "{" [research_params] "}"
        research_params: (metrics | references)*
        // Metrics tracking
        metrics: "metrics" "{" [accuracy_param] [metrics_loss_param] [precision_param] [recall_param] "}"
        accuracy_param: "accuracy:" FLOAT
        metrics_loss_param: "loss:" FLOAT
        precision_param: "precision:" FLOAT
        recall_param: "recall:" FLOAT
        // Paper references
        references: "references" "{" paper_param+ "}"
        paper_param: "paper:" STRING

        // Custom Shape Propagation
        custom_shape: "CustomShape" "(" NAME "," explicit_tuple ")"

        // Math Expressions
        math_expr: term (("+"|"-") term)*
        term: factor (("*"|"/") factor)*
        factor: NUMBER | VARIABLE | "(" math_expr ")" | function_call
        function_call: NAME "(" math_expr ("," math_expr)* ")"

        // Automatic Hyperparameters Optimization
        hpo_expr: "HPO(" (hpo_choice | hpo_range | hpo_log_range) ")"
        hpo_choice: "choice(" value ("," value)* ")"
        hpo_range: "range(" number "," number ("," "step="number)? ")"
        hpo_log_range: "log_range(" number "," number ")"


        // Neural Architecture Search
        layer_choice: "HPO(choice(" layer ("," layer)* "))"

    """
    return  lark.Lark(grammar, start=[start_rule], parser='lalr', lexer='contextual', cache=False)

network_parser = create_parser('network')
layer_parser = create_parser('layer')
research_parser = create_parser('research')

class ModelTransformer(lark.Transformer):
    """
    Transforms the parsed tree NUMBER to a structured dictionary representing the neural network model.
    """
    def layer(self, items):
        return self.visit(items[0])

        
    def wrapper(self, items):
        wrapper_type = items[0]
        layer = items[1]
        params = items[2]
        # Merge layer params with wrapper params
        layer['params'].update(params)
        return {'type': f"{wrapper_type}({layer['type']})", 'params': layer['params']}

    ### Basic Layers & Properties ###################

    def input_layer(self, items):
        for item in items:
            shapes = self._extract_value(item)
        return {'type': 'Input', 'shape': shapes}
    
    def layers(self, items):
        return items

    def flatten(self, items):
        if items:
            params = self._extract_value(items[0])
        else:
            params = None
        return {'type': 'Flatten', 'params': params}

    def dropout(self, items):
        params = self._extract_value(items[0])
        return {'type': 'Dropout', 'params': params}

    def output(self, items):
        return {'type': 'Output', 'params': items[0]}
    
    def regularization(self, items):  
        return {'type': items[0].data.capitalize(), 'params': items[0].children[0]}
    
    def execution_config(self, items):
        params = self._extract_value(items[0])
        return {'type': 'execution_config', 'params':params}

    # In the ModelTransformer class:
    def dense(self, items):
        param_nodes = items[0].children
        param_dict = {}
        ordered_params = []
        named_params = {}

        for child in param_nodes:
            param = self._extract_value(child)
            if isinstance(param, dict):  # Named parameter
                named_params.update(param)
            else:  # Ordered parameter
                ordered_params.append(param)

        # Assign ordered params to 'units' and 'activation'
        if ordered_params:
            param_dict['units'] = ordered_params[0]
            if len(ordered_params) > 1:
                param_dict['activation'] = ordered_params[1]
        
        # Merge named parameters
        param_dict.update(named_params)
        
        return {"type": "Dense", "params": param_dict}
    
    ### Convolutional Layers ####################
    def conv1d(self, items):
        return {'type': 'Conv1D', 'params': items[0]}

    def conv2d(self, items):
        # Recursively flatten a tree or list of items into a single list.
        def flatten(item):
            if isinstance(item, list):
                result = []
                for sub in item:
                    result.extend(flatten(sub))
                return result
            elif isinstance(item, Tree):
                return flatten(item.children)
            else:
                return [item]

        # Flatten the parameter tree.
        flat_items = flatten(items[0])
        
        # Filter out any tokens that are just the literal "Conv2D"
        flat_items = [it for it in flat_items if not (isinstance(it, Token) and it.type == "NAME" and it.value == "Conv2D")]

        ordered_params = []
        named_params = {}
        
        # Process each flattened item:
        for item in flat_items:
            value = self._extract_value(item)
            if isinstance(value, dict):
                named_params.update(value)
            else:
                ordered_params.append(value)
        
        params = {}
        if ordered_params:
            params['filters'] = ordered_params[0]
        if len(ordered_params) > 1:
            params['kernel_size'] = ordered_params[1]
        if len(ordered_params) > 2:
            params['activation'] = ordered_params[2]
        
        # Named parameters override any ordered ones
        params.update(named_params)
        
        return {'type': 'Conv2D', 'params': params}




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
    
    def graph_conv(self, items):
        params = self._extract_value(items[0]) if items else None
        return {'type': 'GraphConv', 'params': params}
    
    ### Loss Parameters ###

    def loss(self, items):
        return items[0].value.strip('"')


    ### Optimization ##############

    def optimizer(self, items):
        name = str(items[0].value)
        # Strip surrounding quotes if present
        if (name.startswith('"') and name.endswith('"')) or (name.startswith("'") and name.endswith("'")):
            name = name[1:-1]
        params = self._extract_value(items[1]) if len(items) > 1 else {}
        
        # if name == "SGD" and "learning_rate" not in params:
           # params["learning_rate"] = 0.01

        return {"type": name, "params": params}

    def schedule(self, items):
        return {
            "type": items[0].value,
            "args": [self._extract_value(x) for x in items[1].children]
        }
    
    ### Training Configurations ############################

    def training_config(self, items):
        for item in items:
            params = self._extract_value(item) if items else None
        return params
    
    def training_params(self, items):
        params = {}
        for item in items:
            if isinstance(item, Tree):
                item = self._extract_value(item)
            elif isinstance(item, dict):
                item = self._extract_value(item)
                params.update(item)
        return params
    
    def epochs_param(self, items):
        return {'epochs': self._extract_value(items[0])}

    def batch_size_param(self, items):
        return {'batch_size': self._extract_value(items[0])}

    def values_list(self, items):
        for x in items:
            params = self._extract_value(x)
        return params
    
    def optimizer_param(self, items):
        return {'optimizer': self._extract_value(items[0])}
    
    def named_optimizer(self, items):
        return {'named_optimizer': self._extract_value(items[0])}

    def learning_rate_param(self,items):
        return {'learning_rate': self._extract_value(items[0])}
    
    def shape(self, items):
        return tuple(items)
    
    ### Wrapper Layers ###

    def wrapper(self, items):
        layer = items[0]
        params = items[1] if len(items) > 1 else {}
        layer['params'].update(params)
        return {'type': f"TimeDistributed({layer['type']})", 'params': layer['params']}

    
    ### Pooling Layers #############################

    def pooling(self, items):
        if isinstance(items, dict):
            return {'type': items['type'], 'params': items}
        return {'type': self._extract_value(items[0]), 'params': self._extract_value(items[0])}
    
    def max_pooling(self, items):
        return self._extract_value(items[0])

    def pool_size(self, items):
        value = self._extract_value(items[0])
        return {'pool_size': value}

    def max_pooling1d(self, items):
        param_nodes = items[0].children
        param_dict = {}
        params = [self._extract_value(child) for child in param_nodes]

        if all(isinstance(p, dict) for p in params):
            for p in params:
                param_dict.update(p)
        else:
            if len(params) >= 1:
                param_dict["pool_size"] = params[0]
            if len(params) >= 2:
                param_dict["strides"] = params[1]
            if len(params) >= 3:
                param_dict["padding"] = params[2]

        return {'type': 'MaxPooling1D', 'params': param_dict}

    def max_pooling2d(self, items):
        param_style = items[0]
        params = {}
        if isinstance(param_style, list):
            ordered_params = [self._extract_value(p) for p in param_style if not isinstance(p, dict)]
            if ordered_params:
                params['pool_size'] = ordered_params[0]
            if len(ordered_params) > 1:
                params['strides'] = ordered_params[1]
            if len(ordered_params) > 2:
                params['padding'] = ordered_params[2]
            for item in param_style:
                if isinstance(item, dict):
                    params.update(item)
        elif isinstance(param_style, dict):
            params = param_style.copy()
        return {'type': 'MaxPooling2D', 'params': params}

    def max_pooling3d(self, items):
        param_nodes = items[0].children
        param_dict = {}
        params = [self._extract_value(child) for child in param_nodes]

        if all(isinstance(p, dict) for p in params):
            for p in params:
                param_dict.update(p)
        else:
            if len(params) >= 1:
                param_dict["pool_size"] = params[0]
            if len(params) >= 2:
                param_dict["strides"] = params[1]
            if len(params) >= 3:
                param_dict["padding"] = params[2]

        return {"type": "MaxPooling3D", "params": param_dict}
    
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

    ### End Basic Layers & Properties #########################

    def batch_norm(self, items):
        params = self._extract_value(items[0]) if items else None
        return {'type': 'BatchNormalization', 'params': params}

    def layer_norm(self, items):
        params = self._extract_value(items[0]) if items else None
        return {'type': 'LayerNormalization', 'params': params}

    def instance_norm(self, items):
        params = self._extract_value(items[0]) if items else None
        return {'type': 'InstanceNormalization', 'params': params}
    def group_norm(self, items):
        params = self._extract_value(items[0]) if items else None
        return {'type': 'GroupNormalization', 'params': params}

    def lstm(self, items):
        return {'type': 'LSTM', 'params': items[0]}

    def gru(self, items):
        return {'type': 'GRU', 'params': items[0]}
    
    ### Recurrent Layers ########################################

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

    ### Everything Research ##################

    def research(self, items):
        name = self._extract_value(items[0]) if items else None
        params = self._extract_value(items[1]) if len(items) > 1 else {}
        return {'type': 'Research', 'name': name, 'params': params}
    
    def research_params(self, items):
        params = {}
        for item in items:
            if isinstance(item, Tree):
                item = self._extract_value(item)
            elif isinstance(item, dict):
                item = self._extract_value(item)
                params.update(item)
        return params
    
    def metrics(self, items):
        # If no metric items were provided, return an empty dictionary.
        if not items:
            return {'metrics': {}}
        result = {}
        for item in items:
            # Skip any None values
            if item is None:
                continue
            # If the item is already a dictionary (as produced by individual metric rules), update the result.
            if isinstance(item, dict):
                result.update(item)
            else:
                # Otherwise, attempt to extract a string value and split it by colon
                val = self._extract_value(item)
                if val and ':' in val:
                    key, v = str(val).split(':', 1)
                    try:
                        result[key.strip()] = float(v.strip())
                    except ValueError:
                        result[key.strip()] = v.strip()
        return {'metrics': result}

    def accuracy_param(self, items):
        return {'accuracy': self._extract_value(items[0])}

    def precision_param(self, items):
        return {'precision': self._extract_value(items[0])}

    def recall_param(self, items):
        return {'recall': self._extract_value(items[0])}
    
    def references(self, items):
        # If no metric items were provided, return an empty dictionary.
        if not items:
            return {'references': {}}
        result = {}
        for item in items:
            # Skip any None values
            if item is None:
                continue
            # If the item is already a dictionary (as produced by individual references rules), update the result.
            if isinstance(item, dict):
                result.update(item)
            else:
                # Otherwise, attempt to extract a string value and split it by colon
                val = self._extract_value(item)
                if val and ':' in val:
                    key, v = str(val).split(':', 1)
                    try:
                        result[key.strip()] = float(v.strip())
                    except ValueError:
                        result[key.strip()] = v.strip()
        return {'references': result}
    
    def paper_param(self, items):
        return self._extract_value(items[0])


    ### NETWORK ACTIVATION ##############################

    def network(self, items):
        name = str(items[0].value)
        input_layer_config = items[1]
        layers_config = items[2]
        loss_config = items[3]
        optimizer_config = items[4]
        training_config = next((item for item in items[5:] if isinstance(item, dict)), None)
        execution_config = next((item for item in items[5:] if isinstance(item, dict) and 'params' in item and 'device' in item['params']), {'params': {'device': 'auto'}})['params']

        # Ensure output_layer exists
        output_layer = next((layer for layer in reversed(items[2]) 
                        if layer['type'] == 'Output'), None)
        
        output_shape = {}
        if output_layer is None:
            output_shape is None
        elif output_layer is not None:
            output_shape = output_layer.get('params', {}).get('units')
        
        

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

    def named_params(self, items):
        params = {}
        for item in items:
            if isinstance(item, Tree):
                item = self._extract_value(item)
            elif isinstance(item, dict):
                item = self._extract_value(item)
                params.update(item)
        return params
    
    def _extract_value(self, item):  # Helper function to extract values from tokens and tuples
        if isinstance(item, Token):
            if item.type == 'NAME':
                return item.value
            if item.type in ('INT', 'FLOAT', 'NUMBER', 'SIGNED_NUMBER'):
                try:
                    return int(item)
                except ValueError:
                    return float(item)
            elif item.type == 'BOOL':
                return item.value.lower() == 'true'
            elif item.type == 'STRING':
                return item.value.strip('"')
            elif item.type == 'WS_INLINE':
                return item.value.strip()
            
        elif isinstance(item, Tree) and item.data == 'number_or_none':
            child = item.children[0]
            if isinstance(child, Token) and child.type == 'STRING' and child.value.upper() == 'NONE':
                return None


        # Add handling for shape tuples
        elif isinstance(item, Tree) and item.data == 'shape':
            return tuple(self._extract_value(child) for child in item.children)
        # Add handling for pool_size tuples
        elif isinstance(item, Tree) and item.data == 'pool_size':
            return tuple(self._extract_value(child) for child in item.children)
        # Add handling for strides tuples
        elif isinstance(item, Tree) and item.data =='strides':
            return tuple(self._extract_value(child) for child in item.children)
        # Add handling for padding tuples
        elif isinstance(item, Tree) and item.data == 'padding':
            return tuple(self._extract_value(child) for child in item.children)
        # Add handling for dilation tuples
        elif isinstance(item, Tree) and item.data == 'dilation':
            return tuple(self._extract_value(child) for child in item.children)
        # Add handling for kernel_size tuples
        elif isinstance(item, list):  # Handles nested lists
            return [self._extract_value(elem) for elem in item]
        elif isinstance(item, dict):  # Handles nested dictionaries
            return {k: self._extract_value(v) for k, v in item.items()}
        elif isinstance(item, Tree):  # Handles all Tree types, not just tuple_
            if item.data == 'tuple_':
                return tuple(self._extract_value(child) for child in item.children)
            else:  # Extract values from other tree types
                return {k: self._extract_value(v) for k, v in zip(item.children[::2], item.children[1::2])}
        elif isinstance(item, Tree) and item.data == 'explicit_tuple':
            return tuple(self._extract_value(child) for child in item.children)
        return item

    def named_float(self, items):
        # items = [NAME, '='|':', FLOAT] 
        return {items[0].value: self._extract_value(items[1])}

    def named_int(self, items):
        # items = [NAME, '=', INT]
        return {items[0].value: self._extract_value(items[1])}

    def named_param(self, items):
        # Generic NAME "=" value rule (e.g., activation="relu")
        # items = [NAME, '=', value]
        return {items[0].value: self._extract_value(items[1])}
    
    def named_string(self, items):
        # items = [NAME, '='|':', STRING]
        return {items[0].value: self._extract_value(items[1])}
    
    def number(self, items):
        return self._extract_value(items[0])
    
    def rate(self, items):
        params = self._extract_value(items[0])
        return {'type': 'rate', 'params': params}
    
    def simple_float(self, items):
        params = self._extract_value(items[0])
        return {'rate': params}
    
    def number_or_none(self, items):
        if not items:
            # Instead of raising an error, return None if no items are provided.
            return None
        value = self._extract_value(items[0])
        if value == "None":
            return None
        try:
            # Convert to int if there is no decimal point; otherwise, convert to float
            return int(value) if '.' not in str(value) else float(value)
        except Exception as e:
            raise ValueError(f"Error converting {value} to a number: {e}")
        
    def value(self, items):
        if isinstance(items[0], Token):
            return items[0].value
        return items[0]

    def explicit_tuple(self, items):
        return tuple(self._extract_value(item) for item in items)

    def bool_value(self, items):
        return self._extract_value(items[0])
    
    def simple_number(self, items):
        return self._extract_value(items[0])

    
    def ordered_params(self, items):

        def dense(self, items):
            params = items[0]
            if isinstance(params, list):
                param_dict = {}
                if len(params) >= 1:
                    param_dict['units'] = self._extract_value(items[0])
                if len(params) >= 2:
                    param_dict['activation'] = self._extract_value(items[1])
                params = param_dict
            return {'type': 'Dense', 'params': params}

        def conv2d(self, items):
            params = items[0]
            if isinstance(params, list):
                param_dict = {}
                if len(params) >= 1:
                    param_dict['filters'] = params[0]
                if len(params) >= 2:
                    param_dict['kernel_size'] = params[1]
                if len(params) >= 3:
                    param_dict['activation'] = params[2]
                params = param_dict
            else:
                param_dict = params
            return {'type': 'Conv2D', 'params': params}

        def max_pooling2d(self, items):
            params = items[0]
            if isinstance(params, list):
                param_dict = {}
                if len(params) >= 1:
                    param_dict['pool_size'] = params[0]
                if len(params) >= 2:
                    param_dict['strides'] = params[1]
                if len(params) >= 3:
                    param_dict['padding'] = params[2]
                params = param_dict
            return {'type': 'MaxPooling2D', 'params': params}
            
        
    def named_kernel_size(self, items):
        return {"kernel_size": self._extract_value(items[0])}

    def named_filters(self, items):
        return {"filters": self._extract_value(items[0])}

    def named_units(self, items):  
        return {"units": self._extract_value(items[0])}

    def activation_param(self, units):
        return {"activation": self._extract_value(units)}

    def named_activation(self, items): 
        return {"activation": self._extract_value(items[0])}

    def named_strides(self, items):
        value = self._extract_value(items[0])
        return {"strides": value}

    def named_padding(self, items):
        value = self._extract_value(items[0])
        return {"padding": value}

    def named_rate(self, items):
        return {"rate": self._extract_value(items[0])}

    def named_dilation_rate(self, items):
        return {"dilation_rate": self._extract_value(items[0])}

    def named_groups(self, items):
        return {"groups": self._extract_value(items[0])}

    def named_size(self, items):
        name = str(items[0])
        value = tuple(int(x) for x in items[2].children)
        return {name: value}

    def named_dropout(self, items):
        return {"dropout": self._extract_value(items[0])}

    def named_return_sequences(self, items):
        return {"return_sequences": self._extract_value(items[0])}

    def named_num_heads(self, items):
        return {"num_heads": self._extract_value(items[0])}

    def named_ff_dim(self, items):
        return {"ff_dim": self._extract_value(items[0])}

    def named_input_dim(self, items):
        return {"input_dim": self._extract_value(items[0])}

    def named_output_dim(self, items):
        return {"output_dim": self._extract_value(items[0])}

    def groups_param(self, items):
        return {'groups': self._extract_value(items[0])}

    def device_param(self, items):
        return {'device': self._extract_value(items[0])}

    ### End named_params ################################################

    
    ### Advanced Layers #################################

    def attention(self, items):
        params = items[0] if items else None
        return {'type': 'Attention', 'params': params}

    def residual(self, items):
        params = items[0] if items else None
        return {'type': 'ResidualConnection', 'params': params}

    def inception(self, items):
        params = items[0] if items else None
        return {'type': 'Inception', 'params': params}

    def graph(self, items):
        return items[0]

    
    def graph_attention(self, items):
        params = self._extract_value(items[0])
        return {'type': 'GraphAttention', 'params': params}
    
    def dynamic(self, items):
        params = items[0] if items else None
        return {'type': 'DynamicLayer', 'params': params}
    
    def noise_layer(self, items):
        return {'type': items[0].data.capitalize(), 'params': items[0].children[0]}

    def normalization_layer(self, items):
        return {'type': items[0].data.capitalize(), 'params': items[0].children[0]}

    def regularization_layer(self, items):
        return {'type': items[0].data.capitalize(), 'params': items[0].children[0]}

    def custom_layer(self, items):
        params = self._extract_value(items[0])
        return {'type': 'Capsule', 'params': params}
    
    def capsule(self, items):
        params = items[0] if items else None
        return {'type': 'CapsuleLayer', 'params': params}

    def squeeze_excitation(self, items):
        params = self._extract_value(items[0]) if items else None
        return {'type': 'SqueezeExcitation', 'params': params}

    def quantum(self, items):
        params = items[0] if items else None
        return {'type': 'QuantumLayer', 'params': params}

    def transformer(self, items):
        params = self._extract_value(items[0])
        return {'type': 'TransformerEncoder', 'params': params}
    
    def embedding(self, items):
        params = items[0] if items else None
        return {'type': 'Embedding', 'params': params}
    
    def lambda_(self, items):
        return {'type': 'Lambda', 'params': {'function': self._extract_value(items[0])}}
    
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
    
    # Custom Shape Propagation
    def custom_shape(self, items):
    # Example: CustomShape(MyLayer, (32, 32))
        layer_name = self._extract_value(items[0])
        custom_dims = self._extract_value(items[1])
        return {"type": "CustomShape", "layer": layer_name, "custom_dims": custom_dims}

### Automatic Hyperparameters Optimization ###

    def hpo_expr(self, items):
        return {"hpo": items[0]}

    def hpo_choice(self, items):
        return {"type": "categorical", "values": [self._extract_value(x) for x in items]}

    def hpo_range(self, items):
        start = self._extract_value(items[0])
        end = self._extract_value(items[1])
        step = self._extract_value(items[2]) if len(items) > 2 else 1
        return {"type": "range", "start": start, "end": end, "step": step}

    def hpo_log_range(self, items):
        low = self._extract_value(items[0])
        high = self._extract_value(items[1])
        return {"type": "log_range", "low": low, "high": high}

### Neural Architecture Search ###

    def layer_choice(self, items):
        return {"hpo_type": "layer_choice", "options": items}


### Frameworks Detection ###
    def parse_network(self, config: str, framework: str = 'auto'):
        tree = network_parser.parse(config)
        model = self.transform(tree)
        
        # Auto-detect framework
        if framework == 'auto':
            framework = self._detect_framework(model)
            
        # Add framework-specific metadata
        model['framework'] = framework
        model['shape_info'] = []
        
        return model

    def _detect_framework(self, model):
        # Detect framework based on layer types and parameters
        for layer in model['layers']:
            if 'torch' in layer.get('params', {}).values():
                return 'pytorch'
            if 'keras' in layer.get('params', {}).values():
                return 'tensorflow'
        return 'tensorflow'  # default


