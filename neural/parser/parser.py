from fastapi import params
import lark
import pysnooper
from lark import Tree, Transformer, Token
from typing import Any, Dict, List, Tuple, Union, Optional, Callable
import json
import plotly.graph_objects as go
import logging
from enum import Enum
from lark.visitors import VisitError


logger = logging.getLogger('neural.parser')
logging.basicConfig(
    level=logging.DEBUG,  # Capture all levels
    format='%(levelname)s: %(message)s'  # Include severity in output
)

def log_by_severity(severity, message):
    """Log a message based on its severity level."""
    if severity == Severity.DEBUG:
        logger.debug(message)
    elif severity == Severity.INFO:
        logger.info(message)
    elif severity == Severity.WARNING:
        logger.warning(message)
    elif severity == Severity.ERROR:
        logger.error(message)
    elif severity == Severity.CRITICAL:
        logger.critical(message)

class Severity(Enum):
    DEBUG = 1    # For development info, not user-facing
    INFO = 2     # Informational, no action needed
    WARNING = 3  # Recoverable issue, parsing can continue
    ERROR = 4    # Non-recoverable, parsing stops
    CRITICAL = 5 # Fatal, immediate halt required


# Custom exception for DSL validation errors
class DSLValidationError(Exception):
    def __init__(self, message, severity=Severity.ERROR, line=None, column=None):
        self.severity = severity
        self.line = line
        self.column = column
        if line and column:
            super().__init__(f"{severity.name} at line {line}, column {column}: {message}")
        else:
            super().__init__(f"{severity.name}: {message}")
        self.message = message  # Store raw message for logging

# Custom error handler for Lark parsing
def custom_error_handler(error):
    if isinstance(error, lark.UnexpectedCharacters):
        msg = f"Syntax error at line {error.line}, column {error.column}: Unexpected character '{error.char}'.\n" \
              f"Expected one of: {', '.join(sorted(error.allowed))}"
        severity = Severity.ERROR  # Syntax errors are typically severe
    elif isinstance(error, lark.UnexpectedToken):
        msg = f"Syntax error at line {error.line}, column {error.column}: Unexpected token '{error}'.\n" \
              f"Expected one of: {', '.join(sorted(error.expected))}"
        severity = Severity.ERROR
    else:
        msg = str(error)
        severity = Severity.ERROR

    log_by_severity(severity, msg)
    if severity.value >= Severity.ERROR.value:
        raise DSLValidationError(msg, severity, error.line, error.column)
    return {"warning": msg, "line": error.line, "column": error.column}  # Return for warnings

def create_parser(start_rule: str = 'network') -> lark.Lark:
    grammar = r"""
        // Layer type tokens (case-insensitive)
        DENSE: "dense"i
        MAXPOOLING1D: "maxpooling1d"i
        MAXPOOLING2D: "maxpooling2d"i
        MAXPOOLING3D: "maxpooling3d"i
        CONV2D: "conv2d"i
        CONV1D: "conv1d"i
        CONV3D: "conv3d"i
        DROPOUT: "dropout"i
        FLATTEN: "flatten"i
        LSTM: "lstm"i
        GRU: "gru"i
        SIMPLERNN: "simplernn"i
        OUTPUT: "output"i
        TRANSFORMER: "transformer"i
        TRANSFORMER_ENCODER: "transformerencoder"i
        TRANSFORMER_DECODER: "transformerdecoder"i
        CONV2DTRANSPOSE: "conv2dtranspose"i
        LSTMCELL: "lstmcell"i
        GRUCELL: "grucell"i
        BATCHNORMALIZATION: "batchnormalization"i
        GAUSSIANNOISE: "gaussiannoise"i
        LAYERNORMALIZATION: "layernormalization"i
        INSTANCENORMALIZATION: "instancenormalization"i
        GROUPNORMALIZATION: "groupnormalization"i


        // Layer type tokens (case-insensitive)
        LAYER_TYPE.2: "dense"i | "conv2d"i | "conv1d"i | "conv3d"i | "dropout"i | "flatten"i | "lstm"i | "gru"i | "simplernn"i | "output"i| "transformer"i | "transformerencoder"i | "transformerdecoder"i | "conv2dtranspose"i | "maxpooling2d"i | "maxpooling1d"i | "maxpooling3d"i | "batchnormalization"i | "gaussiannoise"i | "instancenormalization"i | "groupnormalization"i


        // Basic tokens
        NAME: /[a-zA-Z_][a-zA-Z0-9_]*/
        STRING: /"[^"]*"/ | /'[^']*'/
        INT: /[+-]?[0-9]+/
        FLOAT: /[+-]?[0-9]*\.[0-9]+/ 
        NUMBER: INT | FLOAT 
        TRUE.2: "true"i
        FALSE.2: "false"i
        NONE.2: "none"i
        BOOL: TRUE | FALSE

        // Layer name patterns
        CUSTOM_LAYER: /[A-Z][a-zA-Z0-9]*Layer/  // Matches layer names ending with "Layer"
        MACRO_NAME: /(?!GroupNormalization|InstanceNormalization|LayerNormalization|GaussianNoise|TransformerEncoder|BatchNormalization|Dropout|Flatten|Output|Conv2DTranspose|LSTM|GRU|SimpleRNN|LSTMCell|GRUCell|Dense|Conv1D|Conv2D|Conv3D|MaxPooling1D|MaxPooling2D|MaxPooling3D)(?<!Layer)[A-Z][a-zA-Z0-9]*/

        // Comments and whitespace
        COMMENT: /#[^\n]*/
        WS: /[ \t\f]+/
        _NL: /[\r\n]+/
        _INDENT: /[ \t]+/
        _DEDENT: /\}/

        %ignore COMMENT
        %ignore WS
        %ignore _NL
        

        // Grammar rules
        ?start: network | layer | research

        neural_file: network
        nr_file: network
        rnr_file: research

        
        activation_param: "activation" "=" STRING
        ordered_params: value ("," value)* 
        number1: INT
        explicit_tuple: "(" value ("," value)+ ")"

        research: "research" NAME? "{" [research_params] "}"
        research_params: (metrics | references)*
        metrics: "metrics" "{" [accuracy_param] [metrics_loss_param] [precision_param] [recall_param] "}"
        accuracy_param: "accuracy:" FLOAT
        metrics_loss_param: "loss:" FLOAT
        precision_param: "precision:" FLOAT
        recall_param: "recall:" FLOAT
        references: "references" "{" paper_param+ "}"
        paper_param: "paper:" STRING

        bool_value: BOOL
        named_return_sequences: "return_sequences" "=" bool_value
        named_units: "units" "=" number
        named_activation: "activation" "=" STRING | "activation" "=" hpo_expr
        named_size: NAME ":" explicit_tuple  
        named_filters: "filters" "=" NUMBER
        named_strides: "strides" "=" value
        named_padding: "padding" "=" STRING | "padding" ":" STRING
        named_dilation_rate: "dilation_rate" "=" value
        named_groups: "groups" "=" NUMBER
        named_channels: "channels" "=" NUMBER
        named_num_heads: "num_heads" "=" NUMBER
        named_ff_dim: "ff_dim" "=" NUMBER
        named_input_dim: "input_dim" "=" NUMBER
        named_output_dim: "output_dim" "=" NUMBER
        named_rate: "rate" "=" FLOAT
        named_dropout: "dropout" "=" FLOAT
        named_axis: "axis" "=" NUMBER
        named_momentum: "momentum" "=" FLOAT
        named_epsilon: "epsilon" "=" FLOAT
        named_center: "center" "=" BOOL
        named_scale: "scale" "=" BOOL
        named_beta_initializer: "beta_initializer" "=" STRING
        named_gamma_initializer: "gamma_initializer" "=" STRING
        named_moving_mean_initializer: "moving_mean_initializer" "=" STRING
        named_moving_variance_initializer: "moving_variance_initializer" "=" STRING
        named_training: "training" "=" BOOL
        named_trainable: "trainable" "=" BOOL
        named_use_bias: "use_bias" "=" BOOL
        named_kernel_initializer: "kernel_initializer" "=" STRING
        named_bias_initializer: "bias_initializer" "=" STRING
        named_kernel_regularizer: "kernel_regularizer" "=" STRING
        named_bias_regularizer: "bias_regularizer" "=" STRING
        named_activity_regularizer: "activity_regularizer" "=" STRING
        named_kernel_constraint: "kernel_constraint" "=" STRING
        named_kernel_size: "kernel_size" "=" value
        named_bias_constraint: "bias_constraint" "=" STRING
        named_return_state: "return_state" "=" BOOL
        named_go_backwards: "go_backwards" "=" BOOL
        named_stateful: "stateful" "=" BOOL
        named_time_major: "time_major" "=" BOOL
        named_unroll: "unroll" "=" BOOL
        named_input_shape: "input_shape" "=" value
        named_batch_input_shape: "batch_input_shape" "=" value
        named_dtype: "dtype" "=" STRING
        named_name: "name" "=" STRING
        named_weights: "weights" "=" value
        named_embeddings_initializer: "embeddings_initializer" "=" STRING
        named_mask_zero: "mask_zero" "=" BOOL
        named_input_length: "input_length" "=" NUMBER
        named_embeddings_regularizer: "embeddings_regularizer" "=" STRING
        named_embeddings_constraint: "embeddings_constraint" "=" value
        named_num_layers: "num_layers" "=" NUMBER
        named_bidirectional: "bidirectional" "=" BOOL
        named_merge_mode: "merge_mode" "=" STRING
        named_recurrent_dropout: "recurrent_dropout" "=" FLOAT
        named_noise_shape: "noise_shape" "=" value
        named_seed: "seed" "=" NUMBER
        named_target_shape: "target_shape" "=" value
        named_data_format: "data_format" "=" STRING
        named_interpolation: "interpolation" "=" STRING
        named_crop_to_aspect_ratio: "crop_to_aspect_ratio" "=" BOOL
        named_mask_value: "mask_value" "=" NUMBER
        named_return_attention_scores: "return_attention_scores" "=" BOOL
        named_causal: "causal" "=" BOOL
        named_use_scale: "use_scale" "=" BOOL
        named_key_dim: "key_dim" "=" NUMBER
        named_value_dim: "value_dim" "=" NUMBER
        named_output_shape: "output_shape" "=" value
        named_arguments: "arguments" "=" value
        named_initializer: "initializer" "=" STRING
        named_regularizer: "regularizer" "=" STRING
        named_constraint: "constraint" "=" STRING
        named_l1: "l1" "=" FLOAT
        named_l2: "l2" "=" FLOAT
        named_l1_l2: "l1_l2" "=" tuple_
        named_int: NAME "=" INT | NAME ":" INT
        named_string: NAME "=" STRING | NAME ":" STRING
        named_float: NAME "=" FLOAT | NAME ":" FLOAT
        named_layer: NAME "," explicit_tuple
        simple_number: number1
        simple_float: FLOAT
        named_clipvalue: "clipvalue" "=" FLOAT
        named_clipnorm: "clipnorm" "=" FLOAT
        ?named_param: ( named_layer | named_clipvalue | named_clipnorm | named_units | pool_size | named_kernel_size | named_size | named_activation | named_filters | named_strides | named_padding | named_dilation_rate | named_groups | named_data_format | named_channels | named_return_sequences | named_num_heads | named_ff_dim | named_input_dim | named_output_dim | named_rate | named_dropout | named_axis | named_momentum | named_epsilon | named_center | named_scale | named_beta_initializer | named_gamma_initializer | named_moving_mean_initializer | named_moving_variance_initializer | named_training | named_trainable | named_use_bias | named_kernel_initializer | named_bias_initializer | named_kernel_regularizer | named_bias_regularizer | named_activity_regularizer | named_kernel_constraint | named_bias_constraint | named_return_state | named_go_backwards | named_stateful | named_time_major | named_unroll | named_input_shape | named_batch_input_shape | named_dtype | named_name | named_weights | named_embeddings_initializer | named_mask_zero | named_input_length | named_embeddings_regularizer | named_embeddings_constraint | named_num_layers | named_bidirectional | named_merge_mode | named_recurrent_dropout | named_noise_shape | named_seed | named_target_shape | named_interpolation | named_crop_to_aspect_ratio | named_mask_value | named_return_attention_scores | named_causal | named_use_scale | named_key_dim | named_value_dim | named_output_shape | named_arguments | named_initializer | named_regularizer | named_constraint | named_l1 | named_l2 | named_l1_l2 | named_int | named_float | NAME "=" value | NAME "=" hpo_expr )


        network: "network" NAME "{" input_layer layers [loss] [optimizer] [training_config] [execution_config] "}"
        input_layer: "input" ":" shape ("," shape)*
        layers: "layers" ":" ( layer_or_repeated)*
        loss: "loss" ":" (NAME | STRING) ["(" named_params ")"]
        optimizer: "optimizer:" (NAME | STRING) ["(" named_params ")"]
        layer_or_repeated: layer ["*" INT] 
        ?layer: basic_layer | advanced_layer | special_layer
        config: training_config | execution_config


        shape: "(" [number_or_none ("," number_or_none)* [","]] ")"
        number_or_none: number | NONE


        lambda_: "Lambda" "(" STRING ")"
        wrapper: "TimeDistributed" "(" layer ["," named_params] ")" [layer_block]

        dropout: "Dropout" "(" dropout_params ")"
        dropout_params: FLOAT | named_params
        flatten: "Flatten" "(" [named_params] ")"


        regularization: spatial_dropout1d | spatial_dropout2d | spatial_dropout3d | activity_regularization | l1 | l2 | l1_l2
        l1: "L1(" named_params ")"
        l2: "L2(" named_params ")"
        l1_l2: "L1L2(" named_params ")"

        output: OUTPUT "(" named_params ")"

        conv: conv1d | conv2d | conv3d | conv_transpose | depthwise_conv2d | separable_conv2d
        conv1d: CONV1D "(" param_style1 ")"
        conv2d: CONV2D "(" param_style1 ")"
        conv3d: CONV3D "(" param_style1 ")"
        conv_transpose: conv1d_transpose | conv2d_transpose | conv3d_transpose
        conv1d_transpose: "Conv1DTranspose" "(" param_style1 ")"
        conv2d_transpose: CONV2DTRANSPOSE "(" param_style1 ")"
        conv3d_transpose: "Conv3DTranspose" "(" param_style1 ")"
        depthwise_conv2d: "DepthwiseConv2D" "(" param_style1 ")"
        separable_conv2d: "SeparableConv2D" "(" param_style1 ")"

        pooling: max_pooling | average_pooling | global_pooling | adaptive_pooling
        max_pooling: max_pooling1d | max_pooling2d | max_pooling3d
        max_pooling1d: MAXPOOLING1D "(" named_params ")"
        max_pooling2d: MAXPOOLING2D "(" param_style1 ")"
        max_pooling3d: MAXPOOLING3D "(" named_params ")"
        pool_size: "pool_size" "=" value
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

        ?norm_layer: batch_norm | layer_norm | instance_norm | group_norm
        batch_norm: BATCHNORMALIZATION "(" [named_params] ")"
        layer_norm: LAYERNORMALIZATION "(" [named_params] ")"
        instance_norm: INSTANCENORMALIZATION "(" [named_params] ")"
        group_norm: GROUPNORMALIZATION "(" [named_params] ")"

        conv_rnn: conv_lstm | conv_gru
        conv_lstm: "ConvLSTM2D(" named_params ")"
        conv_gru: "ConvGRU2D(" named_params ")"

        rnn_cell: simple_rnn_cell | lstm_cell | gru_cell
        simple_rnn_cell: "SimpleRNNCell" "(" named_params ")"
        lstm_cell: LSTMCELL "(" named_params ")"
        gru_cell: GRUCELL "(" named_params ")"

        dropout_wrapper_layer: simple_rnn_dropout | gru_dropout | lstm_dropout
        simple_rnn_dropout: "SimpleRNNDropoutWrapper" "(" named_params ")"
        gru_dropout: "GRUDropoutWrapper" "(" named_params ")"
        lstm_dropout: "LSTMDropoutWrapper" "(" named_params ")"
        bidirectional_rnn_layer: bidirectional_simple_rnn_layer | bidirectional_lstm_layer | bidirectional_gru_layer
        bidirectional_simple_rnn_layer: "Bidirectional(SimpleRNN(" named_params "))"
        bidirectional_lstm_layer: "Bidirectional(LSTM(" named_params "))"
        bidirectional_gru_layer: "Bidirectional(GRU(" named_params "))"
        conv_rnn_layer: conv_lstm_layer | conv_gru_layer
        conv_lstm_layer: "ConvLSTM2D(" named_params ")"
        conv_gru_layer: "ConvGRU2D(" named_params ")"
        rnn_cell_layer: simple_rnn_cell_layer | lstm_cell_layer | gru_cell_layer
        simple_rnn_cell_layer: "SimpleRNNCell(" named_params ")"
        lstm_cell_layer: "LSTMCell" "(" named_params ")"
        gru_cell_layer: "GRUCell" "(" named_params ")"

        
        residual: "ResidualConnection" "(" [named_params] ")" [layer_block]
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


        spatial_dropout1d: "SpatialDropout1D(" named_params ")"
        spatial_dropout2d: "SpatialDropout2D(" named_params ")"
        spatial_dropout3d: "SpatialDropout3D(" named_params ")"
        activity_regularization: "ActivityRegularization(" named_params ")"

        activation: activation_with_params | activation_without_params
        activation_with_params: "Activation" "(" STRING "," named_params ")"
        activation_without_params: "Activation" "(" STRING ")"

        training_config: "train" "{" training_params "}"
        training_params: (epochs_param | batch_size_param | optimizer_param | search_method_param | validation_split_param)*
        epochs_param: "epochs:" INT
        batch_size_param: "batch_size:" values_list
        values_list: "[" value ("," value)* "]" | value ("," value)*
        optimizer_param: "optimizer:" named_optimizer
        named_optimizer: "named_optimizer(" learning_rate_param ")"
        learning_rate_param: "learning_rate=" FLOAT
        search_method_param: "search_method:" STRING
        validation_split_param: "validation_split:" FLOAT


        schedule: NAME "(" valparams ")"
        valparams: [value ("," value)*]

        execution_config: "execution" "{" device_param "}"
        device_param: "device:" STRING
        
        CUSTOM_SHAPE: "CustomShape"
        self_defined_shape: CUSTOM_SHAPE named_layer

        math_expr: term (("+"|"-") term)*
        term: factor (("*"|"/") factor)*
        factor: NUMBER | NAME| "(" math_expr ")" | function_call
        function_call: NAME "(" math_expr ("," math_expr)* ")"

        hpo_with_params: hpo_expr ("," named_params)*
        hpo: hpo_expr | layer_choice
        
        // HPO for Hyperparameters
        hpo_expr: "HPO" "(" (hpo_choice | hpo_range | hpo_log_range )* ")"
        hpo_choice: "choice" "(" value ("," value)* ")" 
        hpo_range: "range" "(" number "," number ("," "step="number)? ")"
        hpo_log_range: "log_range" "(" number "," number ")"
        
        // HPO for layer choice
        layer_choice: "HPO" "(" "choice" "(" layer ("," layer)* "))"

        // MACROS AND RELATED RULES
        define: "define" NAME "{" ( layer_or_repeated)*  "}"
        macro_ref: MACRO_NAME "(" [param_style1] ")" [layer_block]
        
        basic_layer: layer_type "(" [param_style1] ")" [layer_block]
        layer_type: DENSE | CONV2D | CONV1D | CONV3D | DROPOUT | FLATTEN | LSTM | GRU | SIMPLERNN | OUTPUT | TRANSFORMER | TRANSFORMER_ENCODER | TRANSFORMER_DECODER | CONV2DTRANSPOSE | LSTMCELL | GRUCELL | MAXPOOLING1D | MAXPOOLING2D | MAXPOOLING3D | BATCHNORMALIZATION | GAUSSIANNOISE | LAYERNORMALIZATION | INSTANCENORMALIZATION | GROUPNORMALIZATION
        ?param_style1: params | hpo_with_params
        params: param ("," param)*
        ?param: named_param | value
        ?value: STRING -> string_value | number | tuple_ | BOOL  
        tuple_: "(" number "," number ")"  
        number: NUMBER  
        named_params: named_param ("," named_param)*
        ?advanced_layer: (attention | transformer | residual | inception | capsule | squeeze_excitation | graph | embedding | quantum | dynamic)
        attention: "Attention" "(" [param_style1] ")" [layer_block]
        transformer: TRANSFORMER "(" [param_style1] ")" [layer_block]
                    | TRANSFORMER_ENCODER "(" [param_style1] ")" [layer_block]
                    | TRANSFORMER_DECODER "(" [param_style1] ")" [layer_block]


        special_layer: custom | macro_ref | wrapper | lambda_

        ?custom_or_macro: custom | macro_ref
        custom: CUSTOM_LAYER "(" param_style1 ")" [layer_block]

        layer_block: "{" (layer_or_repeated)* "}"
        
    """
    return lark.Lark(
        grammar,
        start=start_rule,
        parser='lalr',
        lexer='contextual',
        debug=True,
        cache=True,
        propagate_positions=True,
    )

def safe_parse(parser, text):
    warnings = []
    try:
        tree = parser.parse(text)
        return {"result": tree, "warnings": warnings}
    except (lark.UnexpectedCharacters, lark.UnexpectedToken) as e:
        # Handle Lark syntax errors
        result = custom_error_handler(e)
        if isinstance(result, dict):  # Warning case
            warnings.append(result)
            return {"result": None, "warnings": warnings}
        else:
            # If custom_error_handler raised an exception, propagate it
            raise result from e
    except DSLValidationError as e:
        # Catch and re-raise any DSLValidationError from custom_error_handler
        raise e
    except Exception as e:
        if isinstance(e, DSLValidationError):
            raise e
        else:
            raise DSLValidationError(f"Parsing failed: {str(e)}") from e

network_parser = create_parser('network')
layer_parser = create_parser('layer')
research_parser = create_parser('research')

class ModelTransformer(lark.Transformer):
    def __init__(self):
        super().__init__()
        self.variables = {}
        self.macros = {}
        self.current_macro = None
        self.layer_type_map = {
            'DENSE': 'dense',
            'CONV2D': 'conv2d',
            'CONV1D': 'conv1d',
            'CONV3D': 'conv3d',
            'DROPOUT': 'dropout',
            'FLATTEN': 'flatten',
            'LSTM': 'lstm',
            'GRU': 'gru',
            'SIMPLERNN': 'simplernn',
            'OUTPUT': 'output',
            'TRANSFORMER': 'transformer',
            'TRANSFORMER_ENCODER': 'transformer',
            'TRANSFORMER_DECODER': 'transformer',
            'CONV2DTRANSPOSE': 'conv2d_transpose',
            'LSTMCELL': 'lstmcell',
            'GRUCELL': 'grucell',
            'MAXPOOLING1D': 'maxpooling1d',
            'MAXPOOLING2D': 'maxpooling2d',
            'MAXPOOLING3D': 'maxpooling3d',
            'BATCHNORMALIZATION': 'batch_norm',
            'GAUSSIANNOISE': 'gaussian_noise',
            'LAYERNORMALIZATION': 'layer_norm',
            'INSTANCENORMALIZATION': 'instance_norm',
            'GROUPNORMALIZATION': 'group_norm',
        }
        self.hpo_params = []

    def _track_hpo(self, layer_type, param_name, hpo_data, node):
        self.hpo_params.append({
            'layer_type': layer_type,
            'param_name': param_name,
            'hpo': hpo_data['hpo'],
            'node': node  # Optional: for debugging
        })

    def parse_network_with_hpo(self, config):
        tree = create_parser('network').parse(config)
        model = self.transform(tree)
        return model, self.hpo_params


    def raise_validation_error(self, msg, item=None, severity=Severity.ERROR):
        if item and hasattr(item, 'meta'):
            line, col = item.meta.line, item.meta.column
            full_msg = f"{severity.name} at line {line}, column {col}: {msg}"
        else:
            line, col = None, None
            full_msg = f"{severity.name}: {msg}"
        
        log_by_severity(severity, full_msg)
        if severity.value >= Severity.ERROR.value:
            raise DSLValidationError(msg, severity, line, col)
        return {"warning": msg, "line": line, "column": col}  # Return for warnings

    def _extract_layer_def(self, layer_item):
        """
        Helper method to extract layer definition from an item.
        
        Args:
            layer_item: The layer item to process
        
        Returns:
            dict: The processed layer definition
        """
        if layer_item is None:
            return None
            
        layer_def = self._extract_value(layer_item)
        if not isinstance(layer_def, dict):
            self.raise_validation_error(f"Invalid layer definition: {layer_def}", layer_item)
            
        return layer_def


    def define(self, items):
        macro_name = items[0].value
        layers = self._extract_value(items[1])  # Expecting a list from layer_block
        
        if not layers:
            self.raise_validation_error(f"Macro '{macro_name}' must define at least one layer", items[0])
        
        self.macros[macro_name] = {
            'original': layers if isinstance(layers, list) else [layers],
            'macro': {'type': 'Macro', 'params': macro_name}
        }
        return layers  # Return the layers for potential immediate use

    @pysnooper.snoop()
    def macro_ref(self, items):
        macro_name = items[0].value
        if macro_name not in self.macros:
            self.raise_validation_error(f"Undefined macro '{macro_name}'", items[0])

        params = self._extract_value(items[1]) if len(items) > 1 and items[1].data == 'param_style1' else {}
        sub_layers = self._extract_value(items[2]) if len(items) > 2 and items[2].data == 'layer_block' else []

        macro_def = self.macros[macro_name]['original']
        if isinstance(macro_def, list):
            # If macro defines multiple layers, return them with updated params and sub-layers
            for layer in macro_def:
                layer['params'].update(params)
                if sub_layers:
                    layer.setdefault('sublayers', []).extend(sub_layers)
            return macro_def
        else:
            # Single layer macro
            macro_def['params'].update(params)
            if sub_layers:
                macro_def.setdefault('sublayers', []).extend(sub_layers)
            return macro_def
        
    def layer_block(self, items):
        """Process a block of nested layers."""
        sub_layers = []
        for item in items:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], int):
                layer, count = item
                sub_layers.extend([layer] * count)
            else:
                sub_layers.append(item)
        return sub_layers

    def basic_layer(self, items):
        """
        Parses a basic layer from the given items and returns its information.

        Args:
            items (list): A list of nodes representing the layer type, parameters, and sublayers.

        Returns:
            dict: A dictionary containing the layer type, parameters, and sublayers.

        Raises:
            ValidationError: If the layer type is unsupported.
        """
        layer_type_node = items[0]
        layer_type = layer_type_node.children[0].value.upper()
        params_node = items[1] if len(items) > 1 else None
        params = self._extract_value(params_node) if params_node else None
        sublayers_node = items[2] if len(items) > 2 else None
        sublayers = self._extract_value(sublayers_node) if sublayers_node else []
        
        method_name = self.layer_type_map.get(layer_type)
        if method_name and hasattr(self, method_name):
            try:
                layer_info = getattr(self, method_name)([params])
                layer_info['sublayers'] = sublayers
                return layer_info
            except DSLValidationError as e:
                raise e  # Re-raise DSLValidationError directly
        else:
            self.raise_validation_error(f"Unsupported layer type: {layer_type}", layer_type_node)
            return {'type': layer_type, 'params': params, 'sublayers': sublayers}
        
    def params(self, items):
        return [self._extract_value(item) for item in items]
    
    def param(self, items):
        return self._extract_value(items[0])

    def advanced_layer(self, items):
        return self._extract_value(items[0])
    
    def layers(self, items):
        expanded_layers = []
        for item in items:
            if isinstance(item, list):
                expanded_layers.extend(item)
            elif isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], int):
                layer, count = item
                expanded_layers.extend([layer] * count)
            else:
                expanded_layers.append(item)
        return expanded_layers

    def layer_or_repeated(self, items):
        if len(items) == 2:  # layer and multiplier
            return (items[0], int(items[1]))
        return items[0]  # single layer

    def input_layer(self, items):
        shapes = [self._extract_value(item) for item in items]
        return {'type': 'Input', 'shape': shapes[0] if len(shapes) == 1 else shapes}

    def flatten(self, items):
        params = self._extract_value(items[0]) if items else None
        return {'type': 'Flatten', 'params': params}

    def dropout(self, items):
        param_style = self._extract_value(items[0])
        params = {}
        
        if isinstance(param_style, list):
            merged_params = {}
            for elem in param_style:
                if isinstance(elem, dict):
                    if 'hpo' in elem:
                        merged_params['rate'] = elem
                        self._track_hpo('Dropout', 'rate', elem, items[0])  # Track HPO
                    else:
                        merged_params.update(elem)
                else:
                    merged_params['rate'] = elem
            param_style = merged_params

        if isinstance(param_style, dict):
            params = param_style.copy()
            if 'rate' in params:
                if not isinstance(params['rate'], float) and 'hpo' in params['rate']:
                    self._track_hpo('Dropout', 'rate', params['rate'], items[0])
                else:  # Validate only if not HPO
                    rate = params['rate']
                    if not isinstance(rate, (int, float)):
                        self.raise_validation_error(f"Dropout rate must be a number, got {rate}", items[0], Severity.ERROR)
                    elif not 0 <= rate <= 1:
                        self.raise_validation_error(f"Dropout rate should be between 0 and 1, got {rate}", items[0], Severity.WARNING)
            else:
                self.raise_validation_error("Dropout requires a 'rate' parameter", items[0], Severity.ERROR)
        elif isinstance(param_style, (int, float)):
            params['rate'] = param_style
            if not 0 <= params['rate'] <= 1:
                self.raise_validation_error(f"Dropout rate should be between 0 and 1, got {params['rate']}", items[0], Severity.WARNING)
        else:
            self.raise_validation_error("Invalid parameters for Dropout", items[0], Severity.ERROR)
        
        return {'type': 'Dropout', 'params': params}
    
    def output(self, items):
        params = self._extract_value(items[0])
        if isinstance(params, dict) and 'units' in params and 'hpo' in params['units']:
            self._track_hpo('Output', 'units', params['units'], items[0])  # Track HPO
        return {'type': 'Output', 'params': params}

    def regularization(self, items):
        return {'type': items[0].data.capitalize(), 'params': self._extract_value(items[0].children[0])}

    def execution_config(self, items):
        params = self._extract_value(items[0])
        return {'type': 'execution_config', 'params': params}

    def dense(self, items):
        logger.debug(f"dense called with items: {items}")
        params = {}
        if items and items[0] is not None:
            param_node = items[0]  # From param_style1
            param_values = self._extract_value(param_node) if param_node else []

            ordered_params = []
            named_params = {}
            if isinstance(param_values, list):
                for val in param_values:
                    if isinstance(val, dict):
                        if 'hpo' in val:  # Handle HPO as units
                            named_params['units'] = val
                        else:
                            named_params.update(val)
                    else:
                        ordered_params.append(val)
            elif isinstance(param_values, dict):
                named_params = param_values

            # Map positional arguments
            if ordered_params:
                if len(ordered_params) >= 1:
                    params['units'] = ordered_params[0]
                if len(ordered_params) >= 2:
                    params['activation'] = ordered_params[1]
                if len(ordered_params) > 2:
                    self.raise_validation_error("Dense with more than two positional parameters is not supported", items[0])
            params.update(named_params)

        if 'units' not in params:
            self.raise_validation_error("Dense layer requires 'units' parameter", items[0])
        
        units = params['units']
        if isinstance(units, dict):  # HPO case
            pass  # Allow HPO dict, no further validation needed here
        else:
            if not isinstance(units, (int, float)) or (isinstance(units, float) and not units.is_integer()):
                self.raise_validation_error(f"Dense units must be an integer, got {units}", items[0])
            if units <= 0:
                self.raise_validation_error(f"Dense units must be positive, got {units}", items[0])
            params['units'] = int(units)
        
        if 'activation' in params:
            activation = params['activation']
            if isinstance(activation, dict):  # HPO case
                pass
            elif not isinstance(activation, str):
                self.raise_validation_error(f"Dense activation must be a string or HPO, got {activation}", items[0])
        
        logger.debug(f"Returning: {{'type': 'Dense', 'params': {params}}}")
        return {"type": "Dense", "params": params}
    
    def conv(self, items):
        return items[0]

    def conv1d(self, items):
        params = self._extract_value(items[0])
        if 'filters' in params:
            filters = params['filters']
            if not isinstance(filters, int) or filters <= 0:
                self.raise_validation_error(f"Conv1D filters must be a positive integer, got {filters}", items[0])
        if 'kernel_size' in params:
            ks = params['kernel_size']
            if isinstance(ks, (list, tuple)):
                if not all(isinstance(k, int) and k > 0 for k in ks):
                    self.raise_validation_error(f"Conv1D kernel_size must be positive integers, got {ks}", items[0])
            elif not isinstance(ks, int) or ks <= 0:
                self.raise_validation_error(f"Conv1D kernel_size must be a positive integer, got {ks}", items[0])
        return {'type': 'Conv1D', 'params': params}

    def conv2d(self, items):
        param_style = items[0]
        raw_params = self._extract_value(param_style)
        ordered_params = []
        named_params = {}
        if isinstance(raw_params, list):
            for param in raw_params:
                if isinstance(param, dict):
                    named_params.update(param)
                else:
                    ordered_params.append(param)
        elif isinstance(raw_params, dict):
            named_params = raw_params
        else:
            ordered_params.append(raw_params)
        params = {}
        if ordered_params:
            if len(ordered_params) >= 1:
                params['filters'] = ordered_params[0]
            if len(ordered_params) >= 2:
                params['kernel_size'] = ordered_params[1]
                if isinstance(params['kernel_size'], (list, tuple)):
                    params['kernel_size'] = tuple(params['kernel_size'])
            if len(ordered_params) >= 3:
                params['activation'] = ordered_params[2]
        params.update(named_params)
        if 'filters' in params:
            filters = params['filters']
            if not isinstance(filters, int) or filters <= 0:
                self.raise_validation_error(f"Conv2D filters must be a positive integer, got {filters}", items[0], Severity.ERROR)
        if 'kernel_size' in params:
            ks = params['kernel_size']
            if isinstance(ks, (list, tuple)):
                if not all(isinstance(k, int) for k in ks):
                    self.raise_validation_error(f"Conv2D kernel_size must be integers, got {ks}", items[0], Severity.ERROR)
                elif not all(k > 0 for k in ks):
                    self.raise_validation_error(f"Conv2D kernel_size should be positive integers, got {ks}", items[0], Severity.ERROR)
            elif not isinstance(ks, int) or ks <= 0:
                self.raise_validation_error(f"Conv2D kernel_size must be a positive integer, got {ks}", items[0], Severity.ERROR)
        return {'type': 'Conv2D', 'params': params}

    def conv3d(self, items):
        params = self._extract_value(items[0])
        if 'filters' in params:
            filters = params['filters']
            if not isinstance(filters, int) or filters <= 0:
                self.raise_validation_error(f"Conv3D filters must be a positive integer, got {filters}", items[0])
        if 'kernel_size' in params:
            ks = params['kernel_size']
            if isinstance(ks, (list, tuple)):
                if not all(isinstance(k, int) and k > 0 for k in ks):
                    self.raise_validation_error(f"Conv3D kernel_size must be positive integers, got {ks}", items[0])
            elif not isinstance(ks, int) or ks <= 0:
                self.raise_validation_error(f"Conv3D kernel_size must be a positive integer, got {ks}", items[0])
        return {'type': 'Conv3D', 'params': params}

    def conv1d_transpose(self, items):
        return {'type': 'Conv1DTranspose', 'params': self._extract_value(items[0])}

    def conv2d_transpose(self, items):
        return {'type': 'Conv2DTranspose', 'params': self._extract_value(items[0])}

    def conv3d_transpose(self, items):
        return {'type': 'Conv3DTranspose', 'params': self._extract_value(items[0])}

    def depthwise_conv2d(self, items):
        return {'type': 'DepthwiseConv2D', 'params': self._extract_value(items[0])}

    def separable_conv2d(self, items):
        return {'type': 'SeparableConv2D', 'params': self._extract_value(items[0])}

    def graph_conv(self, items):
        params = self._extract_value(items[0]) if items else None
        return {'type': 'GraphConv', 'params': params}

    def loss(self, items):
        return items[0].value.strip('"')

    def optimizer(self, items):
        params = {}
        opt_type = None
        
        # Extract the value from the first item
        opt_value = self._extract_value(items[0])
        
        if isinstance(opt_value, str):
            # Check if it contains parameters (e.g., "Adam(learning_rate=HPO(...))")
            if '(' in opt_value and ')' in opt_value:
                # Split into type and params string
                opt_type = opt_value[:opt_value.index('(')].strip()
                param_str = opt_value[opt_value.index('(')+1:opt_value.rindex(')')].strip()
                
                # Parse the parameter string (e.g., "learning_rate=HPO(log_range(1e-4, 1e-2))")
                param_parts = param_str.split('=', 1)
                if len(param_parts) == 2:
                    param_name = param_parts[0].strip()  # e.g., "learning_rate"
                    param_value = param_parts[1].strip()  # e.g., "HPO(log_range(1e-4, 1e-2))"
                    
                    # Check if it’s an HPO expression
                    if param_value.startswith('HPO(') and param_value.endswith(')'):
                        hpo_str = param_value[4:-1]  # Extract "log_range(1e-4, 1e-2)"
                        hpo_config = self._parse_hpo(hpo_str, items[0])
                        params[param_name] = hpo_config
                    else:
                        # Handle scalar values if needed (e.g., "learning_rate=0.001")
                        try:
                            params[param_name] = float(param_value)
                        except ValueError:
                            params[param_name] = param_value
            else:
                # Simple optimizer with no params (e.g., "adam")
                opt_type = opt_value.lower() if opt_value.lower() in ['adam', 'sgd', 'rmsprop'] else opt_value
        
        elif isinstance(opt_value, dict):
            # Handle case where optimizer params are already parsed as a dict
            params = opt_value
            opt_type = params.pop('type', None) or 'Adam'  # Default to Adam if no type specified
        
        if not opt_type:
            self.raise_validation_error("Optimizer type must be specified", items[0], Severity.ERROR)
        
        return {'type': opt_type, 'params': params}

    def schedule(self, items):
        return {"type": items[0].value, "args": [self._extract_value(x) for x in items[1].children]}

    def training_config(self, items):
        params = self._extract_value(items[0]) if items else {}
        return {'type': 'training_config', 'params': params}

    def execution_config(self, items):
        params = self._extract_value(items[0]) if items else {}
        return {'type': 'execution_config', 'params': params}

    def training_params(self, items):
        params = {}
        for item in items:
            if isinstance(item, Tree):
                result = self._extract_value(item)
                if isinstance(result, dict):
                    params.update(result)
                else:
                    self.raise_validation_error(f"Expected dictionary from {item.data}, got {result}", item)
            elif isinstance(item, dict):
                params.update(item)
        
        # Ensure validation_split is between 0 and 1 (if applicable)
        if "validation_split" in params:
            val_split = params["validation_split"]
            if not (0 <= val_split <= 1):
                self.raise_validation_error(f"validation_split must be between 0 and 1, got {val_split}")
        
        return params

    def validation_split_param(self, items):
        return {'validation_split': self._extract_value(items[0])}

    def epochs_param(self, items):
        return {'epochs': self._extract_value(items[0])}

    def batch_size_param(self, items):
        return {'batch_size': self._extract_value(items[0])}

    def values_list(self, items):
        values = [self._extract_value(x) for x in items]
        return values[0] if len(values) == 1 else values

    def optimizer_param(self, items):
        return {'optimizer': self._extract_value(items[0])}

    def named_optimizer(self, items):
        return {'named_optimizer': self._extract_value(items[0])}

    def learning_rate_param(self, items):
        return {'learning_rate': self._extract_value(items[0])}

    def shape(self, items):
        return tuple(self._extract_value(item) for item in items)

    def wrapper(self, items):
        wrapper_type = items[0]  # e.g., "TimeDistributed"
        inner_layer = self._extract_value(items[1])  # The wrapped layer
        param_idx = 2
        params = {}
        sub_layers = []

        # Extract additional named parameters if present
        if len(items) > param_idx and isinstance(items[param_idx], Tree) and items[param_idx].data == 'named_params':
            params = self._extract_value(items[param_idx])
            param_idx += 1

        # Extract sub-layers if present
        if len(items) > param_idx and isinstance(items[param_idx], Tree) and items[param_idx].data == 'layer_block':
            sub_layers = self._extract_value(items[param_idx])

        inner_layer['params'].update(params)
        return {'type': f"{wrapper_type}({inner_layer['type']})", 'params': inner_layer['params'], 'sublayers': sub_layers}

    def pooling(self, items):
        return items[0]

    def max_pooling(self, items):
        return self._extract_value(items[0])

    def pool_size(self, items):
        value = self._extract_value(items[0])
        return {'pool_size': value}

    def maxpooling1d(self, items):
        param_nodes = items[0].children
        params = {}
        param_vals = [self._extract_value(child) for child in param_nodes]
        if all(isinstance(p, dict) for p in param_vals):
            for p in param_vals:
                params.update(p)
        else:
            if len(param_vals) >= 1:
                params["pool_size"] = param_vals[0]
            if len(param_vals) >= 2:
                params["strides"] = param_vals[1]
            if len(param_vals) >= 3:
                params["padding"] = param_vals[2]
        for key in ['pool_size', 'strides']:
            if key in params:
                val = params[key]
                if isinstance(val, (list, tuple)):
                    if not all(isinstance(v, int) and v > 0 for v in val):
                        self.raise_validation_error(f"MaxPooling1D {key} must be positive integers, got {val}", items[0])
                elif not isinstance(val, int) or val <= 0:
                    self.raise_validation_error(f"MaxPooling1D {key} must be a positive integer, got {val}", items[0])
        return {'type': 'MaxPooling1D', 'params': params}

    @pysnooper.snoop()
    def maxpooling2d(self, items):
        param_style = self._extract_value(items[0])
        params = {}
        if isinstance(param_style, list):
            ordered_params = [p for p in param_style if not isinstance(p, dict)]
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
        for key in ['pool_size', 'strides']:
            if key in params:
                val = params[key]
                if isinstance(val, (list, tuple)):
                    if not all(isinstance(v, int) and v > 0 for v in val):
                        self.raise_validation_error(f"MaxPooling2D {key.strip("_")} must be positive integers, got {val}", items[0])
                elif not isinstance(val, int) or val <= 0:
                    self.raise_validation_error(f"MaxPooling2D {key.strip("_")} must be a positive integer, got {val}", items[0])
        
        if 'pool_size' in params:
            pool_size = params['pool_size']
            if isinstance(pool_size, (list, tuple)):
                if not all(isinstance(v, int) and v > 0 for v in pool_size):
                    self.raise_validation_error("pool size must be positive", items[0])
            elif not isinstance(pool_size, int) or pool_size <= 0:
                self.raise_validation_error("pool size must be positive", items[0])
        else:
            self.raise_validation_error("Missing required parameter 'pool_size'", items[0])
        
        return {'type': 'MaxPooling2D', 'params': params}

    def maxpooling3d(self, items):
        param_nodes = items[0].children
        params = {}
        param_vals = [self._extract_value(child) for child in param_nodes]
        if all(isinstance(p, dict) for p in param_vals):
            for p in param_vals:
                params.update(p)
        else:
            if len(param_vals) >= 1:
                params["pool_size"] = param_vals[0]
            if len(param_vals) >= 2:
                params["strides"] = param_vals[1]
            if len(param_vals) >= 3:
                params["padding"] = param_vals[2]
        for key in ['pool_size', 'strides']:
            if key in params:
                val = params[key]
                if isinstance(val, (list, tuple)):
                    if not all(isinstance(v, int) and v > 0 for v in val):
                        self.raise_validation_error(f"MaxPooling3D {key} must be positive integers, got {val}", items[0])
                elif not isinstance(val, int) or val <= 0:
                    self.raise_validation_error(f"MaxPooling3D {key} must be a positive integer, got {val}", items[0])
        return {"type": "MaxPooling3D", "params": params}

    def average_pooling1d(self, items):
        return {'type': 'AveragePooling1D', 'params': self._extract_value(items[0])}

    def average_pooling2d(self, items):
        return {'type': 'AveragePooling2D', 'params': self._extract_value(items[0])}

    def average_pooling3d(self, items):
        return {'type': 'AveragePooling3D', 'params': self._extract_value(items[0])}

    def global_max_pooling1d(self, items):
        return {'type': 'GlobalMaxPooling1D', 'params': self._extract_value(items[0])}

    def global_max_pooling2d(self, items):
        return {'type': 'GlobalMaxPooling2D', 'params': self._extract_value(items[0])}

    def global_max_pooling3d(self, items):
        return {'type': 'GlobalMaxPooling3D', 'params': self._extract_value(items[0])}

    def global_average_pooling1d(self, items):
        return {'type': 'GlobalAveragePooling1D', 'params': self._extract_value(items[0])}

    def global_average_pooling2d(self, items):
        return {'type': 'GlobalAveragePooling2D', 'params': self._extract_value(items[0])}

    def global_average_pooling3d(self, items):
        return {'type': 'GlobalAveragePooling3D', 'params': self._extract_value(items[0])}

    def adaptive_max_pooling1d(self, items):
        return {'type': 'AdaptiveMaxPooling1D', 'params': self._extract_value(items[0])}

    def adaptive_max_pooling2d(self, items):
        return {'type': 'AdaptiveMaxPooling2D', 'params': self._extract_value(items[0])}

    def adaptive_max_pooling3d(self, items):
        return {'type': 'AdaptiveMaxPooling3D', 'params': self._extract_value(items[0])}

    def adaptive_average_pooling1d(self, items):
        return {'type': 'AdaptiveAveragePooling1D', 'params': self._extract_value(items[0])}

    def adaptive_average_pooling2d(self, items):
        return {'type': 'AdaptiveAveragePooling2D', 'params': self._extract_value(items[0])}

    def adaptive_average_pooling3d(self, items):
        return {'type': 'AdaptiveAveragePooling3D', 'params': self._extract_value(items[0])}

    ## Normalization ##

    def norm_layer(self, items):
        return self._extract_value(items[0])

    def batch_norm(self, items):
        params = self._extract_value(items[0]) if items else None
        if params and 'axis' in params:
            axis = params['axis']
            if not isinstance(axis, int):
                self.raise_validation_error(f"BatchNormalization axis must be an integer, got {axis}", items[0])
        return {'type': 'BatchNormalization', 'params': params}

    def layer_norm(self, items):
        params = self._extract_value(items[0]) if items else None
        return {'type': 'LayerNormalization', 'params': params}

    def instance_norm(self, items):
        params = self._extract_value(items[0]) if items else None
        return {'type': 'InstanceNormalization', 'params': params}

    @pysnooper.snoop()
    def group_norm(self, items):
        raw_params = self._extract_value(items[0]) if items else None
        return {'type': 'GroupNormalization', 'params': params}

    ############

    @pysnooper.snoop()
    def lstm(self, items):
        params = {}
        if items and items[0] is not None:
            param_node = items[0]  # From param_style1
            param_values = self._extract_value(param_node)
            if isinstance(param_values, list):
                for val in param_values:
                    if isinstance(val, dict):
                        params.update(val)
                    else:
                        # Handle positional units parameter if present
                        if 'units' not in params:
                            params['units'] = val
            elif isinstance(param_values, dict):
                params = param_values

        if 'units' not in params:
            self.raise_validation_error("LSTM requires 'units' parameter", items[0])
        
        units = params['units']
        if isinstance(units, dict) and 'hpo' in units:
            pass  # HPO handled elsewhere
        else:
            if not isinstance(units, (int, float)) or (isinstance(units, float) and not units.is_integer()):
                self.raise_validation_error(f"LSTM units must be an integer, got {units}", items[0])
            if units <= 0:
                self.raise_validation_error(f"LSTM units must be positive, got {units}", items[0])
            params['units'] = int(units)
        
        return {'type': 'LSTM', 'params': params}

    def gru(self, items):
        params = {}
        if items and items[0] is not None:
            param_node = items[0]
            param_values = self._extract_value(param_node)
            if isinstance(param_values, list):
                for val in param_values:
                    if isinstance(val, dict):
                        params.update(val)
                    else:
                        if 'units' not in params:
                            params['units'] = val
            elif isinstance(param_values, dict):
                params = param_values

        if 'units' not in params:
            self.raise_validation_error("GRU requires 'units' parameter", items[0])
        
        units = params['units']
        if isinstance(units, dict) and 'hpo' in units:
            pass
        else:
            if not isinstance(units, (int, float)) or (isinstance(units, float) and not units.is_integer()):
                self.raise_validation_error(f"GRU units must be an integer, got {units}", items[0])
            if units <= 0:
                self.raise_validation_error(f"GRU units must be positive, got {units}", items[0])
            params['units'] = int(units)
        
        return {'type': 'GRU', 'params': params}
    
    def simplernn(self, items):
        params = {}
        if items and items[0] is not None:
            param_node = items[0]
            param_values = self._extract_value(param_node)
            if isinstance(param_values, list):
                for val in param_values:
                    if isinstance(val, dict):
                        params.update(val)
                    else:
                        if 'units' not in params:
                            params['units'] = val
            elif isinstance(param_values, dict):
                params = param_values

        if 'units' not in params:
            self.raise_validation_error("SimpleRNN requires 'units' parameter", items[0])
        
        units = params['units']
        if isinstance(units, dict) and 'hpo' in units:
            pass
        else:
            if not isinstance(units, (int, float)) or (isinstance(units, float) and not units.is_integer()):
                self.raise_validation_error(f"SimpleRNN units must be an integer, got {units}", items[0])
            if units <= 0:
                self.raise_validation_error(f"SimpleRNN units must be positive, got {units}", items[0])
            params['units'] = int(units)
        
        return {'type': 'SimpleRNN', 'params': params}

    def conv_lstm(self, items):
        return {'type': 'ConvLSTM2D', 'params': self._extract_value(items[0])}

    def conv_gru(self, items):
        return {'type': 'ConvGRU2D', 'params': self._extract_value(items[0])}

    def bidirectional_rnn(self, items):
        rnn_layer = items[0]
        bidirectional_params = self._extract_value(items[1])
        rnn_layer['params'].update(bidirectional_params)
        return {'type': f"Bidirectional({rnn_layer['type']})", 'params': rnn_layer['params']}

    def cudnn_gru_layer(self, items):
        return {'type': 'GRU', 'params': self._extract_value(items[0])}

    def bidirectional_simple_rnn_layer(self, items):
        return {'type': 'Bidirectional(SimpleRNN)', 'params': self._extract_value(items[0])}

    def bidirectional_lstm_layer(self, items):
        return {'type': 'Bidirectional(LSTM)', 'params': self._extract_value(items[0])}

    def bidirectional_gru_layer(self, items):
        return {'type': 'Bidirectional(GRU)', 'params': self._extract_value(items[0])}

    def conv_lstm_layer(self, items):
        return {'type': 'ConvLSTM2D', 'params': self._extract_value(items[0])}

    def conv_gru_layer(self, items):
        return {'type': 'ConvGRU2D', 'params': self._extract_value(items[0])}

    ## Cell Layers ##

    def rnn_cell_layer(self, items):
        return {'type': 'RNNCell', 'params': self._extract_value(items[0])}

    def simple_rnn_cell(self, items):
        return {'type': 'SimpleRNNCell', 'params': self._extract_value(items[0])}

    def lstmcell(self, items):
        return {'type': 'LSTMCell', 'params': self._extract_value(items[0])}

    def grucell(self, items):
        params = {}
        if items and items[0] is not None:
            param_node = items[0]  # From param_style1
            param_values = self._extract_value(param_node) if param_node else []
            if isinstance(param_values, list):
                for val in param_values:
                    if isinstance(val, dict):
                        params.update(val)
                    else:
                        params['units'] = val  # Handle positional units if present
            elif isinstance(param_values, dict):
                params = param_values
        if 'units' not in params:
            self.raise_validation_error("GRUCell requires 'units' parameter", items[0])
        return {"type": "GRUCell", "params": params}

    def simple_rnn_dropout(self, items):
        return {"type": "SimpleRNNDropoutWrapper", 'params': self._extract_value(items[0])}

    def gru_dropout(self, items):
        return {"type": "GRUDropoutWrapper", 'params': self._extract_value(items[0])}

    def lstm_dropout(self, items):
        return {"type": "LSTMDropoutWrapper", 'params': self._extract_value(items[0])}

    def research(self, items):
        name = None
        params = {}
        if items and isinstance(items[0], Token) and items[0].type == 'NAME':
            name = self._extract_value(items[0])
            if len(items) > 1:
                params = self._extract_value(items[1])
        else:
            if items:
                params = self._extract_value(items[0])
        return {'type': 'Research', 'name': name, 'params': params}

    def research_params(self, items):
        params = {}
        for item in items:
            if isinstance(item, Tree):
                params.update(self._extract_value(item))
            elif isinstance(item, dict):
                params.update(item)
        return params

    def metrics(self, items):
        if not items:
            return {'metrics': {}}
        result = {}
        for item in items:
            if item is None:
                continue
            val = self._extract_value(item)
            if isinstance(val, dict):
                result.update(val)
            elif isinstance(val, str) and ':' in val:
                key, v = val.split(':', 1)
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

    def paper_param(self, items):
        # Extract the string value from the 'paper:' parameter
        return self._extract_value(items[0])

    def references(self, items):
        papers = [self._extract_value(item) for item in items if item is not None]
        return {'references': papers}
    def metrics_loss_param(self, items):
        return {'loss': self._extract_value(items[0])}

    def network(self, items):
        name = str(items[0].value)
        input_layer_config = self._extract_value(items[1])
        layers_config = self._extract_value(items[2])  # Already handles nested layers via layers method
        loss_config = self._extract_value(items[3])
        optimizer_config = self._extract_value(items[4])
        
        training_config = None
        execution_config = {'params': {'device': 'auto'}}
        
        for item in items[5:]:
            if isinstance(item, dict):
                if item.get('type') == 'training_config':
                    training_config = item.get('params')
                elif item.get('type') == 'execution_config':
                    execution_config = item
        
        output_layer = next((layer for layer in reversed(layers_config) if layer['type'] == 'Output'), None)
        output_shape = output_layer.get('params', {}).get('units') if output_layer else None
        
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
            'execution_config': execution_config.get('params', {'device': 'auto'})
        }

    def search_method_param(self, items):
        value = self._extract_value(items[0])  # Extract "bayesian" from STRING token
        return {'search_method': value}

    def _extract_value(self, item):
        if isinstance(item, Token):
            if item.type == 'NAME':
                return item.value
            if item.type in ('INT', 'FLOAT', 'NUMBER', 'SIGNED_NUMBER'):
                try:
                    return int(item.value)
                except ValueError:
                    return float(item.value)
            elif item.type == 'BOOL':
                return item.value.lower() == 'true'
            elif item.type == 'STRING':
                return item.value.strip('"')
            elif item.type == 'WS_INLINE':
                return item.value.strip()
        elif isinstance(item, Tree) and item.data == 'number_or_none':
            child = item.children[0]
            if isinstance(child, Token) and child.value.upper() in ('NONE', 'None'):
                return None
            else:
                return self._extract_value(child)
        elif isinstance(item, Tree):
            if item.data == 'string_value':
                return self._extract_value(item.children[0])
            elif item.data == 'number':
                return self._extract_value(item.children[0])
            elif item.data == 'bool_value':
                return self._extract_value(item.children[0])
            elif item.data in ('tuple_', 'explicit_tuple'):
                return tuple(self._extract_value(child) for child in item.children)
            else:
                extracted = [self._extract_value(child) for child in item.children]
                if any(isinstance(e, dict) for e in extracted):
                    return extracted
                if len(item.children) % 2 == 0:
                    try:
                        # Check if all keys are strings to form a valid dictionary
                        valid = True
                        pairs = []
                        for k_node, v_node in zip(item.children[::2], item.children[1::2]):
                            key = self._extract_value(k_node)
                            if not isinstance(key, str):
                                valid = False
                                break
                            value = self._extract_value(v_node)
                            pairs.append((key, value))
                        if valid:
                            return dict(pairs)
                        else:
                            return extracted
                    except TypeError:
                        return extracted
                else:
                    return extracted
        elif isinstance(item, list):
            return [self._extract_value(elem) for elem in item]
        elif isinstance(item, dict):
            return {k: self._extract_value(v) for k, v in item.items()}
        return item

    ## Named Parameters ##

    def named_params(self, items):
        params = {}
        for item in items:
            if isinstance(item, Tree):
                params.update(self._extract_value(item))
            elif isinstance(item, dict):
                params.update(item)
            elif isinstance(item, list):
                for i in item:
                    params.update(self._extract_value(i))
        return params

    
    def named_param(self, items):
        return {items[0].value: self._extract_value(items[1])}
    
    def named_float(self, items):
        return {items[0].value: self._extract_value(items[1])}

    def named_int(self, items):
        return {items[0].value: self._extract_value(items[1])}

    def named_string(self, items):
        return {items[0].value: self._extract_value(items[1])}

    def number(self, items):
        return self._extract_value(items[0])

    def rate(self, items):
        return {'rate': self._extract_value(items[0])}

    def simple_float(self, items):
        return self._extract_value(items[0])

    def number_or_none(self, items):
        if not items:
            return None
        value = self._extract_value(items[0])
        if value == "None":
            return None
        try:
            return int(value) if '.' not in str(value) else float(value)
        except Exception as e:
            self.raise_validation_error(f"Error converting {value} to a number: {e}", items[0])

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

    def named_kernel_size(self, items):
        return {"kernel_size": self._extract_value(items[0])}

    def named_filters(self, items):
        return {"filters": self._extract_value(items[0])}

    def named_units(self, items):
        return {"units": self._extract_value(items[0])}

    def activation_param(self, items):
        return {"activation": self._extract_value(items[0])}

    def named_activation(self, items):
        return {"activation": self._extract_value(items[0])}

    def named_strides(self, items):
        return {"strides": self._extract_value(items[0])}

    def named_padding(self, items):
        return {"padding": self._extract_value(items[0])}

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

    def named_input_dim(self, items):
        return {"input_dim": self._extract_value(items[0])}

    def named_output_dim(self, items):
        return {"output_dim": self._extract_value(items[0])}

    def groups_param(self, items):
        return {'groups': self._extract_value(items[0])}

    def device_param(self, items):
        return {'device': self._extract_value(items[0])}


    ### Advanced Layers ###

    def attention(self, items):
        params = self._extract_value(items[0]) if items else None
        sub_layers = self._extract_value(items[1]) if len(items) > 1 and items[1].data == 'layer_block' else []
        return {'type': 'Attention', 'params': params, 'sublayers': sub_layers}

    def residual(self, items):
        params = self._extract_value(items[0]) if items and items[0].data == 'named_params' else {}
        sub_layers = self._extract_value(items[1]) if len(items) > 1 and items[1].data == 'layer_block' else []
        return {'type': 'ResidualConnection', 'params': params, 'sublayers': sub_layers}

    def inception(self, items):
        params = self._extract_value(items[0]) if items else None
        return {'type': 'Inception', 'params': params}

    def graph(self, items):
        return items[0]

    def graph_attention(self, items):
        params = self._extract_value(items[0])
        return {'type': 'GraphAttention', 'params': params}

    def dynamic(self, items):
        params = self._extract_value(items[0]) if items else None
        return {'type': 'DynamicLayer', 'params': params}

    def noise_layer(self, items):
        return {'type': items[0].data.capitalize(), 'params': self._extract_value(items[0].children[0])}

    def normalization_layer(self, items):
        return {'type': items[0].data.capitalize(), 'params': self._extract_value(items[0].children[0])}

    def regularization_layer(self, items):
        return {'type': items[0].data.capitalize(), 'params': self._extract_value(items[0].children[0])}

    def custom_layer(self, items):
        params = self._extract_value(items[0])
        return {'type': 'Capsule', 'params': params}

    def capsule(self, items):
        params = self._extract_value(items[0]) if items else None
        return {'type': 'CapsuleLayer', 'params': params}

    def squeeze_excitation(self, items):
        params = self._extract_value(items[0]) if items else None
        return {'type': 'SqueezeExcitation', 'params': params}

    def quantum(self, items):
        params = self._extract_value(items[0]) if items else None
        return {'type': 'QuantumLayer', 'params': params}


    ## Transformers - Encoders - Decoders ##

    @pysnooper.snoop()
    def transformer(self, items):
        if isinstance(items[0], Token):
            transformer_type = items[0].value
        else:
            self.raise_validation_error("Invalid transformer syntax: missing type identifier", items[0])

        params = {}
        sub_layers = []
        param_idx = 1

        print(items[2])
        print(items[1])

        # Transformer Parameters
        if len(items) > param_idx and isinstance(items[param_idx], Tree) and items[param_idx].data == 'params':
            raw_params = self._extract_value(items[param_idx])
            if isinstance(raw_params, dict):
                params = self._extract_value(raw_params[0])
            elif isinstance(raw_params, list):
                for param in raw_params:
                    if isinstance(param, dict):
                        params.update(param)
        param_idx += 1

        # Extract sub-layers if present
        if len(items) > param_idx and isinstance(items[param_idx], list):
            sub_layers = self._extract_value(items[param_idx])

        # Validation
        for key in ['num_heads', 'ff_dim']:
            if key in params:
                val = params[key]
                if isinstance(val, dict) and 'hpo' in val:
                    continue
                if not isinstance(val, int) or val <= 0:
                    self.raise_validation_error(f"{transformer_type} {key} must be a positive integer, got {val}", items[0])

        return {'type': transformer_type, 'params': params, 'sublayers': sub_layers}

    def named_num_heads(self, items):
        params = self._extract_value(items[0]) if items else None
        return {"num_heads": params }

    def named_ff_dim(self, items):
        params = self._extract_value(items[0]) if items else None
        return {"ff_dim": params}

    def embedding(self, items):
        params = self._extract_value(items[0]) if items else None
        for key in ['input_dim', 'output_dim']:
            if key in params:
                dim = params[key]
                if not isinstance(dim, int) or dim <= 0:
                    self.raise_validation_error(f"Embedding {key} must be a positive integer, got {dim}", items[0])
        return {'type': 'Embedding', 'params': params}

    def lambda_(self, items):
        return {'type': 'Lambda', 'params': {'function': self._extract_value(items[0])}}

    def add(self, items):
        return {'type': 'Add', 'params': self._extract_value(items[0])}

    def subtract(self, items):
        return {'type': 'Subtract', 'params': self._extract_value(items[0])}

    def multiply(self, items):
        return {'type': 'Multiply', 'params': self._extract_value(items[0])}

    def average(self, items):
        return {'type': 'Average', 'params': self._extract_value(items[0])}

    def maximum(self, items):
        return {'type': 'Maximum', 'params': self._extract_value(items[0])}

    def concatenate(self, items):
        return {'type': 'Concatenate', 'params': self._extract_value(items[0])}

    def dot(self, items):
        return {'type': 'Dot', 'params': self._extract_value(items[0])}

    ## Statistical Noises ##

    @pysnooper.snoop()
    def gaussian_noise(self, items):
        raw_params = self._extract_value(items[0])
        params = {}
        
        if isinstance(raw_params, list):
            # Flatten nested lists and merge dictionaries
            for param in raw_params:
                if isinstance(param, dict):
                    params.update(param)
                elif isinstance(param, list):
                    for sub_param in param:
                        params.update(sub_param)
        elif isinstance(raw_params, dict):
            params = raw_params
        
        return {'type': 'GaussianNoise', 'params': params}
    
    def stddev(self, items):
        return {'stddev': self._extract_value(items[1])}

    def gaussian_dropout(self, items):
        return {'type': 'GaussianDropout', 'params': self._extract_value(items[0])}

    def alpha_dropout(self, items):
        return {'type': 'AlphaDropout', 'params': self._extract_value(items[0])}

    def batch_normalization(self, items):
        return {'type': 'BatchNormalization', 'params': self._extract_value(items[0])}

    def layer_normalization(self, items):
        return {'type': 'LayerNormalization', 'params': self._extract_value(items[0])}

    def instance_normalization(self, items):
        return {'type': 'InstanceNormalization', 'params': self._extract_value(items[0])}

    def group_normalization(self, items):
        return {'type': 'GroupNormalization', 'params': self._extract_value(items[0])}

    def spatial_dropout1d(self, items):
        return {'type': 'SpatialDropout1D', 'params': self._extract_value(items[0])}

    def spatial_dropout2d(self, items):
        return {'type': 'SpatialDropout2D', 'params': self._extract_value(items[0])}

    def spatial_dropout3d(self, items):
        return {'type': 'SpatialDropout3D', 'params': self._extract_value(items[0])}

    def activity_regularization(self, items):
        return {'type': 'ActivityRegularization', 'params': self._extract_value(items[0])}

    def l1_l2(self, items):
        return {'type': 'L1L2', 'params': self._extract_value(items[0])}

    
    def custom(self, items):
        layer_type = items[0].value
        params = self._extract_value(items[1]) if items[1].data == 'param_style1' else {}
        sub_layers = self._extract_value(items[2]) if len(items) > 2 and items[2].data == 'layer_block' else []
        return {'type': layer_type, 'params': params, 'sublayers': sub_layers}

    def named_layer(self, items):
        return {'type': items[0].value, 'params': self._extract_value(items[1])}

    def self_defined_shape(self, items):
        layer_name = self._extract_value(items[0])
        custom_dims = self._extract_value(items[1])
        return {"type": "CustomShape", "layer": layer_name, "custom_dims": custom_dims}


    ## HPO ##

    def _parse_hpo(self, hpo_str, item):
        """Helper method to parse HPO expressions like 'log_range(1e-4, 1e-2)'."""
        if hpo_str.startswith('choice('):
            values = [int(v.strip()) for v in hpo_str[7:-1].split(',')]
            return {'hpo': {'type': 'categorical', 'values': values}}
        elif hpo_str.startswith('range('):
            parts = [float(v.strip()) for v in hpo_str[6:-1].split(',')]
            if len(parts) == 3:
                return {'hpo': {'type': 'range', 'start': parts[0], 'end': parts[1], 'step': parts[2]}}
        elif hpo_str.startswith('log_range('):
            parts = [float(v.strip()) for v in hpo_str[10:-1].split(',')]
            if len(parts) == 2:
                return {'hpo': {'type': 'log_range', 'low': parts[0], 'high': parts[1]}}
        self.raise_validation_error(f"Invalid HPO expression: {hpo_str}", item, Severity.ERROR)
        return {}

    def hpo_expr(self, items):
        return {"hpo": self._extract_value(items[0])}
    
    def hpo_with_params(self, items):
        return [self._extract_value(item) for item in items]

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

    def layer_choice(self, items):
        return {"hpo_type": "layer_choice", "options": [self._extract_value(item) for item in items]}

    ######

    def parse_network(self, config: str, framework: str = 'auto'):
        warnings = []
        try:
            parse_result = safe_parse(network_parser, config)
            tree = parse_result["result"]
            if tree is None:
                raise DSLValidationError("Parsing failed due to warnings", Severity.ERROR)
            warnings.extend(parse_result["warnings"])
            
            model = self.transform(tree)
            if framework == 'auto':
                framework = self._detect_framework(model)
            model['framework'] = framework
            model['shape_info'] = []
            model['warnings'] = warnings  # Ensure warnings are always included
            return model
        except (lark.LarkError, DSLValidationError, VisitError) as e:
            log_by_severity(Severity.ERROR, f"Error parsing network: {str(e)}")
            raise

    def _detect_framework(self, model):
        for layer in model['layers']:
            if 'torch' in layer.get('params', {}).values():
                return 'pytorch'
            if 'keras' in layer.get('params', {}).values():
                return 'tensorflow'
        return 'tensorflow'