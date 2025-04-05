import lark
import pysnooper
from lark import Tree, Transformer, Token
from typing import Any, Dict, List, Tuple, Union, Optional, Callable
import json
import plotly.graph_objects as go
import logging
from enum import Enum
from lark.visitors import VisitError
import re


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
    """Exception raised for validation errors in DSL parsing.

    This exception is used to report syntax, semantic, or other validation errors
    encountered during DSL (Domain Specific Language) parsing operations.

    Attributes:
        severity (Severity): The severity level of the validation error
        line (int, optional): The line number where the error occurred
        column (int, optional): The column number where the error occurred
        message (str): The raw error message

    Args:
        message (str): The error description message
        severity (Severity, optional): The severity level. Defaults to Severity.ERROR
        line (int, optional): The line number of the error. Defaults to None
        column (int, optional): The column number of the error. Defaults to None

    Example:
        >>> raise DSLValidationError("Invalid syntax", line=10, column=5)
        ERROR at line 10, column 5: Invalid syntax
    """
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
    if isinstance(error, KeyError):
        msg = "Unexpected end of input (KeyError). The parser did not expect '$END'."
        severity = Severity.ERROR
    elif isinstance(error, lark.UnexpectedCharacters):
        msg = f"Syntax error at line {error.line}, column {error.column}: Unexpected character '{error.char}'.\n" \
              f"Expected one of: {', '.join(sorted(error.allowed))}"
        severity = Severity.ERROR
    elif isinstance(error, lark.UnexpectedToken):
        # Check for end-of-input scenarios more robustly
        if str(error.token) in ['', '$END'] or 'RBRACE' in error.expected:
            msg = "Unexpected end of input - Check for missing closing braces"
            severity = Severity.ERROR
            log_by_severity(severity, msg)
            raise DSLValidationError(msg, severity, error.line, error.column)
        else:
            msg = f"Syntax error at line {error.line}, column {error.column}: Unexpected token '{error.token}'.\n" \
                  f"Expected one of: {', '.join(sorted(error.expected))}"
            severity = Severity.ERROR
    else:
        msg = str(error)
        severity = Severity.ERROR

    log_by_severity(severity, msg)
    if severity.value >= Severity.ERROR.value:
        raise DSLValidationError(msg, severity, error.line, error.column)
    return {"warning": msg, "line": error.line, "column": error.column}

def create_parser(start_rule: str = 'network') -> lark.Lark:
    """
    Create a Lark parser for the Neural DSL grammar with the specified start rule.

    This function initializes a Lark parser with the Neural DSL grammar and configures
    it to start parsing from the specified rule. The parser handles various layer types,
    network configurations, and hyperparameter optimization expressions.

    Args:
        start_rule (str, optional): The grammar rule to start parsing from.
                                    Defaults to 'network'.

    Returns:
        lark.Lark: A configured Lark parser instance ready to parse Neural DSL code.

    Example:
        >>> parser = create_parser('network')
        >>> tree = parser.parse('network MyModel { input: (28, 28, 1) layers: Dense(128) }')
    """
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
        SIMPLE_RNN_DROPOUT_WRAPPER: "simplernndropoutwrapper"i
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
        ACTIVATION: "activation"i
        ADD: "add"i
        SUBSTRACT: "subtract"i
        MULTIPLY: "multiply"i
        AVERAGE: "average"i
        MAXIMUM: "maximum"i
        CONCATENATE: "concatenate"i
        DOT: "dot"i
        TIMEDISTRIBUTED: "timedistributed"i
        RESIDUALCONNECTION: "residualconnection"i
        GLOBALAVERAGEPOOLING2D: "globalaveragepooling2d"i

        // Layer type tokens (case-insensitive)
        LAYER_TYPE.2: "dense"i | "conv2d"i | "conv1d"i | "conv3d"i | "dropout"i | "flatten"i | "lstm"i | "gru"i | "simplernndropoutwrapper"i | "simplernn"i | "output"i| "transformer"i | "transformerencoder"i | "transformerdecoder"i | "conv2dtranspose"i | "maxpooling2d"i | "maxpooling1d"i | "maxpooling3d"i | "batchnormalization"i | "gaussiannoise"i | "instancenormalization"i | "groupnormalization"i | "activation"i | "add"i | "subtract"i | "multiply"i | "average"i | "maximum"i | "concatenate"i | "dot"i | "timedistributed"i | "residualconnection"i | "globalaveragepooling2d"i

        // Basic tokens
        NAME: /[a-zA-Z_][a-zA-Z0-9_]*/
        STRING: /"[^"]*"/ | /'[^']*'/
        INT: /[+-]?[0-9]+/
        FLOAT: /[+-]?[0-9]*\.[0-9]+([eE][+-]?[0-9]+)?/ | /[+-]?[0-9]+[eE][+-]?[0-9]+/
        NUMBER: INT | FLOAT
        TRUE.2: "true"i
        FALSE.2: "false"i
        NONE.2: "none"i
        BOOL: TRUE | FALSE
        AT: "@"

        // Layer name patterns
        CUSTOM_LAYER: /[A-Z][a-zA-Z0-9]*Layer/  // Matches layer names ending with "Layer"
        MACRO_NAME: /^(?!.*Layer$)(?!ResidualConnection|Dot|Average|Maximum|Multiply|Add|Concatenate|substract|TimeDistributed|Activation|GroupNormalization|InstanceNormalization|LayerNormalization|GaussianNoise|TransformerEncoder|TransformerDecoder|BatchNormalization|Dropout|Flatten|Output|Conv2DTranspose|LSTM|GRU|SimpleRNN|LSTMCell|GRUCell|Dense|Conv1D|Conv2D|Conv3D|MaxPooling1D|MaxPooling2D|MaxPooling3D)[A-Z][a-zA-Z0-9]*/

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

        // Research Parsing
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
        named_units: "units" "=" value
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
        named_alpha: "alpha" "=" FLOAT
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
        ?named_param: ( learning_rate_param | momentum_param | named_layer | named_clipvalue | named_clipnorm | named_units | pool_size | named_kernel_size | named_size | named_activation | named_filters | named_strides | named_padding | named_dilation_rate | named_groups | named_data_format | named_channels | named_return_sequences | named_num_heads | named_ff_dim | named_input_dim | named_output_dim | named_rate | named_dropout | named_axis | named_epsilon | named_center | named_scale | named_beta_initializer | named_gamma_initializer | named_moving_mean_initializer | named_moving_variance_initializer | named_training | named_trainable | named_use_bias | named_kernel_initializer | named_bias_initializer | named_kernel_regularizer | named_bias_regularizer | named_activity_regularizer | named_kernel_constraint | named_bias_constraint | named_return_state | named_go_backwards | named_stateful | named_time_major | named_unroll | named_input_shape | named_batch_input_shape | named_dtype | named_name | named_weights | named_embeddings_initializer | named_mask_zero | named_input_length | named_embeddings_regularizer | named_embeddings_constraint | named_num_layers | named_bidirectional | named_merge_mode | named_recurrent_dropout | named_noise_shape | named_seed | named_target_shape | named_interpolation | named_crop_to_aspect_ratio | named_mask_value | named_return_attention_scores | named_causal | named_use_scale | named_key_dim | named_value_dim | named_output_shape | named_arguments | named_initializer | named_regularizer | named_constraint | named_l1 | named_l2 | named_l1_l2 | named_int | named_float | NAME "=" value | NAME "=" hpo_expr | named_alpha)


        network: "network" NAME "{" input_layer layers ";" [loss] [optimizer_param] [training_config] [execution_config] "}"
        input_layer: "input" ":" shape ("," shape)*
        layers: "layers" ":" layer_or_repeated+
        loss: "loss" ":" (NAME | STRING) ["(" param_style1 ")"]

        // Optimizer
        optimizer_param: "optimizer:" (named_optimizer | STRING)
        named_optimizer: NAME "(" [param_style1] ("," [param_style1])* ")"
        EXPONENTIALDECAY: "ExponentialDecay"
        learning_rate_param: "learning_rate=" (FLOAT | hpo_expr | NAME "(" [lr_schedule_args] ")")
        lr_schedule_args: param_style1 ("," param_style1)*
        momentum_param: "momentum=" param_style1
        search_method_param: "search_method:" STRING
        validation_split_param: "validation_split:" FLOAT


        layer_or_repeated: layer ["*" INT]
        ?layer: basic_layer | advanced_layer | special_layer
        config: training_config | execution_config


        shape: "(" [number_or_none ("," number_or_none)* [","]] ")"
        number_or_none: number | NONE


        lambda_: "Lambda" "(" STRING ")"
        wrapper: TIMEDISTRIBUTED "(" layer ["," param_style1] ")" [layer_block]

        dropout: "Dropout" "(" dropout_params ")"
        dropout_params: FLOAT | param_style1
        flatten: "Flatten" "(" [param_style1] ")"


        regularization: spatial_dropout1d | spatial_dropout2d | spatial_dropout3d | activity_regularization | l1 | l2 | l1_l2
        l1: "L1(" param_style1 ")"
        l2: "L2(" param_style1 ")"
        l1_l2: "L1L2(" param_style1 ")"

        output: OUTPUT "(" param_style1 ")"

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
        max_pooling1d: MAXPOOLING1D "(" param_style1 ")"
        max_pooling2d: MAXPOOLING2D "(" param_style1 ")"
        max_pooling3d: MAXPOOLING3D "(" param_style1 ")"
        pool_size: "pool_size" "=" value
        average_pooling: average_pooling1d | average_pooling2d | average_pooling3d
        average_pooling1d: "AveragePooling1D" "(" param_style1 ")"
        average_pooling2d: "AveragePooling2D" "(" param_style1 ")"
        average_pooling3d: "AveragePooling3D" "(" param_style1 ")"
        global_pooling: global_max_pooling | global_average_pooling
        global_max_pooling: global_max_pooling1d | global_max_pooling2d | global_max_pooling3d
        global_max_pooling1d: "GlobalMaxPooling1D" "(" param_style1 ")"
        global_max_pooling2d: "GlobalMaxPooling2D" "(" param_style1 ")"
        global_max_pooling3d: "GlobalMaxPooling3D" "(" param_style1 ")"
        global_average_pooling: global_average_pooling1d | global_average_pooling2d | global_average_pooling3d
        global_average_pooling1d: "GlobalAveragePooling1D" "(" param_style1 ")"
        global_average_pooling2d: "GlobalAveragePooling2D" "(" param_style1 ")"
        global_average_pooling3d: "GlobalAveragePooling3D" "(" param_style1 ")"
        adaptive_pooling: adaptive_max_pooling | adaptive_average_pooling
        adaptive_max_pooling: adaptive_max_pooling1d | adaptive_max_pooling2d | adaptive_max_pooling3d
        adaptive_max_pooling1d: "AdaptiveMaxPooling1D" "(" param_style1 ")"
        adaptive_max_pooling2d: "AdaptiveMaxPooling2D" "(" param_style1 ")"
        adaptive_max_pooling3d: "AdaptiveMaxPooling3D" "(" param_style1 ")"
        adaptive_average_pooling: adaptive_average_pooling1d | adaptive_average_pooling2d | adaptive_average_pooling3d
        adaptive_average_pooling1d: "AdaptiveAveragePooling1D" "(" param_style1 ")"
        adaptive_average_pooling2d: "AdaptiveAveragePooling2D" "(" param_style1 ")"
        adaptive_average_pooling3d: "AdaptiveAveragePooling3D" "(" param_style1 ")"

        ?norm_layer: batch_norm | layer_norm | instance_norm | group_norm
        batch_norm: BATCHNORMALIZATION "(" [param_style1] ")"
        layer_norm: LAYERNORMALIZATION "(" [param_style1] ")"
        instance_norm: INSTANCENORMALIZATION "(" [param_style1] ")"
        group_norm: GROUPNORMALIZATION "(" [param_style1] ")"

        conv_rnn: conv_lstm | conv_gru
        conv_lstm: "ConvLSTM2D(" param_style1 ")"
        conv_gru: "ConvGRU2D(" param_style1 ")"

        rnn_cell: simple_rnn_cell | lstm_cell | gru_cell
        simple_rnn_cell: "SimpleRNNCell" "(" param_style1 ")"
        lstm_cell: LSTMCELL "(" param_style1 ")"
        gru_cell: GRUCELL "(" param_style1 ")"

        dropout_wrapper_layer: simple_rnn_dropout | gru_dropout | lstm_dropout
        simple_rnn_dropout: SIMPLE_RNN_DROPOUT_WRAPPER "(" param_style1 ")"
        gru_dropout: "GRUDropoutWrapper" "(" param_style1 ")"
        lstm_dropout: "LSTMDropoutWrapper" "(" param_style1 ")"
        bidirectional_rnn_layer: bidirectional_simple_rnn_layer | bidirectional_lstm_layer | bidirectional_gru_layer
        bidirectional_simple_rnn_layer: "Bidirectional(SimpleRNN(" param_style1 "))"
        bidirectional_lstm_layer: "Bidirectional(LSTM(" param_style1 "))"
        bidirectional_gru_layer: "Bidirectional(GRU(" param_style1 "))"
        conv_rnn_layer: conv_lstm_layer | conv_gru_layer
        conv_lstm_layer: "ConvLSTM2D" "(" param_style1 ")"
        conv_gru_layer: "ConvGRU2D" "(" param_style1 ")"
        rnn_cell_layer: simple_rnn_cell_layer | lstm_cell_layer | gru_cell_layer
        simple_rnn_cell_layer: "SimpleRNNCell" "(" param_style1 ")"
        lstm_cell_layer: "LSTMCell" "(" param_style1 ")"
        gru_cell_layer: "GRUCell" "(" param_style1 ")"

        // Lambda layers
        merge: add | substract | multiply | average | maximum | concatenate | dot
        add: ADD "("  param_style1 ")"
        substract: SUBSTRACT "(" param_style1 ")"
        multiply: MULTIPLY "(" param_style1 ")"
        average: AVERAGE "(" param_style1 ")"
        maximum: MAXIMUM "(" param_style1 ")"
        concatenate: CONCATENATE "(" param_style1 ")"
        dot: DOT "(" param_style1 ")"

        spatial_dropout1d: "SpatialDropout1D" "(" param_style1 ")"
        spatial_dropout2d: "SpatialDropout2D" "(" param_style1 ")"
        spatial_dropout3d: "SpatialDropout3D" "(" param_style1 ")"
        activity_regularization: "ActivityRegularization" "(" param_style1 ")"


        // Training & Configurations
        training_config: "train" "{" training_params  "}"
        training_params: (epochs_param | batch_size_param | search_method_param | validation_split_param | device)*
        device: "@" NAME
        epochs_param: "epochs:" INT
        batch_size_param: "batch_size:" values_list
        values_list: "[" (value | hpo_expr) ("," (value | hpo_expr))* "]" | (value | hpo_expr) ("," (value | hpo_expr))*

        // Execution Configuration & Devices
        execution_config: "execute" "{" device_param "}"
        device_param: "device:" STRING

        // Customized Shapes
        CUSTOM_SHAPE: "CustomShape"
        self_defined_shape: CUSTOM_SHAPE "(" named_layer ")"

        math_expr: term (("+"|"-") term)*
        term: factor (("*"|"/") factor)*
        factor: NUMBER | NAME| "(" math_expr ")" | function_call
        function_call: NAME "(" math_expr ("," math_expr)* ")"

        hpo_with_params: hpo_expr "," named_params
        hpo: hpo_expr | layer_choice

        // HPO for Hyperparameters
        hpo_expr: "HPO" "(" (hpo_choice | hpo_range | hpo_log_range) ")"
        hpo_choice: "choice" "(" value ("," value)* ")"
        hpo_range: "range" "(" number "," number ("," "step=" number)? ")"
        hpo_log_range: "log_range" "(" number "," number ")"

        // HPO for layer choice
        layer_choice: "HPO" "(" "choice" "(" layer ("," layer)* "))"

        // MACROS AND RELATED RULES
        define: "define" NAME "{" ( layer_or_repeated)*  "}"
        macro_ref: MACRO_NAME "(" [param_style1] ")" [layer_block]

        basic_layer: layer_type "(" [param_style1] ")" [device_spec] [layer_block]
        layer_type: DENSE | CONV2D | CONV1D | CONV3D | DROPOUT | FLATTEN | LSTM | GRU | SIMPLE_RNN_DROPOUT_WRAPPER | SIMPLERNN | OUTPUT | TRANSFORMER | TRANSFORMER_ENCODER | TRANSFORMER_DECODER | CONV2DTRANSPOSE | LSTMCELL | GRUCELL | MAXPOOLING1D | MAXPOOLING2D | MAXPOOLING3D | BATCHNORMALIZATION | GAUSSIANNOISE | LAYERNORMALIZATION | INSTANCENORMALIZATION | GROUPNORMALIZATION | ACTIVATION | ADD | SUBSTRACT | MULTIPLY | AVERAGE | MAXIMUM | CONCATENATE | DOT | TIMEDISTRIBUTED | RESIDUALCONNECTION | GLOBALAVERAGEPOOLING2D | OUTPUT
        ?param_style1:  hpo_param | params 
        hpo_param: hpo_expr | hpo_with_params
        params: param ("," param)*
        ?param: named_param | value
        ?value: STRING -> string_value | number | tuple_ | BOOL
        tuple_: "(" number "," number ")"
        number: NUMBER
        named_params: named_param ("," named_param)*
        device_spec: AT STRING
        ?advanced_layer: (attention | transformer | residual | inception | capsule | squeeze_excitation | graph | embedding | quantum | dynamic)
        attention: "Attention" "(" [param_style1] ")" [layer_block]
        transformer: TRANSFORMER "(" [param_style1] ")" [layer_block]
                    | TRANSFORMER_ENCODER "(" [param_style1] ")" [layer_block]
                    | TRANSFORMER_DECODER "(" [param_style1] ")" [layer_block]
        residual: RESIDUALCONNECTION "(" [param_style1] ")" [layer_block]
        inception: "Inception" "(" [named_params] ")"
        capsule: "CapsuleLayer" "(" [named_params] ")"
        squeeze_excitation: "SqueezeExcitation" "(" [named_params] ")"
        graph: graph_conv | graph_attention
        graph_conv: "GraphConv" "(" [named_params] ")"
        graph_attention: "GraphAttention" "(" [named_params] ")"
        embedding: "Embedding" "(" [named_params] ")"
        quantum: "QuantumLayer" "(" [named_params] ")"
        dynamic: "DynamicLayer" "(" [named_params] ")"


        special_layer: custom | macro_ref | wrapper | lambda_ | self_defined_shape
        ?custom_or_macro: custom | macro_ref
        custom: CUSTOM_LAYER "(" [param_style1] ")" [layer_block]
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
    """
    Safely parse text using the provided parser, handling common parsing errors.

    This function attempts to parse the input text and catches any parsing exceptions,
    converting them to more user-friendly error messages with line and column information.

    Args:
        parser (lark.Lark): The Lark parser to use for parsing.
        text (str): The input text to parse.

    Returns:
        dict: A dictionary containing:
            - result: The parsed tree if successful, None otherwise.
            - warnings: A list of warning messages.

    Raises:
        DSLValidationError: If there are syntax errors or other parsing issues.

    Example:
        >>> parser = create_parser('network')
        >>> try:
        ...     parse_result = safe_parse(parser, 'network MyModel { input: (28, 28, 1) }')
        ...     tree = parse_result["result"]
        ... except DSLValidationError as e:
        ...     print(f"Error: {e}")
    """
    warnings = []
    # Extract the lexer from the parser
    lexer = parser.lexer
    
    # Tokenize the input and log the stream
    logger.debug("Token stream:")
    tokens = list(lexer.lex(text))
    for token in tokens:
        logger.debug(f"Token: {token.type}('{token.value}') at line {token.line}, column {token.column}")
    
    try:
        tree = parser.parse(text)
        logger.debug("Parse successful, tree generated.")
        return {"result": tree, "warnings": warnings}
    except (lark.UnexpectedCharacters, lark.UnexpectedToken) as e:
        log_by_severity(Severity.ERROR, f"Syntax error at line {e.line}, column {e.column}")
        error_msg = f"Syntax error at line {e.line}, column {e.column}"
        details = ""
        if isinstance(e, lark.UnexpectedToken):
            details = f"Unexpected {e.token}, expected one of: {', '.join(sorted(e.expected))}"
        elif isinstance(e, lark.UnexpectedCharacters):
            details = f"Unexpected character '{e.char}', expected one of: {', '.join(sorted(e.allowed))}"
        warnings.append({"message": error_msg, "line": e.line, "column": e.column, "details": details})
        custom_error_handler(e)
    except DSLValidationError as e:
        raise
    except Exception as e:
        log_by_severity(Severity.ERROR, f"Unexpected error while parsing: {str(e)}")
        raise DSLValidationError(f"Parser error: {str(e)}", Severity.ERROR)

network_parser = create_parser('network')
layer_parser = create_parser('layer')
research_parser = create_parser('research')

def split_params(s):
    parts = []
    current = []
    depth = 0
    for c in s:
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        if c == ',' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(c)
    if current:
        parts.append(''.join(current).strip())
    return parts

class ModelTransformer(lark.Transformer):
    """
    Transformer for converting parsed Neural DSL syntax trees into model configurations.

    This class transforms the Lark parse tree into a structured dictionary representation
    of the neural network model. It handles various layer types, network configurations,
    hyperparameter optimization expressions, and validation of parameters.

    The transformer processes each node in the parse tree according to its type and
    converts it into the appropriate Python data structure (dictionaries, lists, etc.)
    that can be used to construct the actual neural network model.
    """
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
            'SIMPLERNNDROPOUTWRAPPER': 'simple_rnn_dropout',
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
            'ACTIVATION': 'activation',
            'ADD': 'add',
            'SUBSTRACT': 'substract',
            'MULTIPLY': 'multiply',
            'AVERAGE': 'average',
            'MAXIMUM': 'maximum',
            'CONCATENATE': 'concatenate',
            'DOT': 'dot',
            'TIMEDISTRIBUTED': 'timedistributed',
            'RESIDUALCONNECTION': 'residual',
            'GLOBALAVERAGEPOOLING2D': 'global_average_pooling2d',
        }
        self.hpo_params = []

    def raise_validation_error(self, msg, item=None, severity=Severity.ERROR):
        """
        Raise a validation error with line and column information from the parse tree node.

        This method extracts position information from the parse tree node and raises
        a DSLValidationError with the provided message and severity level. It helps
        provide detailed error messages to users with exact line and column information.

        Args:
            msg (str): The error message to display to the user.
            item: The parse tree node where the error occurred.
            severity (Severity, optional): The severity level of the error.
                                          Defaults to Severity.ERROR.

        Raises:
            DSLValidationError: An exception containing the error message, severity,
                               and position information.

        Returns:
            dict: For warnings (non-error severity levels), returns a dictionary with
                 warning information including the message and position.
        """
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

        if layer_item is None:
            return None

        layer_def = self._extract_value(layer_item)
        if not isinstance(layer_def, dict):
            self.raise_validation_error(f"Invalid layer definition: {layer_def}", layer_item)

        return layer_def

    def special_layer(self, items):
        """Process special_layer rule by returning the first child (custom, macro_ref, etc.)."""
        return self._extract_value(items[0])

    @pysnooper.snoop()
    def define(self, items):
        """Process macro definition."""
        if len(items) < 1:
            self.raise_validation_error("Macro definition requires a name")

        macro_name = items[0].value
        layers = []

        for item in items[1:]:
            layer = self._extract_value(item)
            if isinstance(layer, tuple) and len(layer) == 2 and isinstance(layer[1], int):
                layer_def, count = layer
                layers.extend([layer_def] * count)
            else:
                layers.append(layer)

        if not layers:
            self.raise_validation_error(f"Empty macro definition for {macro_name}", items[0])

        # Store the macro definition with its layers
        self.macros[macro_name] = {
            'original': layers,
            'macro': {'type': macro_name, 'params': {}, 'sublayers': layers}
        }

        # Return the macro structure instead of the layers list
        return {'type': macro_name, 'params': {}, 'sublayers': layers}

    @pysnooper.snoop()
    def macro_ref(self, items):
        """Process a macro reference."""
        macro_name = items[0].value

        # Check if this is actually a custom layer
        if macro_name.endswith('Layer'):
            params = {}
            if len(items) > 1:
                param_values = self._extract_value(items[1])
                if isinstance(param_values, list):
                    for param in param_values:
                        if isinstance(param, dict):
                            params.update(param)
                elif isinstance(param_values, dict):
                    params = param_values

            return {
                'type': macro_name,
                'params': params,
                'sublayers': []
            }

        # Handle actual macros
        if macro_name not in self.macros:
            self.raise_validation_error(f"Undefined macro '{macro_name}'", items[0])

        params = self._extract_value(items[1]) if len(items) > 1 and items[1].data == 'param_style1' else {}
        sub_layers = self._extract_value(items[2]) if len(items) > 2 and items[2].data == 'layer_block' else []


        return {'type':macro_name, 'params':params, 'sublayers': sub_layers}

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
        layer_type_node = items[0]
        layer_type = layer_type_node.children[0].value.upper()
        params_node = items[1] if len(items) > 1 else None
        device_spec_node = items[2] if len(items) > 2 else None
        sublayers_node = items[3] if len(items) > 3 else None

        raw_params = self._extract_value(params_node) if params_node else None
        device = self._extract_value(device_spec_node) if device_spec_node else None
        sublayers = self._extract_value(sublayers_node) if sublayers_node else []

        method_name = self.layer_type_map.get(layer_type)
        if method_name and hasattr(self, method_name):
            try:
                # Pass raw_params as a single-item list to match method signature
                layer_info = getattr(self, method_name)([raw_params])
                # Ensure 'sublayers' and 'params' are always present
                layer_info['sublayers'] = sublayers if sublayers else layer_info.get('sublayers', [])
                layer_info['params'] = layer_info.get('params', {})  # Default to {} if None
                if device is not None:
                    layer_info['params']['device'] = device
                return layer_info
            except DSLValidationError as e:
                raise e
        else:
            self.raise_validation_error(f"Unsupported layer type: {layer_type}", layer_type_node)
            return {'type': layer_type, 'params': raw_params, 'sublayers': sublayers}

    def device_spec(self, items):
        """Process device specification correctly."""
        if len(items) > 1 and isinstance(items[1], Token) and items[1].type == "STRING":
            return items[1].value.strip('"')
        return self._extract_value(items[0])
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
                        merged_params['rate'] = elem  # Assign HPO to 'rate' without tracking here
                    else:
                        merged_params.update(elem)
                else:
                    merged_params['rate'] = elem
            param_style = merged_params

        if isinstance(param_style, dict):
            params = param_style.copy()
            if 'rate' in params:
                if not isinstance(params['rate'], float) and 'hpo' in params['rate']:
                    self._track_hpo('Dropout', 'rate', params['rate'], items[0])  # Single tracking point
                else:  # Validate only if not HPO
                    rate = params['rate']
                    if not isinstance(rate, (int, float)):
                        self.raise_validation_error(f"Dropout rate must be a number, got {rate}", items[0], Severity.ERROR)
                    elif not 0 <= rate <= 1:
                        self.raise_validation_error(f"Dropout rate should be between 0 and 1, got {rate}", items[0], Severity.ERROR)
            else:
                self.raise_validation_error("Dropout requires a 'rate' parameter", items[0], Severity.ERROR)
        elif isinstance(param_style, (int, float)):
            params['rate'] = param_style
            if not 0 <= params['rate'] <= 1:
                self.raise_validation_error(f"Dropout rate should be between 0 and 1, got {params['rate']}", items[0], Severity.ERROR)
        else:
            self.raise_validation_error("Invalid parameters for Dropout", items[0], Severity.ERROR)

        for param_name, param_value in params.items():
            if isinstance(param_value, dict) and 'hpo' in param_value:
                self._track_hpo('Dropout', param_name, param_value, items[0])

        return {'type': 'Dropout', 'params': params, 'sublayers': []}  # Added sublayers for consistency

    def output(self, items):
        params = {}
        if items and items[0] is not None:
            param_node = items[0]
            param_values = self._extract_value(param_node)

            ordered_params = []
            named_params = {}
            if isinstance(param_values, list):
                for val in param_values:
                    if isinstance(val, dict):
                        if 'hpo' in val and len(named_params) == 0 and len(ordered_params) == 0:
                            # First positional HPO expression becomes 'units'
                            named_params['units'] = val
                        else:
                            named_params.update(val)
                    else:
                        ordered_params.append(val)
            elif isinstance(param_values, dict):
                named_params = param_values

            # Map positional parameters to named
            if ordered_params:
                if len(ordered_params) >= 1:
                    params['units'] = ordered_params[0]
                if len(ordered_params) >= 2:
                    params['activation'] = ordered_params[1]
                if len(ordered_params) > 2:
                    self.raise_validation_error("Output layer accepts at most two positional arguments (units, activation)", items[0])

            params.update(named_params)

        if 'units' not in params:
            self.raise_validation_error("Output layer requires 'units' parameter", items[0])

        # Track HPO if present
        if 'units' in params and isinstance(params['units'], dict) and 'hpo' in params['units']:
            self._track_hpo('Output', 'units', params['units'], items[0])

        # Ensure sublayers is included
        return {'type': 'Output', 'params': params, 'sublayers': []}

    def regularization(self, items):
        return {'type': items[0].data.capitalize(), 'params': self._extract_value(items[0].children[0])}

    def execution_config(self, items):
        params = self._extract_value(items[0])
        return {'type': 'execution_config', 'params': params}

    @pysnooper.snoop()
    def dense(self, items):
        params = {}
        if items and items[0] is not None:
            param_node = items[0]  # From param_style1
            param_values = self._extract_value(param_node) if param_node else []

            ordered_params = []
            named_params = {}
            if isinstance(param_values, list):
                for val in param_values:
                    if isinstance(val, dict):
                        if 'hpo' in val:  # HPO expression
                            if 'units' not in named_params:  # Assign to units if not already set
                                named_params['units'] = val
                            else:
                                self.raise_validation_error("Multiple HPO expressions not supported as positional args", items[0])
                        else:
                            named_params.update(val)  # Named parameter
                    elif isinstance(val, list):
                        ordered_params.extend(val)  # List of positional parameters
                    else:
                        ordered_params.append(val)  # Positional parameter
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
        if isinstance(units, dict) and 'hpo' in units:
            self._track_hpo('Dense', 'units', units, items[0])
        elif isinstance(units, list):
            # Handle list case (e.g., [16, 32, 64]) as categorical if intended for HPO and Return True if all items are true
            if len(units) > 1 and all(isinstance(u, (int, float)) for u in units):
                hpo_config = {
                    'hpo': {
                        'type': 'categorical',
                        'values': units,
                        'original_values': [str(u) for u in units]
                    }
                }
                params['units'] = self._extract_value(hpo_config)
                self._track_hpo('Dense', 'units', hpo_config, items[0])
        else:
            if not isinstance(units, (int, float)):
                self.raise_validation_error(f"Dense units must be a number, got {units}", items[0])
            if units <= 0:
                self.raise_validation_error(f"Dense units must be positive, got {units}", items[0])
            params['units'] = int(units)  # Convert to int if applicable

        if 'activation' in params:
            activation = params['activation']
            if isinstance(activation, dict) and 'hpo' in activation:
                self._track_hpo('Dense', 'activation', activation, items[0])
            elif isinstance(activation, dict) and 'hpo' not in activation:
                params['activation'] = self._extract_value(activation)
            elif not isinstance(activation, str):
                self.raise_validation_error(f"Dense activation must be a string or HPO, got {activation}", items[0])
            else:
                valid_activations = {
                    'relu', 'sigmoid', 'tanh', 'softmax', 'softplus',
                    'softsign', 'selu', 'elu', 'exponential', 'linear'
                }
                if activation.lower() not in valid_activations:
                    self.raise_validation_error(
                        f"Invalid activation function {activation}. Allowed: {', '.join(valid_activations)}",
                        items[0]
                    )

        return {"type": "Dense", "params": params, 'sublayers': []}

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

        for param_name, param_value in params.items():
            if isinstance(param_value, dict) and 'hpo' in param_value:
                self._track_hpo('Conv2D', param_name, param_value, items[0])

        return {'type': 'Conv2D', 'params': params}  # 'sublayers' added by basic_layer

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

    def named_optimizer(self, items):
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        logger.debug(f"named_optimizer called with items: {items}")

        opt_node = items[0]
        opt_value = self._extract_value(opt_node)
        logger.debug(f"opt_value: {opt_value}")

        params = {}
        if len(items) > 1:
            # If items[1] is a list of dictionaries, merge them into a single dictionary
            param_items = self._extract_value(items[1]) or {}
            logger.debug(f"param_items: {param_items}")

            if isinstance(param_items, list):
                for param_dict in param_items:
                    if isinstance(param_dict, dict):
                        params.update(param_dict)
            else:
                params = param_items

            logger.debug(f"merged params: {params}")

        if isinstance(opt_value, str):
            stripped_value = opt_value.strip("'\"")
            logger.debug(f"stripped_value: {stripped_value}")

            if '(' in stripped_value and ')' in stripped_value:
                # Extract optimizer name before '('
                opt_type = stripped_value[:stripped_value.index('(')].strip()
                logger.debug(f"opt_type: {opt_type}")

                # Extract parameters inside parentheses
                param_str = stripped_value[stripped_value.index('(')+1:stripped_value.rindex(')')].strip()
                logger.debug(f"param_str: {param_str}")

                if param_str and not params:  # Only parse if params not provided separately
                    for param in split_params(param_str):
                        param = param.strip()
                        logger.debug(f"processing param: {param}")

                        if '=' in param:
                            param_name, param_value = param.split('=', 1)
                            param_name = param_name.strip()
                            param_value = param_value.strip()
                            logger.debug(f"param_name: {param_name}, param_value: {param_value}")

                            if 'HPO(' in param_value:
                                hpo_start = param_value.index('HPO(')
                                hpo_end = param_value.rindex(')') + 1
                                hpo_str = param_value[hpo_start+4:hpo_end-1]
                                logger.debug(f"hpo_str: {hpo_str}")

                                hpo_config = self._parse_hpo(hpo_str, opt_node)
                                logger.debug(f"hpo_config: {hpo_config}")

                                params[param_name] = hpo_config
                                self._track_hpo('optimizer', param_name, hpo_config, opt_node)
                            else:
                                try:
                                    params[param_name] = float(param_value)
                                except ValueError:
                                    params[param_name] = param_value
            else:
                opt_type = stripped_value  # No parentheses, just the name
                logger.debug(f"opt_type (no parentheses): {opt_type}")
        else:
            logger.debug(f"opt_value is not a string: {type(opt_value)}")
            self.raise_validation_error("Optimizer must be a string", opt_node)

        if not opt_type:
            logger.debug("opt_type is empty")
            self.raise_validation_error("Optimizer type must be specified", opt_node, Severity.ERROR)

        # Validate optimizer name against common PyTorch/TensorFlow optimizers (case-insensitive)
        valid_optimizers = {'Adam', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam'}

        # Find the correct case version of the optimizer
        opt_type_normalized = None
        for valid_opt in valid_optimizers:
            if valid_opt.lower() == opt_type.lower():
                opt_type_normalized = valid_opt
                break

        if not opt_type_normalized:
            logger.debug(f"Invalid optimizer: {opt_type}")
            self.raise_validation_error(
                f"Invalid optimizer '{opt_type}'. Supported optimizers: {', '.join(valid_optimizers)}",
                opt_node,
                Severity.ERROR
            )

        result = {'type': opt_type_normalized, 'params': params}
        logger.debug(f"named_optimizer returning: {result}")
        return result

    def schedule(self, items):
        return {"type": items[0].value, "args": [self._extract_value(x) for x in items[1].children]}

    ## Training And Configurations ##

    def training_config(self, items):
        params = self._extract_value(items[0]) if items else {}
        return {'type': 'training_config', 'params': params}

    def execution_config(self, items):
        """Process execution_config block."""
        params = self._extract_value(items[0]) if items else {'device': 'auto'}
        return params  # Return flat dict directly, no 'type' or 'params' wrappe

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

    def device_spec(self, items):
        """Process device specification correctly."""
        if len(items) > 1 and isinstance(items[1], Token) and items[1].type == "STRING":
            device = items[1].value.strip('"')
        else:
            device = self._extract_value(items[0])

        allowed_devices = ["cpu", "cuda", "tpu", "auto", "cuda:0", "cuda:1"]  # Example allowed devices
        if device.lower() not in [d.lower() for d in allowed_devices]:
            self.raise_validation_error(f"Invalid device '{device}'. Allowed devices are: {allowed_devices}", items[0], Severity.ERROR)
        return device

    def validation_split_param(self, items):
        return {'validation_split': self._extract_value(items[0])}

    def epochs_param(self, items):
        return {'epochs': self._extract_value(items[0])}

    def batch_size_param(self, items):
        value = self._extract_value(items[0])
        if isinstance(value, dict) and 'hpo' in value:
            # Track HPO for batch_size
            self._track_hpo('training_config', 'batch_size', value, items[0])
            return {'batch_size': value}
        elif isinstance(value, list):
            # Handle list case (e.g., [16, 32, 64]) as categorical if intended for HPO
            if len(value) > 1 and all(isinstance(v, (int, float)) for v in value):
                hpo_config = {
                    'hpo': {
                        'type': 'categorical',
                        'values': value,
                        'original_values': [str(v) for v in value]
                    }
                }
                self._track_hpo('training_config', 'batch_size', hpo_config, items[0])
                return {'batch_size': hpo_config}
            return {'batch_size': value[0] if len(value) == 1 else value}
        elif isinstance(value, (int, float)):
            if value <= 0:
                self.raise_validation_error(f"batch_size must be positive, got {value}", items[0])
            return {'batch_size': int(value)}  # Ensure integer
        else:
            self.raise_validation_error(f"Invalid batch_size value: {value}", items[0])

    ###############


    def values_list(self, items):
        values = [self._extract_value(x) for x in items]
        return values[0] if len(values) == 1 else values

    def optimizer_param(self, items):
        return {'optimizer': self._extract_value(items[0])}

    def learning_rate_param(self, items):
        value = self._extract_value(items[0])

        # Handle direct learning rate schedule
        if isinstance(value, dict) and 'type' in value and 'args' in value:
            # This is already a learning rate schedule object from the lr_schedule rule
            pass
        elif isinstance(value, dict) and 'hpo' in value:
            # Track HPO for learning_rate
            self._track_hpo('optimizer', 'learning_rate', value, items[0])
        elif isinstance(value, (int, float)) and value <= 0:
            self.raise_validation_error(f"learning_rate must be positive, got {value}", items[0])
        # Handle string-based learning rate schedule for backward compatibility
        elif isinstance(value, str):
            # Special handling for ExponentialDecay
            if value == 'ExponentialDecay':
                # Create a learning rate schedule for ExponentialDecay
                args = []
                if len(items) > 1:
                    # Try to extract the arguments directly
                    try:
                        # If the next item is a token with parentheses, extract the arguments
                        if hasattr(items[1], 'value') and '(' in items[1].value and ')' in items[1].value:
                            args_str = items[1].value.strip('()')
                            # Split by comma and convert to appropriate types
                            for arg in args_str.split(','):
                                arg = arg.strip()
                                # Try to convert to float if possible
                                try:
                                    args.append(float(arg))
                                except ValueError:
                                    # If not a float, keep as string
                                    args.append(arg)
                        else:
                            # Otherwise use the standard extraction
                            args = self._extract_value(items[1])
                            if not isinstance(args, list):
                                args = [args]
                    except Exception as e:
                        # Fallback to standard extraction
                        args = self._extract_value(items[1])
                        if not isinstance(args, list):
                            args = [args]
                value = {
                    'type': 'ExponentialDecay',
                    'args': args
                }
            # Handle other string-based learning rate schedules
            elif '(' in value and ')' in value:
                schedule_str = value.strip('"\'')
                schedule_type = schedule_str[:schedule_str.index('(')]
                args_str = schedule_str[schedule_str.index('(')+1:schedule_str.rindex(')')]

                # Parse arguments
                args = []
                if args_str:
                    # Split by comma and convert to appropriate types

                        for char in args_str:
                            if char == ',' and paren_level == 0:
                                result.append(current.strip())
                                current = ''
                            else:
                                if char == '(':
                                    paren_level += 1
                                elif char == ')':
                                    paren_level -= 1
                                current += char

                        if current:
                            result.append(current.strip())
                        return result

                arg_list = parse_args(args_str)

                for arg in arg_list:
                    if arg.startswith('HPO('):
                        # Extract HPO parameters
                        hpo_match = re.search(r'HPO\(([^(]+)\(', arg)
                        if hpo_match:
                            hpo_type = hpo_match.group(1).strip()
                            if hpo_type == 'range':
                                params = re.search(r'range\(([^,]+),\s*([^,]+)(?:,\s*step=([^)]+))?\)', arg)
                                if params:
                                    low, high = float(params.group(1)), float(params.group(2))
                                    step = float(params.group(3)) if params.group(3) else None
                                    hpo_dict = {'type': 'range', 'low': low, 'high': high}
                                    if step:
                                        hpo_dict['step'] = step
                                    args.append({'hpo': hpo_dict})
                            elif hpo_type == 'log_range':
                                params = re.search(r'log_range\(([^,]+),\s*([^)]+)\)', arg)
                                if params:
                                    low, high = float(params.group(1)), float(params.group(2))
                                    args.append({'hpo': {'type': 'log_range', 'low': low, 'high': high}})
                            elif hpo_type == 'choice':
                                params = re.search(r'choice\(([^)]+)\)', arg)
                                if params:
                                    choices = [float(x.strip()) for x in params.group(1).split(',')]
                                    args.append({'hpo': {'type': 'choice', 'values': choices}})
                    else:
                        try:
                            args.append(float(arg))
                        except ValueError:
                            args.append(arg)

            value = {
                'type': schedule_type,
                'args': args
            }

        return {'learning_rate': value}

    def momentum_param(self, items):
        value = self._extract_value(items[0])
        if isinstance(value, dict) and 'hpo' in value:
            # Track HPO for momentum
            self._track_hpo('optimizer', 'momentum', value, items[0])
        elif isinstance(value, (int, float)) and not 0 <= value <= 1:
            self.raise_validation_error(f"momentum should be between 0 and 1, got {value}", items[0])
        return {'momentum': value}

    def shape(self, items):
        return tuple(self._extract_value(item) for item in items)

    ## Pooling Layers ##


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

        for param_name, param_value in params.items():
            if isinstance(param_value, dict) and 'hpo' in param_value:
                self._track_hpo('MaxPooling2D', param_name, param_value, items[0])

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
        params = self._extract_value(items[0]) if items else {}
        return {'type': 'GlobalAveragePooling2D', 'params': params or {}, 'sublayers': []}

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
        raw_params = self._extract_value(items[0]) if items and items[0] is not None else None
        params = None  # Default to None for empty parameters

        if raw_params:
            params = {}  # Initialize params dict only if we have parameters
            if isinstance(raw_params, list):
                for item in raw_params:
                    if isinstance(item, dict):
                        params.update(item)
            elif isinstance(raw_params, dict):
                params = raw_params

            # Validate parameters if they exist
            if params:
                if 'axis' in params:
                    axis = params['axis']
                    if not isinstance(axis, int):
                        self.raise_validation_error(
                            f"BatchNormalization axis must be an integer, got {axis}",
                            items[0]
                        )
                if 'momentum' in params:
                    momentum = params['momentum']
                    if not isinstance(momentum, (int, float)) or not 0 <= momentum <= 1:
                        self.raise_validation_error(
                            f"BatchNormalization momentum must be between 0 and 1, got {momentum}",
                            items[0]
                        )

        return {'type': 'BatchNormalization', 'params': params}  # 'sublayers' added by basic_layer

    def named_momentum(self, items):
        return {'momentum': self._extract_value(items[0])}

    def layer_norm(self, items):
        params = self._extract_value(items[0]) if items else None
        return {'type': 'LayerNormalization', 'params': params}

    def instance_norm(self, items):
        params = self._extract_value(items[0]) if items else None
        return {'type': 'InstanceNormalization', 'params': params}


    def group_norm(self, items):
        raw_params = self._extract_value(items[0]) if items else None
        params = {}
        if isinstance(raw_params, list):
            item = raw_params[0] if raw_params else None
            if isinstance(item, dict):
                params.update(item)
            else:
                self.raise_validation_error("Invalid parameters for GroupNormalization", items[0])
        return {'type': 'GroupNormalization', 'params': params}

    ############


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

    def lr_schedule(self, items):
        """
        Process learning rate schedule expressions.

        This method processes learning rate schedule expressions like ExponentialDecay,
        StepDecay, etc. and extracts their parameters.

        Args:
            items: The parse tree items for the learning rate schedule.

        Returns:
            dict: A dictionary containing the schedule type and arguments.
        """
        schedule_type = items[0].value
        args = []
        if len(items) > 1 and hasattr(items[1], 'data') and items[1].data == 'lr_schedule_args':
            args = [self._extract_value(arg) for arg in items[1].children]
        elif len(items) > 1:
            args = self._extract_value(items[1])
            if not isinstance(args, list):
                args = [args]
        return {
            'type': schedule_type,
            'args': args
        }

    def exponentialdecay(self, items):
        """
        Process an ExponentialDecay learning rate schedule.

        This method processes an ExponentialDecay learning rate schedule and extracts its parameters.

        Args:
            items: The parse tree items for the ExponentialDecay schedule.

        Returns:
            dict: A dictionary containing the schedule type and arguments.
        """
        args = []
        if len(items) > 0:
            args = self._extract_value(items[0])
            if not isinstance(args, list):
                args = [args]
        return {
            'type': 'ExponentialDecay',
            'args': args
        }

    def lr_schedule_args(self, items):
        """
        Process learning rate schedule arguments.

        This method processes the arguments for a learning rate schedule.

        Args:
            items: The parse tree items for the learning rate schedule arguments.

        Returns:
            The original items list to be processed by the parent lr_schedule method.
        """
        return items

    def lr_schedule_arg(self, items):
        """
        Process a learning rate schedule argument.

        This method extracts the value from a learning rate schedule argument.

        Args:
            items: The parse tree items for the learning rate schedule argument.

        Returns:
            The extracted value from the learning rate schedule argument.
        """
        return self._extract_value(items[0])

    def conv_lstm(self, items):
        """
        Process a ConvLSTM2D layer.

        This method processes a ConvLSTM2D layer and extracts its parameters.

        Args:
            items: The parse tree items for the ConvLSTM2D layer.

        Returns:
            dict: A dictionary containing the ConvLSTM2D layer configuration.
        """
        return {'type': 'ConvLSTM2D', 'params': self._extract_value(items[0])}

    def conv_gru(self, items):
        """
        Process a ConvGRU2D layer.

        This method processes a ConvGRU2D layer and extracts its parameters.

        Args:
            items: The parse tree items for the ConvGRU2D layer.

        Returns:
            dict: A dictionary containing the ConvGRU2D layer configuration.
        """
        return {'type': 'ConvGRU2D', 'params': self._extract_value(items[0])}

    def bidirectional_rnn(self, items):
        """
        Process a bidirectional RNN layer.

        This method processes a bidirectional RNN layer (like Bidirectional(LSTM),
        Bidirectional(GRU), etc.) and combines the RNN layer parameters with the
        bidirectional wrapper parameters.

        Args:
            items: The parse tree items for the bidirectional RNN layer.

        Returns:
            dict: A dictionary containing the bidirectional RNN layer configuration.
        """
        rnn_layer = items[0]
        bidirectional_params = self._extract_value(items[1])
        rnn_layer['params'].update(bidirectional_params)
        return {'type': f"Bidirectional({rnn_layer['type']})", 'params': rnn_layer['params']}

    def cudnn_gru_layer(self, items):
        """
        Process a CuDNN GRU layer.

        This method processes a CuDNN GRU layer, which is a GPU-optimized version of GRU.
        It maps the CuDNN GRU to the standard GRU layer type with the same parameters.

        Args:
            items: The parse tree items for the CuDNN GRU layer.

        Returns:
            dict: A dictionary containing the GRU layer configuration.
        """
        return {'type': 'GRU', 'params': self._extract_value(items[0])}

    def bidirectional_simple_rnn_layer(self, items):
        """
        Process a bidirectional SimpleRNN layer.

        This method processes a bidirectional SimpleRNN layer and extracts its parameters.

        Args:
            items: The parse tree items for the bidirectional SimpleRNN layer.

        Returns:
            dict: A dictionary containing the bidirectional SimpleRNN layer configuration.
        """
        return {'type': 'Bidirectional(SimpleRNN)', 'params': self._extract_value(items[0])}

    def bidirectional_lstm_layer(self, items):
        """
        Process a bidirectional LSTM layer.

        This method processes a bidirectional LSTM layer and extracts its parameters.

        Args:
            items: The parse tree items for the bidirectional LSTM layer.

        Returns:
            dict: A dictionary containing the bidirectional LSTM layer configuration.
        """
        return {'type': 'Bidirectional(LSTM)', 'params': self._extract_value(items[0])}

    def bidirectional_gru_layer(self, items):
        """
        Process a bidirectional GRU layer.

        This method processes a bidirectional GRU layer and extracts its parameters.

        Args:
            items: The parse tree items for the bidirectional GRU layer.

        Returns:
            dict: A dictionary containing the bidirectional GRU layer configuration.
        """
        return {'type': 'Bidirectional(GRU)', 'params': self._extract_value(items[0])}

    def conv_lstm_layer(self, items):
        """
        Process a ConvLSTM2D layer.

        This method processes a ConvLSTM2D layer and extracts its parameters.
        ConvLSTM2D combines convolutional and LSTM operations for spatiotemporal data.

        Args:
            items: The parse tree items for the ConvLSTM2D layer.

        Returns:
            dict: A dictionary containing the ConvLSTM2D layer configuration.
        """
        return {'type': 'ConvLSTM2D', 'params': self._extract_value(items[0])}

    def conv_gru_layer(self, items):
        """
        Process a ConvGRU2D layer.

        This method processes a ConvGRU2D layer and extracts its parameters.
        ConvGRU2D combines convolutional and GRU operations for spatiotemporal data.

        Args:
            items: The parse tree items for the ConvGRU2D layer.

        Returns:
            dict: A dictionary containing the ConvGRU2D layer configuration.
        """
        return {'type': 'ConvGRU2D', 'params': self._extract_value(items[0])}

    ## Cell Layers ##

    def rnn_cell_layer(self, items):
        """
        Process an RNNCell layer.

        This method processes an RNNCell layer and extracts its parameters.
        RNNCell is a basic RNN cell that can be used in custom RNN architectures.

        Args:
            items: The parse tree items for the RNNCell layer.

        Returns:
            dict: A dictionary containing the RNNCell layer configuration.
        """
        return {'type': 'RNNCell', 'params': self._extract_value(items[0])}

    def simple_rnn_cell(self, items):
        """
        Process a SimpleRNNCell layer.

        This method processes a SimpleRNNCell layer and extracts its parameters.
        SimpleRNNCell is a cell with a single output recurrent unit.

        Args:
            items: The parse tree items for the SimpleRNNCell layer.

        Returns:
            dict: A dictionary containing the SimpleRNNCell layer configuration.
        """
        return {'type': 'SimpleRNNCell', 'params': self._extract_value(items[0])}

    def lstmcell(self, items):
        """
        Process an LSTMCell layer.

        This method processes an LSTMCell layer and extracts its parameters.
        LSTMCell is a cell with LSTM gates that can be used in custom RNN architectures.
        It validates that the units parameter is present.

        Args:
            items: The parse tree items for the LSTMCell layer.

        Returns:
            dict: A dictionary containing the LSTMCell layer configuration.

        Raises:
            DSLValidationError: If the units parameter is missing or invalid.
        """
        raw_params = self._extract_value(items[0])
        if isinstance(raw_params, list):
            params = {}
            for item in raw_params:
                if isinstance(item, dict):
                    params.update(item)
                else:
                    self.raise_validation_error("Invalid parameters for LSTMCell", items[0])
        else:
            params = raw_params

        if 'units' not in params:
            self.raise_validation_error("LSTMCell requires 'units' parameter", items)
        return {'type': 'LSTMCell', 'params': params}

    def grucell(self, items):
        """
        Process a GRUCell layer.

        This method processes a GRUCell layer and extracts its parameters.
        GRUCell is a cell with GRU gates that can be used in custom RNN architectures.
        It validates that the units parameter is present.

        Args:
            items: The parse tree items for the GRUCell layer.

        Returns:
            dict: A dictionary containing the GRUCell layer configuration.

        Raises:
            DSLValidationError: If the units parameter is missing.
        """
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
        """
        Process a SimpleRNNDropoutWrapper layer.

        This method processes a SimpleRNNDropoutWrapper layer and extracts its parameters.
        SimpleRNNDropoutWrapper is a wrapper that adds dropout to a SimpleRNN layer.
        It validates that the units parameter is present.

        Args:
            items: The parse tree items for the SimpleRNNDropoutWrapper layer.

        Returns:
            dict: A dictionary containing the SimpleRNNDropoutWrapper layer configuration.

        Raises:
            DSLValidationError: If the units parameter is missing.
        """
        param_style = self._extract_value(items[0])
        params = {}

        # Handle both positional and named parameters
        if isinstance(param_style, list):
            for param in param_style:
                if isinstance(param, dict):
                    params.update(param)
                else:
                    # Handle positional 'units' parameter
                    if 'units' not in params:
                        params['units'] = param
        elif isinstance(param_style, dict):
            params.update(param_style)

        # Validate required 'units' parameter
        if 'units' not in params:
            self.raise_validation_error("SimpleRNNDropoutWrapper requires 'units' parameter", items[0])

        return {"type": "SimpleRNNDropoutWrapper", 'params': params}

    def gru_dropout(self, items):
        """
        Process a GRUDropoutWrapper layer.

        This method processes a GRUDropoutWrapper layer and extracts its parameters.
        GRUDropoutWrapper is a wrapper that adds dropout to a GRU layer.

        Args:
            items: The parse tree items for the GRUDropoutWrapper layer.

        Returns:
            dict: A dictionary containing the GRUDropoutWrapper layer configuration.
        """
        return {"type": "GRUDropoutWrapper", 'params': self._extract_value(items[0])}

    def lstm_dropout(self, items):
        """
        Process an LSTMDropoutWrapper layer.

        This method processes an LSTMDropoutWrapper layer and extracts its parameters.
        LSTMDropoutWrapper is a wrapper that adds dropout to an LSTM layer.

        Args:
            items: The parse tree items for the LSTMDropoutWrapper layer.

        Returns:
            dict: A dictionary containing the LSTMDropoutWrapper layer configuration.
        """
        return {"type": "LSTMDropoutWrapper", 'params': self._extract_value(items[0])}

    def research(self, items):
        """
        Process a Research block.

        This method processes a Research block in the DSL, which is used to define
        research-specific configurations. It extracts the name and parameters of the
        research block.

        Args:
            items: The parse tree items for the Research block.

        Returns:
            dict: A dictionary containing the Research block configuration, including
                 its type, name (if provided), and parameters.
        """
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
        """
        Process parameters for a Research block.

        This method processes the parameters for a Research block, extracting and
        combining parameters from different sources (Trees or dictionaries).

        Args:
            items: The parse tree items containing the Research parameters.

        Returns:
            dict: A dictionary containing the combined parameters for the Research block.
        """
        params = {}
        for item in items:
            if isinstance(item, Tree):
                params.update(self._extract_value(item))
            elif isinstance(item, dict):
                params.update(item)
        return params

    def metrics(self, items):
        """
        Process metrics specifications.

        This method processes metrics specifications in the DSL, which define the metrics
        to be used for model evaluation. It extracts metric names and their parameters,
        handling both dictionary-based and string-based specifications.

        Args:
            items: The parse tree items containing the metrics specifications.

        Returns:
            dict: A dictionary containing the metrics configuration, with metric names as keys
                 and their parameters as values.
        """
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
        """
        Process an accuracy metric parameter.

        This method processes an accuracy metric parameter in the DSL, extracting
        its value and returning it in a dictionary format.

        Args:
            items: The parse tree items containing the accuracy parameter.

        Returns:
            dict: A dictionary with 'accuracy' as the key and the extracted parameter value.
        """
        return {'accuracy': self._extract_value(items[0])}

    def precision_param(self, items):
        """
        Process a precision metric parameter.

        This method processes a precision metric parameter in the DSL, extracting
        its value and returning it in a dictionary format.

        Args:
            items: The parse tree items containing the precision parameter.

        Returns:
            dict: A dictionary with 'precision' as the key and the extracted parameter value.
        """
        return {'precision': self._extract_value(items[0])}

    def recall_param(self, items):
        """
        Process a recall metric parameter.

        This method processes a recall metric parameter in the DSL, extracting
        its value and returning it in a dictionary format.

        Args:
            items: The parse tree items containing the recall parameter.

        Returns:
            dict: A dictionary with 'recall' as the key and the extracted parameter value.
        """
        return {'recall': self._extract_value(items[0])}

    def paper_param(self, items):
        """
        Process a paper reference parameter.

        This method processes a paper reference parameter in the DSL, extracting
        the reference string value.

        Args:
            items: The parse tree items containing the paper reference.

        Returns:
            str: The extracted paper reference string.
        """
        # Extract the string value from the 'paper:' parameter
        return self._extract_value(items[0])

    def references(self, items):
        """
        Process paper references.

        This method processes paper references in the DSL, extracting and collecting
        all the paper references into a list.

        Args:
            items: The parse tree items containing the paper references.

        Returns:
            dict: A dictionary with 'references' as the key and a list of paper references as the value.
        """
        papers = [self._extract_value(item) for item in items if item is not None]
        return {'references': papers}
    def metrics_loss_param(self, items):
        """
        Process a loss metric parameter.

        This method processes a loss metric parameter in the DSL, extracting
        its value and returning it in a dictionary format.

        Args:
            items: The parse tree items containing the loss parameter.

        Returns:
            dict: A dictionary with 'loss' as the key and the extracted parameter value.
        """
        return {'loss': self._extract_value(items[0])}

    ## Network ##


    def network(self, items):
        """
        Process a network rule and build a complete network configuration.

        This method processes the network rule from the parse tree and constructs a
        structured dictionary representation of the neural network. It handles input shapes,
        layers, loss functions, optimizers, training configurations, and execution settings.

        Args:
            items: The parse tree items for the network rule.

        Returns:
            dict: A structured dictionary containing the complete network configuration,
                  including name, input shapes, layers, loss, optimizer, training, and
                  execution settings.
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        # logger.debug(f"Processing network with items: {items}")

        name = items[0].value
        input = self._extract_value(items[1])
        layers = self._extract_value(items[2])

        # Initialize configs
        loss_config = None
        optimizer_config = None
        training_config = {}
        execution_config = None

        # Process remaining items (loss, optimizer, training_config, execution_config)
        for i, item in enumerate(items[3:], 3):
            value = self._extract_value(item)
            # logger.debug(f"Processing item {i}: {item}, value: {value}")

            if isinstance(item, Tree):
                if item.data == 'loss':
                    loss_config = value
                elif item.data == 'optimizer_param':
                    optimizer_config = value.get('optimizer')
                    # logger.debug(f"Found optimizer_param: {optimizer_config}")
                elif item.data == 'training_config':
                    training_config = value.get('params', value) if isinstance(value, dict) else value
                elif item.data == 'execution_config':
                    execution_config = value
                else:
                    # logger.warning(f"Unrecognized tree data: {item.data}")
                    pass
            elif isinstance(value, str) and i == 3:  # Check if this is the loss value
                loss_config = value  # Directly assign string value for loss
            elif isinstance(item, dict):
                if 'optimizer' in item:
                    optimizer_config = item['optimizer']
                    # logger.debug(f"Found optimizer in dict: {optimizer_config}")
                elif 'type' in item and item.get('type') in {'Adam', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam'}:
                    optimizer_config = item
                    # logger.debug(f"Found optimizer by type: {optimizer_config}")
                elif 'type' in item and item.get('type') == 'training_config':
                    training_config = item.get('params', {})
            else:
                logger.warning(f"Skipping unhandled item at index {i}: {item}")

        # Build the network configuration
        network_config = {
            'name': name,
            'input': input,
            'layers': layers
        }

        if loss_config:
            network_config['loss'] = loss_config

        if optimizer_config:
            network_config['optimizer'] = optimizer_config
            # logger.debug(f"Adding optimizer to network_config: {optimizer_config}")

        if training_config:
            network_config['training'] = training_config

        if execution_config:
            network_config['execution'] = execution_config

        # logger.debug(f"Final network_config: {network_config}")
        return network_config

    #########

    def search_method_param(self, items):
        value = self._extract_value(items[0])  # Extract "bayesian" from STRING token
        return {'search_method': value}

    def _extract_value(self, item):
        """
        Extract a Python value from a parse tree node.

        This method recursively processes parse tree nodes and converts them into
        appropriate Python data structures (strings, numbers, lists, dictionaries, etc.).
        It handles various node types including tokens, trees, and nested structures.

        Args:
            item: A parse tree node (Token, Tree, list, dict, or primitive value).

        Returns:
            The extracted Python value corresponding to the parse tree node.
            - For tokens, returns the token value.
            - For trees, processes according to the tree data type.
            - For lists, processes each element recursively.
            - For dictionaries, processes each value recursively.
            - For other types, returns the value as is.
        """
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

    def activation(self, items):
        """Process activation layer with or without parameters."""
        params = {}
        function_name = None

        # Extract parameters from items[0] (param_style1 result)
        raw_params = self._extract_value(items[0]) if items and items[0] is not None else None

        if raw_params is None:
            self.raise_validation_error("Activation layer requires a function name", items[0])

        # Process raw_params
        if isinstance(raw_params, list):
            ordered_params = [p for p in raw_params if not isinstance(p, dict)]
            named_params = {}
            for param in raw_params:
                if isinstance(param, dict):
                    named_params.update(param)
            if ordered_params:
                function_name = ordered_params[0]  # First positional is function
                if len(ordered_params) > 1:
                    if function_name == 'leaky_relu':
                        params['alpha'] = ordered_params[1]
                    else:
                        self.raise_validation_error(
                            f"Unexpected extra positional parameter {ordered_params[1]} for activation {function_name}",
                            items[0]
                        )
            params.update(named_params)
        elif isinstance(raw_params, dict):
            function_name = raw_params.get('function')
            if not function_name:
                self.raise_validation_error("Activation layer requires 'function' parameter", items[0])
            params = raw_params.copy()
        else:
            function_name = raw_params  # Single string as function name

        if not function_name:
            self.raise_validation_error("Activation layer requires a function name", items[0])

        params['function'] = function_name

        return {'type': 'Activation', 'params': params, 'sublayers': []}

    def named_alpha(self, items):
        return self._extract_value(items[0])

    def attention(self, items):
        params = self._extract_value(items[0]) if items else None
        return {'type': 'Attention', 'params': params, 'sublayers': []}


    def residual(self, items):
        raw_params = items[0] if items else None
        sub_layers = items[1] if len(items) > 1 else []

        params = {}
        if raw_params:
            if isinstance(raw_params, list):
                for p in raw_params:
                    if isinstance(p, dict):
                        params.update(p)
            elif isinstance(raw_params, dict):
                params.update(raw_params)

        return {'type': 'ResidualConnection', 'params': params, 'sublayers': sub_layers}

    def inception(self, items):
        params = self._extract_value(items[0]) if items else {}
        return {'type': 'Inception', 'params': params, 'sublayers': []}

    def graph(self, items):
        return items[0]

    def graph_attention(self, items):
        params = self._extract_value(items[0])
        return {'type': 'GraphAttention', 'params': params, 'sublayers': []}  # Fixed sublayers

    def dynamic(self, items):
        params = self._extract_value(items[0]) if items else None
        return {'type': 'DynamicLayer', 'params': params}

    def noise_layer(self, items):
        return {'type': items[0].data.capitalize(), 'params': self._extract_value(items[0].children[0])}

    def normalization_layer(self, items):
        return {'type': items[0].data.capitalize(), 'params': self._extract_value(items[0].children[0])}

    def regularization_layer(self, items):
        return {'type': items[0].data.capitalize(), 'params': self._extract_value(items[0].children[0])}

    def custom(self, items):
        """Process a custom layer definition."""
        if isinstance(items[0], Token) and items[0].type == 'CUSTOM_LAYER':
            layer_type = items[0].value
        else:
            layer_type = items[0]

        params = {}
        if len(items) > 1:
            param_values = self._extract_value(items[1])
            if isinstance(param_values, list):
                for param in param_values:
                    if isinstance(param, dict):
                        params.update(param)
            elif isinstance(param_values, dict):
                params = param_values

        return {
            'type': layer_type,
            'params': params,
            'sublayers': []
        }

    def capsule(self, items):
        params = self._extract_value(items[0]) if items else None
        return {'type': 'CapsuleLayer', 'params': params}

    def squeeze_excitation(self, items):
        params = self._extract_value(items[0]) if items else {}
        return {'type': 'SqueezeExcitation', 'params': params, 'sublayers': []}

    def quantum(self, items):
        params = self._extract_value(items[0]) if items else None
        return {'type': 'QuantumLayer', 'params': params}


    ## Transformers - Encoders - Decoders ##


    def transformer(self, items):
        if isinstance(items[0], Token):
            transformer_type = items[0].value
        else:
            self.raise_validation_error("Invalid transformer syntax: missing type identifier", items[0])
        params = {}
        sub_layers = []
        param_idx = 1

        if len(items) > param_idx:
            raw_params = self._extract_value(items[param_idx])
            if isinstance(raw_params, list):
                for param in raw_params:
                    if isinstance(param, dict):
                        params.update(param)
            elif isinstance(raw_params, dict):
                params = raw_params
            param_idx += 1

        if len(items) > param_idx:
            sub_layers = self._extract_value(items[param_idx])

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
        """
        Process an Embedding layer and validate its parameters.

        This method extracts and validates parameters for an Embedding layer,
        ensuring that input_dim and output_dim are positive integers.

        Args:
            items: The parse tree items for the Embedding layer.

        Returns:
            dict: A dictionary containing the Embedding layer configuration.

        Raises:
            DSLValidationError: If input_dim or output_dim are not positive integers.
        """
        params = self._extract_value(items[0]) if items else {}
        if params is not None:  # Handle the case where params might be None
            for key in ['input_dim', 'output_dim']:
                if key in params:
                    dim = params[key]
                    if not isinstance(dim, int) or dim <= 0:
                        self.raise_validation_error(f"Embedding {key} must be a positive integer, got {dim}", items[0])
        return {'type': 'Embedding', 'params': params, 'sublayers': []}

    ### Lambda Layers ###

    def merge(self, items):
        return self._extract_value(items[0])

    def lambda_(self, items):
        return {'type': 'Lambda', 'params': {'function': self._extract_value(items[0])}, 'sublayers': []}

    def add(self, items):
        params = self._extract_value(items[0]) if items else None
        return {'type': 'Add', 'params': params, 'sublayers': []}

    def substract(self, items):
        params = self._extract_value(items[0]) if items else None
        return {'type': 'Subtract', 'params': params, 'sublayers': []}

    def multiply(self, items):
        params = self._extract_value(items[0]) if items else None
        return {'type': 'Multiply', 'params': params, 'sublayers': []}

    def average(self, items):
        params = self._extract_value(items[0]) if items else None
        return {'type': 'Average', 'params': params, 'sublayers': []}

    def maximum(self, items):
        params = self._extract_value(items[0]) if items else None
        return {'type': 'Maximum', 'params': params, 'sublayers': []}

    @pysnooper.snoop()
    def concatenate(self, items):
        raw_params = self._extract_value(items[0]) if items and items[0] is not None else None
        params = {}

        if raw_params:
            if isinstance(raw_params, list):
                for param in raw_params:
                    if isinstance(param, dict):
                        params.update(param)
                    elif isinstance(param, list) and param:  # Handle nested list like [[1]]
                        if isinstance(param[0], int):  # Assume single int is axis
                            params['axis'] = param[0]
                        elif isinstance(param[0], dict):  # Handle [{'axis': 1}]
                            params.update(param[0])
                    elif isinstance(param, int):  # Handle positional axis
                        params['axis'] = param
            elif isinstance(raw_params, dict):
                params = raw_params
            elif isinstance(raw_params, int):  # Single integer as axis
                params['axis'] = raw_params

        # Validation
        if 'axis' in params:
            axis = params['axis']
            if not isinstance(axis, int):
                self.raise_validation_error(
                    f"Concatenate axis must be an integer, got {axis}",
                    items[0] if items else None
                )

        return {'type': 'Concatenate', 'params': params, 'sublayers': []}

    def dot(self, items):
        params = self._extract_value(items[0]) if items else None
        return {'type': 'Dot', 'params': params, 'sublayers': []}

    ## Wrappers ##


    def timedistributed(self, items):
        raw_params = self._extract_value(items[0]) if items else None
        if isinstance(raw_params, list):
            params = {}
            for item in raw_params:
                if isinstance(item, dict):
                    params.update(item)
                else:
                    self.raise_validation_error("Invalid parameters for TimeDistributed", items[0])
        else:
            params = raw_params
        return {'type': 'TimeDistributed', 'params': params}


    def wrapper(self, items):
        wrapper_type = items[0]  # e.g., "TimeDistributed"
        inner_layer = self._extract_value(items[1])  # The wrapped layer
        params = inner_layer.get('params', {})  # Start with inner layer's params
        sub_layers = []

        # Handle additional parameters or sublayers
        for i in range(2, len(items)):
            item = items[i]
            if isinstance(item, Tree) and item.data == 'layer_block':
                sub_layers = self._extract_value(item)
            elif isinstance(item, list):  # Named params from param_style1
                for sub_param in item:
                    if isinstance(sub_param, dict):
                        sub_layers = [sub_param]
            elif isinstance(item, dict):
                params.update(item)

        return {
            'type': f"{wrapper_type}({inner_layer['type']})",
            'params': params,
            'sublayers': sub_layers
        }

    ## Statistical Noises ##


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


    def named_layer(self, items):
        layer_name = self._extract_value(items[0])
        dimensions = self._extract_value(items[1])
        return {
            'type': layer_name,
            'params': dimensions
        }

    def self_defined_shape(self, items):
        layer_info = self._extract_value(items[1])  # items[1] is the named_layer result

        #Error handling for negative dims
        if any(dim < 0 for dim in layer_info['params']):
            self.raise_validation_error(f"Invalid dimensions for {layer_info['type']}: {layer_info['params']}", items[1])



        return {
            "type": "CustomShape",
            "layer": layer_info['type'],
            "custom_dims": layer_info['params']
        }

    ## HPO ##

    def _track_hpo(self, layer_type, param_name, hpo_data, node):
        """
        Track hyperparameter optimization (HPO) parameters found during parsing.

        This method records HPO parameters with their layer type, parameter name, and path
        for later use in the hyperparameter optimization system. It maintains a list of all
        HPO parameters found in the network configuration.

        Args:
            layer_type (str): The type of layer or component containing the HPO parameter.
            param_name (str): The name of the parameter using HPO.
            hpo_data (dict): The HPO configuration data.
            node: The parse tree node where the HPO parameter was found.
        """
        path = f"{layer_type}.{param_name}" if param_name else layer_type
        self.hpo_params.append({
            'layer_type': layer_type,
            'param_name': param_name,
            'path': path,
            'hpo': hpo_data['hpo'],
            'node': node  # Optional: for debugging
        })

    def parse_network_with_hpo(self, config):
        """
        Parse a Neural DSL network configuration with hyperparameter optimization (HPO) support.

        This method parses the Neural DSL configuration and tracks all HPO parameters
        found in the network, including those in layers, optimizers, and training settings.
        It creates a structured representation of the network that includes HPO search spaces.

        Args:
            config (str): The Neural DSL configuration text to parse.

        Returns:
            dict: A dictionary containing the parsed network configuration with HPO parameters
                  tracked and properly formatted for the HPO system.

        Raises:
            DSLValidationError: If there are validation errors in the configuration.
            Exception: For other parsing or processing errors.
        """
        try:
            tree = create_parser('network').parse(config)
            model = self.transform(tree)

            # Track HPO parameters in the optimizer
            if 'optimizer' in model and 'params' in model['optimizer']:
                params = model['optimizer']['params']

                # Track HPO parameters in the optimizer
                for param_name, param_value in params.items():
                    if isinstance(param_value, dict) and 'hpo' in param_value:
                        self._track_hpo('optimizer', f'params.{param_name}', param_value, None)
                    # Track HPO parameters in learning rate schedules
                    elif param_name == 'learning_rate' and isinstance(param_value, dict) and 'type' in param_value and 'args' in param_value:
                        # This is a learning rate schedule
                        for i, arg in enumerate(param_value['args']):
                            if isinstance(arg, dict) and 'hpo' in arg:
                                # Track HPO in learning rate schedule
                                self._track_hpo('optimizer', f'params.learning_rate.args[{i}]', arg, None)

                # Process learning rate schedule string
                if 'learning_rate' in params and isinstance(params['learning_rate'], str):
                    lr_value = params['learning_rate']
                    if '(' in lr_value and ')' in lr_value:
                        # Process learning rate schedule string
                        schedule_str = lr_value.strip('"\'')
                        schedule_type = schedule_str[:schedule_str.index('(')]
                        args_str = schedule_str[schedule_str.index('(')+1:schedule_str.rindex(')')]

                        # Parse arguments
                        args = []
                        if args_str:
                            import re
                            # Handle nested parentheses for HPO expressions
                            def parse_args(args_str):
                                result = []
                                current = ''
                                paren_level = 0

                                for char in args_str:
                                    if char == ',' and paren_level == 0:
                                        result.append(current.strip())
                                        current = ''
                                    else:
                                        if char == '(':
                                            paren_level += 1
                                        elif char == ')':
                                            paren_level -= 1
                                        current += char

                                if current:
                                    result.append(current.strip())
                                return result

                            arg_list = parse_args(args_str)

                            for arg in arg_list:
                                if arg.startswith('HPO('):
                                    # Extract HPO parameters
                                    hpo_type = re.search(r'HPO\(([^(]+)\(', arg)
                                    if hpo_type:
                                        hpo_type = hpo_type.group(1).strip()
                                        if hpo_type == 'range':
                                            match = re.search(r'range\(([^,]+),\s*([^,]+)(?:,\s*step=([^)]+))?\)', arg)
                                            if match:
                                                low, high = float(match.group(1)), float(match.group(2))
                                                step = float(match.group(3)) if match.group(3) else None
                                                hpo_dict = {'type': 'range', 'low': low, 'high': high}
                                                if step:
                                                    hpo_dict['step'] = step
                                                args.append({'hpo': hpo_dict})
                                                # Track HPO in learning rate schedule
                                                self._track_hpo('optimizer', f'params.learning_rate.args[{len(args)-1}]', {'hpo': hpo_dict}, None)
                                        elif hpo_type == 'log_range':
                                            match = re.search(r'log_range\(([^,]+),\s*([^)]+)\)', arg)
                                            if match:
                                                low, high = float(match.group(1)), float(match.group(2))
                                                hpo_dict = {'type': 'log_range', 'low': low, 'high': high}
                                                args.append({'hpo': hpo_dict})
                                                # Track HPO in learning rate schedule
                                                self._track_hpo('optimizer', f'params.learning_rate.args[{len(args)-1}]', {'hpo': hpo_dict}, None)
                                        elif hpo_type == 'choice':
                                            match = re.search(r'choice\(([^)]+)\)', arg)
                                            if match:
                                                choices = [float(x.strip()) for x in match.group(1).split(',')]
                                                hpo_dict = {'type': 'choice', 'values': choices}
                                                args.append({'hpo': hpo_dict})
                                                # Track HPO in learning rate schedule
                                                self._track_hpo('optimizer', f'params.learning_rate.args[{len(args)-1}]', {'hpo': hpo_dict}, None)
                                else:
                                    try:
                                        args.append(float(arg))
                                    except ValueError:
                                        args.append(arg)

                        # Replace string with structured representation
                        params['learning_rate'] = {
                            'type': schedule_type,
                            'args': args
                        }

            return model, self.hpo_params
        except VisitError as e:
            # Check if the original exception is a DSLValidationError
            if hasattr(e, 'orig_exc') and isinstance(e.orig_exc, DSLValidationError):
                log_by_severity(Severity.ERROR, f"Error parsing network: {str(e.orig_exc)}")
                # Re-raise the VisitError with DSLValidationError as the cause
                raise e from e.orig_exc
            else:
                log_by_severity(Severity.ERROR, f"Unexpected transformation error: {str(e)}")
                raise
        except DSLValidationError as e:
            # Handle direct DSLValidationError (unlikely here, but for completeness)
            log_by_severity(Severity.ERROR, f"Direct parsing error: {str(e)}")
            raise VisitError("Direct DSL validation error") from e
        except Exception as e:
            log_by_severity(Severity.ERROR, f"Unexpected error: {str(e)}")
            raise

    def _parse_hpo_value(self, value):
        """Parse a single HPO value while retaining original string if numeric."""
        try:
            return float(value) if '.' in value or 'e' in value.lower() else int(value)
        except ValueError:
            return value  # Return as string if not a number

    def _parse_hpo(self, hpo_str, item):
        """
        Parse hyperparameter optimization (HPO) expressions and retain original string values.

        This method parses HPO expressions like choice(), range(), and log_range() into
        structured representations that can be used by the HPO system. It preserves the
        original string values to maintain exact numeric formats.

        Args:
            hpo_str (str): The HPO expression string to parse.
            item: The parse tree node where the HPO expression was found.

        Returns:
            dict: A structured dictionary representing the HPO parameter, including its
                  type (categorical, range, log_range) and values.

        Raises:
            DSLValidationError: If the HPO expression is invalid or contains errors.
        """
        original_str = hpo_str  # Store the original string
        if hpo_str.startswith('choice('):
            values = [v.strip() for v in hpo_str[7:-1].split(',')]  # Keep as strings
            return {
                'hpo': {
                    'type': 'categorical',
                    'values': [self._parse_hpo_value(v) for v in values],
                    'original_values': values  # Store original strings
                }
            }
        elif hpo_str.startswith('range('):
            parts = [v.strip() for v in hpo_str[6:-1].split(',')]
            parsed = [self._parse_hpo_value(p) for p in parts]
            
            # Handle step parameter if present
            if len(parts) >= 3:
                step_part = parts[2]
                if step_part.startswith('step='):
                    step_value = step_part[5:]  # Extract value after 'step='
                    parsed[2] = self._parse_hpo_value(step_value)
            
            hpo_data = {'type': 'range', 'start': parsed[0], 'end': parsed[1]}
            if len(parsed) >= 3:
                hpo_data['step'] = parsed[2]
            hpo_data['original_parts'] = parts  # Store original strings
            return {'hpo': hpo_data}
        elif hpo_str.startswith('log_range('):
            parts = [v.strip() for v in hpo_str[10:-1].split(',')]
            parsed = [self._parse_hpo_value(p) for p in parts]
            return {
                'hpo': {
                    'type': 'log_range',
                    'start': parsed[0],
                    'end': parsed[1],
                    'original_low': parts[0],  # Store original strings
                    'original_high': parts[1]
                }
            }
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
        """
        Parse a Neural DSL network configuration into a structured model representation.

        This method parses the Neural DSL configuration text and converts it into a
        structured dictionary representation that can be used to construct a neural network
        model in the specified framework.

        Args:
            config (str): The Neural DSL configuration text to parse.
            framework (str, optional): The target framework for the model ('tensorflow',
                                      'pytorch', or 'auto'). Defaults to 'auto'.

        Returns:
            dict: A structured dictionary representation of the neural network model,
                  including layers, optimizer, training configuration, and other settings.

        Raises:
            DSLValidationError: If there are validation errors in the configuration.
            Exception: For other parsing or processing errors.
        """
        warnings = []
        try:
            # The safe_parse function will raise DSLValidationError if parsing fails
            try:
                parse_result = safe_parse(network_parser, config)
                tree = parse_result["result"]
                warnings.extend(parse_result.get("warnings", []))
            except Exception as e:
                log_by_severity(Severity.ERROR, f"Error during parsing: {str(e)}")
                raise DSLValidationError(f"Failed to parse network: {str(e)}", Severity.ERROR)

            model = self.transform(tree)
            if framework == 'auto':
                framework = self._detect_framework(model)
            model['framework'] = framework
            model['shape_info'] = []
            model['warnings'] = warnings  # Ensure warnings are always included
            return model
        except VisitError as e:
            if hasattr(e, 'orig_exc') and isinstance(e.orig_exc, DSLValidationError):
                log_by_severity(Severity.ERROR, f"Error parsing network: {str(e.orig_exc)}")
                raise e.orig_exc from e
            else:
                log_by_severity(Severity.ERROR, f"Error parsing network: {str(e)}")
                raise
        except (lark.LarkError, DSLValidationError) as e:
            log_by_severity(Severity.ERROR, f"Error parsing network: {str(e)}")
            raise

    def _detect_framework(self, model):
        """
        Detect the appropriate framework for a model based on its layer parameters.

        This method examines the model's layers to determine whether it should use
        PyTorch or TensorFlow as the backend framework. It looks for framework-specific
        keywords in the layer parameters.

        Args:
            model (dict): The parsed model configuration.

        Returns:
            str: The detected framework ('pytorch' or 'tensorflow'). Defaults to
                 'tensorflow' if no framework-specific indicators are found.
        """
        for layer in model['layers']:
            params = layer.get('params') or {}  # Handle None case
            if 'torch' in params.values():
                return 'pytorch'
            if 'keras' in params.values():
                return 'tensorflow'
        return 'tensorflow'
