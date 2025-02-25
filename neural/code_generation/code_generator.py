from neural.shape_propagation.shape_propagator import ShapePropagator
from neural.parser.parser import ModelTransformer, create_parser
from typing import Any, Dict, Union
import torch
import onnx
from onnx import helper
import numpy as np

def to_number(x: str) -> Union[int, float]:
    try:
        return int(x)
    except ValueError:
        return float(x)

def generate_code(model_data: Dict[str, Any], backend: str) -> str:
    if not isinstance(model_data, dict) or 'layers' not in model_data:
        raise ValueError("Invalid model_data format")

    indent = "    "
    propagator = ShapePropagator(debug=False)
    
    # Layer expansion with proper multiplication handling
    expanded_layers = []
    requires_functional_api = False
    for layer in model_data['layers']:
        # Extract multiplication factor from parser metadata
        multiply = layer.pop('multiply', 1)  # Parser should add 'multiply' key
        
        # Check for architecture requirements
        if layer['type'] in ['TransformerEncoder', 'TransformerDecoder']:
            requires_functional_api = True
        
        # Expand multiplied layers
        expanded_layers.extend([layer.copy() for _ in range(multiply)])
    
    model_data['layers'] = expanded_layers

    if backend == "tensorflow":
        code = "import tensorflow as tf\nfrom tensorflow.keras import layers\n\n"
        
        # Generate TransformerEncoder class only if needed
        if any(l['type'] == 'TransformerEncoder' for l in expanded_layers):
            code += """class TransformerEncoder(layers.Layer):
    def __init__(self, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=ff_dim//num_heads  # Proper dimension calculation
        )
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(ff_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
    
    def call(self, inputs):
        attn_output = self.attn(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)\n\n"""

        # Functional API generation
        code += "# Model construction using Functional API\n"
        input_shape = [None if dim == 'NONE' else dim for dim in model_data['input']['shape']]
        code += f"inputs = tf.keras.Input(shape={tuple(input_shape)})\n"
        code += "x = inputs\n"
        
        # Generate layer calls
        for i, layer in enumerate(expanded_layers):
            layer_type = layer['type']
            params = layer.get('params', {})
            
            if layer_type == 'TransformerEncoder':
                code += (
                    f"x = TransformerEncoder(num_heads={params['num_heads']}, "
                    f"ff_dim={params['ff_dim']}, dropout={params['dropout']})(x)\n"
                )
            elif layer_type == 'Dense':
                activation = params.get('activation', 'linear')  # Default to linear
                code += f"x = layers.Dense({params['units']}, activation='{activation}')(x)\n"
            elif layer_type == 'Output':
                activation = params.get('activation', 'softmax')  # Default for output
                code += f"outputs = layers.Dense({params['units']}, activation='{activation}')(x)\n"
        code += "\nmodel = tf.keras.Model(inputs=inputs, outputs=outputs)\n"
        
        # Compilation
        optimizer_config = model_data['optimizer']
        opt_params = []
        for k, v in optimizer_config.get('params', {}).items():
            if isinstance(v, str):
                opt_params.append(f"{k}='{v}'")
            else:
                opt_params.append(f"{k}={v}")
        
        code += (
            f"\nmodel.compile(\n"
            f"    loss='{model_data['loss']}',\n"
            f"    optimizer=tf.keras.optimizers.{optimizer_config['type']}({', '.join(opt_params)}),\n"
            f"    metrics=['accuracy']\n"
            f")\n"
        )
        
        # Training configuration
        if 'training_config' in model_data:
            tc = model_data['training_config']
            code += (
                f"\nmodel.fit(\n"
                f"    x_train, y_train,\n"
                f"    epochs={tc['epochs']},\n"
                f"    batch_size={tc['batch_size']},\n"
                f"    validation_split={tc['validation_split']},\n"
                f"    verbose=1\n"
                f")\n"
            )
        
        return code
    elif backend == "pytorch":
        code = "import torch\n"
        code += "import torch.nn as nn\n"
        code += "import torch.optim as optim\n\n"
        code += "class NeuralNetworkModel(nn.Module):\n"
        code += f"{indent}def __init__(self):\n"
        code += f"{indent}{indent}super(NeuralNetworkModel, self).__init__()\n"

        input_shape = model_data['input']['shape']
        if not input_shape:
            raise ValueError("Input layer shape is not defined.")
        current_input_shape = input_shape

        layers_code = []
        forward_code_body = []

        for i, layer_config in enumerate(model_data['layers']):
            layer_type = layer_config['type']
            params = layer_config.get('params', {})
            layer_name = f"self.layer{i}"

            if "sub_layers" in layer_config:
                if layer_type == "TimeDistributed":
                    sub_layer = layer_config['sub_layers'][0]
                    sub_type = sub_layer['type']
                    sub_params = sub_layer.get('params', {})
                    if sub_type == "Conv2D":
                        layers_code.append(f"{layer_name}_conv = nn.Conv2d(in_channels={current_input_shape[-1]}, out_channels={sub_params['filters']}, kernel_size={sub_params['kernel_size']})")
                        forward_code_body.append(f"x = self.layer{i}_conv(x)")
                elif layer_type == "Residual":
                    layers_code.append(f"{layer_name}_residual = nn.Sequential(")
                    for sub_layer in layer_config['sub_layers']:
                        if sub_layer['type'] == 'Conv2D':
                            layers_code.append(f"{indent}{indent}nn.Conv2d(in_channels={current_input_shape[-1]}, out_channels={sub_layer['params']['filters']}, kernel_size={sub_layer['params']['kernel_size']}),")
                    layers_code.append(f"{indent})")
                    forward_code_body.append(f"x = x + self.layer{i}_residual(x)")
                current_input_shape = propagator.propagate(current_input_shape, layer_config, framework='pytorch')
                continue

            if layer_type == 'Conv2D':
                filters = params.get('filters')
                kernel_size = params.get('kernel_size')
                activation = params.get('activation', 'relu')
                if not filters or not kernel_size:
                    raise ValueError("Conv2D layer config missing 'filters' or 'kernel_size'.")
                layers_code.append(f"{layer_name}_conv = nn.Conv2d(in_channels={current_input_shape[-1]}, out_channels={filters}, kernel_size={kernel_size})")
                layers_code.append(f"{layer_name}_act = nn.ReLU()" if activation == 'relu' else f"{layer_name}_act = nn.Identity()")
                forward_code_body.append(f"x = self.layer{i}_conv(x)")
                forward_code_body.append(f"x = self.layer{i}_act(x)")
            elif layer_type == 'MaxPooling2D':
                pool_size = params.get('pool_size')
                if not pool_size:
                    raise ValueError("MaxPooling2D layer config missing 'pool_size'.")
                layers_code.append(f"{layer_name}_pool = nn.MaxPool2d(kernel_size={pool_size})")
                forward_code_body.append(f"x = self.layer{i}_pool(x)")
            elif layer_type == 'Flatten':
                layers_code.append(f"{layer_name}_flatten = nn.Flatten()")
                forward_code_body.append(f"x = self.layer{i}_flatten(x)")
            elif layer_type == 'Dense' or layer_type == 'Output':
                units = params.get('units')
                activation = params.get('activation', 'relu' if layer_type == 'Dense' else 'linear')
                if not units:
                    raise ValueError(f"{layer_type} layer config missing 'units'.")
                in_features = np.prod(current_input_shape[1:]) if len(current_input_shape) > 2 else current_input_shape[-1]
                layers_code.append(f"{layer_name}_dense = nn.Linear(in_features={in_features}, out_features={units})")
                act_layer = "nn.ReLU()" if activation == 'relu' else "nn.Softmax(dim=1)" if activation == 'softmax' else "nn.Identity()"
                layers_code.append(f"{layer_name}_act = {act_layer}")
                forward_code_body.append(f"x = self.layer{i}_dense(x)")
                forward_code_body.append(f"x = self.layer{i}_act(x)")
            elif layer_type == 'Dropout':
                rate = params.get('rate')
                if rate is None:
                    raise ValueError("Dropout layer config missing 'rate'.")
                layers_code.append(f"{layer_name}_dropout = nn.Dropout(p={rate})")
                forward_code_body.append(f"x = self.layer{i}_dropout(x)")
            elif layer_type == 'BatchNormalization':
                layers_code.append(f"{layer_name}_bn = nn.BatchNorm2d(num_features={current_input_shape[-1]})")
                forward_code_body.append(f"x = self.layer{i}_bn(x)")
            elif layer_type in ['Transformer', 'TransformerEncoder']:
                num_heads = params.get('num_heads', 8)
                ff_dim = params.get('ff_dim', 512)
                dropout = params.get('dropout', 0.1)
                
                layers_code.append(
                    f"{layer_name} = nn.TransformerEncoderLayer(\n"
                    f"{indent}{indent}d_model={current_input_shape[-1]}, nhead={num_heads},\n"
                    f"{indent}{indent}dim_feedforward={ff_dim}, dropout={dropout}\n"
                    f"{indent})"
                )
                forward_code_body.append(f"x = self.layer{i}(x)")
                current_input_shape = propagator.propagate(current_input_shape, layer_config, 'pytorch')
            else:
                print(f"Warning: Unsupported layer type '{layer_type}' for PyTorch. Skipping.")
                continue

            current_input_shape = propagator.propagate(current_input_shape, layer_config, framework='pytorch')

        for line in layers_code:
            code += f"{indent}{indent}{line}\n"
        code += f"\n{indent}def forward(self, x):\n"
        for line in forward_code_body:
            code += f"{indent}{indent}{line}\n"
        code += f"{indent}{indent}return x\n\n"
        code += "model = NeuralNetworkModel()\n"
        code += "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
        code += "model.to(device)\n\n"

        loss_value = model_data['loss'] if isinstance(model_data['loss'], str) else model_data['loss']['value'].strip('"')
        optimizer_value = model_data['optimizer']['type'] if isinstance(model_data['optimizer'], dict) else model_data['optimizer'].strip('"')
        loss_fn = "nn.CrossEntropyLoss()" if "crossentropy" in loss_value.lower() else "nn.MSELoss()"
        code += f"loss_fn = {loss_fn}\n"
        code += f"optimizer = optim.{optimizer_value}(model.parameters(), lr=0.001)\n"
        return code

    elif backend == "onnx":
        return export_onnx(model_data, "model.onnx")

    else:
        raise ValueError(f"Unsupported backend: {backend}. Choose 'tensorflow', 'pytorch', or 'onnx'.")

def save_file(filename: str, content: str) -> None:
    """Save content to a file."""
    try:
        with open(filename, 'w') as f:
            f.write(content)
    except Exception as e:
        raise IOError(f"Error writing file: {filename}. {e}") from e
    print(f"Successfully saved file: {filename}")

def load_file(filename: str) -> Any:
    """Load and parse a neural config file."""
    with open(filename, 'r') as f:
        content = f.read()
    if filename.endswith('.neural') or filename.endswith('.nr'):
        return create_parser('network').parse(content)
    elif filename.endswith('.rnr'):
        return create_parser('research').parse(content)
    else:
        raise ValueError(f"Unsupported file type: {filename}")

def generate_onnx(model_data: Dict[str, Any]) -> onnx.ModelProto:
    """Generate ONNX model from model_data."""
    output_shape = model_data.get('output_shape', [None, 10])  # Default output shape if not provided
    graph = helper.make_graph(
        nodes=[],
        name="NeuralModel",
        inputs=[helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, model_data['input']['shape'])],
        outputs=[helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, output_shape)],
    )
    model = helper.make_model(graph, producer_name="Neural")
    propagator = ShapePropagator()
    current_input_shape = model_data['input']['shape']

    for layer in model_data['layers']:
        current_input_shape = propagator.propagate(current_input_shape, layer, framework='tensorflow')  # Use TF as proxy
        if layer['type'] == 'Conv2D':
            params = layer.get('params', {})
            node = helper.make_node(
                'Conv', ['input'], [f"conv_{id(layer)}"],
                kernel_shape=[params['kernel_size']] * 2 if isinstance(params['kernel_size'], int) else params['kernel_size'],
                pads=[0, 0, 0, 0],
                strides=[params.get('strides', 1)] * 2 if isinstance(params.get('strides', 1), int) else params.get('strides', [1, 1])
            )
            graph.node.append(node)
    onnx.checker.check_model(model)
    return model

def export_onnx(model_data: Dict[str, Any], filename: str = "model.onnx") -> str:
    """Export model to ONNX format."""
    model = generate_onnx(model_data)
    onnx.save(model, filename)
    return f"ONNX model saved to {filename}"



