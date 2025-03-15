import logging
from neural.shape_propagation.shape_propagator import ShapePropagator
from neural.parser.parser import ModelTransformer, create_parser
from typing import Any, Dict, Union
import torch
import onnx
from onnx import helper, TensorProto
import numpy as np
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def to_number(x: str) -> Union[int, float]:
    try:
        return int(x)
    except ValueError:
        return float(x)

def generate_code(model_data: Dict[str, Any], backend: str) -> str:
    if not isinstance(model_data, dict) or 'layers' not in model_data or 'input' not in model_data:
        raise ValueError("Invalid model_data format: must be a dict with 'layers' and 'input' keys")

    indent = "    "
    propagator = ShapePropagator(debug=False)
    current_input_shape = model_data['input']['shape']
    
    # Process expanded layers before modifying model_data
    expanded_layers = []
    for layer in model_data.get('layers', []):
        if not isinstance(layer, dict) or 'type' not in layer:
            raise ValueError(f"Invalid layer format: {layer}")
        multiply = layer.get('multiply', 1)
        if not isinstance(multiply, int) or multiply < 1:
            raise ValueError(f"Invalid 'multiply' value: {multiply}")
        layer_copy = layer.copy()
        if 'multiply' in layer_copy:
            del layer_copy['multiply']
        for _ in range(multiply):
            expanded_layers.append(layer_copy.copy())

    # Store original layers
    original_layers = model_data['layers']
    model_data['layers'] = expanded_layers

    if backend == "tensorflow":
        optimizer_config = model_data.get('optimizer', {'type': 'Adam'})
        optimizer_type = optimizer_config['type'] if isinstance(optimizer_config, dict) else optimizer_config

        code = "import tensorflow as tf\nfrom tensorflow.keras import layers\n"
        code += f"from tensorflow.keras.optimizers import {optimizer_type}\n\n"

        # Add input shape handling
        input_shape = tuple(dim for dim in model_data['input']['shape'][1:])  # Remove batch dimension
        code += f"# Input layer with shape {input_shape}\n"
        code += f"inputs = layers.Input(shape={input_shape})\n"
        code += "x = inputs\n\n"

        for layer in expanded_layers:
            layer_type = layer['type']
            params = layer.get('params', {})
            
            if layer_type == "Residual":
                code += "# Residual block\n"
                code += "residual_input = x\n"
                for sub_layer in layer.get('sub_layers', []):
                    sub_type = sub_layer['type']
                    sub_params = sub_layer.get('params', {})
                    layer_code = generate_tensorflow_layer(sub_type, sub_params)
                    if layer_code:
                        code += f"x = {layer_code}\n"
                code += "x = layers.Add()([x, residual_input])\n"
            else:
                layer_code = generate_tensorflow_layer(layer_type, params)
                if layer_code:
                    code += f"x = {layer_code}\n"
            current_input_shape = propagator.propagate(current_input_shape, layer)

        code += "\n# Build model\n"
        code += "model = tf.keras.Model(inputs=inputs, outputs=x)\n"
        
        opt_params = []
        if isinstance(optimizer_config, dict):
            for k, v in optimizer_config.get('params', {}).items():
                opt_params.append(f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}")
        loss_value = model_data['loss']['value'] if isinstance(model_data['loss'], dict) else model_data.get('loss', 'categorical_crossentropy')
        code += f"# Compile model with {optimizer_type} optimizer and {loss_value} loss\n"
        code += f"model.compile(loss='{loss_value}', optimizer={optimizer_type}({', '.join(opt_params)}))\n"
        
        if 'training_config' in model_data:
            tc = model_data['training_config']
            code += "# Training configuration\n"
            code += (
                f"model.fit(\n    x_train, y_train,\n"
                f"    epochs={tc.get('epochs', 10)},\n"
                f"    batch_size={tc.get('batch_size', 32)},\n"
                f"    validation_split={tc.get('validation_split', 0.2)},\n"
                f"    verbose=1\n)\n"
            )
            if 'training_config' in model_data and model_data['training_config'].get('mixed_precision', False):
                code = "from tensorflow.keras.mixed_precision import set_global_policy\n" + code
                code += "set_global_policy('mixed_float16')\n"
            if 'training_config' in model_data and 'save_path' in model_data['training_config']:
                code += f"model.save('{model_data['training_config']['save_path']}')\n"
        return code

    elif backend == "pytorch":
        optimizer_config = model_data.get('optimizer', {'type': 'Adam'})
        optimizer_type = optimizer_config['type'] if isinstance(optimizer_config, dict) else optimizer_config
        code = "import torch\nimport torch.nn as nn\nimport torch.optim as optim\n\n"
        code += "# Neural network model definition\n"
        code += "class NeuralNetworkModel(nn.Module):\n"
        code += f"{indent}def __init__(self):\n"
        code += f"{indent}{indent}super(NeuralNetworkModel, self).__init__()\n"

        layers_code = []
        forward_code_body = []

        layer_counts = {}  # Track layer counts for unique naming
        current_channels = None  # Track channels globally for BatchNorm
        
        for i, layer in enumerate(expanded_layers):
            layer_type = layer['type']
            params = layer.get('params', {}).copy()
            
            if layer_type == "Residual":
                sub_layers = []
                for sub_layer in layer.get('sub_layers', []):
                    sub_type = sub_layer['type']
                    sub_params = sub_layer.get('params', {}).copy()
                    # Propagate data_format from parent layer
                    if 'data_format' in params:
                        sub_params['data_format'] = params['data_format']
                    # Track channels for BatchNorm
                    if sub_type == "Conv2D":
                        current_channels = sub_params.get("filters")
                    elif sub_type == "BatchNormalization" and current_channels:
                        sub_params["filters"] = current_channels
                    sub_layer_code = generate_pytorch_layer(sub_type, sub_params, current_input_shape)
                    if sub_layer_code:
                        sub_layers.append(sub_layer_code)
                
                if sub_layers:
                    layers_code.append(f"self.layer{i}_residual = nn.Sequential(")
                    for sub_layer in sub_layers:
                        layers_code.append(f"{indent}{sub_layer},")
                    layers_code.append(")")
                    forward_code_body.append(f"identity = x")
                    forward_code_body.append(f"x = self.layer{i}_residual(x)")
                    forward_code_body.append(f"x = x + identity")
            else:
                if layer_type not in layer_counts:
                    layer_counts[layer_type] = 0
                
                # Generate layer name with counter
                layer_name = f"layer{i}_{layer_type.lower()}_{layer_counts[layer_type]}"
                layer_counts[layer_type] += 1
                
                if layer_type == "Conv2D":
                    current_channels = params.get("filters")  # Update current channels
                elif layer_type == "BatchNormalization" and current_channels:
                    params["filters"] = current_channels
                
                layer_code = generate_pytorch_layer(layer_type, params, current_input_shape)
                if layer_code:
                    layers_code.append(f"self.{layer_name} = {layer_code}")
                    forward_code_body.append(f"x = self.{layer_name}(x)")

            current_input_shape = propagator.propagate(current_input_shape, layer)

        # Restore original layers
        model_data['layers'] = original_layers

        # Generate class code
        for line in layers_code:
            code += f"{indent}{indent}{line}\n"
        code += f"\n{indent}# Forward pass\n"
        code += f"{indent}def forward(self, x):\n"
        for line in forward_code_body:
            code += f"{indent}{indent}{line}\n"
        code += f"{indent}{indent}return x\n\n"

        code += "# Model instantiation\nmodel = NeuralNetworkModel()\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nmodel.to(device)\n\n"

        loss_value = model_data.get('loss', 'crossentropy') if isinstance(model_data.get('loss'), str) else model_data.get('loss', {'value': 'crossentropy'})['value']
        loss_fn = "nn.CrossEntropyLoss()" if "crossentropy" in loss_value.lower() else "nn.MSELoss()"
        code += f"# Loss function\nloss_fn = {loss_fn}\n"
        
        opt_params = []
        if isinstance(optimizer_config, dict):
            for k, v in optimizer_config.get('params', {'lr': 0.001}).items():
                opt_params.append(f"{k}={repr(v)}")
        code += f"# Optimizer\noptimizer = optim.{optimizer_type}(model.parameters(), {', '.join(opt_params)})\n"

        if 'training_config' in model_data:
            tc = model_data['training_config']
            code += "\n# Mixed precision training setup\n"
            code += "scaler = GradScaler()\n"
            code += "for epoch in range(num_epochs):\n"
            code += f"{indent}for batch_idx, (data, target) in enumerate(train_loader):\n"
            code += f"{indent}{indent}data, target = data.to(device), target.to(device)\n"
            code += f"{indent}{indent}optimizer.zero_grad()\n"
            code += f"{indent}{indent}with torch.cuda.amp.autocast():\n"
            code += f"{indent}{indent}{indent}output = model(data)\n"
            code += f"{indent}{indent}{indent}loss = criterion(output, target)\n"
            code += f"{indent}{indent}scaler.scale(loss).backward()\n"
            code += f"{indent}{indent}scaler.step(optimizer)\n"
            code += f"{indent}{indent}scaler.update()\n"
            if 'training_config' in model_data and 'save_path' in model_data['training_config']:
                code += f"torch.save(model.state_dict(), '{model_data['training_config']['save_path']}')\n"
            
            code += "# Note: Replace 'dataset' with your actual dataset\n"

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
        raise IOError(f"Error writing file: {filename}. {e}")
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

def generate_onnx(model_data):
    """Generate ONNX model"""
    # Create nodes for each layer
    nodes = []
    current_input = "input"
    
    for i, layer in enumerate(model_data['layers']):
        layer_type = layer['type']
        params = layer.get('params', {})
        output_name = f"layer_{i}_output"
        
        if layer_type == "Conv2D":
            nodes.append(helper.make_node(
                'Conv',
                inputs=[current_input],
                outputs=[output_name],
                kernel_shape=params.get('kernel_size', [3, 3]),
                strides=params.get('strides', [1, 1])
            ))
        # Add other layer types as needed
        
        current_input = output_name

    # Create graph with nodes
    graph = helper.make_graph(
        nodes=nodes,
        name="NeuralModel",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, model_data["input"]["shape"])],
        outputs=[helper.make_tensor_value_info(current_input, TensorProto.FLOAT, None)],
        initializer=[]
    )
    
    # Create model
    model = helper.make_model(graph, producer_name="Neural")
    model.opset_import[0].version = 13
    
    return model

def export_onnx(model_data: Dict[str, Any], filename: str = "model.onnx") -> str:
    """Export model to ONNX format."""
    model = generate_onnx(model_data)
    onnx.save(model, filename)
    return f"ONNX model saved to {filename}"

def generate_tensorflow_layer(layer_type, params):
    """Generate TensorFlow layer code"""
    if layer_type == "TransformerEncoder":
        num_heads = params.get("num_heads", 8)
        ff_dim = params.get("ff_dim", 512)
        dropout = params.get("dropout", 0.1)
        code = [
            "# TransformerEncoder block",
            f"x = layers.LayerNormalization(epsilon=1e-6)(x)",
            f"attn_output = layers.MultiHeadAttention(num_heads={num_heads}, key_dim={ff_dim})(x, x)",
            f"x = layers.Add()([x, attn_output])",
            f"x = layers.LayerNormalization(epsilon=1e-6)(x)",
            f"x = layers.Dense({ff_dim}, activation='relu')(x)",
            f"x = layers.Dense({ff_dim})(x)",
            f"x = layers.Dropout({dropout})(x)"
        ]
        return "\n".join(code)
    elif layer_type == "BatchNormalization":
        momentum = params.get("momentum", 0.99)
        epsilon = params.get("epsilon", 0.001)
        if momentum == 0.99 and epsilon == 0.001:
            return "layers.BatchNormalization()"
        return f"layers.BatchNormalization(momentum={momentum}, epsilon={epsilon})"
    elif layer_type == "Conv2D":
        filters = params.get("filters", 32)
        kernel_size = params.get("kernel_size", (3, 3))
        if isinstance(kernel_size, (tuple, list)):
            kernel_size = kernel_size[0]
        padding = params.get("padding", "same")
        activation = params.get("activation", None)
        code = f"layers.Conv2D(filters={filters}, kernel_size={kernel_size}, padding='{padding}'"
        if activation:
            code += f", activation='{activation}'"
        code += ")"
        return code
    elif layer_type == "Dense":
        units = params.get("units", 64)
        activation = params.get("activation", None)
        code = f"layers.Dense(units={units}"
        if activation:
            code += f", activation='{activation}'"
        code += ")"
        return code
    elif layer_type == "MaxPooling2D":
        pool_size = params.get("pool_size", (2, 2))
        if isinstance(pool_size, (tuple, list)):
            pool_size = pool_size
        strides = params.get("strides", None)
        if strides:
            return f"layers.MaxPooling2D(pool_size={pool_size}, strides={strides})"
        return f"layers.MaxPooling2D(pool_size={pool_size})"
    elif layer_type == "AveragePooling2D":
        pool_size = params.get("pool_size", (2, 2))
        if isinstance(pool_size, (tuple, list)):
            pool_size = pool_size[0] if isinstance(pool_size[0], int) else pool_size
        return f"layers.AveragePooling2D(pool_size={pool_size})"
    elif layer_type == "Flatten":
        return "layers.Flatten()"
    elif layer_type == "LSTM":
        units = params.get("units", 128)
        return_sequences = params.get("return_sequences", False)
        return f"layers.LSTM(units={units}, return_sequences={str(return_sequences)})"
    elif layer_type == "GRU":
        units = params.get("units", 64)
        return_sequences = params.get("return_sequences", False)
        return f"layers.GRU(units={units}, return_sequences={str(return_sequences)})"
    elif layer_type == "Dropout":
        rate = params.get("rate", 0.5)
        return f"layers.Dropout(rate={rate})"
    elif layer_type == "Output":
        units = params.get("units", 10)
        activation = params.get("activation", "softmax")
        return f"layers.Dense(units={units}, activation='{activation}')"
    else:
        warnings.warn(f"Unsupported layer type '{layer_type}' for tensorflow. Skipping.", UserWarning)
        return None

def generate_pytorch_layer(layer_type, params, input_shape=None):
    """Generate PyTorch layer code"""
    if layer_type == "Conv2D":
        data_format = params.get("data_format", "channels_last")
        in_channels = input_shape[1] if data_format == "channels_first" else input_shape[3]
        in_channels = in_channels if input_shape and len(input_shape) > 3 else 3
        out_channels = params.get("filters", 32)
        kernel_size = params.get("kernel_size", 3)
        # Handle both tuple/list and integer kernel sizes
        if isinstance(kernel_size, (tuple, list)):
            kernel_size = kernel_size[0]  # Use first element for both dimensions
        return f"nn.Conv2d(in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size})"
    elif layer_type == "BatchNormalization":
        data_format = params.get("data_format", "channels_last")
        if input_shape and len(input_shape) > 3:
            num_features = input_shape[1] if data_format == "channels_first" else input_shape[3]
        else:
            # Use the number of filters from previous Conv2D layer if available
            num_features = params.get("filters", 64)  # Default to 64 features
        momentum = params.get("momentum", 0.9)
        eps = params.get("epsilon", 0.001)
        # Only include momentum and eps if they differ from defaults
        if momentum == 0.9 and eps == 0.001:
            return f"nn.BatchNorm2d(num_features={num_features})"
        return f"nn.BatchNorm2d(num_features={num_features}, momentum={momentum}, eps={eps})"
    elif layer_type == "Dense":
        in_features = np.prod(input_shape[1:]) if input_shape else 64
        out_features = params.get("units", 64)
        activation = params.get("activation", None)
        layers = [f"nn.Linear(in_features={in_features}, out_features={out_features})"]
        if activation:
            if activation == "relu":
                layers.append("nn.ReLU()")
            elif activation == "tanh":
                layers.append("nn.Tanh()")
            elif activation == "softmax":
                layers.append("nn.Softmax(dim=1)")
            elif activation == "invalid":
                layers.append("nn.Identity()")
        return "nn.Sequential(" + ", ".join(layers) + ")"
    elif layer_type == "MaxPooling2D":
        pool_size = params.get("pool_size", 2)
        # Handle both tuple/list and integer pool sizes
        if isinstance(pool_size, (tuple, list)):
            pool_size = pool_size if len(pool_size) == 2 else (pool_size[0], pool_size[0])
        strides = params.get("strides", None)
        if strides:
            return f"nn.MaxPool2d(kernel_size={pool_size}, stride={strides})"
        return f"nn.MaxPool2d(kernel_size={pool_size})"
    elif layer_type == "AveragePooling2D":
        pool_size = params.get("pool_size", 2)
        # Handle both tuple/list and integer pool sizes
        if isinstance(pool_size, (tuple, list)):
            pool_size = pool_size if len(pool_size) == 2 else (pool_size[0], pool_size[0])
        return f"nn.AvgPool2d(kernel_size={pool_size})"
    elif layer_type == "Flatten":
        return "nn.Flatten()"
    elif layer_type == "Dropout":
        rate = params.get("rate", 0.5)
        return f"nn.Dropout(p={rate})"
    elif layer_type == "Output":
        in_features = np.prod(input_shape[1:]) if input_shape else 64
        out_features = params.get("units", 10)
        activation = params.get("activation", "softmax")
        layers = [f"nn.Linear(in_features={in_features}, out_features={out_features})"]
        if activation == "softmax":
            layers.append("nn.Softmax(dim=1)")
        return "nn.Sequential(" + ", ".join(layers) + ")"
    elif layer_type == "LSTM":
        input_size = input_shape[-1] if input_shape else 32
        hidden_size = params.get("units", 128)
        return f"nn.LSTM(input_size={input_size}, hidden_size={hidden_size}, batch_first=True)"
    elif layer_type == "GRU":
        input_size = params.get("input_size", 128)
        hidden_size = params.get("units", 64)
        return f"nn.GRU(input_size={input_size}, hidden_size={hidden_size}, batch_first=True)"
    elif layer_type == "TransformerEncoder":
        d_model = params.get("d_model", 512)
        nhead = params.get("num_heads", 8)
        dim_feedforward = params.get("ff_dim", 2048)
        dropout = params.get("dropout", 0.1)
        return f"nn.TransformerEncoderLayer(d_model={d_model}, nhead={nhead}, dim_feedforward={dim_feedforward}, dropout={dropout})"
    else:
        warnings.warn(f"Unsupported layer type '{layer_type}' for pytorch. Skipping.", UserWarning)
        return None
    
def generate_optimized_dsl(config, best_params):
    transformer = ModelTransformer()
    model_dict, hpo_params = transformer.parse_network_with_hpo(config)
    lines = config.strip().split('\n')
    for i, line in enumerate(lines):
        for hpo in hpo_params:
            param_key = f"{hpo['layer_type'].lower()}_{hpo['param_name']}"
            if param_key in best_params and 'HPO(' in line:
                hpo_str = str(hpo['node'].children[0])  # Original HPO expression
                lines[i] = line.replace(f"HPO({hpo_str})", str(best_params[param_key]))
    return '\n'.join(lines)

