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
    
    # Layer expansion with proper multiplication handling
    expanded_layers = []
    for layer in model_data['layers']:
        if not isinstance(layer, dict) or 'type' not in layer:
            raise ValueError(f"Invalid layer format: {layer}")
        multiply = layer.pop('multiply', 1)  
        if not isinstance(multiply, int) or multiply < 1:
            raise ValueError(f"Invalid 'multiply' value: {multiply}")
        
        expanded_layers.extend([layer.copy() for _ in range(multiply)])
        
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
            
            layer_code = generate_tensorflow_layer(layer_type, params)
            if layer_code:
                code += f"x = {layer_code}(x)\n"

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

        input_shape = model_data['input']['shape']
        if not input_shape:
            raise ValueError("Input layer shape is not defined.")
        current_input_shape = input_shape

        layers_code = []
        forward_code_body = []

        # Add mixed precision imports if needed
        if model_data.get("training_config", {}).get("mixed_precision"):
            code += "from torch.cuda.amp import autocast, GradScaler\n"

        for i, layer_config in enumerate(model_data['layers']):
            layer_type = layer_config['type']
            params = layer_config.get('params', {})
            layer_name = f"self.layer{i}"

            if "sub_layers" in layer_config:
                sub_layers = layer_config.get('sub_layers', [])
                if layer_type == "TimeDistributed":
                    if not sub_layers:
                        raise ValueError("TimeDistributed requires sub_layers")
                    sub_layer = sub_layers[0]
                    sub_type = sub_layer.get('type', '')
                    sub_params = sub_layer.get('params', {})
                    if sub_type == "Conv2D":
                        layers_code.append(
                            f"# Time-distributed Conv2D layer\n"
                            f"{layer_name}_conv = nn.Conv2d(in_channels={current_input_shape[-1]}, "
                            f"out_channels={sub_params.get('filters', 32)}, kernel_size={sub_params.get('kernel_size', 3)})"
                        )
                        forward_code_body.append(f"x = self.layer{i}_conv(x)")
                elif layer_type == "Residual":
                    if not sub_layers:
                        raise ValueError("Residual requires sub_layers")
                    layers_code.append(f"# Residual block\n{layer_name}_residual = nn.Sequential(")
                    for sub_layer in sub_layers:
                        if sub_layer.get('type') == 'Conv2D':
                            sub_params = sub_layer.get('params', {})
                            layers_code.append(
                                f"{indent}{indent}nn.Conv2d(in_channels={current_input_shape[-1]}, "
                                f"out_channels={sub_params.get('filters', 32)}, kernel_size={sub_params.get('kernel_size', 3)}),"
                            )
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
                layers_code.append(
                    f"# Convolutional layer with {filters} filters\n"
                    f"{layer_name}_conv = nn.Conv2d(in_channels={current_input_shape[-1]}, out_channels={filters}, kernel_size={kernel_size})"
                )
                layers_code.append(f"{layer_name}_act = nn.ReLU()" if activation == 'relu' else f"{layer_name}_act = nn.Identity()")
                forward_code_body.append(f"x = self.layer{i}_conv(x)")
                forward_code_body.append(f"x = self.layer{i}_act(x)")
            elif layer_type == 'MaxPooling2D':
                pool_size = params.get('pool_size', 2)
                layers_code.append(f"# MaxPooling2D layer\n{layer_name}_pool = nn.MaxPool2d(kernel_size={pool_size})")
                forward_code_body.append(f"x = self.layer{i}_pool(x)")
            elif layer_type == 'Flatten':
                layers_code.append(f"# Flatten layer\n{layer_name}_flatten = nn.Flatten()")
                forward_code_body.append(f"x = self.layer{i}_flatten(x)")
            elif layer_type == 'MaxPooling2D':
                pool_size = params.get('pool_size', 2)
                layers_code.append(f"{layer_name}_pool = nn.MaxPool2d(kernel_size={pool_size})")
            elif layer_type == 'AveragePooling2D':
                pool_size = params.get('pool_size', 2)
                layers_code.append(f"{layer_name}_pool = nn.AvgPool2d(kernel_size={pool_size})")
            elif layer_type in ['LSTM', 'GRU']:
                units = params.get('units', 64)
                return_seq = params.get('return_sequences', False)
                code += f"{layer_name} = nn.{layer_type}(units={units}, return_sequences={return_seq})(x)\n"            
            elif layer_type == 'BatchNormalization':
                momentum = params.get('momentum', 0.99)
                epsilon = params.get('epsilon', 1e-3)
                layers_code.append(f"{layer_name}_bn = nn.BatchNorm2d(num_features={current_input_shape[-1]}, momentum={momentum}, eps={epsilon})")
            elif layer_type in ['Dense', 'Output']:
                units = params.get('units')
                activation = params.get('activation', 'relu' if layer_type == 'Dense' else 'linear')
                if not units:
                    raise ValueError(f"{layer_type} layer config missing 'units'.")
                in_features = np.prod(current_input_shape[1:]) if len(current_input_shape) > 2 else current_input_shape[-1]
                layers_code.append(
                    f"# {'Output' if layer_type == 'Output' else 'Dense'} layer with {units} units\n"
                    f"{layer_name}_dense = nn.Linear(in_features={in_features}, out_features={units})"
                )
                act_layer = "nn.ReLU()" if activation == 'relu' else "nn.Softmax(dim=1)" if activation == 'softmax' else "nn.Identity()"
                layers_code.append(f"{layer_name}_act = {act_layer}")
                forward_code_body.append(f"x = self.layer{i}_dense(x)")
                forward_code_body.append(f"x = self.layer{i}_act(x)")
            elif layer_type == 'Dropout':
                rate = params.get('rate', 0.5)
                layers_code.append(f"# Dropout layer with rate {rate}\n{layer_name}_dropout = nn.Dropout(p={rate})")
                forward_code_body.append(f"x = self.layer{i}_dropout(x)")
            elif layer_type == 'BatchNormalization':
                layers_code.append(f"# BatchNormalization layer\n{layer_name}_bn = nn.BatchNorm2d(num_features={current_input_shape[-1]})")
                forward_code_body.append(f"x = self.layer{i}_bn(x)")
            elif layer_type in ['Transformer', 'TransformerEncoder']:
                num_heads = params.get('num_heads', 8)
                ff_dim = params.get('ff_dim', 512)
                dropout = params.get('dropout', 0.1)
                layers_code.append(
                    f"# TransformerEncoder layer\n{layer_name} = nn.TransformerEncoderLayer(\n"
                    f"{indent}{indent}d_model={current_input_shape[-1]}, nhead={num_heads},\n"
                    f"{indent}{indent}dim_feedforward={ff_dim}, dropout={dropout}\n{indent})"
                )
                forward_code_body.append(f"x = self.layer{i}(x)")
            else:
                logger.warning(f"Unsupported layer type '{layer_type}' for PyTorch. Skipping.")
                continue

            current_input_shape = propagator.propagate(current_input_shape, layer_config, framework='pytorch')

        for line in layers_code:
            code += f"{indent}{indent}{line}\n"
        code += f"\n{indent}# Forward pass\n"
        if model_data.get("training_config", {}).get("mixed_precision"):
            code += f"{indent}def forward(self, x):\n"
            code += f"{indent}{indent}with autocast():\n"
            for line in forward_code_body:
                code += f"{indent}{indent}{indent}{line}\n"
            code += f"{indent}{indent}return x\n\n"
        else:
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
            code += (
                "\n# Training loop\n"
                f"for epoch in range({tc.get('epochs', 10)}):\n"
                f"{indent}model.train()\n"
                f"{indent}for batch_x, batch_y in DataLoader(dataset, batch_size={tc.get('batch_size', 32)}):\n"
                f"{indent}{indent}batch_x, batch_y = batch_x.to(device), batch_y.to(device)\n"
                f"{indent}{indent}optimizer.zero_grad()\n"
                f"{indent}{indent}outputs = model(batch_x)\n"
                f"{indent}{indent}loss = loss_fn(outputs, batch_y)\n"
                f"{indent}{indent}loss.backward()\n"
                f"{indent}{indent}optimizer.step()\n"
                f"{indent}print(f'Epoch {{epoch+1}}, Loss: {{loss.item()}}')\n"
            )
            if 'training_config' in model_data and model_data['training_config'].get('mixed_precision', False):
                code += "\nscaler = GradScaler()\n"
                code += "\nwith autocast():\n    # Your training code here\n"
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
    if layer_type == "Conv2D":
        filters = params.get("filters", 32)
        kernel_size = params.get("kernel_size", (3, 3))
        return f"layers.Conv2D(filters={filters}, kernel_size={kernel_size})"
    elif layer_type == "Dense":
        units = params.get("units", 64)
        return f"layers.Dense(units={units})"
    elif layer_type == "BatchNormalization":
        return "layers.BatchNormalization()"
    elif layer_type == "MaxPooling2D":
        pool_size = params.get("pool_size", (2, 2))
        strides = params.get("strides", None)
        if strides:
            return f"layers.MaxPooling2D(pool_size={pool_size}, strides={strides})"
        return f"layers.MaxPooling2D(pool_size={pool_size})"
    elif layer_type == "AveragePooling2D":
        pool_size = params.get("pool_size", (2, 2))
        strides = params.get("strides", None)
        if strides:
            return f"layers.AveragePooling2D(pool_size={pool_size}, strides={strides})"
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
    else:
        warnings.warn(f"Unsupported layer type '{layer_type}' for tensorflow. Skipping.", UserWarning)
        return None

def generate_pytorch_layer(layer_type, params, input_shape=None):
    """Generate PyTorch layer code"""
    if layer_type == "Conv2D":
        in_channels = input_shape[3] if input_shape else 3
        out_channels = params.get("filters", 32)
        kernel_size = params.get("kernel_size", (3, 3))
        return f"nn.Conv2d(in_channels={in_channels}, out_channels={out_channels}, kernel_size={kernel_size[0]})"
    elif layer_type == "BatchNormalization":
        num_features = params.get("filters", 64)
        return f"nn.BatchNorm2d(num_features={num_features})"
    elif layer_type == "Dense":
        in_features = input_shape[-1] if input_shape else 64
        out_features = params.get("units", 64)
        return f"nn.Linear(in_features={in_features}, out_features={out_features})"
    elif layer_type == "LSTM":
        input_size = input_shape[-1] if input_shape else 32
        hidden_size = params.get("units", 128)
        return f"nn.LSTM(input_size={input_size}, hidden_size={hidden_size}, batch_first=True)"
    elif layer_type == "GRU":
        input_size = input_shape[-1] if input_shape else 32
        hidden_size = params.get("units", 64)
        return f"nn.GRU(input_size={input_size}, hidden_size={hidden_size}, batch_first=True)"
    else:
        warnings.warn(f"Unsupported layer type '{layer_type}' for pytorch. Skipping.")
        return None