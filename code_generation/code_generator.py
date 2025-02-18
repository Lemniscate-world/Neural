from shape_propagation.shape_propagator import propagate_shape
from typing import Any, Dict, List, Tuple, Union, Optional, Callable
import torch

def NUMBER(x):
    try:
        return int(x)
    except ValueError:
        return float(x)


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
        train_loader = torch.utils.data.DataLoader(MNIST(), batch_size=batch_size, shuffle=True)
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
            elif layer_type in ['Attention', 'TransformerEncoder', 'Residual', 'InceptionModule', 'CapsuleLayer', 'SqueezeExcitation', 'GraphConv', 'Embedding', 'QuantumLayer', 'DynamicLayer']:
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


def load_file(filename):
    with open(filename, 'r') as f:
        content = f.read()
        
    if filename.endswith('.neural') or filename.endswith('.nr'):
        # Parse as a network file.
        return create_parser('neural_file').parse(content)
    elif filename.endswith('.rnr'):
        # Parse as a research file.
        return create_parser('rnr_file').parse(content)
    else:
        raise ValueError(f"Unsupported file type: {filename}")
