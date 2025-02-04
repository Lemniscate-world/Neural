import lark
import numpy as np

grammar = r"""
    network: "network" NAME "{" input_layer layers output_layer loss optimizer training_config? "}"
    input_layer: "input:" "(" INT "," INT "," INT ")"
    layers: "layers:" layer+
    layer: conv2d_layer | dense_layer | output_layer | dropout_layer | flatten_layer | max_pooling2d_layer
    conv2d_layer: "Conv2D(" "filters=" INT "," "kernel_size=(" INT "," INT ")," "activation=" ESCAPED_STRING ")"
    dense_layer: "Dense(" "units=" INT "," "activation=" ESCAPED_STRING ")"
    output_layer: "Output(" "units=" INT "," "activation=" ESCAPED_STRING ")"
    dropout_layer: "Dropout(" "rate=" FLOAT ")"
    flatten_layer: "Flatten()"
    max_pooling2d_layer: "MaxPooling2D(" "pool_size" "=" "(" INT "," INT ")" ")"
    training_config: "train" "{" ("epochs:" INT)? ("batch_size:" INT)?  "}"
    loss: "loss:" ESCAPED_STRING
    optimizer: "optimizer:" ESCAPED_STRING

    %import common.CNAME -> NAME
    %import common.INT
    %import common.FLOAT
    %import common.ESCAPED_STRING
    %import common.WS
    %ignore WS
"""

parser = lark.Lark(grammar, start='network')

class ModelTransformer(lark.Transformer):
    def layer(self, items):
        """
        Process a layer in the neural network model.

        This method extracts information about a single layer from the parsed items
        and returns a dictionary containing the layer's type and parameters.

        Parameters:
        items (list): A list containing the parsed information for the layer.
                      The first item is expected to be a dictionary with layer information.

        Returns:
        dict: A dictionary containing two keys:
              'type': The type of the layer (e.g., 'Conv2D', 'Dense', etc.)
              'params': A dictionary containing all the parameters for the layer
        """
        layer_info = items[0]  # The first item should be the layer information
        if isinstance(layer_info, lark.Tree):
        # If it's a Tree, the type is in the data attribute
            return {
                'type': layer_info.data,
                'params': layer_info.children[0] if layer_info.children else {}
            }
        else:
        # If it's not a Tree, the type is in the first item
            return {
                'type': items[0],
                'params': items[1] if len(items) > 1 else {}
            }
    
    def input_layer(self, items):
        # Convert tokens to integers explicitly
        shape = tuple(int(str(item)) for item in items)
        return {
            'type': 'Input',
            'shape': shape,
            }
    
    def conv2d_layer(self, items):
        # Debug Print
        print("conv2D items:", items)
        """
        Parses and processes a Conv2D layer configuration from the parsed items.

        Parameters:
        items (list): A list containing the parsed information for the Conv2D layer.
                      The list should contain four elements:
                      - The number of filters for the Conv2D layer.
                      - The height of the kernel for the Conv2D layer.
                      - The width of the kernel for the Conv2D layer.
                      - The activation function for the Conv2D layer as a string.

        Returns:
        dict: A dictionary containing the following keys:
              'type': The type of the layer, which is 'Conv2D'.
              'filters': The number of filters for the Conv2D layer.
              'kernel_size': A tuple containing the height and width of the kernel for the Conv2D layer.
              'activation': The activation function for the Conv2D layer.
        """

        # Parse items properly - items should be [filters, kernel_h, kernel_w, activation]
        filters = int(str(items[0]))
        kernel_h = int(str(items[1]))
        kernel_w = int(str(items[2]))
        activation = str(items[3]).strip('"')
        
        return {
            'type': 'Conv2D',
            'filters': filters,
            'kernel_size': (kernel_h, kernel_w),
            'activation': activation
        }
    
    def dense_layer(self, items):
        return {
                'type': 'Dense',
                'units': int(items[0]),
                'activation': items[1].strip('"')
            }
    
    def output_layer(self, items):
        """
        Parses and processes the configuration of an output layer in a neural network.

        Parameters:
        items (list): A list containing the parsed information for the output layer.
                      The list should contain two elements:
                      - The number of units for the output layer.
                      - The activation function for the output layer as a string.

        Returns:
        dict: A dictionary containing the following keys:
              'type': The type of the layer, which is 'Output'.
              'shape': A tuple representing the shape of the output data, containing the number of units.
              'units': The number of units for the output layer.
              'activation': The activation function for the output layer.
        """
        units = int(items[0])
        return {
                'type': 'Output',
                'shape': (units,),  # Output shape is a tuple with the number of units
                'units': int(items[0]),
                'activation': items[1].strip('"')
            }
    
    def loss(self, items):
        return {
            'type': 'Loss',
            'value': items[0]
        }
    
    def optimizer(self, items):
        return {'type': 'Optimizer',
                'value': items[0]
            }
    
    def layers(self, items):
        parsed_layers = []
        for item in items:
            if isinstance(item, lark.Tree):
                layer_data = self.layer(item)
                parsed_layers.append(layer_data)
            else:
                parsed_layers.append(item)
        return {'type': 'Layers', 'layers': parsed_layers}

    
    def dropout_layer(self, items):
        return {
            'type': 'Dropout',
            'rate': float(items[0])
        }
    
    def flatten_layer(self, items):
        return {'type': 'Flatten'}
    
    def training_config(self, items):
        return {
            'type': 'TrainingConfig',
            'epochs': int(items[0]['epochs']) if 'epochs' in items[0] else None,
            'batch_size': int(items[0]['batch_size']) if 'batch_size' in items[0] else None
        }
    
    def shape(self, items):
        return tuple(items)

    def max_pooling2d_layer(self, items):
        return {
            'type': 'MaxPooling2D',
            'pool_size': (int(items[0].value), int(items[1].value))
        }

    def network(self, items):
        name = str(items[0])
        input_shape = items[1]['shape']  # Input layer configuration
        layers = items[2]['layers']  # Layers configuration
        # Use the separate output layer if provided
        output_layer = next((item for item in items if isinstance(item, dict) and item.get('type') == 'Output'), None)
        
        # If no explicit output layer was found, check if there's an output layer in the layers
        if output_layer is None:
            output_layer = next((layer for layer in reversed(layers) if layer['type'] == 'Output'), None)
        
        # If still no output layer, use a default
        if output_layer is None:
            output_layer = {
                'type': 'Output', 
                'units': 1, 
                'activation': 'linear'
            }
        
        # Determine output shape
        if 'shape' in output_layer:
            output_shape = output_layer['shape']
        elif 'units' in output_layer:
            output_shape = (output_layer['units'],)
        else:
            output_shape = (1,)
        
        # Find loss and optimizer
        loss = next((item for item in items if isinstance(item, dict) and item.get('type') == 'Loss'), None)
        optimizer = next((item for item in items if isinstance(item, dict) and item.get('type') == 'Optimizer'), None)
        
        # Find training config (if exists)
        training_config = next((item for item in items if isinstance(item, dict) and item.get('type') == 'TrainingConfig'), {})
        
        return {
            'name': name,
            'input_shape': input_shape,
            'layers': layers,
            'output_layer': output_layer,
            'output_shape': output_shape,
            'loss': loss,
            'optimizer': optimizer,
            'training_config': training_config
        }

# Example code
code = """
network MyModel {
    input: (28, 28, 1)
    layers:
        Conv2D(filters=32, kernel_size=(3, 3), activation="relu")
        Dense(units=128, activation="relu")
        Output(units=10, activation="softmax")
    loss: "categorical_crossentropy"
    optimizer: "adam"
}
"""


def propagate_shape(input_shape, layer):
    """
    Propagates the input shape through a neural network layer to calculate the output shape.

    This function determines the output shape of a layer given its input shape and layer configuration.
    It supports Conv2D, Flatten, Dense, and Dropout layers.

    Parameters:
    input_shape (tuple): The shape of the input to the layer. For Conv2D, it should be a 3-tuple (height, width, channels).
    layer (dict): A dictionary containing the layer configuration. Must include a 'type' key specifying the layer type.

    Returns:
    tuple: The output shape of the layer after processing the input.

    Raises:
    TypeError: If the layer parameter is not a dictionary.
    KeyError: If the layer dictionary is missing required keys.
    ValueError: If the layer type is unsupported or if the input shape is invalid for the given layer type.
    """
    # Defensive type checking
    if not isinstance(layer, dict):
        raise TypeError(f"Layer must be a dictionary, got {type(layer)}")

    # Extract actual layer type, handling nested dictionary case
    layer_type = layer['type'] if isinstance(layer['type'], str) else layer['type']['type']

    print("Layer Type:", layer_type)
    print("Input Shape:", input_shape)
    print("Full Layer:", layer)

    # Helper to check input shape is 3D for convolution and pooling layers
    def check_3d_input(layer_type):
        if len(input_shape) != 3:
            raise ValueError(f"{layer_type} requires 3D input shape (h,w,c), got {input_shape}")
        return input_shape[0], input_shape[1], input_shape[2]

    if layer_type == 'Conv2D':
        # Handle both direct and nested dictionary cases for layer parameters
        if isinstance(layer['type'], dict):
            filters = layer['type']['filters']
            kernel_size = layer['type']['kernel_size']
        else:
            filters = layer.get('filters')
            kernel_size = layer.get('kernel_size')

        h, w, c = check_3d_input('Conv2D')
        
        # Validate required parameters
        if filters is None or kernel_size is None:
            raise KeyError("Conv2D layer missing required parameters")

        kernel_h, kernel_w = kernel_size

        # Simple shape calculation without padding (assuming valid padding)
        return (h - kernel_h + 1, w - kernel_w + 1, filters)

    elif layer_type == 'MaxPooling2D':
        h, w, c = check_3d_input('MaxPooling2D')
        
        # Handle both direct and nested dictionary cases for pool_size
        if isinstance(layer['type'], dict):
            pool_size = layer['type']['pool_size']
        else:
            pool_size = layer.get('pool_size')
        
        # Validate required parameters
        if pool_size is None:
            raise KeyError("MaxPooling2D layer missing pool_size parameter")
        
        pool_h, pool_w = pool_size
        return (h // pool_h, w // pool_w, c)

    elif layer_type == 'Flatten':
        # Flatten converts multi-dimensional input to 1D
        return (np.prod(input_shape),)

    elif layer_type == 'Dense':
        # Handle both direct and nested dictionary cases for units
        if isinstance(layer['type'], dict):
            units = layer['type'].get('units')
        else:
            units = layer.get('units')

        # Validate required parameters
        if units is None:
            raise KeyError("Dense layer missing units parameter")
        return (units,)

    elif layer_type == 'Dropout':
        # Dropout doesn't change the shape
        return input_shape

    elif layer_type == 'Output':
        # Handle both direct and nested dictionary cases for units
        if isinstance(layer['type'], dict):
            units = layer['type'].get('units')
        else:
            units = layer.get('units')

        if units is None:
            raise KeyError("Output layer missing units parameter")
        return (units,)

    else:
        raise ValueError(f"Unsupported layer type: {layer_type}") 

def generate_code(model_data, backend="tensorflow"):
    """
    This function generates code for creating and training a neural network model based on the given model data and backend.

    Parameters:
    model_data (dict): A dictionary containing the model configuration and data. It should have the following keys:
                       - 'input': A dictionary containing the input shape of the model. It should have a 'shape' key.
                       - 'layers': A list of dictionaries, each representing a layer in the model. Each layer dictionary should have a 'type' key.
                       - 'loss': A dictionary containing the loss function for the model. It should have a 'value' key.
                       - 'optimizer': A dictionary containing the optimizer for the model. It should have a 'value' key.
                       - 'training_config' (optional): A dictionary containing the training configuration for the model. It should have 'epochs' and 'batch_size' keys.

    backend (str): The backend framework to use for generating the code. It can be either 'tensorflow' or 'pytorch'.

    Returns:
    str: The generated code for creating and training the neural network model based on the given model data and backend.

    Raises:
    ValueError: If the specified backend is not supported.
    """
    if backend == "tensorflow":
        code = "import tensorflow as tf\n"
        code += "model = tf.keras.Sequential([\n"
        for layer in model_data['layers']:
            if layer['type'] == 'Conv2D':
                code += f"    tf.keras.layers.Conv2D({layer['filters']}, {layer['kernel_size']}, activation='{layer['activation']}'),\n"
            elif layer['type'] == 'Dense':
                code += f"    tf.keras.layers.Dense({layer['units']}, activation='{layer['activation']}'),\n"
            elif layer['type'] == 'Flatten':
                code += "    tf.keras.layers.Flatten(),\n"
            elif layer['type'] == 'Dropout':
                code += f"    tf.keras.layers.Dropout({layer['rate']}),\n"
        code += "])\n"

        # Remove quotes from optimizer and loss values
        optimizer = str(model_data['optimizer']['value']).strip('"')
        loss = str(model_data['loss']['value']).strip('"')
        code += f"model.compile(loss='{loss}', optimizer='{optimizer}')\n"

        # Extract training configuration
        if 'training' in model_data and model_data['training_config']:
            epochs = model_data['training_config'].get('epochs', 10) # Default to 10 epochs
            batch_size = model_data['training_config'].get('batch_size', 32) # Default to batch size of 32
            code += f"model.fit(data, epochs={epochs}, batch_size={batch_size})\n"
        return code

    elif backend == "pytorch":
        # PyTorch code generation logic 
        code = "import torch\n"
        code += "class Model(torch.nn.Module):\n"
        code += "    def __init__(self):\n"
        code += "        super(Model, self).__init__()\n"
        input_shape = model_data['input']['shape']
        for layer in model_data['layers']:
            output_shape = propagate_shape(input_shape, layer)
            if layer['type'] == 'Conv2D':
                code += f"        self.conv = torch.nn.Conv2d({input_shape[2]}, {layer['filters']}, {layer['kernel_size']}, padding=1)\n"
                code += f"        self.relu = torch.nn.ReLU()\n"
                input_shape = output_shape
                code += f"        self.pool = torch.nn.MaxPool2d({layer['kernel_size']})\n"
            elif layer['type'] == 'Flatten':
                code += "        self.flatten = torch.nn.Flatten()\n"
                input_shape = output_shape
                code += f"        self.fc = torch.nn.Linear({np.prod(input_shape)}, {layer['units']})\n"
                code += f"        self.relu = torch.nn.ReLU()\n"
                input_shape = output_shape
                code += f"        self.softmax = torch.nn.Softmax(dim=1)\n"
                code += f"    def forward(self, x):\n"
                code += f"        x = self.conv(x)\n"
                code += f"        x = self.relu(x)\n"
                code += f"        x = self.pool(x)\n"
                code += f"        x = self.flatten(x)\n"
                code += f"        x = self.fc(x)\n"
                code += f"        x = self.relu(x)\n"
                code += f"        x = self.softmax(x)\n"
                code += f"        return x\n"
        code += "model = Model()\n"
        code += f"model.to('{backend}')')\n"
        code += f"loss_fn = torch.nn.CrossEntropyLoss()\n"
        code += f"optimizer = torch.optim.{model_data['optimizer']['value']}()\n"

        if 'training_config' in model_data:
            code += f"for epoch in range({model_data['training_config']['epochs']}):\n"
            code += "    for batch_idx, (data, target) in enumerate(data):\n"
            code += "        optimizer.zero_grad()\n"
            code += "        output = model(data)\n"
            code += "        loss = loss_fn(output, target)\n"
            code += "        loss.backward()\n"
            code += "        optimizer.step()\n"
            code += "print('Finished Training')"
        return code
    else:
        raise ValueError("Unsupported backend")
    return code

