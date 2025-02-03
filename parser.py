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
    max_pooling2d_layer: "MaxPooling2D(" "pool_size=(" INT "," INT "))"
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
        return {
            'type': layer_info['type'],
            'params': layer_info
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
        return {
            'type': 'Layers',
            'layers': [items['params'] if 'params' in items else items for items in items]  
        }
    
    def dropout_layer(self, items):
        return {
            'type': 'Dropout',
            'rate': float(items[0])
        }
    
    def flatten_layer(self, items):
        return {'type': 'Flatten'}
    
    def training_config(self, items):
        return {
            'epochs': int(items[0]['epochs']) if 'epochs' in items[0] else None,
            'batch_size': int(items[0]['batch_size']) if 'batch_size' in items[0] else None
        }
    
    def shape(self, items):
        return tuple(items)

    def network(self, items):
        """
        Parses and processes the network configuration from the parsed items.

        Parameters:
        items (list): A list containing the parsed information for the network.
                    The list should contain the following elements in order:
                    - The name of the network (ignored).
                    - The input layer configuration.
                    - The layers configuration.
                    - The output layer configuration.
                    - The loss function.
                    - The optimizer.
                    - The training configuration (optional).

        Returns:
        dict: A dictionary containing the following keys:
            - 'input_shape': A tuple representing the shape of the input data.
            - 'layers': A list of dictionaries, each representing a layer in the network.
            - 'output_shape': A tuple representing the shape of the output data.
            - 'loss': A dictionary containing the loss function for the network.
            - 'optimizer': A dictionary containing the optimizer for the network.
            - 'training_config': A dictionary containing the training configuration for the network.
        """
        input_shape = items[1]['shape']  # Input layer configuration
        layers = items[2]['layers']  # Layers configuration
        output_shape = items[3]['shape']  # Output layer configuration
        loss = items[4]      # Loss function
        optimizer = items[5]  # Optimizer
        training_config = items[6] if len(items) > 6 else {}  # Training configuration (optional)
        
        return {
            'input_shape': input_shape,
            'layers': layers,
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
    # Debug Print
    print("Layer Received:", layer)
    print("Input shape", input_shape)
    print(f"Processing shape {input_shape} for layer {layer}")

    if not isinstance(layer, dict):
        raise TypeError(f"Layer must be a dictionary, got {type(layer)}")

    if 'type' not in layer:
        raise KeyError("Layer dictionary must contain 'type' key")

    if layer['type'] == 'Conv2D':
        # Validate Conv2D parameters
        required_keys = ['filters', 'kernel_size']
        for key in required_keys:
            if key not in layer:
                raise KeyError(f"Conv2D layer missing required parameter: {key}")

        filters = layer['filters']
        kernel_h, kernel_w = layer['kernel_size']

        if len(input_shape) != 3:
            raise ValueError(f"Conv2D requires 3D input shape (h,w,c), got {input_shape}")

        h, w, c = input_shape
        # Simple shape calculation without padding
        return (h - kernel_h + 1, w - kernel_w + 1, filters)

    elif layer['type'] == 'MaxPooling2D':
        # For MaxPooling2D, it ensures the correct parameters are provided 
        # and calculates how the input shape changes after applying max pooling.
        # Validate MaxPooling parameters
        required_keys = ['pool_size']
        for key in required_keys:
            if key not in layer:
                raise KeyError(f"MaxPooling2D layer missing required parameter: {key}")
        
        pool_h, pool_w = layer['pool_size']

        if len(input_shape) != 3:
            raise ValueError(f"MaxPooling2D requires 3D input shape (h,w,c), got {input_shape}")
        
        h, w, c = input_shape
        return (h // pool_h, w // pool_w, c)

    elif layer['type'] == 'Output':
        if 'units' not in layer:
            raise KeyError("Output layer missing required parameter: units")
        return (layer['units'],)

    elif layer['type'] == 'Flatten':
        return (np.prod(input_shape),)

    elif layer['type'] == 'Dense':
        if 'units' not in layer:
            raise KeyError("Dense layer missing required parameter: units")
        return (layer['units'],)

    elif layer['type'] == 'Dropout':
        return input_shape

    else:
        raise ValueError(f"Unsupported layer type: {layer['type']}")


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

