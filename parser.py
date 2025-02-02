import lark
import numpy as np

grammar = r"""
    network: "network" NAME "{" input_layer layers output_layer loss optimizer training_config? "}"
    input_layer: "input:" "(" INT "," INT "," INT ")"
    layers: "layers:" layer+
    layer: conv2d_layer | dense_layer | output_layer | dropout_layer | flatten_layer
    conv2d_layer: "Conv2D(" "filters=" INT "," "kernel_size=(" INT "," INT ")," "activation=" ESCAPED_STRING ")"
    dense_layer: "Dense(" "units=" INT "," "activation=" ESCAPED_STRING ")"
    output_layer: "Output(" "units=" INT "," "activation=" ESCAPED_STRING ")"
    dropout_layer: "Dropout(" "rate=" FLOAT ")"
    flatten_layer: "Flatten()"
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
        return {'type': 'Input',
                'shape': tuple(items)
            }
    
    def conv2d_layer(self, items):
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
        return {
            'type': 'Conv2D',
            'filters': int(items[0]),
            'kernel_size': (int(items[1]), int(items[2])),
            'activation': items[3].strip('"')
        }
    
    def dense_layer(self, items):
        return {
                'type': 'Dense',
                'units': int(items[0]),
                'activation': items[1].strip('"')
            }
    
    def output_layer(self, items):
        return {
                'type': 'Output',
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
            'layers': items
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
        loss = items[4]['loss']      # Loss function
        optimizer = items[5]['optimizer']  # Optimizer
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
    Propagates the shape of the input through a given layer.

    Parameters:
    input_shape (tuple): The shape of the input data. For a Conv2D layer, it should be (height, width, channels).
                         For a Dense layer, it should be (units,). For a Flatten layer, it should be the shape after
                         the previous layer.
    layer (dict): A dictionary representing the layer. It should contain the 'type' key, which can be one of 'Conv2D',
                  'Flatten', or 'Dense'. For 'Conv2D' layers, it should also contain 'filters' and 'kernel_size' keys.
                  For 'Dense' layers, it should contain 'units' key.

    Returns:
    tuple: The shape of the output data after passing through the given layer. For a Conv2D layer, it will be
           (height, width, filters). For a Flatten layer, it will be (units,). For a Dense layer, it will be (units,).
    """
    if layer['type'] == 'Conv2D':
        filters = layer['filters']
        kernel_h, kernel_w = layer['kernel_size']
        h, w, _ = input_shape
        return (h - kernel_h + 1, w - kernel_w + 1, filters)
    elif layer['type'] == 'Flatten':
        return (np.prod(input_shape),)
    elif layer['type'] == 'Dense':
        return (layer['units'],)


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
        code += f"model.compile(optimizer='{model_data['optimizer']['value']}', loss='{model_data['loss']['value']}')\n"
        if 'training_config' in model_data:
            code += f"model.fit(data, epochs={model_data['training_config']['epochs']}, batch_size={model_data['training_config']['batch_size']})\n"
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

