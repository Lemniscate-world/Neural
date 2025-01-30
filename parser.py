import lark

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
    def network(self, items):
        return {
            'type': 'Network',
            'name': items[0],
            'input': items[1],
            'layers': items[2],
            'output': items[3],
            'loss': items[4],
            'optimizer': items[5]
        }
    
    def input_layer(self, items):
        return {'type': 'Input', 'shape': tuple(items)}
    
    def conv2d_layer(self, items):
        return {'type': 'Conv2D', 'filters': items[0], 'kernel_size': (items[1], items[2]), 'activation': items[3]}
    
    def dense_layer(self, items):
        return {'type': 'Dense', 'units': items[0], 'activation': items[1]}
    
    def output_layer(self, items):
        return {'type': 'Output', 'units': items[0], 'activation': items[1]}
    
    def loss(self, items):
        return {'type': 'Loss', 'value': items[0]}
    
    def optimizer(self, items):
        return {'type': 'Optimizer', 'value': items[0]}
    
    def layers(self, items):
        return items


# Exemple de code à analyser
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

# Analyser et transformer
tree = parser.parse(code)
transformer = ModelTransformer()
model_data = transformer.transform(tree)

print(model_data)

def propagate_shape(input_shape, layer):
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
        # Add PyTorch code generation logic here
        pass

switch_code = generate_code(model_data)
print(tensorflow_code)