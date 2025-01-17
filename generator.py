def generate_tensorflow_code(model_data):
    model_code = "import tensorflow as tf\n"
    model_code += "model = tf.keras.Sequential([\n"
    for layer in model_data['layers']:
        if layer['type'] == 'Conv2D':
            model_code += f"    tf.keras.layers.Conv2D({layer['filters']}, {layer['kernel_size']}, activation='{layer['activation']}'),\n"
        elif layer['type'] == 'Dense':
            model_code += f"    tf.keras.layers.Dense({layer['units']}, activation='{layer['activation']}'),\n"
        elif layer['type'] == 'Output':
            model_code += f"    tf.keras.layers.Dense({layer['units']}, activation='{layer['activation']}')\n"
    model_code += "])\n"
    model_code += f"model.compile(optimizer='{model_data['optimizer_function']}', loss='{model_data['loss_function']}')\n"
    return model_code

tensorflow_code = generate_tensorflow_code(model_data)
print(tensorflow_code)
