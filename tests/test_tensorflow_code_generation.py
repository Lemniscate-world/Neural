from html import parser
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lark
from neural.parser.parser import create_parser, ModelTransformer, generate_code
from neural.shape_propagation.shape_propagator import  ShapePropagator

test_code = """
network MyModel {
    input: (28, 28, 1)
    layers:
        Conv2D(filters=32, kernel_size=(3, 3), activation="relu")
        Flatten()
        Dense(units=128, activation="relu")
        Dropout(rate=0.5)
        Output(units=10, activation="softmax")
    loss: "categorical_crossentropy"
    optimizer: "adam"
    train {
        epochs: 10
        batch_size: 32
    }
}
"""
parser = create_parser("network")
tree = parser.parse(test_code)
print("Parsed tree: ", tree) # Debug Print
transformer = ModelTransformer()
model_data = transformer.transform(tree)
print("Transformed data: ", model_data) # Debug Print

# Debug prints
print("Input shape:", model_data['input_shape'])
print("First layer:", model_data['layers'][0])

propagate_shape = ShapePropagator()

# Validate shapes
input_shape = model_data['input_shape']
for layer in model_data['layers']:
    input_shape = propagate_shape(input_shape, layer)
    print(f"Layer {layer['type']} output shape: {input_shape}")

# Generate TensorFlow code
print(generate_code(model_data, backend="tensorflow"))