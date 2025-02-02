import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lark
from parser import parser, ModelTransformer, propagate_shape, generate_code

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

tree = parser.parse(test_code)
transformer = ModelTransformer()
model_data = transformer.transform(tree)
print("Transformed data: ", model_data) # Debug Print

# Validate shapes
input_shape = model_data['input']['shape']
for layer in model_data['layers']:
    input_shape = propagate_shape(input_shape, layer)
    print(f"Layer {layer['type']} output shape: {input_shape}")

# Generate TensorFlow code
print(generate_code(model_data, backend="tensorflow"))