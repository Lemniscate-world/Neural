import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from parser import propagate_shape, parser, ModelTransformer


def test_parser():
    code = """
    network MyModel {
        input: (28, 28, 1)
        layers:
            Conv2D(filters=32, kernel_size=(3, 3), activation="relu")
            MaxPooling2D(pool_size=(2, 2))
            Flatten()
            Dense(units=128, activation="relu")
            Output(units=10, activation="softmax")
        loss: "categorical_crossentropy"
        optimizer: "adam"
    }
    """
    tree = parser.parse(code)
    print(tree.pretty())
    transformer = ModelTransformer()
    model_data = transformer.transform(tree)
    print(model_data)
    assert model_data["name"] == "MyModel"
    assert model_data["input_shape"] == (28, 28, 1)
    assert len(model_data["layers"]) == 5  # Now including MaxPooling2D
    assert model_data["layers"][0]["type"] == "Conv2D"
    assert model_data["layers"][1]["type"] == "MaxPooling2D"
    assert model_data["layers"][2]["type"] == "Flatten"
    assert model_data["layers"][3]["type"] == "Dense"
    assert model_data["layers"][4]["type"] == "Output"
    assert model_data["output_shape"] == (10,)
    assert model_data["loss"]["value"] == '"categorical_crossentropy"'
    assert model_data["optimizer"]["value"] == '"adam"'