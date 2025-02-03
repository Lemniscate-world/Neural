def test_shape_propagation():
    input_shape = (28, 28, 1)
    layers = [
        {"type": "Conv2D", "filters": 32, "kernel_size": (3, 3)},
        {"type": "MaxPooling2D", "pool_size": (2, 2)},
        {"type": "Flatten"},
        {"type": "Dense", "units": 128},
        {"type": "Output", "units": 10}
    ]
    for layer in layers:
        input_shape = propagate_shape(input_shape, layer)
    assert input_shape == (10,)