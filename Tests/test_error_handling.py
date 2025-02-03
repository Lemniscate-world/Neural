def test_invalid_shape():
    input_shape = (28, 28)
    layers = [{"type": "Conv2D", "filters": 32, "kernel_size": (3, 3)}]
    with pytest.raises(ValueError):
        propagate_shape(input_shape, layers[0])