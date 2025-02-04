def my_custom_layer(items):
    return {"type": "CustomLayer", "custom_param": int(items[0])}

def register(register_fn):
    register_fn("CustomLayer", my_custom_layer)
