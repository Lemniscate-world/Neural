import importlib.util
import os

LAYER_PLUGINS = {}

def register_layer(name, handler):
    """ Register a custom layer with a parsing handler. """
    LAYER_PLUGINS[name] = handler

def load_plugins(plugin_dir="plugins"):
    """ Load all Python files in the plugin directory dynamically. """
    if not os.path.exists(plugin_dir):
        os.makedirs(plugin_dir)

    for filename in os.listdir(plugin_dir):
        if filename.endswith(".py"):
            module_name = filename[:-3]
            module_path = os.path.join(plugin_dir, filename)
            
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, "register"):
                module.register(register_layer)

load_plugins()
