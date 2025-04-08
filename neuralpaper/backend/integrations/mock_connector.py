"""
Mock connector for NeuralPaper.ai in production environments where Neural DSL is not available.
This provides basic functionality for the API to work without the actual Neural DSL.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger("neuralpaper.mock_connector")

class MockConnector:
    """
    Mock connector class that simulates Neural DSL and NeuralDbg functionality
    """
    
    def __init__(self, models_dir: str = None):
        """
        Initialize the mock connector
        
        Args:
            models_dir: Directory containing model files
        """
        self.models_dir = models_dir or os.path.join(os.path.dirname(__file__), "../models")
        logger.info(f"Initialized mock connector with models directory: {self.models_dir}")
    
    def load_model(self, model_id: str) -> Tuple[str, Dict[str, Any]]:
        """
        Load a model from file
        
        Args:
            model_id: ID of the model to load
            
        Returns:
            Tuple of (dsl_code, annotations)
        """
        model_path = os.path.join(self.models_dir, f"{model_id}.neural")
        annotations_path = os.path.join(self.models_dir, f"{model_id}.annotations.json")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'r') as f:
            dsl_code = f.read()
        
        annotations = {}
        if os.path.exists(annotations_path):
            with open(annotations_path, 'r') as f:
                annotations = json.load(f)
        
        logger.info(f"Loaded model: {model_id}")
        return dsl_code, annotations
    
    def parse_dsl(self, dsl_code: str, backend: str = 'tensorflow') -> Dict[str, Any]:
        """
        Parse Neural DSL code (mock implementation)
        
        Args:
            dsl_code: Neural DSL code
            backend: Target backend (tensorflow or pytorch)
            
        Returns:
            Dictionary with model_data, shape_history, and trace_data
        """
        logger.info(f"Parsing DSL code for backend: {backend}")
        
        # Extract basic information from the DSL code
        lines = dsl_code.split('\n')
        model_name = "Unknown"
        input_shape = [1, 28, 28, 1]  # Default shape
        layers = []
        
        for line in lines:
            line = line.strip()
            
            # Extract model name
            if "network" in line and "{" in line:
                model_name = line.split("network")[1].split("{")[0].strip()
            
            # Extract input shape
            if "input:" in line:
                shape_str = line.split("input:")[1].strip()
                if "(" in shape_str and ")" in shape_str:
                    shape_parts = shape_str.strip("()").split(",")
                    input_shape = [int(p.strip()) for p in shape_parts]
            
            # Extract layers
            if "(" in line and ")" in line and not "input:" in line and not line.startswith("#"):
                layer_type = line.split("(")[0].strip()
                if layer_type:
                    layers.append({
                        "type": layer_type,
                        "params": {"mock": True}
                    })
        
        # Generate mock shape history
        shape_history = []
        current_shape = input_shape
        
        for i, layer in enumerate(layers):
            layer_type = layer["type"]
            output_shape = list(current_shape)  # Copy current shape
            
            # Simulate shape changes based on layer type
            if layer_type in ["Conv2D", "MaxPooling2D"]:
                # Reduce spatial dimensions
                if len(output_shape) >= 3:
                    output_shape[-3] = max(1, output_shape[-3] // 2)
                    output_shape[-2] = max(1, output_shape[-2] // 2)
            elif layer_type == "Dense":
                # Change to 1D
                output_shape = [output_shape[0], 10]
            elif layer_type == "Flatten":
                # Flatten to 1D
                if len(output_shape) >= 3:
                    flat_size = output_shape[-3] * output_shape[-2] * output_shape[-1]
                    output_shape = [output_shape[0], flat_size]
            
            shape_history.append({
                "layer_id": f"layer_{i}",
                "layer_type": layer_type,
                "input_shape": current_shape,
                "output_shape": output_shape
            })
            
            current_shape = output_shape
        
        return {
            "model_data": {
                "name": model_name,
                "input": {"shape": input_shape},
                "layers": layers
            },
            "shape_history": shape_history,
            "trace_data": {"mock": True}
        }
    
    def generate_code(self, dsl_code: str, backend: str = 'tensorflow') -> str:
        """
        Generate code from Neural DSL (mock implementation)
        
        Args:
            dsl_code: Neural DSL code
            backend: Target backend (tensorflow or pytorch)
            
        Returns:
            Generated code
        """
        logger.info(f"Generating {backend} code")
        
        if backend == "tensorflow":
            return self._generate_tensorflow_code(dsl_code)
        else:
            return self._generate_pytorch_code(dsl_code)
    
    def _generate_tensorflow_code(self, dsl_code: str) -> str:
        """Generate TensorFlow code from DSL"""
        model_info = self.parse_dsl(dsl_code, "tensorflow")
        model_name = model_info["model_data"]["name"]
        
        code = f"""
import tensorflow as tf
from tensorflow.keras import layers, models

def create_{model_name.lower()}():
    \"\"\"
    Create {model_name} model
    Generated from Neural DSL
    \"\"\"
    model = models.Sequential()
    
    # Input layer
    model.add(layers.InputLayer(input_shape={model_info["model_data"]["input"]["shape"][1:]}))
    
"""
        
        # Add layers
        for i, layer in enumerate(model_info["model_data"]["layers"]):
            layer_type = layer["type"]
            if layer_type == "Conv2D":
                code += f"    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))\n"
            elif layer_type == "MaxPooling2D":
                code += f"    model.add(layers.MaxPooling2D((2, 2)))\n"
            elif layer_type == "Dense":
                code += f"    model.add(layers.Dense(10, activation='softmax'))\n"
            elif layer_type == "Flatten":
                code += f"    model.add(layers.Flatten())\n"
            elif layer_type == "Dropout":
                code += f"    model.add(layers.Dropout(0.5))\n"
            else:
                code += f"    # Unsupported layer: {layer_type}\n"
        
        code += f"""
    return model

# Create model
model = create_{model_name.lower()}()

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Print summary
model.summary()
"""
        
        return code
    
    def _generate_pytorch_code(self, dsl_code: str) -> str:
        """Generate PyTorch code from DSL"""
        model_info = self.parse_dsl(dsl_code, "pytorch")
        model_name = model_info["model_data"]["name"]
        
        code = f"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class {model_name}(nn.Module):
    \"\"\"
    {model_name} model
    Generated from Neural DSL
    \"\"\"
    def __init__(self):
        super({model_name}, self).__init__()
        
"""
        
        # Add layers
        in_channels = model_info["model_data"]["input"]["shape"][-1]
        for i, layer in enumerate(model_info["model_data"]["layers"]):
            layer_type = layer["type"]
            if layer_type == "Conv2D":
                out_channels = 32
                code += f"        self.conv{i+1} = nn.Conv2d({in_channels}, {out_channels}, kernel_size=3, padding=1)\n"
                in_channels = out_channels
            elif layer_type == "MaxPooling2D":
                code += f"        self.pool{i+1} = nn.MaxPool2d(2)\n"
            elif layer_type == "Dense":
                code += f"        self.fc{i+1} = nn.Linear(in_features=512, out_features=10)\n"
            elif layer_type == "Flatten":
                code += f"        self.flatten = nn.Flatten()\n"
            elif layer_type == "Dropout":
                code += f"        self.dropout{i+1} = nn.Dropout(0.5)\n"
            else:
                code += f"        # Unsupported layer: {layer_type}\n"
        
        code += f"""
    def forward(self, x):
"""
        
        # Add forward pass
        for i, layer in enumerate(model_info["model_data"]["layers"]):
            layer_type = layer["type"]
            if layer_type == "Conv2D":
                code += f"        x = F.relu(self.conv{i+1}(x))\n"
            elif layer_type == "MaxPooling2D":
                code += f"        x = self.pool{i+1}(x)\n"
            elif layer_type == "Dense":
                code += f"        x = self.fc{i+1}(x)\n"
            elif layer_type == "Flatten":
                code += f"        x = self.flatten(x)\n"
            elif layer_type == "Dropout":
                code += f"        x = self.dropout{i+1}(x)\n"
            else:
                code += f"        # Unsupported layer: {layer_type}\n"
        
        code += f"""        return F.log_softmax(x, dim=1)

# Create model
model = {model_name}()

# Print model
print(model)
"""
        
        return code
    
    def start_debug_session(self, dsl_code: str, backend: str = 'tensorflow') -> Dict[str, Any]:
        """
        Start a mock debug session
        
        Args:
            dsl_code: Neural DSL code
            backend: Target backend (tensorflow or pytorch)
            
        Returns:
            Dictionary with session_id and dashboard_url
        """
        logger.info(f"Starting mock debug session for backend: {backend}")
        
        return {
            "session_id": f"mock_debug_session",
            "dashboard_url": "http://localhost:8050",
            "process_id": 12345
        }
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models
        
        Returns:
            List of model metadata
        """
        models = []
        
        try:
            for file in os.listdir(self.models_dir):
                if file.endswith('.neural'):
                    model_id = os.path.splitext(file)[0]
                    annotations_path = os.path.join(self.models_dir, f"{model_id}.annotations.json")
                    
                    model_info = {
                        "id": model_id,
                        "name": model_id.capitalize(),
                        "has_annotations": os.path.exists(annotations_path)
                    }
                    
                    if os.path.exists(annotations_path):
                        with open(annotations_path, 'r') as f:
                            annotations = json.load(f)
                            model_info["name"] = annotations.get("name", model_info["name"])
                            model_info["description"] = annotations.get("description", "")
                    
                    models.append(model_info)
        except Exception as e:
            logger.error(f"Error listing models: {e}")
        
        return models
