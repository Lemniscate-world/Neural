import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add Neural to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Import Neural components
from neural.parser.parser import create_parser, ModelTransformer
from neural.code_generation.code_generator import generate_code
from neural.shape_propagation.shape_propagator import ShapePropagator

class NeuralConnector:
    """
    Connector class for integrating Neural DSL and NeuralDbg with NeuralPaper.ai
    """
    
    def __init__(self, models_dir: str = None):
        """
        Initialize the connector
        
        Args:
            models_dir: Directory containing model files
        """
        self.models_dir = models_dir or os.path.join(os.path.dirname(__file__), "../models")
        self.parser = create_parser('network')
        self.transformer = ModelTransformer()
        self.propagator = ShapePropagator(debug=False)
    
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
        
        return dsl_code, annotations
    
    def parse_dsl(self, dsl_code: str, backend: str = 'tensorflow') -> Dict[str, Any]:
        """
        Parse Neural DSL code
        
        Args:
            dsl_code: Neural DSL code
            backend: Target backend (tensorflow or pytorch)
            
        Returns:
            Dictionary with model_data, shape_history, and trace_data
        """
        tree = self.parser.parse(dsl_code)
        model_data = self.transformer.transform(tree)
        
        # Generate shape propagation data
        self.propagator = ShapePropagator(debug=False)
        input_shape = model_data['input']['shape']
        shape_history = []
        
        for i, layer in enumerate(model_data['layers']):
            output_shape = self.propagator.propagate(input_shape, layer, backend)
            shape_history.append({
                "layer_id": f"layer_{i}",
                "layer_type": layer['type'],
                "input_shape": input_shape,
                "output_shape": output_shape
            })
            input_shape = output_shape
        
        return {
            "model_data": model_data,
            "shape_history": shape_history,
            "trace_data": self.propagator.get_trace()
        }
    
    def generate_code(self, dsl_code: str, backend: str = 'tensorflow') -> str:
        """
        Generate code from Neural DSL
        
        Args:
            dsl_code: Neural DSL code
            backend: Target backend (tensorflow or pytorch)
            
        Returns:
            Generated code
        """
        tree = self.parser.parse(dsl_code)
        model_data = self.transformer.transform(tree)
        return generate_code(model_data, backend)
    
    def start_debug_session(self, dsl_code: str, backend: str = 'tensorflow') -> Dict[str, Any]:
        """
        Start a NeuralDbg debug session
        
        Args:
            dsl_code: Neural DSL code
            backend: Target backend (tensorflow or pytorch)
            
        Returns:
            Dictionary with session_id and dashboard_url
        """
        # Save DSL to temporary file
        temp_file = Path("temp_model.neural")
        temp_file.write_text(dsl_code)
        
        # Start NeuralDbg in background
        cmd = ["python", "-m", "neural.cli", "debug", str(temp_file), "--backend", backend]
        process = subprocess.Popen(cmd)
        
        return {
            "session_id": f"debug_{process.pid}",
            "dashboard_url": "http://localhost:8050",
            "process_id": process.pid
        }
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models
        
        Returns:
            List of model metadata
        """
        models = []
        
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
        
        return models
