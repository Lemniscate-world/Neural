#!/usr/bin/env python3
"""
Initialize the database with sample models.
This script copies the sample models from the models directory to the database.
"""

import os
import sys
import json
import shutil
from pathlib import Path

# Add project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Import Neural connector
from neuralpaper.backend.integrations.neural_connector import NeuralConnector

# Sample model metadata
SAMPLE_MODELS = [
    {
        "id": "resnet",
        "name": "ResNet-18",
        "description": "Deep Residual Network with skip connections for image classification",
        "category": "Computer Vision",
        "complexity": "Medium",
    },
    {
        "id": "transformer",
        "name": "Transformer",
        "description": "Attention-based sequence model for NLP tasks",
        "category": "Natural Language Processing",
        "complexity": "High",
    },
]

def main():
    """Initialize the database with sample models."""
    print("Initializing database with sample models...")
    
    # Create models directory if it doesn't exist
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
    os.makedirs(models_dir, exist_ok=True)
    
    # Copy sample models
    for model in SAMPLE_MODELS:
        model_id = model["id"]
        print(f"Processing model: {model_id}")
        
        # Check if model files exist
        neural_file = os.path.join(models_dir, f"{model_id}.neural")
        annotations_file = os.path.join(models_dir, f"{model_id}.annotations.json")
        
        if not os.path.exists(neural_file):
            print(f"Warning: Model file not found: {neural_file}")
            continue
        
        if not os.path.exists(annotations_file):
            print(f"Warning: Annotations file not found: {annotations_file}")
            
            # Create empty annotations file
            with open(annotations_file, "w") as f:
                json.dump({
                    "name": model["name"],
                    "description": model["description"],
                    "category": model.get("category", "Uncategorized"),
                    "complexity": model.get("complexity", "Medium"),
                    "sections": []
                }, f, indent=2)
            
            print(f"Created empty annotations file: {annotations_file}")
        else:
            # Update annotations file with metadata
            with open(annotations_file, "r") as f:
                annotations = json.load(f)
            
            # Update metadata
            annotations["name"] = model["name"]
            annotations["description"] = model["description"]
            annotations["category"] = model.get("category", "Uncategorized")
            annotations["complexity"] = model.get("complexity", "Medium")
            
            # Write updated annotations
            with open(annotations_file, "w") as f:
                json.dump(annotations, f, indent=2)
            
            print(f"Updated annotations file: {annotations_file}")
    
    print("Database initialization complete.")

if __name__ == "__main__":
    main()
