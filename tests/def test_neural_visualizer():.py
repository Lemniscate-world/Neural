from visualizer import NeuralVisualizer

# Add absolute path to the parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add neural visualizer path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "neural"))

# Test the NeuralVisualizer class

def test_neural_visualizer():
    mock_model_data = {
        "input": {"shape": (28, 28, 1)},
        "layers": [
            {"type": "Conv2D", "params": {"filters": 32, "kernel_size": (3,3), "activation": "relu"}},
            {"type": "MaxPooling2D", "params": {"pool_size": (2,2)}},
            {"type": "Dense", "params": {"units": 128, "activation": "relu"}}
        ],
        "output_layer": {"type": "Output", "params": {"units": 10, "activation": "softmax"}}
    }
    
    visualizer = NeuralVisualizer(mock_model_data)
    result = visualizer.model_to_d3_json()
    
    assert len(result["nodes"]) == 5  # Input + 3 hidden layers + output
    assert len(result["links"]) == 4  # Connections between layers
    assert result["nodes"][0]["type"] == "Input"
    assert result["nodes"][-1]["type"] == "Output"
    assert result["links"][-1]["target"] == "output"
