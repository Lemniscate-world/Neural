from visualizer import NeuralVisualizer

# Add absolute path to the parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add neural visualizer path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "neural"))

# Test the NeuralVisualizer class


def test_neural_visualizer_basic():
    """Test standard model with input, multiple layers, and output."""
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
    
    # Nodes: input + 3 layers + output = 5 nodes
    assert len(result["nodes"]) == 5
    # Links: 3 links between input/layers and one final link = 4 links
    assert len(result["links"]) == 4
    # Vérification du type des noeuds
    assert result["nodes"][0]["type"] == "Input"
    assert result["nodes"][-1]["type"] == "Output"
    # Vérifier que le lien final va bien vers 'output'
    assert result["links"][-1]["target"] == "output"
    # Vérifier la séquence des liens pour les couches
    expected_sources = ["input", "layer1", "layer2"]
    for idx, link in enumerate(result["links"][:-1]):
        assert link["source"] == expected_sources[idx]
        assert link["target"] == f"layer{idx+1}"
        
def test_neural_visualizer_no_layers():
    """Test le cas où aucune couche cachée n'est fournie."""
    mock_model_data = {
        "input": {"shape": (28, 28, 1)},
        "layers": [],
        "output_layer": {"type": "Output", "params": {"units": 10, "activation": "softmax"}}
    }
    visualizer = NeuralVisualizer(mock_model_data)
    result = visualizer.model_to_d3_json()
    
    # Seulement input et output doivent être présents
    assert len(result["nodes"]) == 2
    # Pas de liens créés si aucune couche n'est présente
    assert len(result["links"]) == 0

def test_neural_visualizer_missing_input():
    """Test lorsque la clé 'input' est absente du modèle."""
    mock_model_data = {
        "layers": [
            {"type": "Dense", "params": {"units": 64, "activation": "relu"}}
        ],
        "output_layer": {"type": "Output", "params": {"units": 10, "activation": "softmax"}}
    }
    visualizer = NeuralVisualizer(mock_model_data)
    result = visualizer.model_to_d3_json()
    
    # Même si 'input' manque, un nœud d'input par défaut doit être créé
    assert result["nodes"][0]["type"] == "Input"
    assert result["nodes"][0]["shape"] is None
    # Devrait y avoir 3 nœuds: input, une couche, output
    assert len(result["nodes"]) == 3
    # Et 2 liens : de input à la couche, puis de la couche à output
    assert len(result["links"]) == 2

def test_neural_visualizer_empty_model_data():
    """Test avec un dictionnaire de modèle vide."""
    mock_model_data = {}
    visualizer = NeuralVisualizer(mock_model_data)
    result = visualizer.model_to_d3_json()
    
    # Devrait créer des nœuds par défaut pour input et output
    assert len(result["nodes"]) == 2
    # Aucun lien ne doit être créé
    assert len(result["links"]) == 0
    # Vérification des types par défaut
    assert result["nodes"][0]["type"] == "Input"
    assert result["nodes"][-1]["type"] == "Output"
    
def test_neural_visualizer_unknown_layer():
    """Test lorsqu'une couche ne fournit pas les clés 'type' et 'params'."""
    mock_model_data = {
        "input": {"shape": (32, 32, 3)},
        "layers": [
            {}  # La couche doit par défaut être 'Unknown' et params être {}.
        ],
        "output_layer": {"type": "Output", "params": {"units": 10, "activation": "softmax"}}
    }
    visualizer = NeuralVisualizer(mock_model_data)
    result = visualizer.model_to_d3_json()
    
    # Vérifier que la couche par défaut a le type 'Unknown' et des paramètres vides
    assert result["nodes"][1]["type"] == "Unknown"
    assert result["nodes"][1].get("params", {}) == {}
    # Le nombre total de nœuds doit être de 3 (input, layer inconnue, output)
    assert len(result["nodes"]) == 3
    # Le nombre de liens doit être 2 (input->layer, layer->output)
    assert len(result["links"]) == 2
