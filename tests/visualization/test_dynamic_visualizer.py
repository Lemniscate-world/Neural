import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from neural.visualization.dynamic_visualizer.api import app


class TestDynamicVisualizer:
    
    @pytest.fixture
    def client(self):
        """Create a test client for the Flask app."""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    @pytest.fixture
    def sample_neural_code(self):
        """Sample Neural DSL code for testing."""
        return """
        network TestNet {
            input: (None, 28, 28, 1)
            layers:
                Conv2D(filters=32, kernel_size=(3,3), activation="relu")
                MaxPooling2D(pool_size=(2,2))
                Flatten()
                Dense(128, "relu")
                Output(10, "softmax")
            loss: "categorical_crossentropy"
            optimizer: "adam"
        }
        """
    
    @pytest.fixture
    def sample_visualization_data(self):
        """Sample visualization data for testing."""
        return {
            "nodes": [
                {"id": "input", "type": "Input", "shape": [None, 28, 28, 1]},
                {"id": "layer1", "type": "Conv2D", "params": {"filters": 32, "kernel_size": [3, 3], "activation": "relu"}},
                {"id": "layer2", "type": "MaxPooling2D", "params": {"pool_size": [2, 2]}},
                {"id": "layer3", "type": "Flatten", "params": {}},
                {"id": "layer4", "type": "Dense", "params": {"units": 128, "activation": "relu"}},
                {"id": "output", "type": "Output", "params": {"units": 10, "activation": "softmax"}}
            ],
            "links": [
                {"source": "input", "target": "layer1"},
                {"source": "layer1", "target": "layer2"},
                {"source": "layer2", "target": "layer3"},
                {"source": "layer3", "target": "layer4"},
                {"source": "layer4", "target": "output"}
            ]
        }
    
    @patch('neural.visualization.dynamic_visualizer.api.create_parser')
    @patch('neural.visualization.dynamic_visualizer.api.ModelTransformer')
    @patch('neural.visualization.dynamic_visualizer.api.NeuralVisualizer')
    def test_parse_network_success(self, mock_neural_visualizer, mock_model_transformer, 
                                  mock_create_parser, client, sample_neural_code, sample_visualization_data):
        """Test the /parse endpoint with successful parsing."""
        # Setup mocks
        mock_parser = MagicMock()
        mock_create_parser.return_value = mock_parser
        
        mock_tree = MagicMock()
        mock_parser.parse.return_value = mock_tree
        
        mock_transformer = MagicMock()
        mock_model_transformer.return_value = mock_transformer
        
        mock_model_data = MagicMock()
        mock_transformer.transform.return_value = mock_model_data
        
        mock_visualizer = MagicMock()
        mock_neural_visualizer.return_value = mock_visualizer
        
        mock_visualizer.model_to_d3_json.return_value = sample_visualization_data
        
        # Make the request
        response = client.post('/parse', data=sample_neural_code)
        
        # Check that the response is successful
        assert response.status_code == 200
        
        # Check that the response data is correct
        response_data = json.loads(response.data)
        assert response_data == sample_visualization_data
        
        # Check that the parser was created with the correct start rule
        mock_create_parser.assert_called_once_with('network')
        
        # Check that the parser was called with the code
        mock_parser.parse.assert_called_once_with(sample_neural_code)
        
        # Check that the transformer was called with the parse tree
        mock_transformer.transform.assert_called_once_with(mock_tree)
        
        # Check that the visualizer was created with the model data
        mock_neural_visualizer.assert_called_once_with(mock_model_data)
        
        # Check that model_to_d3_json was called
        mock_visualizer.model_to_d3_json.assert_called_once()
    
    @patch('neural.visualization.dynamic_visualizer.api.create_parser')
    def test_parse_network_parsing_error(self, mock_create_parser, client, sample_neural_code):
        """Test the /parse endpoint with a parsing error."""
        # Setup mocks
        mock_parser = MagicMock()
        mock_create_parser.return_value = mock_parser
        
        # Make the parser raise an exception
        mock_parser.parse.side_effect = Exception("Parsing error")
        
        # Make the request
        response = client.post('/parse', data=sample_neural_code)
        
        # Check that the response is an error
        assert response.status_code == 400
        
        # Check that the response data contains the error message
        response_data = json.loads(response.data)
        assert 'error' in response_data
        assert 'Parsing error' in response_data['error']
        
        # Check that the parser was created with the correct start rule
        mock_create_parser.assert_called_once_with('network')
        
        # Check that the parser was called with the code
        mock_parser.parse.assert_called_once_with(sample_neural_code)
    
    def test_parse_network_empty_request(self, client):
        """Test the /parse endpoint with an empty request."""
        # Make the request with empty data
        response = client.post('/parse', data='')
        
        # Check that the response is an error
        assert response.status_code == 400
        
        # Check that the response data contains an error message
        response_data = json.loads(response.data)
        assert 'error' in response_data


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
