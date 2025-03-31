import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from neural.visualization.static_visualizer.visualizer import NeuralVisualizer


class TestNeuralVisualizer:
    
    @pytest.fixture
    def sample_model_data(self):
        """Sample model data for testing."""
        return {
            'input': {
                'shape': (None, 28, 28, 1)
            },
            'layers': [
                {
                    'type': 'Conv2D',
                    'params': {
                        'filters': 32,
                        'kernel_size': (3, 3),
                        'activation': 'relu'
                    }
                },
                {
                    'type': 'MaxPooling2D',
                    'params': {
                        'pool_size': (2, 2)
                    }
                },
                {
                    'type': 'Flatten',
                    'params': {}
                },
                {
                    'type': 'Dense',
                    'params': {
                        'units': 128,
                        'activation': 'relu'
                    }
                }
            ],
            'output_layer': {
                'type': 'Output',
                'params': {
                    'units': 10,
                    'activation': 'softmax'
                }
            }
        }
    
    def test_init(self, sample_model_data):
        """Test initializing the NeuralVisualizer."""
        visualizer = NeuralVisualizer(sample_model_data)
        assert visualizer.model_data == sample_model_data
        assert visualizer.figures == []
    
    def test_model_to_d3_json(self, sample_model_data):
        """Test converting model data to D3 visualization format."""
        visualizer = NeuralVisualizer(sample_model_data)
        d3_data = visualizer.model_to_d3_json()
        
        # Check that we have the correct structure
        assert 'nodes' in d3_data
        assert 'links' in d3_data
        
        # Check that we have the correct number of nodes
        # Input + 4 layers + output = 6 nodes
        assert len(d3_data['nodes']) == 6
        
        # Check that we have the correct number of links
        # 5 connections between 6 nodes
        assert len(d3_data['links']) == 5
        
        # Check that the input node is correct
        input_node = d3_data['nodes'][0]
        assert input_node['id'] == 'input'
        assert input_node['type'] == 'Input'
        assert input_node['shape'] == (None, 28, 28, 1)
        
        # Check that the output node is correct
        output_node = d3_data['nodes'][-1]
        assert output_node['id'] == 'output'
        assert output_node['type'] == 'Output'
        assert 'units' in output_node['params']
        assert output_node['params']['units'] == 10
        
        # Check that the links are correct
        # First link should be from input to layer1
        first_link = d3_data['links'][0]
        assert first_link['source'] == 'input'
        assert first_link['target'] == 'layer1'
        
        # Last link should be from layer4 to output
        last_link = d3_data['links'][-1]
        assert last_link['source'] == 'layer4'
        assert last_link['target'] == 'output'
    
    @patch('plotly.graph_objects.Figure')
    def test_create_3d_visualization(self, mock_figure, sample_model_data):
        """Test creating a 3D visualization of the model."""
        visualizer = NeuralVisualizer(sample_model_data)
        
        # Create a sample shape history
        shape_history = [
            ('Input', (None, 28, 28, 1)),
            ('Conv2D', (None, 26, 26, 32)),
            ('MaxPooling2D', (None, 13, 13, 32)),
            ('Flatten', (None, 5408)),
            ('Dense', (None, 128)),
            ('Output', (None, 10))
        ]
        
        # Mock the Figure.add_trace method
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        # Call the method
        result = visualizer.create_3d_visualization(shape_history)
        
        # Check that the Figure was created
        mock_figure.assert_called_once()
        
        # Check that add_trace was called for each shape in the history
        assert mock_fig.add_trace.call_count == len(shape_history)
        
        # Check that update_layout was called
        mock_fig.update_layout.assert_called_once()
        
        # Check that the result is the mock figure
        assert result == mock_fig


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
