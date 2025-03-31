import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from neural.shape_propagation.shape_propagator import ShapePropagator


class TestShapeVisualization:
    
    @pytest.fixture
    def propagator(self):
        """Create a ShapePropagator instance for testing."""
        return ShapePropagator(debug=False)
    
    @pytest.fixture
    def sample_layers(self):
        """Sample layers for testing."""
        return [
            {
                "type": "Conv2D",
                "params": {
                    "filters": 32,
                    "kernel_size": (3, 3),
                    "padding": "valid",
                    "stride": 1
                }
            },
            {
                "type": "MaxPooling2D",
                "params": {
                    "pool_size": (2, 2),
                    "padding": "valid",
                    "stride": 2
                }
            },
            {
                "type": "Flatten",
                "params": {}
            },
            {
                "type": "Dense",
                "params": {
                    "units": 128
                }
            },
            {
                "type": "Output",
                "params": {
                    "units": 10
                }
            }
        ]
    
    def test_visualize_layer(self, propagator):
        """Test the _visualize_layer method."""
        # Initial state
        assert propagator.current_layer == 0
        assert len(propagator.shape_history) == 0
        
        # Call the method
        layer_name = "Conv2D"
        shape = (1, 26, 26, 32)
        propagator._visualize_layer(layer_name, shape)
        
        # Check that the current_layer was incremented
        assert propagator.current_layer == 1
        
        # Check that the shape_history was updated
        assert len(propagator.shape_history) == 1
        assert propagator.shape_history[0] == (layer_name, shape)
    
    def test_create_connection(self, propagator):
        """Test the _create_connection method."""
        # Initial state
        assert len(propagator.layer_connections) == 0
        
        # Call the method
        from_id = 0
        to_id = 1
        propagator._create_connection(from_id, to_id)
        
        # Check that the layer_connections was updated
        assert len(propagator.layer_connections) == 1
        assert propagator.layer_connections[0] == (from_id, to_id)
    
    @patch('plotly.graph_objects.Figure')
    @patch('graphviz.Digraph')
    def test_generate_report(self, mock_digraph, mock_figure, propagator):
        """Test the generate_report method."""
        # Setup mock objects
        mock_dot = MagicMock()
        mock_digraph.return_value = mock_dot
        propagator.dot = mock_dot
        
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        # Add some shape history
        propagator.shape_history = [
            ('Input', (1, 28, 28, 3)),
            ('Conv2D', (1, 26, 26, 32)),
            ('MaxPooling2D', (1, 13, 13, 32)),
            ('Flatten', (1, 5408)),
            ('Dense', (1, 128)),
            ('Output', (1, 10))
        ]
        
        # Call the method
        report = propagator.generate_report()
        
        # Check that the report has the expected keys
        assert 'dot_graph' in report
        assert 'plotly_chart' in report
        assert 'shape_history' in report
        
        # Check that the dot_graph is the mock object
        assert report['dot_graph'] == mock_dot
        
        # Check that the plotly_chart is the mock object
        assert report['plotly_chart'] == mock_fig
        
        # Check that the shape_history is correct
        assert report['shape_history'] == propagator.shape_history
        
        # Check that add_trace was called
        mock_fig.add_trace.assert_called_once()
        
        # Check that update_layout was called
        mock_fig.update_layout.assert_called_once()
    
    def test_propagate_visualization(self, propagator, sample_layers):
        """Test that propagate correctly visualizes layers."""
        # Initial state
        assert propagator.current_layer == 0
        assert len(propagator.shape_history) == 0
        assert len(propagator.layer_connections) == 0
        
        # Propagate through the layers
        input_shape = (1, 28, 28, 3)
        shape = input_shape
        for layer in sample_layers:
            shape = propagator.propagate(shape, layer, framework="tensorflow")
        
        # Check that the shape_history has the correct number of entries
        assert len(propagator.shape_history) == len(sample_layers)
        
        # Check that the layer_connections has the correct number of entries
        # There should be len(sample_layers) - 1 connections
        assert len(propagator.layer_connections) == len(sample_layers) - 1
        
        # Check that the current_layer is correct
        assert propagator.current_layer == len(sample_layers)
        
        # Check that the final shape is correct
        assert shape == (1, 10)
    
    @patch('json.dumps')
    def test_get_shape_data(self, mock_dumps, propagator):
        """Test the get_shape_data method."""
        # Setup mock
        mock_dumps.return_value = '{"shape_data": "test"}'
        
        # Add some shape history
        propagator.shape_history = [
            ('Input', (1, 28, 28, 3)),
            ('Conv2D', (1, 26, 26, 32))
        ]
        
        # Call the method
        result = propagator.get_shape_data()
        
        # Check that json.dumps was called with the correct data
        expected_data = [
            {"layer": "Input", "output_shape": (1, 28, 28, 3)},
            {"layer": "Conv2D", "output_shape": (1, 26, 26, 32)}
        ]
        mock_dumps.assert_called_once()
        args, _ = mock_dumps.call_args
        assert args[0] == expected_data
        
        # Check that the result is the mock return value
        assert result == '{"shape_data": "test"}'


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
