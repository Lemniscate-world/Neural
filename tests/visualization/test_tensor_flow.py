import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from neural.dashboard.tensor_flow import create_animated_network


class TestTensorFlow:
    
    @pytest.fixture
    def sample_layer_data(self):
        """Sample layer data for testing."""
        return [
            {"layer": "Input", "output_shape": (1, 28, 28, 3)},
            {"layer": "Conv2D", "output_shape": (1, 26, 26, 32)},
            {"layer": "MaxPooling2D", "output_shape": (1, 13, 13, 32)},
            {"layer": "Flatten", "output_shape": (1, 5408)},
            {"layer": "Dense", "output_shape": (1, 128)},
            {"layer": "Output", "output_shape": (1, 10)}
        ]
    
    @patch('networkx.DiGraph')
    @patch('networkx.drawing.nx_agraph.graphviz_layout')
    @patch('plotly.graph_objects.Figure')
    def test_create_animated_network(self, mock_figure, mock_graphviz_layout, mock_digraph, sample_layer_data):
        """Test creating an animated network visualization."""
        # Setup mocks
        mock_graph = MagicMock()
        mock_digraph.return_value = mock_graph
        
        mock_pos = {
            "Input": (0, 0),
            "Conv2D": (1, 0),
            "MaxPooling2D": (2, 0),
            "Flatten": (3, 0),
            "Dense": (4, 0),
            "Output": (5, 0)
        }
        mock_graphviz_layout.return_value = mock_pos
        
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        # Call the function
        result = create_animated_network(sample_layer_data)
        
        # Check that the DiGraph was created
        mock_digraph.assert_called_once()
        
        # Check that nodes were added to the graph
        assert mock_graph.add_node.call_count == len(sample_layer_data)
        
        # Check that edges were added to the graph
        # There should be len(sample_layer_data) - 1 edges
        assert mock_graph.add_edge.call_count == len(sample_layer_data) - 1
        
        # Check that graphviz_layout was called with the graph
        mock_graphviz_layout.assert_called_once_with(mock_graph, prog="dot", args="-Grankdir=TB")
        
        # Check that the Figure was created
        mock_figure.assert_called_once()
        
        # Check that add_trace was called twice (once for edges, once for nodes)
        assert mock_fig.add_trace.call_count == 2
        
        # Check that update_layout was called
        mock_fig.update_layout.assert_called_once()
        
        # Check that the result is the mock figure
        assert result == mock_fig
    
    def test_create_animated_network_empty_data(self):
        """Test creating an animated network with empty data."""
        # Call the function with empty data
        result = create_animated_network([])
        
        # Check that a Figure was returned
        assert result is not None
        
        # Call the function with None
        result = create_animated_network(None)
        
        # Check that a Figure was returned
        assert result is not None


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
