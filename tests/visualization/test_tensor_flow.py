import os
import sys
import pytest
from unittest.mock import patch, MagicMock
import plotly.graph_objects as go
import networkx as nx

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from neural.dashboard.tensor_flow import create_animated_network, create_layer_computation_timeline


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
    
    @pytest.fixture
    def complex_layer_data(self):
        """Complex layer data with branching architecture."""
        return [
            {"layer": "Input", "output_shape": (1, 224, 224, 3)},
            {"layer": "Conv2D_1", "output_shape": (1, 112, 112, 64)},
            {"layer": "Conv2D_2a", "output_shape": (1, 56, 56, 128)},
            {"layer": "Conv2D_2b", "output_shape": (1, 56, 56, 128)},
            {"layer": "Concat", "output_shape": (1, 56, 56, 256)},
            {"layer": "Dense", "output_shape": (1, 1000)},
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
        assert isinstance(result, go.Figure)
        
        # Call the function with None
        result = create_animated_network(None)
        
        # Check that a Figure was returned
        assert result is not None
        assert isinstance(result, go.Figure)
    
    @patch('networkx.drawing.nx_agraph.graphviz_layout')
    def test_create_animated_network_integration(self, mock_graphviz_layout, sample_layer_data):
        """Integration test with real NetworkX and Plotly objects."""
        # Mock only the graphviz layout to avoid requiring graphviz installation
        mock_pos = {layer["layer"]: (i, 0) for i, layer in enumerate(sample_layer_data)}
        mock_graphviz_layout.return_value = mock_pos
        
        # Call the function with real data
        result = create_animated_network(sample_layer_data)
        
        # Verify the result
        assert isinstance(result, go.Figure)
        assert len(result.data) == 2  # One trace for edges, one for nodes
        
        # Check that all layers are represented in the visualization
        node_trace = result.data[1]
        assert len(node_trace.text) == len(sample_layer_data)
        for layer in sample_layer_data:
            assert layer["layer"] in node_trace.text
    
    @patch('networkx.drawing.nx_agraph.graphviz_layout')
    def test_node_attributes(self, mock_graphviz_layout, sample_layer_data):
        """Test that node attributes are correctly set."""
        # Mock the graphviz layout
        mock_pos = {layer["layer"]: (i, 0) for i, layer in enumerate(sample_layer_data)}
        mock_graphviz_layout.return_value = mock_pos
        
        # Create a real graph to inspect node attributes
        with patch.object(nx, 'DiGraph', wraps=nx.DiGraph) as wrapped_digraph:
            result = create_animated_network(sample_layer_data)
            graph = wrapped_digraph.call_args[0][0] if wrapped_digraph.call_args else wrapped_digraph.return_value
            
            # Check that each node has the correct output_shape attribute
            for layer in sample_layer_data:
                node_name = layer["layer"]
                if hasattr(graph, 'nodes') and callable(graph.nodes) and node_name in graph.nodes:
                    assert graph.nodes[node_name].get('output_shape') == layer["output_shape"]
    
    @patch('networkx.drawing.nx_agraph.graphviz_layout')
    def test_complex_network_structure(self, mock_graphviz_layout, complex_layer_data):
        """Test visualization of a more complex network structure."""
        # Mock the graphviz layout
        mock_pos = {layer["layer"]: (i, 0) for i, layer in enumerate(complex_layer_data)}
        mock_graphviz_layout.return_value = mock_pos
        
        # Call the function with complex data
        result = create_animated_network(complex_layer_data)
        
        # Verify the result
        assert isinstance(result, go.Figure)
        
        # Check that all layers are represented
        node_trace = result.data[1]
        for layer in complex_layer_data:
            assert layer["layer"] in node_trace.text
        
        # Check edge count (should be number of layers - 1 for sequential networks)
        edge_trace = result.data[0]
        # Count non-None x values and divide by 3 (each edge has 3 points: start, end, None)
        edge_count = sum(1 for x in edge_trace.x if x is not None) / 2
        assert edge_count == len(complex_layer_data) - 1

    @patch('time.sleep')  # Patch sleep to speed up tests
    @patch('networkx.drawing.nx_agraph.graphviz_layout')
    def test_create_animated_network_with_progress(self, mock_graphviz_layout, mock_sleep, sample_layer_data):
        """Test that progress reporting works correctly."""
        # Mock the graphviz layout
        mock_pos = {layer["layer"]: (i, 0) for i, layer in enumerate(sample_layer_data)}
        mock_graphviz_layout.return_value = mock_pos
        
        # Capture print output
        with patch('builtins.print') as mock_print:
            # Call the function with progress enabled
            result = create_animated_network(sample_layer_data, show_progress=True)
            
            # Verify progress was printed
            assert mock_print.call_count > 0
            
            # Check for progress percentage in prints
            progress_prints = [call[0][0] for call in mock_print.call_args_list 
                              if isinstance(call[0][0], str) and "%" in call[0][0]]
            assert len(progress_prints) > 0
            
            # Check for 100% completion
            assert any("100" in p for p in progress_prints)
        
        # Verify the result is still correct
        assert isinstance(result, go.Figure)
        
        # Check that node sizes and colors are set
        node_trace = result.data[1]
        assert len(node_trace.marker.size) == len(sample_layer_data)
        assert len(node_trace.marker.color) == len(sample_layer_data)
        
        # Check that hover texts are created
        assert all(isinstance(text, str) and "Shape:" in text for text in node_trace.hovertext)

    @pytest.mark.parametrize("show_progress", [True, False])
    def test_create_animated_network_progress_parameter(self, sample_layer_data, show_progress):
        """Test that the show_progress parameter works correctly."""
        with patch('builtins.print') as mock_print:
            result = create_animated_network(sample_layer_data, show_progress=show_progress)
            
            # If show_progress is True, print should be called; otherwise, it shouldn't
            if show_progress:
                assert mock_print.call_count > 0
            else:
                assert mock_print.call_count == 0
            
        # Result should be a figure regardless of progress setting
        assert isinstance(result, go.Figure)

    def test_create_layer_computation_timeline(self, sample_layer_data):
        """Test creating a computation timeline visualization."""
        # Add execution times to sample data
        for i, layer in enumerate(sample_layer_data):
            layer["execution_time"] = 0.1 * (i + 1)
        
        # Create the timeline
        result = create_layer_computation_timeline(sample_layer_data)
        
        # Verify the result
        assert isinstance(result, go.Figure)
        
        # Check that all layers are included
        assert len(result.data) > 0


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
