import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from neural.dashboard.dashboard import update_graph, update_resource_graph


class TestDashboardVisualization:
    
    @pytest.fixture
    def sample_trace_data(self):
        """Sample trace data for testing."""
        return [
            {
                "layer": "Conv2D",
                "input_shape": (1, 28, 28, 3),
                "output_shape": (1, 26, 26, 32),
                "execution_time": 0.005,
                "flops": 1000000,
                "memory": 100000
            },
            {
                "layer": "MaxPooling2D",
                "input_shape": (1, 26, 26, 32),
                "output_shape": (1, 13, 13, 32),
                "execution_time": 0.001,
                "flops": 10000,
                "memory": 5000
            },
            {
                "layer": "Flatten",
                "input_shape": (1, 13, 13, 32),
                "output_shape": (1, 5408),
                "execution_time": 0.0005,
                "flops": 5000,
                "memory": 1000
            },
            {
                "layer": "Dense",
                "input_shape": (1, 5408),
                "output_shape": (1, 128),
                "execution_time": 0.002,
                "flops": 500000,
                "memory": 50000
            },
            {
                "layer": "Output",
                "input_shape": (1, 128),
                "output_shape": (1, 10),
                "execution_time": 0.0008,
                "flops": 1000,
                "memory": 500
            }
        ]
    
    @patch('neural.dashboard.dashboard.go.Figure')
    @patch('neural.dashboard.dashboard.TRACE_DATA', new_property=lambda self: None)
    def test_update_graph_basic(self, mock_figure, sample_trace_data):
        """Test the update_graph function with basic visualization."""
        # Setup mocks
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        # Set the trace data
        from neural.dashboard.dashboard import TRACE_DATA
        TRACE_DATA = sample_trace_data
        
        # Call the function
        result = update_graph(1, "basic")
        
        # Check that the Figure was created
        mock_figure.assert_called_once()
        
        # Check that add_trace was called
        mock_fig.add_trace.assert_called_once()
        
        # Check that update_layout was called
        mock_fig.update_layout.assert_called()
        
        # Check that the result is a list containing the mock figure
        assert result == [mock_fig]
    
    @patch('neural.dashboard.dashboard.go.Figure')
    @patch('neural.dashboard.dashboard.TRACE_DATA', new_property=lambda self: None)
    def test_update_graph_stacked(self, mock_figure, sample_trace_data):
        """Test the update_graph function with stacked visualization."""
        # Setup mocks
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        # Set the trace data
        from neural.dashboard.dashboard import TRACE_DATA
        TRACE_DATA = sample_trace_data
        
        # Call the function
        result = update_graph(1, "stacked")
        
        # Check that the Figure was created
        mock_figure.assert_called_once()
        
        # Check that add_trace was called multiple times
        assert mock_fig.add_trace.call_count > 1
        
        # Check that update_layout was called
        mock_fig.update_layout.assert_called()
        
        # Check that the result is a list containing the mock figure
        assert result == [mock_fig]
    
    @patch('neural.dashboard.dashboard.go.Figure')
    @patch('neural.dashboard.dashboard.TRACE_DATA', new_property=lambda self: None)
    def test_update_graph_horizontal(self, mock_figure, sample_trace_data):
        """Test the update_graph function with horizontal visualization."""
        # Setup mocks
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        # Set the trace data
        from neural.dashboard.dashboard import TRACE_DATA
        TRACE_DATA = sample_trace_data
        
        # Call the function
        result = update_graph(1, "horizontal")
        
        # Check that the Figure was created
        mock_figure.assert_called_once()
        
        # Check that add_trace was called
        mock_fig.add_trace.assert_called_once()
        
        # Check that update_layout was called
        mock_fig.update_layout.assert_called()
        
        # Check that the result is a list containing the mock figure
        assert result == [mock_fig]
    
    @patch('neural.dashboard.dashboard.go.Figure')
    @patch('neural.dashboard.dashboard.TRACE_DATA', new_property=lambda self: None)
    def test_update_graph_box(self, mock_figure, sample_trace_data):
        """Test the update_graph function with box plot visualization."""
        # Setup mocks
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        # Set the trace data
        from neural.dashboard.dashboard import TRACE_DATA
        TRACE_DATA = sample_trace_data
        
        # Call the function
        result = update_graph(1, "box")
        
        # Check that the Figure was created
        mock_figure.assert_called_once()
        
        # Check that add_trace was called
        mock_fig.add_trace.assert_called_once()
        
        # Check that update_layout was called
        mock_fig.update_layout.assert_called()
        
        # Check that the result is a list containing the mock figure
        assert result == [mock_fig]
    
    @patch('neural.dashboard.dashboard.go.Figure')
    @patch('neural.dashboard.dashboard.TRACE_DATA', new_property=lambda self: None)
    def test_update_graph_gantt(self, mock_figure, sample_trace_data):
        """Test the update_graph function with gantt chart visualization."""
        # Setup mocks
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        # Set the trace data
        from neural.dashboard.dashboard import TRACE_DATA
        TRACE_DATA = sample_trace_data
        
        # Call the function
        result = update_graph(1, "gantt")
        
        # Check that the Figure was created
        mock_figure.assert_called_once()
        
        # Check that add_trace was called for each layer
        assert mock_fig.add_trace.call_count == len(sample_trace_data)
        
        # Check that update_layout was called
        mock_fig.update_layout.assert_called()
        
        # Check that the result is a list containing the mock figure
        assert result == [mock_fig]
    
    @patch('neural.dashboard.dashboard.go.Figure')
    @patch('neural.dashboard.dashboard.TRACE_DATA', new_property=lambda self: None)
    @patch('neural.dashboard.dashboard.np.random.rand')
    def test_update_graph_heatmap(self, mock_rand, mock_figure, sample_trace_data):
        """Test the update_graph function with heatmap visualization."""
        # Setup mocks
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        mock_rand.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5],
                                  [0.6, 0.7, 0.8, 0.9, 1.0],
                                  [0.1, 0.2, 0.3, 0.4, 0.5],
                                  [0.6, 0.7, 0.8, 0.9, 1.0],
                                  [0.1, 0.2, 0.3, 0.4, 0.5]]
        
        # Set the trace data
        from neural.dashboard.dashboard import TRACE_DATA
        TRACE_DATA = sample_trace_data
        
        # Call the function
        result = update_graph(1, "heatmap")
        
        # Check that the Figure was created
        mock_figure.assert_called_once()
        
        # Check that add_trace was called
        mock_fig.add_trace.assert_called_once()
        
        # Check that update_layout was called
        mock_fig.update_layout.assert_called()
        
        # Check that the result is a list containing the mock figure
        assert result == [mock_fig]
    
    @patch('neural.dashboard.dashboard.go.Figure')
    @patch('neural.dashboard.dashboard.TRACE_DATA', new_property=lambda self: None)
    def test_update_graph_thresholds(self, mock_figure, sample_trace_data):
        """Test the update_graph function with thresholds visualization."""
        # Setup mocks
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        # Set the trace data
        from neural.dashboard.dashboard import TRACE_DATA
        TRACE_DATA = sample_trace_data
        
        # Call the function
        result = update_graph(1, "thresholds")
        
        # Check that the Figure was created
        mock_figure.assert_called_once()
        
        # Check that add_trace was called
        mock_fig.add_trace.assert_called_once()
        
        # Check that update_layout was called
        mock_fig.update_layout.assert_called()
        
        # Check that the result is a list containing the mock figure
        assert result == [mock_fig]
    
    @patch('neural.dashboard.dashboard.go.Figure')
    @patch('neural.dashboard.dashboard.psutil')
    @patch('neural.dashboard.dashboard.torch')
    def test_update_resource_graph(self, mock_torch, mock_psutil, mock_figure):
        """Test the update_resource_graph function."""
        # Setup mocks
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        mock_psutil.cpu_percent.return_value = 50
        mock_psutil.virtual_memory.return_value.percent = 75
        
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 1024 ** 3  # 1 GB
        
        # Call the function
        result = update_resource_graph(1)
        
        # Check that the Figure was created
        mock_figure.assert_called_once()
        
        # Check that add_trace was called
        mock_fig.add_trace.assert_called_once()
        
        # Check that update_layout was called
        mock_fig.update_layout.assert_called_once()
        
        # Check that the result is a list containing the mock figure
        assert result == [mock_fig]
    
    @patch('neural.dashboard.dashboard.go.Figure')
    @patch('neural.dashboard.dashboard.psutil')
    @patch('neural.dashboard.dashboard.torch')
    def test_update_resource_graph_no_gpu(self, mock_torch, mock_psutil, mock_figure):
        """Test the update_resource_graph function without GPU."""
        # Setup mocks
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        mock_psutil.cpu_percent.return_value = 50
        mock_psutil.virtual_memory.return_value.percent = 75
        
        mock_torch.cuda.is_available.return_value = False
        
        # Call the function
        result = update_resource_graph(1)
        
        # Check that the Figure was created
        mock_figure.assert_called_once()
        
        # Check that add_trace was called
        mock_fig.add_trace.assert_called_once()
        
        # Check that update_layout was called
        mock_fig.update_layout.assert_called_once()
        
        # Check that the result is a list containing the mock figure
        assert result == [mock_fig]


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
