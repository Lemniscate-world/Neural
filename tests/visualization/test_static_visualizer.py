import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Try different import paths based on the actual project structure
try:
    from neural.visualization.static_visualizer.visualizer import NeuralVisualizer
except ModuleNotFoundError:
    try:
        # Alternative import path if the module structure is different
        from neural.visualization.static_visualizer.visualizer import NeuralVisualizer
    except ModuleNotFoundError:
        # If that fails too, try to find the module in the project
        import importlib.util
        import glob
        # Search for the visualizer module
        visualizer_files = glob.glob(os.path.join(os.path.dirname(__file__), "../../**/*visualizer*.py"), recursive=True)
        if visualizer_files:
            print(f"Found potential visualizer files: {visualizer_files}")
            # Try to import the first one found
            spec = importlib.util.spec_from_file_location("visualizer", visualizer_files[0])
            visualizer_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(visualizer_module)
            NeuralVisualizer = getattr(visualizer_module, "NeuralVisualizer")
        else:
            raise ImportError("Could not find NeuralVisualizer class in the project")


class TestStaticVisualizer:

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
        """Test initializing the StaticVisualizer."""
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

    def test_save_architecture_diagram(self, sample_model_data):
        """Test saving architecture diagram to file."""
        import os
        visualizer = NeuralVisualizer(sample_model_data)

        # Print current working directory
        print(f"Current working directory: {os.getcwd()}")

        # Call the method with a test filename
        test_filename = "test_architecture.png"
        print(f"Attempting to save to: {os.path.abspath(test_filename)}")

        visualizer.save_architecture_diagram(test_filename)

        # Check that the file was actually created
        assert os.path.exists(test_filename), f"File {test_filename} was not created"
        print(f"Successfully created file: {os.path.abspath(test_filename)}")

        # Get file size
        file_size = os.path.getsize(test_filename)
        print(f"File size: {file_size} bytes")

        # Clean up
        if os.path.exists(test_filename):
            os.remove(test_filename)
            print(f"Removed test file: {test_filename}")

    def test_save_shape_visualization(self, sample_model_data):
        """Test saving shape visualization to HTML file."""
        import os
        visualizer = NeuralVisualizer(sample_model_data)

        # Print current working directory
        print(f"Current working directory: {os.getcwd()}")

        # Create a shape history for visualization
        shape_history = [
            ('Input', (None, 28, 28, 1)),
            ('Conv2D', (None, 26, 26, 32)),
            ('MaxPooling2D', (None, 13, 13, 32)),
            ('Flatten', (None, 5408)),
            ('Dense', (None, 128)),
            ('Output', (None, 10))
        ]

        # Create the figure
        fig = visualizer.create_3d_visualization(shape_history)

        # Save to a test file
        test_filename = "test_shapes.html"
        print(f"Attempting to save to: {os.path.abspath(test_filename)}")

        visualizer.save_shape_visualization(fig, test_filename)

        # Check that the file was actually created
        assert os.path.exists(test_filename), f"File {test_filename} was not created"
        print(f"Successfully created file: {os.path.abspath(test_filename)}")

        # Get file size
        file_size = os.path.getsize(test_filename)
        print(f"File size: {file_size} bytes")

        # Clean up
        if os.path.exists(test_filename):
            os.remove(test_filename)
            print(f"Removed test file: {test_filename}")


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
