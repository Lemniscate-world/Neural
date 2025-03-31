import os
import sys
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from neural.cli import cli, visualize


class TestCLIVisualization:
    
    @pytest.fixture
    def runner(self):
        """Create a CliRunner for testing."""
        return CliRunner()
    
    @pytest.fixture
    def sample_neural_file(self):
        """Create a sample .neural file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.neural', delete=False) as f:
            f.write(b"""
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
            """)
            temp_path = f.name
        
        yield temp_path
        
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @patch('neural.cli.create_parser')
    @patch('neural.cli.ModelTransformer')
    @patch('neural.cli.ShapePropagator')
    @patch('neural.dashboard.tensor_flow.create_animated_network')
    def test_visualize_command_html(self, mock_create_animated_network, mock_shape_propagator, 
                                   mock_model_transformer, mock_create_parser, runner, sample_neural_file):
        """Test the visualize command with HTML output."""
        # Setup mocks
        mock_parser = MagicMock()
        mock_create_parser.return_value = mock_parser
        
        mock_tree = MagicMock()
        mock_parser.parse.return_value = mock_tree
        
        mock_transformer = MagicMock()
        mock_model_transformer.return_value = mock_transformer
        
        mock_model_data = {
            'input': {'shape': (None, 28, 28, 1)},
            'layers': [
                {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'}},
                {'type': 'MaxPooling2D', 'params': {'pool_size': (2, 2)}},
                {'type': 'Flatten', 'params': {}},
                {'type': 'Dense', 'params': {'units': 128, 'activation': 'relu'}},
                {'type': 'Output', 'params': {'units': 10, 'activation': 'softmax'}}
            ]
        }
        mock_transformer.transform.return_value = mock_model_data
        
        mock_propagator = MagicMock()
        mock_shape_propagator.return_value = mock_propagator
        
        mock_dot = MagicMock()
        mock_fig = MagicMock()
        mock_report = {
            'dot_graph': mock_dot,
            'plotly_chart': mock_fig,
            'shape_history': [
                ('Input', (None, 28, 28, 1)),
                ('Conv2D', (None, 26, 26, 32)),
                ('MaxPooling2D', (None, 13, 13, 32)),
                ('Flatten', (None, 5408)),
                ('Dense', (None, 128)),
                ('Output', (None, 10))
            ]
        }
        mock_propagator.generate_report.return_value = mock_report
        
        mock_animated_fig = MagicMock()
        mock_create_animated_network.return_value = mock_animated_fig
        
        # Create a temporary directory for the test
        with runner.isolated_filesystem():
            # Run the command
            result = runner.invoke(cli, ['visualize', sample_neural_file, '--format', 'html', '--no-cache'])
            
            # Check that the command executed successfully
            assert result.exit_code == 0
            
            # Check that the parser was created with the correct start rule
            mock_create_parser.assert_called_once_with(start_rule='network')
            
            # Check that the parser was called with the file content
            mock_parser.parse.assert_called_once()
            
            # Check that the transformer was called with the parse tree
            mock_transformer.transform.assert_called_once_with(mock_tree)
            
            # Check that the propagator was created
            mock_shape_propagator.assert_called_once()
            
            # Check that propagate was called for each layer
            assert mock_propagator.propagate.call_count == len(mock_model_data['layers'])
            
            # Check that generate_report was called
            mock_propagator.generate_report.assert_called_once()
            
            # Check that the dot graph was rendered
            mock_dot.render.assert_called_once_with('architecture', cleanup=True)
            
            # Check that the plotly chart was saved
            mock_fig.write_html.assert_called_once_with('shape_propagation.html')
            
            # Check that create_animated_network was called
            mock_create_animated_network.assert_called_once()
            
            # Check that the animated network was saved
            mock_animated_fig.write_html.assert_called_once_with('tensor_flow.html')
    
    @patch('neural.cli.create_parser')
    @patch('neural.cli.ModelTransformer')
    @patch('neural.cli.ShapePropagator')
    def test_visualize_command_png(self, mock_shape_propagator, mock_model_transformer, 
                                  mock_create_parser, runner, sample_neural_file):
        """Test the visualize command with PNG output."""
        # Setup mocks
        mock_parser = MagicMock()
        mock_create_parser.return_value = mock_parser
        
        mock_tree = MagicMock()
        mock_parser.parse.return_value = mock_tree
        
        mock_transformer = MagicMock()
        mock_model_transformer.return_value = mock_transformer
        
        mock_model_data = {
            'input': {'shape': (None, 28, 28, 1)},
            'layers': [
                {'type': 'Conv2D', 'params': {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'}},
                {'type': 'MaxPooling2D', 'params': {'pool_size': (2, 2)}},
                {'type': 'Flatten', 'params': {}},
                {'type': 'Dense', 'params': {'units': 128, 'activation': 'relu'}},
                {'type': 'Output', 'params': {'units': 10, 'activation': 'softmax'}}
            ]
        }
        mock_transformer.transform.return_value = mock_model_data
        
        mock_propagator = MagicMock()
        mock_shape_propagator.return_value = mock_propagator
        
        mock_dot = MagicMock()
        mock_fig = MagicMock()
        mock_report = {
            'dot_graph': mock_dot,
            'plotly_chart': mock_fig,
            'shape_history': []
        }
        mock_propagator.generate_report.return_value = mock_report
        
        # Create a temporary directory for the test
        with runner.isolated_filesystem():
            # Run the command
            result = runner.invoke(cli, ['visualize', sample_neural_file, '--format', 'png', '--no-cache'])
            
            # Check that the command executed successfully
            assert result.exit_code == 0
            
            # Check that the parser was created with the correct start rule
            mock_create_parser.assert_called_once_with(start_rule='network')
            
            # Check that the parser was called with the file content
            mock_parser.parse.assert_called_once()
            
            # Check that the transformer was called with the parse tree
            mock_transformer.transform.assert_called_once_with(mock_tree)
            
            # Check that the propagator was created
            mock_shape_propagator.assert_called_once()
            
            # Check that propagate was called for each layer
            assert mock_propagator.propagate.call_count == len(mock_model_data['layers'])
            
            # Check that generate_report was called
            mock_propagator.generate_report.assert_called_once()
            
            # Check that the dot graph was rendered with the correct format
            mock_dot.render.assert_called_once_with('architecture', cleanup=True)
            assert mock_dot.format == 'png'
    
    @patch('neural.cli.create_parser')
    @patch('neural.cli.ModelTransformer')
    @patch('neural.cli.ShapePropagator')
    def test_visualize_command_with_cache(self, mock_shape_propagator, mock_model_transformer, 
                                         mock_create_parser, runner, sample_neural_file):
        """Test the visualize command with caching."""
        # Setup mocks
        mock_parser = MagicMock()
        mock_create_parser.return_value = mock_parser
        
        mock_tree = MagicMock()
        mock_parser.parse.return_value = mock_tree
        
        mock_transformer = MagicMock()
        mock_model_transformer.return_value = mock_transformer
        
        mock_model_data = {
            'input': {'shape': (None, 28, 28, 1)},
            'layers': []
        }
        mock_transformer.transform.return_value = mock_model_data
        
        # Create a temporary directory for the test
        with runner.isolated_filesystem():
            # Create a cache directory and a cached file
            cache_dir = Path(".neural_cache")
            cache_dir.mkdir(exist_ok=True)
            
            # Create a dummy cached file
            import hashlib
            file_hash = hashlib.sha256(Path(sample_neural_file).read_bytes()).hexdigest()
            cache_file = cache_dir / f"viz_{file_hash}_svg"
            with open(cache_file, 'w') as f:
                f.write("dummy cached content")
            
            # Run the command
            result = runner.invoke(cli, ['visualize', sample_neural_file, '--format', 'svg', '--cache'])
            
            # Check that the command executed successfully
            assert result.exit_code == 0
            
            # Check that the parser was not called
            mock_create_parser.assert_not_called()
            
            # Check that the transformer was not called
            mock_transformer.transform.assert_not_called()
            
            # Check that the propagator was not created
            mock_shape_propagator.assert_not_called()
            
            # Check that the cached file was copied
            assert os.path.exists('architecture.svg')
    
    @patch('neural.cli.create_parser')
    def test_visualize_command_unsupported_file_type(self, mock_create_parser, runner):
        """Test the visualize command with an unsupported file type."""
        # Create a temporary file with an unsupported extension
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"This is not a neural file")
            temp_path = f.name
        
        try:
            # Run the command
            result = runner.invoke(cli, ['visualize', temp_path])
            
            # Check that the command failed
            assert result.exit_code != 0
            
            # Check that the parser was not called
            mock_create_parser.assert_not_called()
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    @patch('neural.cli.create_parser')
    @patch('neural.cli.ModelTransformer')
    def test_visualize_command_parsing_error(self, mock_model_transformer, mock_create_parser, 
                                           runner, sample_neural_file):
        """Test the visualize command with a parsing error."""
        # Setup mocks
        mock_parser = MagicMock()
        mock_create_parser.return_value = mock_parser
        
        # Make the parser raise an exception
        mock_parser.parse.side_effect = Exception("Parsing error")
        
        # Run the command
        result = runner.invoke(cli, ['visualize', sample_neural_file])
        
        # Check that the command failed
        assert result.exit_code != 0
        
        # Check that the parser was called
        mock_create_parser.assert_called_once()
        
        # Check that the parser was called with the file content
        mock_parser.parse.assert_called_once()
        
        # Check that the transformer was not called
        mock_model_transformer.transform.assert_not_called()


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])
