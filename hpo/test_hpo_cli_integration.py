import os
import sys
import tempfile
import pytest
from unittest.mock import patch, MagicMock, ANY
from click.testing import CliRunner

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from neural.cli import cli

class TestHPOCLIIntegration:

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def sample_neural_file(self):
        with tempfile.NamedTemporaryFile(suffix='.neural', delete=False) as f:
            f.write(b"""
            network HPOTest {
                input: (28, 28, 1)
                layers:
                    Dense(HPO(choice(64, 128, 256)))
                    Dropout(HPO(range(0.2, 0.5, step=0.1)))
                    Output(10)
                loss: "categorical_crossentropy"
                optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
            }
            """)
            temp_path = f.name

        yield temp_path

        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @patch('neural.hpo.hpo.optimize_and_return')
    @patch('neural.code_generation.code_generator.generate_optimized_dsl')
    @patch('neural.code_generation.code_generator.generate_code')
    def test_compile_with_hpo(self, mock_generate_code, mock_generate_optimized_dsl, mock_optimize, runner, sample_neural_file):
        """Test the compile command with HPO enabled."""
        # Setup mocks
        mock_optimize.return_value = {
            'dense_units': 128,
            'dropout_rate': 0.3,
            'learning_rate': 0.001
        }
        mock_generate_optimized_dsl.return_value = "optimized dsl content"
        mock_generate_code.return_value = "generated code content"

        # Run the CLI command
        result = runner.invoke(cli, ['compile', sample_neural_file, '--hpo', '--dry-run'])

        # Verify the command executed successfully
        assert result.exit_code == 0

        # Verify our mocks were called correctly
        mock_optimize.assert_called_once()
        mock_generate_optimized_dsl.assert_called_once()
        mock_generate_code.assert_called_once()

        # Check output contains expected content
        assert "Best parameters:" in result.output
        assert "Generated code:" in result.output

    @patch('neural.hpo.hpo.optimize_and_return')
    @patch('neural.code_generation.code_generator.generate_optimized_dsl')
    @patch('subprocess.run')
    def test_run_with_hpo(self, mock_subprocess, mock_generate_optimized_dsl, mock_optimize, runner, sample_neural_file):
        """Test the run command with HPO enabled."""
        # Setup mocks
        mock_optimize.return_value = {
            'dense_units': 128,
            'dropout_rate': 0.3,
            'learning_rate': 0.001
        }
        mock_generate_optimized_dsl.return_value = "optimized dsl content"

        # Run the CLI command
        result = runner.invoke(cli, ['run', sample_neural_file, '--hpo'])

        # Verify the command executed successfully
        assert result.exit_code == 0

        # Verify our mocks were called correctly
        mock_optimize.assert_called_once()
        mock_generate_optimized_dsl.assert_called_once()

    @patch('neural.hpo.hpo.optimize_and_return')
    def test_hpo_with_different_dataset(self, mock_optimize, runner, sample_neural_file):
        """Test HPO with a different dataset."""
        # Setup mock
        mock_optimize.return_value = {'dense_units': 128}

        # Run the CLI command
        result = runner.invoke(cli, ['compile', sample_neural_file, '--hpo', '--dataset', 'CIFAR10', '--dry-run'])

        # Verify the command executed successfully
        assert result.exit_code == 0

        # Verify our mock was called with the correct dataset
        mock_optimize.assert_called_once_with(
            mock.ANY,  # config content
            n_trials=3,  # default value
            dataset_name='CIFAR10',
            backend='tensorflow'  # default value
        )

    @patch('neural.hpo.hpo.optimize_and_return')
    def test_hpo_with_unsupported_dataset(self, mock_optimize, runner, sample_neural_file):
        """Test HPO with an unsupported dataset."""
        # Setup mock
        mock_optimize.return_value = {'dense_units': 128}

        # Run the CLI command
        result = runner.invoke(cli, ['compile', sample_neural_file, '--hpo', '--dataset', 'UNSUPPORTED', '--dry-run'])

        # Command should still execute but with a warning
        assert result.exit_code == 0

        # Warning should be logged
        assert "may not be supported" in result.output

    def test_verbose_mode(self, runner, sample_neural_file):
        """Test that verbose mode is properly configured."""
        # This test will verify that the verbose flag is properly passed to configure_logging
        with patch('neural.cli.configure_logging') as mock_configure_logging:
            # Run with verbose flag
            runner.invoke(cli, ['-v', 'version'])
            mock_configure_logging.assert_called_with(True)

            # Run without verbose flag
            runner.invoke(cli, ['version'])
            mock_configure_logging.assert_called_with(False)
