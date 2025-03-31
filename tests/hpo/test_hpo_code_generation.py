import os
import sys
from unittest.mock import patch

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from neural.code_generation.code_generator import generate_optimized_dsl

class TestHPOCodeGeneration:

    def test_generate_optimized_dsl_basic(self):
        """Test basic HPO parameter replacement in DSL."""
        config = """
        network BasicTest {
            input: (28, 28, 1)
            layers:
                Dense(HPO(choice(64, 128, 256)))
                Output(10)
            loss: "categorical_crossentropy"
            optimizer: Adam(learning_rate=0.001)
        }
        """

        best_params = {
            'dense_units': 128
        }

        optimized = generate_optimized_dsl(config, best_params)

        # Verify HPO expression was replaced
        assert "HPO(choice(64, 128, 256))" not in optimized
        assert "Dense(128)" in optimized

    def test_generate_optimized_dsl_learning_rate(self):
        """Test learning rate HPO parameter replacement."""
        config = """
        network LRTest {
            input: (28, 28, 1)
            layers:
                Dense(128)
                Output(10)
            loss: "categorical_crossentropy"
            optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
        }
        """

        best_params = {
            'learning_rate': 0.001
        }

        optimized = generate_optimized_dsl(config, best_params)

        # Verify HPO expression was replaced
        assert "HPO(log_range(1e-4, 1e-2))" not in optimized
        assert "learning_rate=0.001" in optimized

    def test_generate_optimized_dsl_multiple_params(self):
        """Test multiple HPO parameter replacements."""
        config = """
        network MultiTest {
            input: (28, 28, 1)
            layers:
                Dense(HPO(choice(64, 128, 256)))
                Dropout(HPO(range(0.2, 0.5, step=0.1)))
                Output(10)
            loss: "categorical_crossentropy"
            optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
        }
        """

        best_params = {
            'dense_units': 128,
            'dropout_rate': 0.3,
            'learning_rate': 0.001
        }

        optimized = generate_optimized_dsl(config, best_params)

        # Verify HPO expressions were replaced
        assert "HPO(choice(64, 128, 256))" not in optimized
        assert "HPO(range(0.2, 0.5, step=0.1))" not in optimized
        assert "HPO(log_range(1e-4, 1e-2))" not in optimized

        assert "Dense(128)" in optimized
        assert "Dropout(0.3)" in optimized
        assert "learning_rate=0.001" in optimized

    def test_generate_optimized_dsl_missing_params(self):
        """Test handling of missing parameters in best_params."""
        config = """
        network MissingTest {
            input: (28, 28, 1)
            layers:
                Dense(HPO(choice(64, 128, 256)))
                Dropout(HPO(range(0.2, 0.5, step=0.1)))
                Output(10)
            loss: "categorical_crossentropy"
            optimizer: Adam(learning_rate=HPO(log_range(1e-4, 1e-2)))
        }
        """

        # Missing dropout_rate
        best_params = {
            'dense_units': 128,
            'learning_rate': 0.001
        }

        # Should log a warning but not fail
        optimized = generate_optimized_dsl(config, best_params)

        # Verify available params were replaced
        assert "Dense(128)" in optimized
        assert "learning_rate=0.001" in optimized

        # Dropout HPO should still be present
        assert "HPO(range(0.2, 0.5, step=0.1))" in optimized

    @patch('neural.code_generation.code_generator.logger')
    def test_generate_optimized_dsl_invalid_hpo(self, mock_logger):
        """Test handling of invalid HPO expressions."""
        config = """
        network InvalidTest {
            input: (28, 28, 1)
            layers:
                Dense(HPO(invalid_type(64, 128)))
                Output(10)
            loss: "categorical_crossentropy"
        }
        """

        best_params = {
            'dense_units': 128
        }

        # Should log a warning but not fail
        optimized = generate_optimized_dsl(config, best_params)

        # Verify logger was called with warning
        mock_logger.warning.assert_called()

        # Original config should be returned with HPO still present
        assert "HPO(invalid_type(64, 128))" in optimized

    def test_generate_optimized_dsl_quoted_strings(self):
        """Test handling of quoted strings in HPO expressions."""
        config = """
        network QuotedTest {
            input: (28, 28, 1)
            layers:
                Dense(128, activation=HPO(choice("relu", "tanh", "sigmoid")))
                Output(10)
            loss: "categorical_crossentropy"
        }
        """

        best_params = {
            'dense_activation': "relu"
        }

        optimized = generate_optimized_dsl(config, best_params)

        # Verify HPO expression was replaced
        assert 'HPO(choice("relu", "tanh", "sigmoid"))' not in optimized
        assert 'activation="relu"' in optimized or "activation='relu'" in optimized
