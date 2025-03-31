import os
import sys

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

    def test_generate_optimized_dsl_nonexistent_param(self):
        """Test handling of nonexistent parameters."""
        config = """
        network InvalidTest {
            input: (28, 28, 1)
            layers:
                Dense(HPO(choice(64, 128)))
                Output(10)
            loss: "categorical_crossentropy"
        }
        """

        # Parameter name doesn't match what's in the config
        best_params = {
            'nonexistent_param': 128
        }

        # Should not fail
        optimized = generate_optimized_dsl(config, best_params)

        # Original HPO should still be present
        assert "HPO(choice(64, 128))" in optimized

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

        # This test is just to verify the code doesn't crash with quoted strings
        optimized = generate_optimized_dsl(config, best_params)

        # Verify the output contains the network name
        assert "QuotedTest" in optimized
