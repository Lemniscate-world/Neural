import os
import sys
import pytest
from click.testing import CliRunner
from neural.cli import cli, logger
from neural.parser.parser import ModelTransformer, DSLValidationError
import shutil
import logging
from pathlib import Path

@pytest.fixture
def runner():
    """Fixture for running CLI commands."""
    return CliRunner()

@pytest.fixture
def sample_neural(tmp_path):
    """Create a temporary .neural file for testing."""
    file_path = tmp_path / "sample.neural"
    content = """
    network TestNet {
        input: (28, 28, 1)
        layers:
            Conv2D(32, 3, activation="relu")
            Dense(10, activation="softmax")
        loss: "categorical_crossentropy"
        optimizer: "adam"
    }
    """
    file_path.write_text(content)
    return str(file_path)

@pytest.fixture
def invalid_neural(tmp_path):
    """Create a temporary invalid .neural file."""
    file_path = tmp_path / "invalid.neural"
    content = """
    network InvalidNet {
        input: (28, "invalid", 1)  # Invalid shape
        layers:
            Dense(-10)  # Negative units
    }
    """
    file_path.write_text(content)
    return str(file_path)

# Compile Command Tests
def test_compile_command(runner, sample_neural):
    """Test compile with a valid .neural file."""
    output_file = "sample_tensorflow.py"
    result = runner.invoke(cli, ["compile", sample_neural, "--backend", "tensorflow", "--output", output_file])
    assert result.exit_code == 0, f"Command failed: {result.output}"
    assert os.path.exists(output_file), "Output file not found"
    assert "Output written to" in result.output
    with open(output_file, 'r') as f:
        assert "tensorflow" in f.read().lower(), "Generated code should reference TensorFlow"

def test_compile_pytorch_backend(runner, sample_neural):
    """Test compile with PyTorch backend."""
    output_file = "sample_pytorch.py"
    result = runner.invoke(cli, ["compile", sample_neural, "--backend", "pytorch", "--output", output_file])
    assert result.exit_code == 0
    assert os.path.exists(output_file)
    with open(output_file, 'r') as f:
        assert "torch" in f.read().lower(), "Generated code should reference PyTorch"

def test_compile_dry_run(runner, sample_neural):
    """Test compile with --dry-run option."""
    result = runner.invoke(cli, ["compile", sample_neural, "--backend", "tensorflow", "--dry-run"])
    assert result.exit_code == 0
    assert "Generated code (dry run):" in result.output
    assert not os.path.exists("sample_tensorflow.py"), "Dry run should not create output file"

def test_compile_invalid_file(runner):
    """Test compile with a non-existent file."""
    result = runner.invoke(cli, ["compile", "nonexistent.neural", "--backend", "tensorflow"])
    assert result.exit_code != 0
    assert "does not exist" in result.output

def test_compile_invalid_backend(runner, sample_neural):
    """Test compile with an unsupported backend."""
    result = runner.invoke(cli, ["compile", sample_neural, "--backend", "invalid"])
    assert result.exit_code != 0
    assert "Invalid choice" in result.output

def test_compile_invalid_syntax(runner, invalid_neural):
    """Test compile with invalid .neural syntax."""
    result = runner.invoke(cli, ["compile", invalid_neural, "--backend", "tensorflow"])
    assert result.exit_code != 0
    assert "Parsing/transforming" in result.output or "Error" in result.output

# Run Command Tests
def test_run_command(runner, sample_neural):
    """Test run with a compiled .py file."""
    output_file = "sample_tensorflow.py"
    runner.invoke(cli, ["compile", sample_neural, "--backend", "tensorflow", "--output", output_file])
    result = runner.invoke(cli, ["run", output_file, "--backend", "tensorflow"])
    assert result.exit_code == 0  # Assuming the generated code runs successfully
    assert "Execution completed successfully" in result.output

def test_run_invalid_file(runner):
    """Test run with a non-.py file."""
    result = runner.invoke(cli, ["run", "sample.neural", "--backend", "tensorflow"])
    assert result.exit_code != 0
    assert "Expected a .py file" in result.output

# Visualize Command Tests
def test_visualize_command(runner, sample_neural):
    """Test visualize with a valid .neural file."""
    result = runner.invoke(cli, ["visualize", sample_neural, "--format", "png"])
    assert result.exit_code == 0
    assert os.path.exists("architecture.png"), "Visualization file not created"
    assert "Visualization saved as architecture.png" in result.output

def test_visualize_html_format(runner, sample_neural):
    """Test visualize with HTML format."""
    result = runner.invoke(cli, ["visualize", sample_neural, "--format", "html"])
    assert result.exit_code == 0
    assert os.path.exists("shape_propagation.html")
    assert os.path.exists("tensor_flow.html")
    assert "Visualizations generated" in result.output

def test_visualize_cache(runner, sample_neural):
    """Test visualize with caching."""
    # First run to generate cache
    runner.invoke(cli, ["visualize", sample_neural, "--format", "png"])
    # Second run to use cache
    result = runner.invoke(cli, ["visualize", sample_neural, "--format", "png"])
    assert result.exit_code == 0
    assert "Using cached visualization" in result.output

def test_visualize_no_cache(runner, sample_neural):
    """Test visualize with --no-cache."""
    result = runner.invoke(cli, ["visualize", sample_neural, "--format", "png", "--no-cache"])
    assert result.exit_code == 0
    assert "Using cached visualization" not in result.output

def test_visualize_invalid_file(runner):
    """Test visualize with a non-existent file."""
    result = runner.invoke(cli, ["visualize", "nonexistent.neural", "--format", "png"])
    assert result.exit_code != 0
    assert "does not exist" in result.output

# Clean Command Tests
def test_clean_command(runner, sample_neural):
    """Test clean command after generating files."""
    runner.invoke(cli, ["compile", sample_neural, "--backend", "tensorflow"])
    runner.invoke(cli, ["visualize", sample_neural, "--format", "png"])
    result = runner.invoke(cli, ["clean"])
    assert result.exit_code == 0
    assert not os.path.exists("sample_tensorflow.py")
    assert not os.path.exists("architecture.png")
    assert not os.path.exists(".neural_cache")
    assert "Removed" in result.output

def test_clean_no_files(runner):
    """Test clean with no files to remove."""
    result = runner.invoke(cli, ["clean"])
    assert result.exit_code == 0
    assert "No files to clean" in result.output

# Version Command Test
def test_version_command(runner):
    """Test the version command."""
    result = runner.invoke(cli, ["version"])
    assert result.exit_code == 0
    assert "Neural CLI v0.1.0" in result.output
    assert "Python" in result.output
    assert "Click" in result.output
    assert "Lark" in result.output

# Debug Command Tests
def test_debug_command_basic(runner, sample_neural):
    """Test debug command without options."""
    result = runner.invoke(cli, ["debug", sample_neural, "--backend", "tensorflow"])
    assert result.exit_code == 0
    assert "Debugging" in result.output
    assert "Debug session completed" in result.output

def test_debug_gradients(runner, sample_neural):
    """Test debug with --gradients option."""
    result = runner.invoke(cli, ["debug", sample_neural, "--gradients"])
    assert result.exit_code == 0
    assert "Gradient flow trace (simulated)" in result.output

def test_debug_step(runner, sample_neural, mocker):
    """Test debug with --step option (mock user input)."""
    mocker.patch('click.confirm', return_value=True)  # Simulate 'yes' to continue
    result = runner.invoke(cli, ["debug", sample_neural, "--step"])
    assert result.exit_code == 0
    assert "Step debugging mode" in result.output
    assert "Step 1: Conv2D" in result.output
    assert "Step 2: Dense" in result.output

def test_debug_invalid_file(runner):
    """Test debug with a non-existent file."""
    result = runner.invoke(cli, ["debug", "nonexistent.neural"])
    assert result.exit_code != 0
    assert "does not exist" in result.output

# No-Code Command Test (Mocked)
def test_no_code_command(runner, mocker):
    """Test no-code command (mock dashboard run)."""
    mocker.patch('neural.dashboard.dashboard.app.run_server', side_effect=lambda *args, **kwargs: None)  # Mock GUI launch
    result = runner.invoke(cli, ["no-code"])
    assert result.exit_code == 0
    assert "Launching no-code interface at http://localhost:8051" in result.output

# Verbose Option Tests
def test_verbose_flag(runner, sample_neural, caplog):
    """Test --verbose flag increases log level."""
    caplog.set_level(logging.DEBUG)
    result = runner.invoke(cli, ["compile", sample_neural, "--verbose"])
    assert result.exit_code == 0
    assert "Verbose mode enabled" in caplog.text
    assert "DEBUG" in caplog.text