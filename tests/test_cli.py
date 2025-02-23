import os
import pysnooper
import pytest
from click.testing import CliRunner
from neural.cli import cli

@pytest.fixture
def runner():
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

@pysnooper.snoop()
def test_compile_command(runner, sample_neural):
    """Test the compile command with a sample .neural file."""
    output_file = "sample_tensorflow.py"
    with runner.isolated_filesystem():
        # Copy the sample file into the isolated filesystem
        with open(sample_neural, 'r') as src:
            content = src.read()
        isolated_sample = "sample.neural"
        with open(isolated_sample, 'w') as dest:
            dest.write(content)
        
        # Invoke the compile command within the isolated environment
        result = runner.invoke(cli, ["compile", isolated_sample, "--backend", "tensorflow", "--output", output_file])
        
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert os.path.exists(output_file), "Output file not found"
        assert "Compiled" in result.output
def test_compile_invalid_file(runner):
    """Test compile with a non-existent file."""
    result = runner.invoke(cli, ["compile", "nonexistent.neural", "--backend", "tensorflow"])
    assert result.exit_code != 0
    assert "does not exist" in result.output

def test_version_command(runner):
    """Test the version command."""
    result = runner.invoke(cli, ["version"])
    assert result.exit_code == 0
    assert "Neural CLI v0.1" in result.output