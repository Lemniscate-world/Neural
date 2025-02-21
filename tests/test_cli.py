import click.testing
import pytest
import os

def test_compile_command():
    runner = click.testing.CliRunner()
    result = runner.invoke(cli, ["compile", "tests/sample.neural", "--backend", "tensorflow"])
    assert result.exit_code == 0
    assert os.path.exists("sample_tensorflow.py")