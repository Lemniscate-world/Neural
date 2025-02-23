#!/usr/bin/env python
import os
import sys
import subprocess
import click
import logging
import hashlib
import shutil

# Add the parent directory of 'neural' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neural.parser.parser import create_parser, ModelTransformer
from neural.code_generation.code_generator import generate_code

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """Neural CLI: A compiler-like interface for .neural and .nr files."""
    pass

@cli.command()
@click.argument('file', type=click.Path(exists=True))
@click.option('--backend', default='tensorflow', help='Target backend: tensorflow or pytorch', type=click.Choice(['tensorflow', 'pytorch']))
@click.option('--verbose', is_flag=True, help='Show verbose output')
@click.option('--output', default=None, help='Output file path for generated code or visualizations')
def compile(file, backend, verbose, output):
    """Compile a .neural or .nr file into an executable Python script."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    ext = os.path.splitext(file)[1].lower()
    if ext in ['.neural', '.nr']:
        parser_instance = create_parser('network')
    elif ext == '.rnr':
        parser_instance = create_parser('research')
    else:
        click.echo(f"Unsupported file type: {ext}")
        sys.exit(1)

    with open(file, 'r') as f:
        content = f.read()
    
    try:
        tree = parser_instance.parse(content)
    except Exception as e:
        click.echo(f"Error parsing {file}: {e}")
        sys.exit(1)

    transformer = ModelTransformer()
    try:
        model_data = transformer.transform(tree)
    except Exception as e:
        click.echo(f"Error transforming {file}: {e}")
        sys.exit(1)

    code = generate_code(model_data, backend)
    output_file = output or os.path.splitext(file)[0] + f"_{backend}.py"
    with open(output_file, 'w') as f:
        f.write(code)
    click.echo(f"Compiled {file} to {output_file} for backend {backend}")

# Other commands (run, visualize, etc.) remain as-is...
# Here's an abbreviated version for brevity; include your full commands as needed
@cli.command()
@click.argument('file', type=click.Path(exists=True))
def run(file):
    """Run an executable neural model."""
    subprocess.run([sys.executable, file], check=True)

@cli.command()
def version():
    """Show the version of Neural CLI and dependencies."""
    click.echo(f"Neural CLI v0.1")
    click.echo(f"Python: {sys.version}")
    click.echo(f"Click: {click.__version__}")

if __name__ == '__main__':
    cli()