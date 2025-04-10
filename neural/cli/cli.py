#!/usr/bin/env python
import os
import sys
import subprocess
import click
import logging
from typing import Optional
import hashlib
import shutil
from pathlib import Path
from lark import exceptions


# Import CLI aesthetics
from .cli_aesthetics import (
    print_neural_logo, print_command_header, print_success,
    print_error, print_warning, print_info, Spinner,
    progress_bar, animate_neural_network, Colors,
    print_help_command
)

# Import welcome message
from .welcome_message import show_welcome_message

# Import version from version module
from .version import __version__

def configure_logging(verbose=False):
    """Configure logging levels based on verbosity"""
    # Set environment variables to suppress debug messages from dependencies
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow messages
    os.environ['PYTHONWARNINGS'] = 'ignore'    # Suppress Python warnings
    os.environ['MPLBACKEND'] = 'Agg'           # Non-interactive matplotlib backend

    # First, set all loggers to ERROR level by default
    logging.basicConfig(
        level=logging.ERROR,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Get the root logger and set its level
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.ERROR)

    # Configure our own logger
    neural_logger = logging.getLogger('neural')
    if verbose:
        neural_logger.setLevel(logging.DEBUG)
        # Create a formatter that includes more details
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        # Create a new handler with this formatter
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        # Remove existing handlers and add our new one
        neural_logger.handlers = []
        neural_logger.addHandler(handler)
    else:
        neural_logger.setLevel(logging.INFO)
        # Create a simpler formatter
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        # Create a new handler with this formatter
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        # Remove existing handlers and add our new one
        neural_logger.handlers = []
        neural_logger.addHandler(handler)

    # Explicitly silence noisy libraries by setting them to CRITICAL level
    for logger_name in [
        'graphviz', 'matplotlib', 'tensorflow', 'jax', 'tf', 'absl',
        'pydot', 'PIL', 'torch', 'urllib3', 'requests', 'h5py',
        'filelock', 'numba', 'asyncio', 'parso', 'werkzeug',
        'matplotlib.font_manager', 'matplotlib.ticker', 'optuna',
        'dash', 'plotly', 'ipykernel', 'traitlets', 'click'
    ]:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)
        # Also disable propagation to the root logger
        logging.getLogger(logger_name).propagate = False

    # Redirect stderr to /dev/null to suppress any remaining debug messages
    # that might be printed directly to stderr
    if not verbose:
        # Only do this in non-verbose mode
        try:
            # Save the original stderr
            original_stderr = sys.stderr
            # Open /dev/null for writing
            null_fd = open(os.devnull, 'w')
            # Replace stderr with /dev/null
            sys.stderr = null_fd
            # Register a cleanup function to restore stderr when the program exits
            import atexit
            def restore_stderr():
                sys.stderr = original_stderr
                null_fd.close()
            atexit.register(restore_stderr)
        except Exception:
            # If anything goes wrong, just continue with the original stderr
            pass

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import lightweight modules directly
from neural.parser.parser import create_parser, ModelTransformer, DSLValidationError

# Import lazy loaders for heavy modules
from .lazy_imports import (
    # Lazy Neural modules
    shape_propagator as shape_propagator_module,
    tensor_flow as tensor_flow_module,
    hpo as hpo_module,
    code_generator as code_generator_module,
    # Get actual module function
    get_module
)

# Create logger for this module
logger = logging.getLogger(__name__)

# Supported datasets (extend as implemented)
SUPPORTED_DATASETS = {"MNIST", "CIFAR10", "CIFAR100", "ImageNet"}

# Global CLI context for shared options
@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.version_option(version=__version__, prog_name="Neural")
def cli(verbose: bool):
    """Neural CLI: A compiler-like interface for .neural and .nr files."""
    configure_logging(verbose)

    # Show welcome message if it's the first time the CLI is run
    # The welcome message already includes the logo, so we only need to show
    # the logo separately if the welcome message is not shown
    welcome_shown = False
    if not hasattr(cli, '_welcome_shown'):
        welcome_shown = show_welcome_message()
        setattr(cli, '_welcome_shown', True)

    # Show logo if welcome message wasn't displayed
    if not welcome_shown:
        print_neural_logo(__version__)

    if verbose:
        logger.debug("Verbose mode enabled")

# Override the help command to use our custom help formatter
@cli.command(help="Show this message and exit.", add_help_option=False)
@click.pass_context
def help(ctx):
    """Show help for commands."""
    print_help_command(ctx, cli.commands)

# Compile command
@cli.command()
@click.argument('file', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option('--backend', '-b', default='tensorflow', help='Target backend', type=click.Choice(['tensorflow', 'pytorch', 'onnx'], case_sensitive=False))
@click.option('--dataset', default='MNIST', help='Dataset name (e.g., MNIST, CIFAR10)')
@click.option('--output', '-o', default=None, help='Output file path (defaults to <file>_<backend>.py)')
@click.option('--dry-run', is_flag=True, help='Preview generated code without writing to file')
@click.option('--hpo', is_flag=True, help='Enable hyperparameter optimization')
def compile(file: str, backend: str, dataset: str, output: Optional[str], dry_run: bool, hpo: bool):
    """Compile a .neural or .nr file into an executable Python script.

    Example: neural compile my_model.neural --backend pytorch --output model.py --hpo
    """
    print_command_header("compile")
    print_info(f"Compiling {file} for {backend} backend")

    # Validate file type
    ext = os.path.splitext(file)[1].lower()
    start_rule = 'network' if ext in ['.neural', '.nr'] else 'research' if ext == '.rnr' else None
    if not start_rule:
        print_error(f"Unsupported file type: {ext}")
        print_info(f"Supported file types: .neural, .nr, .rnr")
        sys.exit(1)

    # Parse the Neural DSL file
    with Spinner("Parsing Neural DSL file"):
        parser_instance = create_parser(start_rule=start_rule)
        with open(file, 'r') as f:
            content = f.read()

        try:
            tree = parser_instance.parse(content)
            model_data = ModelTransformer().transform(tree)
        except (exceptions.UnexpectedCharacters, exceptions.UnexpectedToken, DSLValidationError) as e:
            print_error(f"Parsing failed: {e}")
            # Show the line with the error
            try:
                if hasattr(e, 'line') and hasattr(e, 'column') and e.line is not None and e.column is not None:
                    lines = content.split('\n')
                    line_num = int(e.line) - 1
                    if 0 <= line_num < len(lines):
                        error_line = lines[line_num]
                        print(f"\nLine {e.line}:")
                        print(f"{error_line}")
                        print(f"{' ' * max(0, int(e.column) - 1)}^")
                        print(f"Error at column {e.column}")
            except (TypeError, ValueError, AttributeError):
                # If there's any issue with the error attributes, just show the error message
                pass
            sys.exit(1)

    # Run hyperparameter optimization if requested
    if hpo:
        print_info("Running hyperparameter optimization")
        if dataset not in SUPPORTED_DATASETS:
            print_warning(f"Dataset '{dataset}' may not be supported")
            print(f"Supported datasets: {', '.join(sorted(SUPPORTED_DATASETS))}")

        # Lazy load the HPO module
        optimize_and_return = get_module(hpo_module).optimize_and_return
        generate_optimized_dsl = get_module(code_generator_module).generate_optimized_dsl

        with Spinner(f"Optimizing hyperparameters for {dataset} dataset"):
            best_params = optimize_and_return(content, n_trials=3, dataset_name=dataset, backend=backend)

        print_success("Hyperparameter optimization complete!")
        print(f"\n{Colors.CYAN}Best Parameters:{Colors.ENDC}")
        for param, value in best_params.items():
            print(f"  {Colors.BOLD}{param}:{Colors.ENDC} {value}")

        with Spinner("Generating optimized DSL code"):
            content = generate_optimized_dsl(content, best_params)

    # Generate code
    with Spinner(f"Generating {backend} code"):
        try:
            # Lazy load the code generator module
            generate_code = get_module(code_generator_module).generate_code
            code = generate_code(model_data, backend)
        except Exception as e:
            print_error(f"Code generation failed: {e}")
            sys.exit(1)

    # Output the generated code
    output_file = output or f"{os.path.splitext(file)[0]}_{backend}.py"
    if dry_run:
        print_info("Generated code (dry run)")
        print(f"\n{Colors.CYAN}" + "="*50 + f"{Colors.ENDC}")
        print(code)
        print(f"{Colors.CYAN}" + "="*50 + f"{Colors.ENDC}")
        print_warning("Dry run - code not saved to file")
    else:
        with Spinner(f"Writing code to {output_file}"):
            with open(output_file, 'w') as f:
                f.write(code)

        print_success(f"Compilation successful!")
        print(f"\n{Colors.CYAN}Output:{Colors.ENDC}")
        print(f"  {Colors.BOLD}File:{Colors.ENDC} {output_file}")
        print(f"  {Colors.BOLD}Backend:{Colors.ENDC} {backend}")
        print(f"  {Colors.BOLD}Size:{Colors.ENDC} {len(code)} bytes")

        # Show a brief animation
        print("\nNeural network structure:")
        animate_neural_network(2)

# Run command
@cli.command()
@click.argument('file', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option('--backend', '-b', default='tensorflow', help='Backend to run', type=click.Choice(['tensorflow', 'pytorch'], case_sensitive=False))
@click.option('--dataset', default='MNIST', help='Dataset name (e.g., MNIST, CIFAR10)')
@click.option('--hpo', is_flag=True, help='Enable HPO for .neural files')
def run(file: str, backend: str, dataset: str, hpo: bool):
    ext = os.path.splitext(file)[1].lower()
    if ext == '.py':
        logger.info(f"Running {file} with {backend} backend")
        try:
            subprocess.run([sys.executable, file], check=True)
            logger.info("Execution completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Execution failed with exit code {e.returncode}")
            sys.exit(e.returncode)
    elif ext in ['.neural', '.nr'] and hpo:
        logger.info(f"Optimizing and running {file} with {backend} backend")
        start_rule = 'network' if ext in ['.neural', '.nr'] else None
        if not start_rule:
            logger.error(f"Unsupported file type for HPO: {ext}")
            sys.exit(1)

        with open(file, 'r') as f:
            content = f.read()

        if dataset not in SUPPORTED_DATASETS:
            logger.warning("Dataset '%s' may not be supported. Supported: %s", dataset, SUPPORTED_DATASETS)

        # Optimize and generate code
        # Lazy load the HPO and code generator modules
        optimize_and_return = get_module(hpo_module).optimize_and_return
        generate_optimized_dsl = get_module(code_generator_module).generate_optimized_dsl
        generate_code = get_module(code_generator_module).generate_code

        best_params = optimize_and_return(content, n_trials=3, dataset_name=dataset, backend=backend)
        logger.info("Best parameters found: %s", best_params)
        optimized_config = generate_optimized_dsl(content, best_params)

        output_file = f"{os.path.splitext(file)[0]}_optimized_{backend}.py"
        parser_instance = create_parser(start_rule=start_rule)
        try:
            tree = parser_instance.parse(optimized_config)
            model_data = ModelTransformer().transform(tree)
            code = generate_code(model_data, backend, best_params=best_params)
            with open(output_file, 'w') as f:
                f.write(code)
            logger.info(f"Compiled optimized {file} to {output_file}")
        except Exception as e:
            logger.error(f"Optimization or code generation failed: {e}")
            sys.exit(1)

        # Run the compiled file
        try:
            subprocess.run([sys.executable, output_file], check=True)
            logger.info("Execution completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Execution failed with exit code {e.returncode}")
            sys.exit(e.returncode)
    else:
        logger.error(f"Expected a .py file, got {ext}. Use 'compile' first or add --hpo for .neural files.")
        sys.exit(1)

# Visualize command
@cli.command()
@click.argument('file', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option('--format', '-f', default='html', help='Output format', type=click.Choice(['html', 'png', 'svg'], case_sensitive=False))
@click.option('--cache/--no-cache', default=True, help='Use cached visualizations if available')
def visualize(file: str, format: str, cache: bool):
    """Visualize network architecture and shape propagation.

    Example: neural visualize my_model.neural --format png --no-cache
    """
    print_command_header("visualize")
    print_info(f"Visualizing {file} in {format} format")

    ext = os.path.splitext(file)[1].lower()
    start_rule = 'network' if ext in ['.neural', '.nr'] else 'research' if ext == '.rnr' else None
    if not start_rule:
        print_error(f"Unsupported file type: {ext}")
        sys.exit(1)

    cache_dir = Path(".neural_cache")
    cache_dir.mkdir(exist_ok=True)
    file_hash = hashlib.sha256(Path(file).read_bytes()).hexdigest()
    cache_file = cache_dir / f"viz_{file_hash}_{format}"

    if cache and cache_file.exists():
        print_info(f"Using cached visualization")
        with Spinner("Copying cached visualization"):
            shutil.copy(cache_file, f"architecture.{format}")
        print_success(f"Cached visualization copied to architecture.{format}")
        return

    # Parse the Neural DSL file
    with Spinner("Parsing Neural DSL file"):
        parser_instance = create_parser(start_rule=start_rule)
        with open(file, 'r') as f:
            content = f.read()

        try:
            tree = parser_instance.parse(content)
            model_data = ModelTransformer().transform(tree)
        except Exception as e:
            print_error(f"Processing {file} failed: {e}")
            sys.exit(1)

    # Lazy load the shape propagator module
    ShapePropagator = get_module(shape_propagator_module).ShapePropagator

    # Propagate shapes through the network
    propagator = ShapePropagator()
    input_shape = model_data['input']['shape']
    if not input_shape:
        print_error("Input shape not defined in model")
        sys.exit(1)

    print_info("Propagating shapes through the network...")
    shape_history = []
    total_layers = len(model_data['layers'])

    for i, layer in enumerate(model_data['layers']):
        progress_bar(i, total_layers, prefix='Progress:', suffix=f'Layer: {layer["type"]}', length=40)
        input_shape = propagator.propagate(input_shape, layer, model_data.get('framework', 'tensorflow'))
        shape_history.append({"layer": layer['type'], "output_shape": input_shape})

    progress_bar(total_layers, total_layers, prefix='Progress:', suffix='Complete', length=40)

    # Generate visualizations
    with Spinner("Generating visualizations"):
        report = propagator.generate_report()
        dot = report['dot_graph']
        dot.format = format if format != 'html' else 'svg'
        dot.render('architecture', cleanup=True)

        if format == 'html':
            report['plotly_chart'].write_html('shape_propagation.html')
            # Lazy load the tensor flow module
            create_animated_network = get_module(tensor_flow_module).create_animated_network
            create_animated_network(shape_history).write_html('tensor_flow.html')

    # Show success message with animation
    if format == 'html':
        print_success("Visualizations generated successfully!")
        print(f"{Colors.CYAN}Files created:{Colors.ENDC}")
        print(f"  - {Colors.GREEN}architecture.svg{Colors.ENDC} (Network architecture)")
        print(f"  - {Colors.GREEN}shape_propagation.html{Colors.ENDC} (Parameter count chart)")
        print(f"  - {Colors.GREEN}tensor_flow.html{Colors.ENDC} (Data flow animation)")

        # Show a brief animation of a neural network
        print("\nNeural network data flow animation:")
        animate_neural_network(3)
    else:
        print_success(f"Visualization saved as architecture.{format}")

    # Cache the visualization
    if cache:
        with Spinner("Caching visualization for future use"):
            shutil.copy(f"architecture.{format}", cache_file)
        print_info(f"Visualization cached for future use")

# Clean command
@cli.command()
def clean():
    """Remove generated files (e.g., .py, .png, .svg, .html, cache)."""
    print_command_header("clean")
    print_info("Cleaning up generated files...")

    extensions = ['.py', '.png', '.svg', '.html']
    removed_files = []

    with Spinner("Scanning for generated files"):
        for file in os.listdir('.'):
            if any(file.endswith(ext) for ext in extensions):
                os.remove(file)
                removed_files.append(file)

    if removed_files:
        print_success(f"Removed {len(removed_files)} generated files")
        for file in removed_files[:5]:  # Show first 5 files
            print(f"  - {file}")
        if len(removed_files) > 5:
            print(f"  - ...and {len(removed_files) - 5} more")

    if os.path.exists(".neural_cache"):
        with Spinner("Removing cache directory"):
            shutil.rmtree(".neural_cache")
        print_success("Removed cache directory")

    if not removed_files and not os.path.exists(".neural_cache"):
        print_warning("No files to clean")

# Version command
@cli.command()
def version():
    """Show the version of Neural CLI and dependencies."""
    # We don't need to print the logo here because it's already printed by the CLI
    # Get additional dependency versions
    import lark

    # Create a table-like output for dependencies
    print(f"\n{Colors.CYAN}System Information:{Colors.ENDC}")
    print(f"  {Colors.BOLD}Python:{Colors.ENDC}      {sys.version.split()[0]}")
    print(f"  {Colors.BOLD}Platform:{Colors.ENDC}    {sys.platform}")

    print(f"\n{Colors.CYAN}Core Dependencies:{Colors.ENDC}")
    print(f"  {Colors.BOLD}Click:{Colors.ENDC}       {click.__version__}")
    print(f"  {Colors.BOLD}Lark:{Colors.ENDC}        {lark.__version__}")

    print(f"\n{Colors.CYAN}ML Frameworks:{Colors.ENDC}")
    for pkg in ('torch', 'tensorflow', 'jax', 'optuna'):
        try:
            ver = __import__(pkg).__version__
            print(f"  {Colors.BOLD}{pkg.capitalize()}:{Colors.ENDC}" + " " * (12 - len(pkg)) + f"{ver}")
        except ImportError:
            print(f"  {Colors.BOLD}{pkg.capitalize()}:{Colors.ENDC}" + " " * (12 - len(pkg)) + f"{Colors.YELLOW}Not installed{Colors.ENDC}")

    # Show a brief animation
    print("\nNeural is ready to build amazing neural networks!")

# Debug command
@cli.command()
@click.argument('file', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option('--gradients', is_flag=True, help='Analyze gradient flow')
@click.option('--dead-neurons', is_flag=True, help='Detect dead neurons')
@click.option('--anomalies', is_flag=True, help='Detect training anomalies')
@click.option('--step', is_flag=True, help='Enable step debugging mode')
@click.option('--backend', '-b', default='tensorflow', help='Backend for runtime', type=click.Choice(['tensorflow', 'pytorch'], case_sensitive=False))
@click.option('--dataset', default='MNIST', help='Dataset name (e.g., MNIST, CIFAR10)')
def debug(file: str, gradients: bool, dead_neurons: bool, anomalies: bool, step: bool, backend: str, dataset: str):
    """Debug a neural network model with NeuralDbg.

    Example: neural debug my_model.neural --gradients --step --backend pytorch
    """
    print_command_header("debug")
    print_info(f"Debugging {file} with NeuralDbg (backend: {backend})")

    ext = os.path.splitext(file)[1].lower()
    start_rule = 'network' if ext in ['.neural', '.nr'] else 'research' if ext == '.rnr' else None
    if not start_rule:
        print_error(f"Unsupported file type: {ext}")
        sys.exit(1)

    if dataset not in SUPPORTED_DATASETS:
        print_warning(f"Dataset '{dataset}' may not be supported.")
        print(f"Supported datasets: {', '.join(sorted(SUPPORTED_DATASETS))}")

    # Parse the Neural DSL file
    with Spinner("Parsing Neural DSL file"):
        parser_instance = create_parser(start_rule=start_rule)
        with open(file, 'r') as f:
            content = f.read()

        try:
            tree = parser_instance.parse(content)
            model_data = ModelTransformer().transform(tree)
        except Exception as e:
            print_error(f"Processing {file} failed: {e}")
            sys.exit(1)

    # Shape propagation for baseline
    print_info("Analyzing model architecture...")
    with Spinner("Propagating shapes through the network"):
        # Lazy load the shape propagator module
        ShapePropagator = get_module(shape_propagator_module).ShapePropagator
        propagator = ShapePropagator(debug=True)
        input_shape = model_data['input']['shape']
        for layer in model_data['layers']:
            input_shape = propagator.propagate(input_shape, layer, backend)
        trace_data = propagator.get_trace()

    print_success("Model analysis complete!")

    # Debugging modes
    if gradients:
        print(f"\n{Colors.CYAN}Gradient Flow Analysis{Colors.ENDC}")
        print_warning("Gradient flow analysis is simulated (runtime integration coming soon)")
        print("\nGradient flow trace:")
        for entry in trace_data:
            print(f"  Layer {Colors.BOLD}{entry['layer']}{Colors.ENDC}: mean_activation = {entry.get('mean_activation', 'N/A')}")

    if dead_neurons:
        print(f"\n{Colors.CYAN}Dead Neuron Detection{Colors.ENDC}")
        print_warning("Dead neuron detection is simulated (runtime integration coming soon)")
        print("\nDead neuron trace:")
        for entry in trace_data:
            print(f"  Layer {Colors.BOLD}{entry['layer']}{Colors.ENDC}: active_ratio = {entry.get('active_ratio', 'N/A')}")

    if anomalies:
        print(f"\n{Colors.CYAN}Anomaly Detection{Colors.ENDC}")
        print_warning("Anomaly detection is simulated (runtime integration coming soon)")
        print("\nAnomaly trace:")
        anomaly_found = False
        for entry in trace_data:
            if 'anomaly' in entry:
                print(f"  Layer {Colors.BOLD}{entry['layer']}{Colors.ENDC}: {entry['anomaly']}")
                anomaly_found = True
        if not anomaly_found:
            print("  No anomalies detected")

    if step:
        print(f"\n{Colors.CYAN}Step Debugging Mode{Colors.ENDC}")
        print_info("Stepping through network layer by layer...")

        for i, layer in enumerate(model_data['layers']):
            input_shape = propagator.propagate(input_shape, layer, backend)
            print(f"\n{Colors.BOLD}Step {i+1}/{len(model_data['layers'])}{Colors.ENDC}: {layer['type']}")
            print(f"  Output Shape: {input_shape}")
            if 'params' in layer and layer['params']:
                print(f"  Parameters: {layer['params']}")

            if click.confirm("Continue?", default=True):
                continue
            else:
                print_info("Debugging paused by user")
                break

    # Show completion message with animation
    print("\n" + "="*50)
    print_success("Debug session completed!")
    print_info("Full runtime debugging with NeuralDbg coming soon!")
    animate_neural_network(2)

# No-code command
@cli.command(name='no-code')
@click.option('--port', default=8051, help='Web interface port', type=int)
def no_code(port: int):
    """Launch the no-code interface for building models.

    Example: neural no-code --port 8051
    """
    print_command_header("no-code")
    print_info("Launching the Neural no-code interface...")

    # Import dashboard module
    with Spinner("Loading dashboard components"):
        from neural.dashboard.dashboard import app

    # Display server information
    print_success(f"Dashboard ready!")
    print(f"\n{Colors.CYAN}Server Information:{Colors.ENDC}")
    print(f"  {Colors.BOLD}URL:{Colors.ENDC}         http://localhost:{port}")
    print(f"  {Colors.BOLD}Interface:{Colors.ENDC}   Neural No-Code Builder")
    print(f"\n{Colors.YELLOW}Press Ctrl+C to stop the server{Colors.ENDC}")

    # Start the server
    try:
        app.run_server(debug=False, host="localhost", port=port)
    except Exception as e:
        print_error(f"Failed to launch no-code interface: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print_info("\nServer stopped by user")

# Remove standalone HPO command (integrated into run/compile)
# If you want to keep it separate, uncomment and adjust:
# @cli.command()
# @click.argument('file')
# @click.option('--n-trials', default=20)
# @click.option('--dataset', default='MNIST')
# def hpo(file, n_trials, dataset):
#     """Run hyperparameter optimization"""
#     with open(file) as f:
#         config = f.read()
#     best_params = optimize_and_return(config, n_trials, dataset)
#     click.echo(f"Best parameters: {best_params}")

if __name__ == '__main__':
    cli()
