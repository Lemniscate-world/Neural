# Neural CLI

<p align="center">
  <img src="../../docs/images/cli_diagram.png" alt="CLI Architecture" width="600"/>
</p>

This directory contains the command-line interface for Neural, providing a user-friendly way to interact with the Neural DSL. The CLI is designed to be intuitive, visually appealing, and efficient, with optimizations for fast startup time and responsive user experience.

## Components

### `cli.py`

The main CLI implementation using the Click library. It provides commands for:

- **Visualizing** neural networks
- **Compiling** Neural DSL code to different backends
- **Running** compiled models
- **Debugging** models with NeuralDbg
- **Launching** the no-code interface

### `cli_aesthetics.py`

Provides visual enhancements for the CLI, including:

- **ASCII art** logos and headers
- **Colored output** for different message types
- **Progress bars** for long-running operations
- **Spinners** for operations with indeterminate duration
- **Animations** for neural network visualization

### `welcome_message.py`

Displays a welcome message when the CLI is first run, providing:

- **Introduction** to Neural
- **Command overview** with examples
- **Visual preview** of Neural's capabilities

### `lazy_imports.py`

Implements lazy loading for heavy dependencies, improving startup time by:

- **Deferring imports** until they're actually needed
- **Caching attributes** to avoid repeated lookups
- **Suppressing warnings** from dependencies

## Usage

The CLI can be used directly after installing the Neural package:

```bash
# Show version information
neural version

# Visualize a neural network
neural visualize my_model.neural

# Compile a model to TensorFlow
neural compile my_model.neural --backend tensorflow

# Run a compiled model
neural run my_model_tensorflow.py

# Debug a model with NeuralDbg
neural debug my_model.neural

# Launch the no-code interface
neural no-code
```

## CLI Architecture

The Neural CLI is built using the Click framework and follows a command-group pattern:

```
neutral (main command group)
├── version     - Display version information
├── help        - Show help information
├── compile     - Compile Neural DSL to backend code
├── run         - Run compiled models
├── visualize   - Generate visualizations
├── debug       - Launch the debugging dashboard
├── no-code     - Launch the no-code interface
└── clean       - Clean generated files
```

Each command has its own set of options and arguments, providing a consistent and intuitive interface for users.

## Command Details

### `compile`

Compiles Neural DSL code to executable code for a specific backend.

```bash
neural compile model.neural --backend tensorflow --output model_tf.py --hpo
```

Options:
- `--backend, -b`: Backend to compile to (tensorflow, pytorch, jax)
- `--output, -o`: Output file path
- `--hpo`: Enable hyperparameter optimization
- `--dataset`: Dataset to use for HPO
- `--dry-run`: Show generated code without saving

### `visualize`

Generates visualizations of neural network architecture.

```bash
neural visualize model.neural --format html
```

Options:
- `--format, -f`: Output format (html, png, svg)
- `--output, -o`: Output directory

### `debug`

Launches the NeuralDbg dashboard for debugging and analyzing models.

```bash
neural debug model.neural --gradients --anomalies
```

Options:
- `--gradients`: Enable gradient flow analysis
- `--dead-neurons`: Enable dead neuron detection
- `--anomalies`: Enable anomaly detection
- `--step`: Enable step debugging mode
- `--port`: Dashboard server port

## Performance Optimization

The CLI has been optimized for performance, particularly focusing on startup time. The main optimizations include:

1. **Lazy Loading**: Heavy dependencies like TensorFlow, PyTorch, and JAX are loaded only when needed
2. **Attribute Caching**: Frequently accessed attributes are cached to avoid repeated lookups
3. **Warning Suppression**: Debug messages and warnings are suppressed to improve the user experience
4. **Modular Design**: Commands are loaded on-demand to minimize initial loading time
5. **Efficient Resource Management**: Resources are allocated and released as needed

These optimizations have significantly improved the startup time of the CLI, especially for simple commands like `version` and `help` that don't require the heavy ML frameworks.

## Integration with Other Components

The CLI integrates with all other components of the Neural framework:

- **Parser**: Parses Neural DSL code into model representations
- **Code Generation**: Generates code for different backends
- **Shape Propagation**: Validates tensor shapes in models
- **Visualization**: Creates visualizations of model architecture
- **Dashboard**: Launches the debugging dashboard
- **HPO**: Performs hyperparameter optimization

## Extending the CLI

The CLI is designed to be extensible. To add a new command:

1. Define a new command function in `cli.py`
2. Decorate it with `@cli.command()`
3. Implement the command logic

Example:

```python
@cli.command()
@click.argument('file', type=click.Path(exists=True))
@click.option('--option', '-o', help='Command option')
def new_command(file, option):
    """Description of the new command."""
    # Command implementation
    print_info(f"Processing {file} with option {option}")
    # ...
```
