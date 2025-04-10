# NeuralDbg Dashboard

<p align="center">
  <img src="../../docs/images/dashboard_overview.png" alt="Dashboard Overview" width="600"/>
</p>

## Overview

NeuralDbg is a powerful debugging and visualization dashboard for neural networks. It provides real-time monitoring, profiling, and analysis tools to help you understand and optimize your models. The dashboard offers an interactive web interface that displays detailed information about your model's architecture, execution, and performance.

## Key Features

1. **Real-Time Execution Monitoring**: Track activations, gradients, memory usage, and FLOPs as your model runs.
2. **Shape Propagation Debugging**: Visualize tensor transformations at each layer.
3. **Gradient Flow Analysis**: Detect vanishing and exploding gradients.
4. **Dead Neuron Detection**: Identify inactive neurons in deep networks.
5. **Anomaly Detection**: Spot NaNs, extreme activations, and weight explosions.
6. **Step Debugging Mode**: Pause execution and inspect tensors manually.
7. **Performance Profiling**: Analyze computational and memory bottlenecks.

## Components

### 1. Dashboard Server (`server.py`)

The main server component that:
- Hosts the web interface
- Processes and serves data from the model
- Manages WebSocket connections for real-time updates

### 2. Data Collectors

Components that collect data from the running model:
- **Activation Collector**: Captures layer activations
- **Gradient Collector**: Captures gradient information
- **Memory Profiler**: Tracks memory usage
- **FLOP Counter**: Counts floating-point operations
- **Anomaly Detector**: Identifies unusual patterns

### 3. Visualizers

Components that create visualizations for the dashboard:
- **Network Visualizer**: Shows the model architecture
- **Activation Visualizer**: Visualizes layer activations
- **Gradient Visualizer**: Shows gradient flow
- **Performance Visualizer**: Displays performance metrics

### 4. Web Interface

The frontend components of the dashboard:
- **Interactive UI**: Built with Dash and Plotly
- **Real-Time Updates**: Uses WebSockets for live data
- **Customizable Views**: Configurable dashboard layouts
- **Export Options**: Save visualizations and reports

## Usage

### Starting the Dashboard

```python
from neural.dashboard.dashboard import start_dashboard
from neural.parser.parser import create_parser, ModelTransformer

# Parse a neural network definition
parser = create_parser('network')
with open('model.neural', 'r') as f:
    neural_code = f.read()

parsed = parser.parse(neural_code)
model_data = ModelTransformer().transform(parsed)

# Start the dashboard
start_dashboard(model_data, port=8050)
```

Or use the CLI:

```bash
neural debug model.neural
```

### Dashboard Modes

The dashboard supports several debugging modes:

1. **Standard Mode**: General monitoring and visualization
   ```bash
   neural debug model.neural
   ```

2. **Gradient Analysis Mode**: Focus on gradient flow
   ```bash
   neural debug --gradients model.neural
   ```

3. **Dead Neuron Detection**: Identify inactive neurons
   ```bash
   neural debug --dead-neurons model.neural
   ```

4. **Anomaly Detection**: Find unusual patterns
   ```bash
   neural debug --anomalies model.neural
   ```

5. **Step Debugging**: Pause and inspect tensors
   ```bash
   neural debug --step model.neural
   ```

## Dashboard Interface

### 1. Overview Panel

<p align="center">
  <img src="../../docs/images/dashboard_overview_panel.png" alt="Overview Panel" width="400"/>
</p>

Provides a high-level summary of the model:
- Model architecture
- Layer count and types
- Parameter count
- Memory requirements
- Computational complexity

### 2. Execution Trace Panel

<p align="center">
  <img src="../../docs/images/dashboard_execution_trace.png" alt="Execution Trace" width="400"/>
</p>

Shows the execution flow of the model:
- Layer-by-layer execution
- Timing information
- Activation statistics
- Memory usage

### 3. Gradient Flow Panel

<p align="center">
  <img src="../../docs/images/dashboard_gradient_flow.png" alt="Gradient Flow" width="400"/>
</p>

Visualizes gradient information:
- Gradient magnitudes
- Vanishing/exploding gradient detection
- Gradient flow through layers

### 4. Anomaly Detection Panel

<p align="center">
  <img src="../../docs/images/dashboard_anomaly_detection.png" alt="Anomaly Detection" width="400"/>
</p>

Highlights unusual patterns:
- NaN values
- Extreme activations
- Weight explosions
- Dead neurons

### 5. Performance Panel

<p align="center">
  <img src="../../docs/images/dashboard_performance.png" alt="Performance Panel" width="400"/>
</p>

Provides performance metrics:
- Memory usage
- FLOP counts
- Execution time
- Bottleneck identification

## Integration with Other Components

The Dashboard module integrates with:

1. **Parser**: Uses the model representation from the parser.
2. **Shape Propagation**: Uses shape information for visualizations.
3. **Visualization**: Uses visualization components for the dashboard.
4. **CLI**: Accessible through the CLI interface.

## Extension Points

The Dashboard module is designed to be extensible:

1. **Custom Panels**: Add new dashboard panels for specific analyses.
2. **Data Collectors**: Implement new data collectors for additional metrics.
3. **Visualization Types**: Add new visualization types to the dashboard.
4. **Export Formats**: Support additional export formats for reports.

## Resources

- [Dash Documentation](https://dash.plotly.com/)
- [Plotly Documentation](https://plotly.com/python/)
- [WebSocket Documentation](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API)
- [Neural DSL Reference](../../docs/DSL.md)
- [Dashboard Tutorial](../../docs/tutorials/dashboard.md)
