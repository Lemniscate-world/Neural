# Neural Visualizer

The Neural Visualizer provides powerful tools to visualize neural network architectures, shape propagation, and tensor flow in multiple formats.

## Overview

The visualization module consists of two main components:

1. **Static Visualizer**: Generates self-contained visualizations of neural network architecture in PNG, SVG, or HTML formats
   - Creates standalone files that can be shared and viewed without a server
   - Includes both fixed diagrams and self-contained interactive visualizations
   - Uses Matplotlib, Graphviz, and Plotly for rendering

2. **Dynamic Visualizer**: Provides a web application with D3.js for exploring network architecture
   - Runs as a service with real-time updates
   - Allows direct manipulation of the visualization
   - Connects to a backend server for processing Neural DSL code

## Installation

The visualizer is included with the Neural package. Ensure you have the required dependencies:

```bash
pip install tensorflow matplotlib graphviz plotly numpy
```

For the dynamic visualizer, you'll also need:
```bash
pip install flask flask-cors
```

## Usage

### Command Line

```bash
neural visualize model.neural --format png
```

Options:
- `--format, -f`: Output format (`html`, `png`, `svg`) [default: html]
- `--cache/--no-cache`: Use cached visualizations [default: True]

### Output Files

When using the HTML format (default), three visualization files are generated:

1. `architecture.svg` - Network architecture diagram
2. `shape_propagation.html` - Interactive 3D parameter count and shape visualization
3. `tensor_flow.html` - Animated data flow visualization

## Static Visualizer

The static visualizer generates visualizations that can be viewed without a server:

1. **Architecture Diagrams**: Clean, structured diagrams showing the network structure
2. **Interactive Shape Visualizations**: Self-contained HTML files with 3D interactive plots
3. **Tensor Flow Animations**: Pre-rendered animations of data flow

### Example

```python
from neural.parser.parser import create_parser, ModelTransformer
from neural.visualization.static_visualizer.visualizer import NeuralVisualizer

# Parse a neural network definition
parser = create_parser('network')
parsed = parser.parse(neural_code)
model_data = ModelTransformer().transform(parsed)

# Create visualizations
visualizer = NeuralVisualizer(model_data)
d3_data = visualizer.model_to_d3_json()  # Used by both static and dynamic visualizers

# For 3D visualization of shape propagation
shape_history = [('Input', (None, 28, 28, 1)),
                ('Conv2D', (None, 26, 26, 32)),
                ('MaxPooling2D', (None, 13, 13, 32)),
                ('Flatten', (None, 5408)),
                ('Dense', (None, 128)),
                ('Output', (None, 10))]
fig = visualizer.create_3d_visualization(shape_history)
fig.write_html('shape_propagation.html')  # Self-contained interactive HTML
```

## Dynamic Visualizer

The dynamic visualizer provides a web application that allows you to:
- Drag and rearrange network components
- Zoom in/out of the architecture
- Hover over nodes to see detailed information
- Update visualizations in real-time as you modify the Neural DSL code

### Running the Dynamic Visualizer

1. Start the visualization server:
```bash
cd neural/visualization/dynamic_visualizer
./start.sh
```

2. Open your browser to `http://localhost:8000`

3. Enter your Neural DSL code in the editor and click "Visualize"

### API Integration

The dynamic visualizer includes a Flask API server that parses Neural DSL code and returns visualization data:

```python
# Example API request
import requests

response = requests.post('http://localhost:5000/parse',
                        headers={'Content-Type': 'text/plain'},
                        data=neural_code)
visualization_data = response.json()
```

## Visualization Types

### Architecture Diagram

Shows the network structure with layer types and connections.

### Shape Propagation

Interactive 3D visualization of tensor shapes through the network:

```python
def create_3d_visualization(self, shape_history):
    fig = go.Figure()

    for i, (name, shape) in enumerate(shape_history):
        fig.add_trace(go.Scatter3d(
            x=[i]*len(shape),
            y=list(range(len(shape))),
            z=shape,
            mode='markers+text',
            text=[str(d) for d in shape],
            name=name
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='Layer Depth',
            yaxis_title='Dimension Index',
            zaxis_title='Dimension Size'
        )
    )
    return fig
```

### Tensor Flow

Animated visualization of data as it flows through your network.

## Integration with Dashboard

The visualizer integrates with the Neural Dashboard to provide real-time execution tracing, including:

- Layer-wise execution trace
- Memory & FLOP profiling
- Gradient flow analysis
- Dead neuron detection
- Anomaly detection

## Performance

The visualization system has been optimized for performance:

```
# Before optimization: ~500ms for complex models
# After optimization: ~150ms for the same models
```

## Examples

See the `examples` directory for sample visualizations and code.
