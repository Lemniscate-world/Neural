import tensorflow as tf
import torch
from torch.utils.tensorboard import SummaryWriter
from neural.code_generation.code_generator import generate_code

class TensorBoardLogger:
    def __init__(self, log_dir="runs/neural"):
        self.writer = SummaryWriter(log_dir)

    def log_metrics(self, metrics, step):
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)

    def log_model(self, model, step):
        if isinstance(model, tf.keras.Model):
            model.summary(print_fn=lambda x: self.writer.add_text("model_summary", x))
        elif isinstance(model, torch.nn.Module):
            self.writer.add_graph(model, torch.randn(1, *model.input_shape))

# Update ShapePropagator or training logic to log to TensorBoard
class ShapePropagator:
    def __init__(self, debug=False):
        # ... Existing init ...
        self.tensorboard_logger = TensorBoardLogger()

    def propagate(self, input_shape, layer, framework):
        # ... Existing propagate logic ...
        resources = self.monitor_resources()
        self.tensorboard_logger.log_metrics({
            "cpu_usage": resources["cpu_usage"],
            "memory_usage": resources["memory_usage"],
            "gpu_memory": resources["gpu_memory"],
            "io_usage": resources["io_usage"],
            "execution_time": trace_entry["execution_time"]
        }, self.current_layer)
        # Log model architecture if framework supports
        if self.current_layer == 0:
            self.tensorboard_logger.log_model(self.model, 0)  # Assume model is available

# CLI train command
@cli.command()
@click.argument('file', type=click.Path(exists=True))
@click.option('--backend', default='tensorflow', help='Target backend: tensorflow or pytorch')
@click.option('--log-dir', default='runs/neural', help='TensorBoard log directory')
def train(file, backend, log_dir):
    """Train a neural network model and log to TensorBoard."""
    from neural.parser.parser import create_parser
    parser_instance = create_parser('network' if os.path.splitext(file)[1].lower() in ['.neural', '.nr'] else 'research')
    with open(file, 'r') as f:
        content = f.read()
    tree = parser_instance.parse(content)
    model_data = ModelTransformer().transform(tree)
    code = generate_code(model_data, backend)
    save_file(f"model_{backend}.py", code)
    
    # Load and train the model (simplified)
    if backend == "tensorflow":
        import tensorflow as tf
        model = tf.keras.models.load_model(f"model_{backend}.py")  # Adjust for your code structure
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        model.fit(..., callbacks=[tensorboard_callback])  # Add your dataset
    elif backend == "pytorch":
        import torch
        model = torch.load(f"model_{backend}.py")  # Adjust for PyTorch
        writer = SummaryWriter(log_dir)
        # Training loop with TensorBoard logging
        for epoch in range(model_data['training_config'].get('epochs', 10)):
            for batch in train_loader:  # Assume train_loader is defined
                # ... Training logic ...
                writer.add_scalar("loss", loss.item(), epoch)
        writer.close()
    click.echo(f"Training completed. Logs available in {log_dir}")