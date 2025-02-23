import click
from neural.parser.parser import create_parser, ModelTransformer
from neural.code_generation.code_generator import generate_code
import re

class NeuralChat:
    def __init__(self):
        self.parser = create_parser("network")
        self.transformer = ModelTransformer()
        self.config = "network MyNet {\n    input: (28, 28, 1)\n    layers:\n"
        self.backend = "tensorflow"

    def process_command(self, command: str) -> str:
        command = command.lower().strip()

        # Basic NLP rules
        if "create model" in command or "new model" in command:
            name_match = re.search(r"named\s+(\w+)", command)
            name = name_match.group(1) if name_match else "MyNet"
            self.config = f"network {name} {{\n    input: (28, 28, 1)\n    layers:\n"
            return f"Started new model: {name}"

        elif "add layer" in command or "add" in command:
            if "conv2d" in command:
                filters = re.search(r"(\d+)\s*filters", command)
                filters = filters.group(1) if filters else "32"
                kernel = re.search(r"kernel\s*(\d+)", command) or re.search(r"size\s*(\d+)", command)
                kernel = kernel.group(1) if kernel else "3"
                activation = re.search(r"(relu|softmax|sigmoid)", command)
                activation = activation.group(1) if activation else "relu"
                self.config += f"        Conv2D(filters={filters}, kernel_size={kernel}, activation=\"{activation}\")\n"
                return f"Added Conv2D(filters={filters}, kernel_size={kernel}, activation=\"{activation}\")"
            elif "dense" in command:
                units = re.search(r"(\d+)\s*units", command)
                units = units.group(1) if units else "128"
                self.config += f"        Dense(units={units}, activation=\"relu\")\n"
                return f"Added Dense(units={units}, activation=\"relu\")"
            elif "nested" in command or "residual" in command:
                self.config += f"        Residual(Conv2D(64, 3), Dense(128))\n"
                return "Added Residual block with Conv2D and Dense"

        elif "set backend" in command:
            backend = re.search(r"(tensorflow|pytorch)", command)
            if backend:
                self.backend = backend.group(1)
                return f"Backend set to {self.backend}"
            return "Please specify tensorflow or pytorch"

        elif "compile" in command or "generate" in command:
            self.config += "    loss: \"categorical_crossentropy\"\n    optimizer: \"adam\"\n}\n"
            tree = self.parser.parse(self.config)
            model_data = self.transformer.transform(tree)
            code = generate_code(model_data, self.backend)
            output_file = f"model_{self.backend}.py"
            with open(output_file, "w") as f:
                f.write(code)
            self.config = "network MyNet {\n    input: (28, 28, 1)\n    layers:\n"
            return f"Compiled model to {output_file}"

        elif "visualize" in command:
            tree = self.parser.parse(self.config + "}\n")
            model_data = self.transformer.transform(tree)
            from neural.dashboard.dashboard import app
            app.run_server(debug=True, port=8050)
            return "Launched visualization dashboard at http://localhost:8050"

        return "Command not recognized. Try: 'create model', 'add layer', 'compile', 'visualize'."

# Example usage in CLI
def chat():
    chat = NeuralChat()
    click.echo("Welcome to NeuralChat! Type commands or 'exit' to quit.")
    while True:
        command = click.prompt("> ", type=str)
        if command.lower() == "exit":
            break
        response = chat.process_command(command)
        click.echo(response)