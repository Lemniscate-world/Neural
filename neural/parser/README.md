# Neural Parser

<p align="center">
  <img src="../../docs/images/parser_diagram.png" alt="Parser Diagram" width="600"/>
</p>

## Overview

The Neural Parser is responsible for parsing Neural DSL code and transforming it into an intermediate representation that can be used by other components of the Neural framework. It uses the Lark parsing library to define the grammar and parse Neural DSL code.

## Components

### 1. Grammar Definition (`grammar.lark`)

The grammar file defines the syntax of the Neural DSL language using the Lark grammar format. It includes rules for:

- Network definitions
- Layer declarations
- Parameter specifications
- Training configurations
- Hyperparameter settings

### 2. Parser Implementation (`parser.py`)

The main parser implementation that:
- Creates a Lark parser instance with the Neural DSL grammar
- Parses Neural DSL code into a parse tree
- Validates the syntax and structure of the code

### 3. Model Transformer (`transformer.py`)

Transforms the parse tree into a structured model representation that can be used by:
- Code generators
- Shape propagators
- Visualizers
- Debuggers

### 4. Validation (`validation.py`)

Performs semantic validation of the model definition:
- Checks for required parameters
- Validates parameter types and values
- Ensures model structure is valid

## Usage

```python
from neural.parser.parser import create_parser, ModelTransformer

# Create a parser for Neural DSL
parser = create_parser()

# Parse Neural DSL code
neural_code = """
network MNIST {
  input: (28, 28, 1)
  layers:
    Conv2D(32, kernel_size=3, activation="relu")
    MaxPooling2D(pool_size=2)
    Flatten()
    Dense(128, activation="relu")
    Output(10, activation="softmax")
}
"""

# Parse the code into a parse tree
tree = parser.parse(neural_code)

# Transform the parse tree into a model representation
transformer = ModelTransformer()
model_data = transformer.transform(tree)

# Now model_data can be used by other components
```

## Extension Points

The parser is designed to be extensible in several ways:

1. **Grammar Extensions**: The grammar can be extended to support new language features by modifying the `grammar.lark` file.

2. **Custom Transformers**: You can create custom transformers that inherit from `ModelTransformer` to add specialized processing for specific model types.

3. **Validation Rules**: Additional validation rules can be added to ensure model definitions meet specific requirements.

## Related Components

- **Code Generation**: Uses the model representation to generate code for different backends.
- **Shape Propagation**: Uses the model representation to infer and validate tensor shapes.
- **Visualization**: Uses the model representation to generate visualizations of the model architecture.

## Resources

- [Lark Parser Documentation](https://lark-parser.readthedocs.io/en/latest/)
- [Neural DSL Reference](../../docs/DSL.md)
- [Grammar Tutorial](../../docs/tutorials/grammar.md)
