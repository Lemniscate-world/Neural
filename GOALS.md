01/25 - 02/25 Grammar, Parser, Tests

30 01 25

- Add more layer types to grammar(Dropout, Flatten, LSTM)
- What is shape propagation? (Answered)
- What Lexer Does Lark Use ? (Answered)
- Added Shape Validation Function (propagate_shape)
- Add Custom activations to grammar (LeakyReLU, Swish)
- Add training configurations to grammar (epochs, batch_size)
- What is Code Generation ? (Answered)
- Adding and Extending generate_code function to support backend (TensorFlow) and Training Configuration (Epochs, Batch Size)
- Understanding tf.keras.Sequential()
- Added Documentation (Differences between Neural and Tensorflow/Pytorch)
- Added Test Docs
(Done)

- Add backend logic for pytorch
- Implement other backends features and sources
- Writing a test logic documentation, test for validating the parser, shape propagation, and code generation.

- Expand grammar to support GANs/Transformers
- Add shape validation rules
- Add more layers types to the shape propagation system and the grammar.
- Make a dictionary of layers disponible in the grammar and in  the shape propagation system
- Add recurrent/attention layers to parser.py
- Implement tensor shape propagation for parser.py
- Implement Better error messages with Lark's Diagnostics for parser.py
  
- Code Generation
  - Add PyTorch/JAX backends
  - ONNX export scaffolding
  
- Tooling
  - cli iNTERFACE
  - VSCode Extension

- Modernizing the Stack
    - MyPy type annotations
    - Pydantic models for layer configs
    - LSP server for IDE support

- Testing Strategy
  - golden tests using popular architectures
  - Tree-sitter for syntax highlighting
