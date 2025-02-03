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

31 01 25
- Added backend logic for pytorch
- Added error handling for backend logic supplier (generate_code)
- Understanding deeply ModelTransformer
- What is lark.Transformer? (Answered)
- Test on Case1.py
(Done)

02 02 25
- I am adding a unit test for ModelTransformer class and methods
- I am adding methods that convert layer nodes to dictionaries
- Added code documentation
- I am adding a debug print to verify the transformation is working
- Understanding dictionaries properly.
- I'm modifying the layer method (ModelTransformer) to extract the layer type from the Tree nodes correctly. Instead of accessing the data attribute directly, I can use the children attribute to retrieve the layer type. (parser.py)
(Done)

- TESTING THE MVP and debugging (Writing a test logic documentation, test for validating the parser, shape propagation, and code generation.)
- Write more test cases for edge cases
- Expand grammar to support GANs/Transformers
- Add shape validation rules
- Is my grammar good for a mvp ?
- What should be done for mvp ?
- Other interesting backend features
- Testing and debugging for faster performance - Adding some advantages of traditional backends and frameworks
- Add more layers types to the shape propagation system and the grammar.
- Make a dictionary of layers disponible in the grammar and in  the shape propagation system
- Pytorch COnversion to Tensorflow for MVP
- Advanced for other versions
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
