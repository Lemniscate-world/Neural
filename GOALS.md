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
- TESTING THE MVP and debugging (1) (Writing a test logic documentation, test for validating the parser, shape propagation, and code generation.) (Done)

Performance Optimization Features
  - Automatic Mixed Precision (Train models using mixed float16 and float32 precision to reduce memory usage and speed up training.)
  - Just-In-Time (JIT) Compilation (Compile models and training loops at runtime for optimized execution.)
  - Kernel Fusion (Combine multiple operations into a single kernel to reduce memory bandwidth and improve performance.)

Advanced Model Design Features 
  -  Neural Architecture Search (NAS) (Automatically search for the best model architecture for a given task.)
  -  Dynamic Computation Graphs (Allow models to change their architecture dynamically during training or inference.)
  -  Symbolic Shape Inference (Infer tensor shapes symbolically, even for dynamic models.)

Deployment and Production Features
- Export models to TensorFlow, PyTorch, ONNX, Core ML, TensorRT, and more with a single command.
- Optimize models for low-latency, real-time inference on edge devices.
- Provide tools for interactive debugging of models (e.g., tensor values, gradients).
- Visual Model Editor
- Automatically search for the best hyperparameters (e.g., learning rate, batch size).
- A repository of pre-trained models that users can easily import and fine-tune.
- Support for quantum neural networks and hybrid classical-quantum models. ++++++






- Write more test cases for edge cases
- Add more test cases for edge cases and custom layers.

Integrate continuous integration (CI) for automated testing.

Write documentation for users and contributors.

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
  
- TESTING THE MVP and debugging (2) (Writing a test logic documentation, test for validating the parser, shape propagation, and code generation.) 

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
