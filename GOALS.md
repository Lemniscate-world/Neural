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

04 02 25
- GPU-optimized inference
  To make Neural a high-performance framework, we will:
✅ Implement GPU acceleration using CUDA & TensorRT (for NVIDIA GPUs)
✅ Support multi-backend inference (CPU, GPU, and possibly future hardware like TPUs)
✅ Introduce automatic device selection based on available hardware
- Implement ONNX export for deployment
- Add visualizer
- Make neural generate research paper (No AI) on model i'm using or training with benchmarks and stats etc in the its IDE + visualizer and interactive solver and nodes explorers with details
- Create very basic deepseek model with neural and even fine-tune to showcase all neural functionalities.and in the same time testing if they are working.

05 02 25
- Debugging all day, great improvements but still bugs.
06 02 25
- 15h of Debugging (Still, some errors)
- 


- Creating executable neural files
- Neural Syntax
- Enhance parallelization in neural
- Arxiv and JOSS apI
- Shape propagations and code generations in separate files
- Add mathematics formulations to shape propagation on research paper and visualizations
- Create parallelization layers and unique parallelization techniques
- Add all pytorch and tensorflow syntax to neural
- Add more researchs capabilities for neural
- Create my bennchmark module with all possible and disponibles statistic and analysis capabilities
- Syntax highlight neural IDE
- Mathematical formulas and modeling  extraction and visualization on the dashboard
- High-level API
- Implement and create my own neural tester
- Make neural more faster - Faster analyzer, analyzer code with the goal to make everything faster
- compiler optimizations
- Integrate Hugging Face
- AutoML features.
- Compiler stack for deployment
- Create a memory file to resume for neural
- Create a standalone interpreter for neural
- Ability to load models and parses them into different backends  
- Add Multi GPU training
- DSL for image processing (Add image processing)
- Optimize and enhance Automatic Shape Validation to the best
- Optimize for low-latency real-time inference
- Create my own neural machine learning model based on graph theory, with benchmarks and research papers wto showcase neural functionalities.
- Create model for autoevolving music with neural
- Add custom graph-based debugging
- explainability, and rich metric visualizations.
-  Engineered for production: prebuilt Docker containers, native support for running with Ray on Kubernetes, export models to Torchscript and Triton, upload to HuggingFace with one command.
- Optimized for scale and efficiency: automatic batch size selection, distributed training (DDP, DeepSpeed), parameter efficient fine-tuning (PEFT), 4-bit quantization (QLoRA), and larger-than-memory datasets.
- Implement hyperparameter tuning visualization
- Live tracking of training progress
- My own model form y MWI Qubits simulator
- For high frequency trading too
- For BloomDb branch (probabilistic and fuzzy data sets)
- Refactor my code (1)
- Create a model made of some models definitions by combining their .neural and .nr files
- Make  a no-code interface for neural

How can i enhance shape validation and propagation ?
How can i enhance Code Generation ?
How can i enhance the plugin system ?
How can i enhance the training configuration ?
How can i make neural the less boilerplate possible ?
How can i enhance my grammar ?
Create requirements.txt
Write full neural language syntax documentation
Create requirements-dev.txt


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


- Add Dynamic Layer
  - Purpose: Allow the layer to adjust its behavior based on input characteristics.
Why It’s Unique: Dynamically choose between different sub-layers (e.g., Dense with 128 units vs. 64 units) depending on the input, making the model adaptable to varying data conditions.
- QuantumLayer

Purpose: Integrate quantum computing concepts into the network to explore hybrid classical-quantum models.
Why It’s Unique: Provides a pathway to experiment with quantum neural networks, leveraging libraries like PennyLane.
- Symbolic Layers or Special Configurations:
- While not a layer per se, consider extending the input configuration to support symbolic shapes (e.g., input: (None, 28, 28, 1)) to handle dynamic batch sizes and enhance error checking.
- Plugin-Defined Layers:
Enable a plugin system where community members can contribute custom layers, optimizers, or even entirely new paradigms. This ensures that Neural remains flexible and can rapidly adapt to emerging trends.
Convolutional Layers
Conv1D, Conv2D, Conv3D
Perform convolution operations over 1-dimensional, 2-dimensional, or 3-dimensional data, respectively.

Transposed Convolution (Deconvolution) Layers
Used for upsampling in tasks like image generation or segmentation.

Separable Convolution Layers
Factorize convolutions to reduce computation (e.g., depthwise separable convolutions).

Dilated (Atrous) Convolution Layers
Increase the receptive field without increasing the number of parameters.

 Pooling Layers
MaxPooling Layers
Take the maximum value over a defined window (common in CNNs).

AveragePooling Layers
Compute the average over a window.

Global Max/Average Pooling
Reduce each feature map to a single value by taking the max or average across the entire spatial dimension.

Adaptive Pooling Layers
Adjust the pooling region so the output size is fixed regardless of the input size.

Normalization Layers
Batch Normalization
Normalizes activations across the current batch to stabilize and speed up training.

Layer Normalization
Normalizes across the features in each individual data point, often used in RNNs or Transformers.

Instance Normalization
Normalizes each individual sample separately, popular in style transfer tasks.

Group Normalization
Divides channels into groups and normalizes within each group.

 Regularization Layers
Dropout
Randomly drops a subset of neurons during training to reduce overfitting.

Spatial Dropout
Specifically drops entire feature maps in convolutional layers.

Activation Layers
(Sometimes the activation is integrated into other layers, but they can also be separate.)

ReLU, Leaky ReLU, ELU, SELU
Various forms of rectified linear activations.

Sigmoid, Tanh
Classic activations for bounded outputs.

Softmax
Converts a vector into a probability distribution, typically used in classification.

 Recurrent and Sequence Layers
Simple RNN
Basic recurrent layer that processes sequences.

LSTM (Long Short-Term Memory)
Addresses the vanishing gradient problem, ideal for long sequences.

GRU (Gated Recurrent Unit)
A simpler variant of LSTM that often achieves similar performance.

Bidirectional RNN/LSTM/GRU
Processes sequences in both forward and backward directions for richer context.

ConvLSTM
Combines convolution with LSTM for spatiotemporal data.

Attention and Transformer Layers
Attention Layers
Mechanisms that allow the network to focus on different parts of the input.

Self-Attention / Multi-Head Attention
Core components of Transformer models that enable parallel processing of sequence data.

Transformer Encoder and Decoder Layers
Build upon self-attention with feedforward networks and layer normalization.

 Specialized and Advanced Layers
Residual (Skip Connection) Layers/Blocks
Allow gradients to flow more easily through deep networks (e.g., ResNet blocks).

Inception Modules
Use multiple filter sizes in parallel for richer feature extraction.

Capsule Layers
Aim to capture hierarchical relationships between features.

Squeeze-and-Excitation Layers
Reweight channel-wise features adaptively.

Spatial Transformer Networks
Provide networks the ability to spatially transform feature maps.

Graph Convolutional Layers
Extend convolutions to graph-structured data.

Embedding Layers
Map discrete tokens (like words) into continuous vector representations.

Lambda Layers or Custom Function Layers
Allow you to define custom operations as part of the network.

Attention Variants for Vision and Beyond
Such as non-local blocks or axial attention for handling long-range dependencies in images.



Add GPU-optimized inference
🔹 Implement symbolic shape inference
🔹 Enable multi-backend support (TensorFlow, PyTorch, JAX)

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
