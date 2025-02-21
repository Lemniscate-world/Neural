# MNIST Classifier Guide

This example shows how to define, compile, run, and debug a CNN for MNIST digit classification.

### Steps

1. **Save the Model**  
   Create `examples/mnist.neural` with the content above.

2. **Compile for TensorFlow**  
   ```bash
   python neural.py compile examples/mnist.neural --backend tensorflow --output examples/mnist_tf.py