# Neural Examples

<p align="center">
  <img src="../docs/images/examples_overview.png" alt="Examples Overview" width="600"/>
</p>

## Overview

This directory contains example Neural DSL files demonstrating various use cases and model architectures. These examples serve as both documentation and starting points for your own models. Each example includes detailed comments explaining the model architecture and implementation details.

## Example Categories

The examples are organized into the following categories:

### 1. Basic Examples

Simple examples to help you get started with Neural DSL:

- **[MNIST Classifier](basic/mnist_classifier.neural)**: A simple convolutional neural network for classifying MNIST digits
- **[Fashion MNIST Classifier](basic/fashion_mnist_classifier.neural)**: A classifier for the Fashion MNIST dataset
- **[CIFAR-10 Classifier](basic/cifar10_classifier.neural)**: A convolutional neural network for classifying CIFAR-10 images
- **[Iris Classifier](basic/iris_classifier.neural)**: A simple neural network for classifying Iris flowers

### 2. Computer Vision Examples

Examples focused on computer vision tasks:

- **[ResNet](computer_vision/resnet.neural)**: Implementation of the ResNet architecture
- **[VGG](computer_vision/vgg.neural)**: Implementation of the VGG architecture
- **[MobileNet](computer_vision/mobilenet.neural)**: Implementation of the MobileNet architecture
- **[Object Detection](computer_vision/object_detection.neural)**: Object detection model using SSD
- **[Segmentation](computer_vision/segmentation.neural)**: Image segmentation model using U-Net

### 3. Natural Language Processing Examples

Examples focused on NLP tasks:

- **[Text Classification](nlp/text_classification.neural)**: Text classification model using LSTM
- **[Named Entity Recognition](nlp/ner.neural)**: Named entity recognition model
- **[Sentiment Analysis](nlp/sentiment_analysis.neural)**: Sentiment analysis model
- **[Language Model](nlp/language_model.neural)**: Simple language model
- **[Transformer](nlp/transformer.neural)**: Implementation of the Transformer architecture

### 4. Generative Models

Examples of generative models:

- **[Variational Autoencoder](generative/vae.neural)**: Implementation of a variational autoencoder
- **[GAN](generative/gan.neural)**: Implementation of a generative adversarial network
- **[StyleGAN](generative/stylegan.neural)**: Implementation of StyleGAN
- **[Diffusion Model](generative/diffusion.neural)**: Implementation of a diffusion model

### 5. Reinforcement Learning

Examples of reinforcement learning models:

- **[DQN](reinforcement_learning/dqn.neural)**: Implementation of Deep Q-Network
- **[A2C](reinforcement_learning/a2c.neural)**: Implementation of Advantage Actor-Critic
- **[PPO](reinforcement_learning/ppo.neural)**: Implementation of Proximal Policy Optimization

### 6. Multi-Modal Models

Examples of models that combine multiple modalities:

- **[Image Captioning](multi_modal/image_captioning.neural)**: Model for generating captions for images
- **[Visual Question Answering](multi_modal/vqa.neural)**: Model for answering questions about images
- **[Audio-Visual Fusion](multi_modal/audio_visual.neural)**: Model that combines audio and visual inputs

## Using the Examples

### Running an Example

To run an example, use the Neural CLI:

```bash
# Compile the example to TensorFlow
neural compile examples/basic/mnist_classifier.neural --backend tensorflow

# Run the compiled model
neural run mnist_classifier_tensorflow.py

# Visualize the model architecture
neural visualize examples/basic/mnist_classifier.neural
```

### Modifying an Example

You can use these examples as starting points for your own models:

1. Copy the example file to your working directory
2. Modify the model architecture, hyperparameters, or training configuration
3. Compile and run the modified model

```bash
# Copy an example
cp examples/basic/mnist_classifier.neural my_model.neural

# Edit the model
nano my_model.neural

# Compile and run the modified model
neural compile my_model.neural --backend tensorflow
neural run my_model_tensorflow.py
```

## Example Structure

Each example follows a consistent structure:

```
network ModelName {
  // Input specification
  input: (input_shape)

  // Layer definitions
  layers:
    Layer1(params)
    Layer2(params)
    ...
    Output(params)

  // Training configuration
  loss: loss_function
  optimizer: optimizer_type(params)
  metrics: [metric1, metric2, ...]

  // Hyperparameters
  batch_size: value
  epochs: value
  ...
}
```

## Contributing Examples

We welcome contributions of new examples! To contribute:

1. Create a new Neural DSL file in the appropriate category directory
2. Include detailed comments explaining the model architecture and implementation
3. Add the example to this README
4. Submit a pull request

Please follow these guidelines when contributing examples:

- Use clear, descriptive names for models and variables
- Include detailed comments explaining the model architecture
- Follow the existing example structure
- Test the example to ensure it works with the Neural CLI

## Resources

- [Neural DSL Reference](../docs/DSL.md)
- [Layer Reference](../docs/layers.md)
- [Optimizer Reference](../docs/optimizers.md)
- [Training Configuration](../docs/training.md)
- [Hyperparameter Specification](../docs/hyperparameters.md)
