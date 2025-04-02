# FashionMNIST-Classifier

A neural network model built with TensorFlow and trained on the Fashion MNIST dataset from Zalando. This model classifies images of clothing items into 10 categories using a simple feedforward architecture.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [License](#license)

## Introduction
Fashion MNIST is a dataset of grayscale images of clothing items, serving as a more complex alternative to the classic MNIST digit classification dataset. This repository provides an implementation of a neural network model to classify these images into predefined categories.

## Dataset
The dataset consists of 70,000 images (60,000 for training and 10,000 for testing) belonging to the following categories:

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

Each image has a resolution of 28x28 pixels in grayscale.

## Installation
To run this project locally, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/FashionMNIST-Classifier.git
   cd FashionMNIST-Classifier
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To train and test the model, run:

```bash
python clasificador.py
```

This script will:
- Load the Fashion MNIST dataset.
- Preprocess the images.
- Define and train a neural network model.
- Evaluate the model's performance on the test set.

## Model Architecture
The model is a simple feedforward neural network consisting of:

- Input layer (28x28 flattened to 784 neurons)
- Hidden layers with ReLU activation
- Output layer with 10 neurons (softmax activation)

## Results
After training, the model achieves an accuracy of approximately X% on the test dataset (adjust based on results).

## Future Improvements
- Implement convolutional neural networks (CNNs) for improved accuracy.
- Experiment with data augmentation techniques.
- Optimize hyperparameters using Grid Search or Bayesian Optimization.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

