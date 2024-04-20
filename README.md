# MNIST Digit Recognition Model

## Overview

This repository contains a simple neural network model built using TensorFlow and Keras to recognize hand-written digits from the MNIST dataset.

## Model Architecture

The neural network model consists of three dense (fully connected) layers:
1. Input layer with 784 neurons (flattened 28x28 images)
2. Two hidden layers, each with 128 neurons and ReLU activation function
3. Output layer with 10 neurons and softmax activation function

## Dataset

The MNIST dataset is used for training and testing the model. It contains 60,000 training images and 10,000 testing images of hand-written digits from 0 to 9.

## Training

The model is trained for 3 epochs using stochastic gradient descent (SGD) as the optimizer and categorical crossentropy as the loss function. Training accuracy is displayed for each epoch.

## Evaluation

After training, the model is evaluated using the test dataset to calculate accuracy. The model achieves an accuracy of approximately [INSERT ACCURACY HERE]% on the test set.

## Predictions

The model makes predictions on the test dataset, and 25 samples along with their predicted values are displayed. Correct predictions are labeled in green, while incorrect predictions are labeled in red.

## Sample Visualization

Below is a visualization of 25 samples from the test dataset along with their predicted values:

[INSERT IMAGE OF SAMPLE PREDICTIONS HERE]

## Additional Information

For more detailed information, you can refer to the Python script `mnist_model.py` in the repository.

