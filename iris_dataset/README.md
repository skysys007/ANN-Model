# Iris Flower Classification Using Neural Networks

## Overview

The Iris dataset is a classic benchmark dataset introduced by Ronald A. Fisher in 1936. It is widely used for testing and demonstrating machine learning algorithms, especially in supervised classification tasks.

This project implements a **neural network from scratch** to classify iris flowers into their respective species using numerical feature measurements. The implementation focuses on understanding the **core building blocks of neural networks**, including forward propagation, backpropagation, optimization techniques, and regularization.

---

## Dataset Description

* **Total Samples:** 150
* **Number of Classes:** 3
* **Samples per Class:** 50

Each row in the dataset represents one iris flower sample.

### Attributes

| Index | Attribute Name | Description         |
| ----: | -------------- | ------------------- |
|     1 | Serial ID      | Unique identifier   |
|     2 | Sepal Length   | Length of the sepal |
|     3 | Sepal Width    | Width of the sepal  |
|     4 | Petal Length   | Length of the petal |
|     5 | Petal Width    | Width of the petal  |
|     6 | Species        | Flower class label  |

---

## Input and Output

### Input Features (4)

1. Sepal Length
2. Sepal Width
3. Petal Length
4. Petal Width

### Output Classes (3)

1. Iris-setosa
2. Iris-versicolor
3. Iris-virginica

---

## Neural Network Architecture

The neural network is implemented using a modular, object-oriented design. The architecture includes:

* Fully Connected Dense Layers (`Layer_Dense`)
* Activation Functions

  * ReLU (Rectified Linear Unit)
  * Softmax
* Forward Propagation
* Backpropagation using Gradient Descent
* Multiple Optimization Algorithms
* Regularization Techniques

---

## Activation Functions

* **ReLU (Rectified Linear Unit):**
  Used in hidden layers to introduce non-linearity and reduce vanishing gradient issues.

* **Softmax:**
  Used in the output layer to convert logits into class probabilities.

---

## Optimization Algorithms

The project includes implementations of the following optimizers:

* Stochastic Gradient Descent (SGD)
* SGD with Momentum
* AdaGrad
* RMSProp
* Adam Optimizer

Each optimizer can be tested independently to compare convergence behavior and performance.

---

## Loss Function

* **Categorical Cross-Entropy Loss**

This loss function is used for multi-class classification problems and works in conjunction with the Softmax activation function.

---

## Regularization

To reduce overfitting and improve generalization, the model supports:

* **L1 Regularization**
* **L2 Regularization**

Both regularization techniques are applied to trainable parameters during optimization.

---

## Learning Objectives

This project is intended to:

* Build a neural network from scratch without external ML frameworks
* Understand forward and backward propagation in detail
* Explore different optimization strategies
* Learn the impact of activation functions and regularization
* Gain hands-on experience with multi-class classification

---
## Final Outcome
The model:
* loads the iris dataset 
* takes 4 input features
* uses 64 hidden layer neurons to prevent overfitting
* uses Softmax Activation function 
* uses Cross Categorical Entropy Loss to calculate Loss
* uses the Adam Optimizer
* after training, it achieved an accuracy of 0.9867 and loss of 0.039 after 200 epochs
 
## References

* Fisher, R. A. (1936). *The Use of Multiple Measurements in Taxonomic Problems*
* Iris Dataset: UCI Machine Learning Repository
