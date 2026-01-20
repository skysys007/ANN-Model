import matplotlib.pyplot as plt
import nnfs 
from nnfs.datasets import vertical_data
nnfs.init()
import numpy as np
#Dense Layer
class Layer_Dense:

    #Constructor to initialize weights and Biases
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    #Forward Pass
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

#ReLU function
class Activation_ReLU:

    #Forward Pass
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

#SoftMax Function
class Activation_Softmax:
    
    #Forward Pass
    def forward(self, inputs):
        #Exponentiate Values
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        #Normalization
        probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# Common Loss Class
class Loss:
    def calculate(self, output, y):
        # Calculates the data and regularization losses
        # given model's output and ground truth values
        sample_losses = self.forward(output, y)
        # Mean of Loss
        data_loss = np.mean(sample_losses)
        # return Loss
        return data_loss
    
# This class inherits the Loss class and performs all error calculations
class Loss_Categorical_Cross_Entropy(Loss):
    # Forward Pass
    def forward(self, y_pred, y_true):
        # Number of Samples in a Batch
        samples = len(y_pred)

        # Clip data
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        # Check for vector or a list of vectors
        # for categorical labels/sparse vector
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples), 
                y_true
            ]
        # for one hot encoded label
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped*y_true, 
                axis = 1
            )

        # Loss
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

X, y = vertical_data(samples=100, classes=3)
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

loss_function = Loss_Categorical_Cross_Entropy()

# Helper Variables
lowest_loss = 9999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()


for iteration in range(10000):
    #Update weights with some small random values
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)
    dense2.weights += 0.05 * np.random.randn(3, 3)
    dense2.biases += 0.05 * np.random.randn(1, 3)
    #Perform a forward Pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss = loss_function.calculate(activation2.output, y)

    predictions = np.argmax(activation2.output, axis = 1)
    accuracy = np.mean(predictions==y)
    #if loss is smaller - print and save the weights and biases
    if loss<lowest_loss:
        print('New set of weights foundm iteration:', iteration, 'loss:', loss, 'acc:', accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss 
    # else revert the changes if the loss is not decreasing with the current random updation
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()







# plt.scatter(X[:, 0], X[:, 1], c = y, s = 40, cmap='brg')

# plt.show()