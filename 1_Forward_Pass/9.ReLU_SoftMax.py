import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

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

#Create Non Linear Spiral Data
X, y = spiral_data(samples=100, classes=3)

#Layer Dense with 2 input features and 3 outputs
dense1 = Layer_Dense(2, 3)
#ReLU for dense 1
activation1 = Activation_ReLU()

#Layer Dense with 3 input features and 3 outputs
dense2 = Layer_Dense(3, 3)
#ReLU for dense 2 for output feeding/generating a probability Distribution 
activation2 = Activation_Softmax()

#Forward pass of Dense 1
dense1.forward(X)
#feeding outputs of dense 1 as inputs for ReLU function
activation1.forward(dense1.output)

#Forward pass of Dense 2 which takes the ReLU output from Layer 1 as input
dense2.forward(activation1.output)
#feeding outputs of dense 2 as inputs for SoftMax function
activation2.forward(dense2.output)

#Output from 5 rows of the final output / Probability Distribution  
print(activation2.output[:5])




