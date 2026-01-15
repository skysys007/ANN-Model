import numpy as np
import nnfs
nnfs.init
import matplotlib.pyplot as plt 
from nnfs.datasets import spiral_data

#OBJECT LAYERS

np.random.seed(0)
X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

#Initializing the weights and biases randomly b/w -1 and 1(the shorter the range the better)
#we use smaller initial values, we do so the maintain the scale of the dataset, such that the meaning stays the same but the scale is maintained
#for example : -0.1 to +0.1
#bias is usually set to 0, but in some cases when the neuron's output is zero we dont set it as zero

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):#Forward method
        self.output = np.dot(inputs, self.weights) + self.biases

X, Y = spiral_data(samples = 100, classes=3)
dense1 = Layer_Dense(2, 3)
dense1.forward(X)
print(dense1.output[:5])
#the shape can be defined here itself, so that we dont have to do the transpose everyttime during the forward pass.



 

