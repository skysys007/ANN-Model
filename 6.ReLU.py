import numpy as np
import nnfs
nnfs.init
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []
# for i in inputs:
#     if i>0:
#         output.append(i)
#     else:
#         output.append(0)

##this can be written more simply as
# NO HIDDEN LAYER
for i in inputs:
    output.append(max(0, i)) #ReLU Activation Function
print(output)

inputs1 = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output1 = np.maximum(0, inputs)#numpy's shorthand for taking the weighted sum 
print(output1)


inputs2 = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]

#ReLU function Class
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


X, y = spiral_data(samples=100, classes=3)

#Dense Layer CLass
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):#Forward method
        self.output = np.dot(inputs, self.weights) + self.biases
    
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense1.forward(X)#Forward pass of layer 1 resulting in output(Weighted sum)
activation1.forward(dense1.output)#(Activation Layer deciding whether the Neuron fires or not)

print(activation1.output[:5])#printing 5 rows of the output