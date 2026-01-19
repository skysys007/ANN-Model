import numpy as np
#Training in Batches
layer_outputs = np.array([[4.8, 1.21, 2.385],
                         [8.9, -1.81, 0.2],
                         [1.41, 1.051, 0.026]])

print("Sum without axis: ")
print(np.sum(layer_outputs))

print("Sum with axis 0: ")
print(np.sum(layer_outputs, axis=0))
#Axis 0 means to sum row wise, along axis 0, meaning the O/P will have the same size as the axis

#In a 2D arrays axis 0 refers to the rows and axis 2 refers to the columns
#the values from all other dimensions are summed to form it
#basically it sums the values from each row from the corresponding column

#but we want sum of rows instead
print("Sum with axis 1: ")
print(np.sum(layer_outputs, axis=1))
#here, it sums the value from each column from the corresponfing column
#now we want to keep the dimensions from the layer output as it is a batch, and right now the sum is a vector
print(np.sum(layer_outputs, axis=1, keepdims=True))
#now we can divide the outputs and the sum of outputs sample wise

# SoftMax Class
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values/np.sum(exp_values, axis=1, keepdims = True)
        self.output = probabilities

softmax = Activation_Softmax()
softmax.forward([[-2, -1, 0]])
softmax.forward([[0.5, 1, 1.5]])
print(softmax.output)
print(np.sum(softmax.output))


