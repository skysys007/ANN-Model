import numpy as np

# Passed in gradients from the next Layer
# Vectors of 1's for simplicity
d_values = np.array([[1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0]])

# We have three sets of weights - one set for each neuron
# Similiarly for derivative of weights, inputs need to be transposed to match with the gradient of next layer
inputs = np.array([[1, 2, 3, 2.5],
                   [2, 5, -1, 2], 
                   [-1.5, 2.7, 3.3, -0.8]])
# we have 4 inputs so 4 weights
weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T


biases = [[2, 3, 0.5]]

# Forward pass
layer_outputs = np.dot(inputs, weights)+biases
relu_outputs = np.maximum(0, layer_outputs)

#Derivative of ReLU Activation function
drelu = relu_outputs.copy()
drelu[layer_outputs<=0] = 0

# Dense Layer
dinputs = np.dot(drelu, weights.T)
dweights = np.dot(inputs.T, drelu)
dbiases = np.sum(d_values, axis=0, keepDims = True)


weights += -0.001 * dweights
inputs += -0.001 * dinputs

print(weights)
print(biases)


# z = np.array([[1, 2, -3, -4], 
#               [2, -7, -1, 3], 
#               [-1, 2, 5, -1]])
# dl_values = np.array([[1, 2, 3, 4],
#                      [5, 6, 7, 8],
#                      [9, 10, 11, 12]])

# # d_relu = np.zeros_like(z)
# # d_relu[z>0] = 1

# #for simplicity we copy the values from the dl_values and then set the values less than or equals zero as 0
# drelu = dl_values.copy()
# drelu[z<=0] = 0

# # sum the weights and multiply by the passed in gradient from the next layer
# dinputs = np.dot(d_values, weights.T)
# dweights = np.dot(inputs.T, d_values)
# dbiases = np.sum(d_values, axis=0, keepdims=True)

# print(dinputs)
# print(dweights)
# print(dbiases)
# print(drelu)
# #the chain rule
# # d_relu *= dl_values
# # print(d_relu)

