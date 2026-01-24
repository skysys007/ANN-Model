import numpy as np

class Layer_Dense:

    # Constructor to initialize weights and Biases
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward Pass
    def forward(self, inputs):
        # we will need to calculate partial derivative of inputs later on, so we store it in the class props
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward Pass
    def backward(self, dvalues):
        # Gradients of params
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbias = np.sum(dvalues, axis=0, keepdims=True)
        # Gradients of values
        self.dinputs = np.dot(dvalues, self.weights.T)

# ReLU function
class Activation_ReLU:
    # Forward Pass
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        # Remember input values
        self.inputs = inputs

    # Backward Pass
        def backward(self, dvalues):
            self.dinputs = dvalues.copy()
            self.dinputs[dvalues<=0] = 0 