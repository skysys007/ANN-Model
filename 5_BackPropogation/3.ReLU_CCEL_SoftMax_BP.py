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

    # Backward Pass
    def backward(self, dvalues, y_true):
        #Number of Samples
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Gradient of CCEL function
        self.dinputs = - y_true/dvalues
        # Normalize Gradient
        self.dinputs /= samples