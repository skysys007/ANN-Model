import numpy as np
import nnfs
nnfs.init()
from nnfs.datasets import spiral_data

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
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradients of values
        self.dinputs = np.dot(dvalues, self.weights.T)

# ReLU function
class Activation_ReLU:
    # Forward Pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    # Backward Pass
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0 

#SoftMax Function
class Activation_Softmax:
    #Forward Pass
    def forward(self, inputs):
        #Exponentiate Values
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        #Normalization
        probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
    #Backward Pass
    def backward(self, dvalues):
        #Create an unitialized array
        self.dinputs = np.empty_like(dvalues)

        # Iterate for outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calc Jacobian Matrix
            # diagflat to convert it into a diagonal matrix
            jacobian_matrix =(
                np.diagflat(single_output) - np.dot(single_output, single_output.T)
                )
            # Calcuate sample wise gradients and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


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
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

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
        self.dinputs = self.dinputs / samples

class Activation_SoftMax_Loss_CategoricalCrossentropy():
    # Create activation function and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_Categorical_Cross_Entropy() 

    # Forward Pass
    def forward(self, inputs, y_true):
            # O/P Layer Activation Function
            self.activation.forward(inputs)
            # Set output to O/P layer activation function's output
            self.output = self.activation.output
            # calculate the loss from the O/P and return it
            return self.loss.calculate(self.output, y_true)
    # Backward Pass    
    def backward(self, dvalues, y_true):
            # No. of Samples
            samples = len(dvalues)
            # Convert into hot encoded values
            if len(y_true.shape) == 2:
                y_true = np.argmax(y_true, axis=1)
            #copy so we can safely modify
            self.dinputs = dvalues.copy()
            # Calculate Gradient(P.D of loss function w.r.t predicted O/P)
            # which is y_pred - ytrue(which is always 1)
            # since y_true is 1 we subtract the y_pred with 1
            self.dinputs[range(samples), y_true] -= 1
            # Normalize Gradient
            self.dinputs = self.dinputs / samples

softmax_outputs = np.array([
    [0.7, 0.1, 0.2],
    [0.1, 0.5, 0.4],
    [0.02, 0.9, 0.08]
])

class_targets = np.array([0, 1, 1])

# Derivate calculated using CCEL and SM function class
softmax_loss = Activation_SoftMax_Loss_CategoricalCrossentropy()
softmax_loss.backward(softmax_outputs, class_targets)
dvalues1 = softmax_loss.dinputs

# Derivative Calculated by backpropogating step by step
activation = Activation_Softmax()
activation.output = softmax_outputs
loss = Loss_Categorical_Cross_Entropy()
loss.backward(softmax_outputs, class_targets)
activation.backward(loss.dinputs)
dvalues2 = activation.dinputs

# Print Gradients
print("Gradient: Combined Loss and SoftMax: ")
print(dvalues1)
print("Gradient: Seperate Loss and SoftMax: ")
print(dvalues2)

# Create data
# Forward Pass
X, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
loss_activation = Activation_SoftMax_Loss_CategoricalCrossentropy()
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y)
print(loss_activation.output[:5])
print('loss: ', loss)

# Convert into hod encoded values
predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)
# Accuracy
print("acc: ", accuracy)

# Backward Pass
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

# Print Gradients
print(dense1.dweights)
print("")
print(dense1.dbiases)
print("")
print(dense2.dweights)
print("")
print(dense2.dbiases)
