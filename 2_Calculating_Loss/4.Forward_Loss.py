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
        elif len(y_true.shape) == 1:
            correct_confidences = np.sum(
                y_pred_clipped*y_true, 
                axis = 1
            )

        # Loss
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

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
# Loss Function
loss_function = Loss_Categorical_Cross_Entropy()

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

#Calculate the loss by a forward pass through the loss function which takes the act func 2 as input and returns loss
Loss = loss_function.calculate(activation2.output, y)
# Print Loss Function
print('Loss:', Loss)

# the prediction done by the model
predictions = np.argmax(activation2.output, axis=1)
# for hot encoded values
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
# accuracy  = No of Correct predictions/No. of Total Predictions 
accuracy = np.mean(predictions == y)
#print accuracy
print('ACC: ', accuracy)



