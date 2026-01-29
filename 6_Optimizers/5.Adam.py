import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        self.inputs = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(
            zip(self.output, dvalues)
        ):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = (
                np.diagflat(single_output)
                - np.dot(single_output, single_output.T)
            )
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_Categorical_Cross_Entropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs /= samples


class Activation_SoftMax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_Categorical_Cross_Entropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs /= samples

# Scholastic Gradient Descent
class Optimizer_Adam:
    # Initialize optimizer - set learning rate
    # initial learning rate - 1.0
    def __init__(self, learning_rate = .001, decay = 0., epsilon = 1e-7, beta_1 = 0.9, beta_2 = 0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Pre Update Parameters
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1/(1 + self.iterations*self.decay))

    # Update Parameters 
    # set learning rate to current learning rate
    def update_params(self, Layer):
        # if layer doesn't contain cache arrays initialize one with zeros
        if not hasattr(Layer, 'weight_cache'):
            Layer.weight_cache = np.zeros_like(Layer.weights)
            Layer.weight_momentums = np.zeros_like(Layer.weights)
            Layer.bias_cache = np.zeros_like(Layer.biases)
            Layer.bias_momentums = np.zeros_like(Layer.biases)
        
        # Update momentums with current gradients
        Layer.weight_momentums = self.beta_1*Layer.weight_momentums+(1-self.beta_1)*Layer.dweights
        Layer.bias_momentums = self.beta_1*Layer.bias_momentums+(1-self.beta_1)*Layer.dbiases

        # Corrected Momentums 
        # self.iteration is 0, but we need to set it to 1 here
        weight_momentums_corrected = Layer.weight_momentums/(1-(self.beta_1)**self.iterations+1)
        bias_momentums_corrected = Layer.bias_momentums/(1-(self.beta_1)**self.iterations+1)
        # Updated cache with squared gradients
        Layer.weight_cache = self.beta_2 * Layer.weight_cache+(1-self.beta_2)*Layer.dweights**2      
        Layer.bias_cache = self.beta_2 * Layer.bias_cache+(1-self.beta_2)*Layer.dbiases**2   
        # Corrected Caches
        weight_cache_corrected = Layer.weight_cache / (1-self.beta_2**(self.iterations+1))  
        bias_cache_corrected = Layer.bias_cache / (1-self.beta_2**(self.iterations+1))   

        # Vanilla SGD + normalization with MSR cache

        Layer.weights += - self.current_learning_rate*weight_momentums_corrected/(np.sqrt(weight_cache_corrected)+ self.epsilon)  
        Layer.biases += - self.current_learning_rate*bias_momentums_corrected/(np.sqrt(bias_cache_corrected)+ self.epsilon)  

    # After Updating parameters
    def post_update_params(self):
        self.iterations+=1

X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 3)
loss_activation = Activation_SoftMax_Loss_CategoricalCrossentropy()

#setting the learning rate to 0.85
optimizer = Optimizer_Adam(learning_rate=0.02, decay=1e-5)

for epoch in range(15001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)
    
    if not epoch % 100:
        print('epoch:' , epoch 
              , 'acc: ' , accuracy 
              , 'loss: ' , loss
              , 'lr: ', optimizer.current_learning_rate
              )

    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()  