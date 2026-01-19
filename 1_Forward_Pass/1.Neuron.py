import numpy as np
inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2
output = [inputs[0]*weights[0] 
          +inputs[1]*weights[1] 
          +inputs[2]*weights[2]
          +inputs[3]*weights[3]
          +bias]
print(output)
#Or the output can also be calculated using dot product of two vectors
#the dot product multiplies each corresponding neuron and sums it all up to form a scalar   

outputs = np.dot(weights, inputs) + bias
print(outputs)