import math

layer_outputs = [4.8, 1.21, 2.385]
exp_values = []
E = math.e
#EXPONENTIATION
for outputs in layer_outputs:
    exp_values.append(E**outputs)

print(exp_values)

#NORMALIZATION
norm_base = sum(exp_values)#denominator for sum of values
norm_values = []#numerator for exponentiated values
for values in exp_values:
    norm_values.append(values/norm_base)
print(norm_values)
print(sum(norm_values))#adds up to 1


#with numpy
import numpy as np
exp_values = np.exp(layer_outputs)
print(exp_values)
norm_values = exp_values/np.sum(exp_values)
print(norm_values)
print(np.sum(norm_values))
#numpy makes it easier for us to calculate and read the code
