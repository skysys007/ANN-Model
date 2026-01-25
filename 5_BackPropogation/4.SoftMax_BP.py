
import numpy as np
softmax_outputs = [0.7, 0.1, 0.2]
# reshape(rows, cols) rows = -1(figure ou no of rows), cols = 1(only one column)
softmax_outputs = np.array(softmax_outputs).reshape(-1, 1)
print(softmax_outputs)
# Kronecker's delta equals 1 if both inputs are equal, and 0 otherwise
print(np.eye(softmax_outputs.shape[0]))
print(softmax_outputs*np.eye(softmax_outputs.shape[0]))
# or for even more simpler outcome, we can use np.diagflat
# np.diagflat creates an array using the input vector as the diagonal values
# this is the first part of the eqn, Si,j * &j,k
print(np.diagflat(softmax_outputs))

#Second Eq'n [Si,j * Si,k]
print(np.dot(softmax_outputs, softmax_outputs.T))
print(np.diagflat(softmax_outputs)-np.dot(softmax_outputs,softmax_outputs.T))

