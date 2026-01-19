import math
import numpy as np
#example of O/P from a neural network output layer
softmax_output = [0.7, 0.1, 0.2]
#Ground truth
target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0])*target_output[0] +
         math.log(softmax_output[1])*target_output[1] +
         math.log(softmax_output[2])*target_output[2])

# or simply 
loss = math.log(softmax_output[0])
print(loss)

#Batches
softmax_output = np.array([[0.7, 0.1, 0.2],
                          [0.1, 0.5, 0.4],
                          [0.02, 0.9, 0.08]])

class_targets = [0, 1, 1]

print(softmax_output[[0, 1, 2], class_targets])
#applying range, so that we do not have to manually type values
print(softmax_output[
    range(len(softmax_output)), class_targets
])

# Applying Negative log to this list
neg_loss = -np.log(softmax_output[
    range(len(softmax_output)), class_targets
])
#taking mean of losses 
average_loss = np.mean(neg_loss)
print(average_loss)

# Updating class targets for each row
class_targets = np.array([[1, 0, 0], 
                 [0, 1, 0],
                 [0, 1, 0]])


# print(-np.log(0)) -- Error
# np.mean([1, 2, 3, -np.log(0)]) -- divide by zero error

# to encounter the softmax output coming as 0, we add a very small value to the softmax output, which resolves the log(0) issue
print(-np.log(1-1e-7) )
# now, the output is clipped from 1e-7(->0+) to 1 - 1e-7(->1-) to rescale the values to resolve the negative loss issue
y_pred = -np.log(1-1e-7)#confidence 1 , loss -> 0
y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)


 
