import numpy as np

softmax_output = np.array([[0.7, 0.1, 0.2],
                          [0.1, 0.5, 0.4],
                          [0.02, 0.9, 0.08]])

class_targets = np.array([[1, 0, 0], 
                 [0, 1, 0],
                 [0, 1, 0]])

# Checking for sparse & hot encoded values
# only if categorical labels
if len(class_targets.shape) == 1:
    correct_confidences = softmax_output[
        range(len(softmax_output)), 
        class_targets
    ]
# Mask Values - only for one hot encoded labels
elif len(class_targets.shape) == 2:
    correct_confidences = np.sum(
        softmax_output * class_targets, 
        axis = 1)

neg_loss = -np.log(correct_confidences)
average_loss = np.mean(neg_loss)
print(average_loss)