import numpy as np
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
    
