import numpy as np
from sklearn.datasets import load_iris

# Load the Iris Dataset
data = load_iris()

X = data.data # Shape : (150, 4) 
Y = data.target # Shape : (150, )
# y is already integer encoded, perfect CCEL

# Normalizing the input features
X = (X - X.mean(axis=0))/X.std(axis = 0)
# Dividing the difference of current value and mean with standard deviation 
# helps to normalize the data which helps the model to learn faster