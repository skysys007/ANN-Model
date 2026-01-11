import numpy as np
import nnfs
import matplotlib.pyplot as plt 
nnfs.init()
#sets the seed to 0 and creates a float32 datatype, which overrides the original dpt product from NUMPY
from nnfs.datasets import spiral_data

X, y  = spiral_data(samples=100, classes = 3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap ="brg")
plt.show()
