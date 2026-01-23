# Forward Pass
x = [1.0, -2.0, 3.0]
w = [-3.0, -1.0, 1.0]
b = 1.0

# Multiplying inputs by weights
xw0 = x[0]*w[0]
xw1 = x[1]*w[1]
xw2 = x[2]*w[2]

# Weighted Sum of weighted inputs and bias
z = xw0 + xw1 + xw2 + b

# ReLU Activation Function
y = max(0, z)

# Backward Pass 

# The derivative from the next layer 
d_value = 1.0

# Derivative of ReLU
drelu_dz = d_value* (1.0 if z > 0 else 0.0)
print(drelu_dz)

# Partial Derivatives of Weighted Sum
dsum_xw0 = 1
dsum_xw1 = 1
dsum_xw2 = 1
dsum_b = 1
drelu_dxw0 = drelu_dz * dsum_xw0
drelu_dxw1 = drelu_dz * dsum_xw1
drelu_dxw2 = drelu_dz * dsum_xw2
drelu_db = drelu_dz * dsum_b
print(drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)

# Partial Derivatives of the Multiplication function
dmul_dx0 = w[0]
dmul_dx1 = w[1]
dmul_dx2 = w[2]
dmul_dw0 = x[0]
dmul_dw1 = x[1]
dmul_dw2 = x[2]

# Partial Derivative of the whole function w.r.t respective parameter
drelu_dx0 = drelu_dz*dmul_dx0
drelu_dx1 = drelu_dz*dmul_dx1
drelu_dx2 = drelu_dz*dmul_dx2
drelu_dw0 = drelu_dz*dmul_dw0
drelu_dw1 = drelu_dz*dmul_dw1
drelu_dw2 = drelu_dz*dmul_dw2
print(drelu_dx0, drelu_dx1, drelu_dx2, drelu_dw0, drelu_dw1, drelu_dw2)








