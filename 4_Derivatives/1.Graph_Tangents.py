import matplotlib.pyplot as plt
import numpy as np

# A linear function
def f(x):
    return 2*x
x = np.array(range(5))
y = f(x)

print(x)
print(y)

plt.plot(x, y)
plt.show()

# Slope - the change in Y wrt to change in X
print((y[1] - y[0]/x[1] - x[0]))

def f(x):
    return 2*x**2;
# granularity in the graph using numpy's arrange, allowing us to plot with smaller steps
x = np.arange(0, 5, 0.0001)
y = f(x)
# plotting the y = 2x^2 graph 
plt.plot(x, y)
# colors for the tangent line
colors = ['k', 'g', 'r', 'b', 'c']
# function for the tangent line
def approximate_tangent_line(x, approximate_derivative):
    return (approximate_derivative*x)+b
# loop for drawing the tangent line in range of x[0,5]
for i in range(5):
    p2_delta = 0.0001
    x1 = i;
    x2 = x1+p2_delta
    y1 = f(x1)
    y2 = f(x2)
    print((x1, y1),(x2, y2))
    
    #approximate derivatice
    approximate_derivative = (y2-y1)/(x2-x1)
    # intercept for the tangent
    b = y2 - (approximate_derivative*x2)
    # the line range of the plotted tangent line
    to_plot = [x1-0.9, x1, x1+0.9]

    # plotting the tanget lines at (x[i], y[i])
    plt.scatter(x1, y1, c=colors[i])
    plt.plot([point for point in to_plot], 
             [approximate_tangent_line(point, approximate_derivative)
              for point in to_plot], 
              c = colors[i])
    print('Approx Derivative for f(x)', 'where x = ', x1 ,' is ', approximate_derivative)
# plotting the graph with tangent lines 
plt.show() 
                
                

