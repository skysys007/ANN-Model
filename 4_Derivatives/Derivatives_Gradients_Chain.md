## Math Summary: Derivatives & Gradients

Here is a summary of the calculus rules and solutions implemented in this project, derived from *Neural Networks from Scratch*.

### 1. Basic Derivative Solutions
Let's look at the basic building blocks for finding derivatives.

**Derivative of a Constant**
The derivative of a constant equals 0.

$$
\frac{d}{dx} 1 = 0
$$

$$
\frac{d}{dx} m = 0
$$

**Derivative of $x$**
The derivative of $x$ with respect to $x$ is 1.

$$
\frac{d}{dx} x = 1
$$

**Derivative of a Linear Function**
The derivative of a linear function equals its slope.

$$
\frac{d}{dx} (mx + b) = m
$$

### 2. Basic Derivative Rules
These rules describe how to handle more complex operations.

**Constant Multiple Rule**
The derivative of a constant multiple of a function equals the constant multiple of the function's derivative.

$$
\frac{d}{dx} [k \cdot f(x)] = k \cdot \frac{d}{dx} f(x)
$$

**Sum & Subtraction Rule**
The derivative of a sum (or subtraction) of functions equals the sum (or subtraction) of their derivatives.

$$
\frac{d}{dx} [f(x) + g(x)] = \frac{d}{dx} f(x) + \frac{d}{dx} g(x) = f'(x) + g'(x)
$$

$$
\frac{d}{dx} [f(x) - g(x)] = \frac{d}{dx} f(x) - \frac{d}{dx} g(x) = f'(x) - g'(x)
$$

**Power Rule (Exponentiation)**
The derivative of an exponentiation:

$$
\frac{d}{dx} x^n = n \cdot x^{n-1}
$$

---

### 3. Partial Derivatives
When dealing with multiple inputs (like in a Neural Network), we use partial derivatives.

**Partial Derivative of a Sum**
The partial derivative of a sum with respect to any input equals 1.

$$
f(x, y) = x + y \quad \to \quad \frac{\partial}{\partial x} f(x, y) = 1
$$

$$
\frac{\partial}{\partial y} f(x, y) = 1
$$

**Partial Derivative of Multiplication**
The partial derivative of the multiplication operation with 2 inputs, with respect to any input, equals the other input.

$$
f(x, y) = x \cdot y \quad \to \quad \frac{\partial}{\partial x} f(x, y) = y
$$

$$
\frac{\partial}{\partial y} f(x, y) = x
$$

**Derivative of Max Function (ReLU)**
The partial derivative of the max function of 2 variables with respect to any of them is 1 if that variable is the biggest, and 0 otherwise.

$$
f(x, y) = \max(x, y) \quad \to \quad \frac{\partial}{\partial x} f(x, y) = 1(x > y)
$$

For a single variable (ReLU activation), the derivative is 1 if the variable is greater than 0, and 0 otherwise.

$$
f(x) = \max(x, 0) \quad \to \quad \frac{d}{dx} f(x) = 1(x > 0)
$$

---

### 4. The Chain Rule & Gradients

**The Chain Rule**
The derivative of chained functions equals the product of the partial derivatives of the subsequent functions.

$$
\frac{d}{dx} f(g(x)) = \frac{d}{dg(x)} f(g(x)) \cdot \frac{d}{dx} g(x) = f'(g(x)) \cdot g'(x)
$$

The same applies to partial derivatives. For example:

$$
\frac{\partial}{\partial x} f(g(y, h(x, z))) = f'(g(y, h(x, z))) \cdot g'(y, h(x, z)) \cdot h'(x, z)
$$

**The Gradient**
The gradient is a vector of all possible partial derivatives. An example of a triple-input function:

$$
\nabla f(x, y, z) = \begin{bmatrix} 
\frac{\partial}{\partial x} f(x, y, z) \\ 
\frac{\partial}{\partial y} f(x, y, z) \\ 
\frac{\partial}{\partial z} f(x, y, z) 
\end{bmatrix} 
= 
\begin{bmatrix} 
\frac{\partial}{\partial x} \\ 
\frac{\partial}{\partial y} \\ 
\frac{\partial}{\partial z} 
\end{bmatrix} f(x, y, z)
$$