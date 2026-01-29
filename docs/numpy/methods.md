# Neural Networks from Scratch (NNFS)
<<<<<<< HEAD

## Functions Used up to Chapter 10

This document lists **all important functions, methods, and operations** used in the book *Neural Networks from Scratch* up to **Chapter 10**, with **brief descriptions** .
=======
## Functions Used up to Chapter 10

This document lists all important **functions, methods, and operations** used in the book *Neural Networks from Scratch* (up to Chapter 10), with brief explanations.
>>>>>>> de56031 (methods)

---

## 1. Core NumPy Functions

### `np.array()`
<<<<<<< HEAD

Creates an array object to store vectors, matrices, and tensors.

### `np.dot(A, B)`

Performs matrix multiplication.
[
Z = XW + b
]
Used in dense layers to compute neuron outputs.

### `np.random.randn()`

Generates random values from a normal distribution.
Used to initialize weights.

### `np.zeros()` / `np.ones()`

Creates arrays filled with zeros or ones.
Used for biases and gradient placeholders.

### `np.maximum(a, b)`

Returns element-wise maximum.
Used in ReLU activation:
[
ReLU(x) = \max(0, x)
]

### `np.sum(x, axis, keepdims)`

Adds elements across dimensions.
Used in softmax normalization and bias gradients.

### `np.exp(x)`

Computes exponential values.
Used in softmax activation:
[
softmax(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
]

### `np.log(x)`

Computes natural logarithm.
Used in categorical cross-entropy loss.

### `np.mean(x)`

Computes average.
Used to calculate mean loss and accuracy.

### `np.argmax(x, axis)`

Returns index of maximum value.
Used for prediction and accuracy calculation.

### `np.clip(x, min, max)`

Limits values to a fixed range.
Prevents numerical instability in `log()`.

### `np.copy()`

Creates a copy of an array.
Used to safely modify gradients.

### `np.arange(n)`

Creates a sequence of integers.
Used in optimized loss backpropagation.

### `np.eye(n)`

Creates an identity matrix.
Used in categorical cross-entropy backward pass.

### `np.diagflat(x)`

Creates a diagonal matrix.
=======
Creates an array to store vectors, matrices, and tensors.

---

### `np.dot(A, B)`
Matrix multiplication.

$$
Z = X \cdot W + b
$$

Used in dense (fully connected) layers.

---

### `np.random.randn()`
Generates random values from a normal distribution.
Used for weight initialization.

---

### `np.zeros()` / `np.ones()`
Creates arrays filled with zeros or ones.
Used for biases and gradient storage.

---

### `np.maximum(a, b)`
Element-wise maximum.
Used in ReLU activation:

$$
\text{ReLU}(x) = \max(0, x)
$$

---

### `np.sum(x, axis, keepdims=True)`
Sums array elements along a given axis.
Used in softmax normalization and bias gradients.

---

### `np.exp(x)`
Computes the exponential.
Used in softmax activation:

$$
\sigma(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
$$

---

### `np.log(x)`
Computes the natural logarithm.
Used in categorical cross-entropy loss.

---

### `np.mean(x)`
Computes the mean value.
Used for average loss and accuracy.

---

### `np.argmax(x, axis=1)`
Returns index of the maximum value.
Used for predictions and accuracy calculation.

---

### `np.clip(x, min, max)`
Clips values into a fixed range.
Prevents numerical instability in `log()`.

---

### `np.copy(x)`
Creates a copy of an array.
Used when modifying gradients safely.

---

### `np.arange(n)`
Creates a sequence of integers from `0` to `n-1`.
Used in optimized backpropagation.

---

### `np.eye(n)`
Creates an identity matrix.
Used in categorical cross-entropy backward pass.

---

### `np.diagflat(x)`
Creates a diagonal matrix from a vector.
>>>>>>> de56031 (methods)
Used in softmax Jacobian matrix.

---

<<<<<<< HEAD
## 2. Dataset Utilities (nnfs)

### `nnfs.init()`

Initializes random seeds and default data types.
Ensures reproducibility.

### `spiral_data(samples, classes)`

Generates spiral-shaped classification data.
Used for visualization and training.

### `vertical_data(samples, classes)`

Generates vertical-line classification data.
Used in early chapters.
=======
## 2. NNFS Utility Functions

### `nnfs.init()`
Initializes random seed and default data type (`float32`).
Ensures reproducibility.

---

### `spiral_data(samples, classes)`
Generates spiral-shaped classification data.

---

### `vertical_data(samples, classes)`
Generates vertically separable classification data.
>>>>>>> de56031 (methods)

---

## 3. Dense Layer

### `Layer_Dense.__init__(n_inputs, n_neurons)`
<<<<<<< HEAD

Initializes weights and biases.

### `Layer_Dense.forward(inputs)`

Computes forward pass:
[
output = XW + b
]

### `Layer_Dense.backward(dvalues)`

Computes gradients:

* `dweights = X^T · dvalues`
* `dbiases = sum(dvalues)`
* `dinputs = dvalues · W^T`
=======
Initializes weights and biases.

---

### `Layer_Dense.forward(inputs)`
Forward pass:

$$
\text{output} = X \cdot W + b
$$

---

### `Layer_Dense.backward(dvalues)`
Computes gradients:

$$
\frac{\partial L}{\partial W} = X^T \cdot \text{dvalues}
$$

$$
\frac{\partial L}{\partial b} = \sum \text{dvalues}
$$

$$
\frac{\partial L}{\partial X} = \text{dvalues} \cdot W^T
$$
>>>>>>> de56031 (methods)

---

## 4. Activation Functions

<<<<<<< HEAD
### ReLU Activation

#### `Activation_ReLU.forward(inputs)`

Applies ReLU function.

#### `Activation_ReLU.backward(dvalues)`

Passes gradient only where input > 0.

---

### Softmax Activation

#### `Activation_Softmax.forward(inputs)`

Converts logits into probabilities.

#### `Activation_Softmax.backward(dvalues)`

Uses Jacobian matrix:
[
J = \text{diag}(p) - p p^T
]
=======
### ReLU

#### `Activation_ReLU.forward(inputs)`
Applies ReLU function.

#### `Activation_ReLU.backward(dvalues)`
Passes gradients only where input $> 0$.

---

### Softmax

#### `Activation_Softmax.forward(inputs)`
Converts raw scores into probabilities.

#### `Activation_Softmax.backward(dvalues)`
Uses Jacobian matrix:

$$
J = \text{diag}(p) - p \cdot p^T
$$
>>>>>>> de56031 (methods)

---

## 5. Loss Functions

### Base Loss Class

#### `Loss.calculate(output, y)`
<<<<<<< HEAD

Computes mean loss across samples.
=======
Computes mean loss over all samples.
>>>>>>> de56031 (methods)

---

### Categorical Cross-Entropy Loss

#### `Loss_CategoricalCrossentropy.forward(y_pred, y_true)`

<<<<<<< HEAD
Loss formula:
[
L = -\sum y \log(\hat{y})
]

#### `Loss_CategoricalCrossentropy.backward(dvalues, y_true)`

Computes gradient of loss w.r.t predictions.

---

## 6. Combined Softmax + Loss (Chapter 10)

### `Activation_Softmax_Loss_CategoricalCrossentropy.forward()`

Runs softmax and loss together for efficiency.
=======
$$
L = -\sum y_{\text{true}} \cdot \log(y_{\text{pred}})
$$

---

#### `Loss_CategoricalCrossentropy.backward(dvalues, y_true)`
Computes gradient of loss with respect to predictions.

---

## 6. Combined Softmax + Cross-Entropy (Chapter 10)

### `Activation_Softmax_Loss_CategoricalCrossentropy.forward()`
Runs softmax and loss together for numerical stability.

---
>>>>>>> de56031 (methods)

### `Activation_Softmax_Loss_CategoricalCrossentropy.backward(dvalues, y_true)`

Optimized gradient:
<<<<<<< HEAD
[
\frac{\partial L}{\partial z} = \hat{y} - y
]

---

## 7. Optimizer (SGD)

### `Optimizer_SGD.__init__(learning_rate, decay, momentum)`

Initializes optimizer parameters.

### `pre_update_params()`

Applies learning rate decay.

### `update_params(layer)`

Updates weights and biases:
[
W = W - \eta \nabla W
]

### `post_update_params()`

=======

$$
\frac{\partial L}{\partial z} = \hat{y} - y
$$

---

## 7. Optimizer: Stochastic Gradient Descent (SGD)

### `Optimizer_SGD.__init__(learning_rate, decay, momentum)`
Initializes optimizer parameters.

---

### `pre_update_params()`
Applies learning rate decay.

---

### `update_params(layer)`

$$
W = W - \eta \cdot \nabla W
$$

$$
b = b - \eta \cdot \nabla b
$$

*(Where $\eta$ is the learning rate)*

---

### `post_update_params()`
>>>>>>> de56031 (methods)
Increments iteration counter.

---

## 8. Accuracy Calculation

<<<<<<< HEAD
### `Accuracy.calculate(predictions, y)`

Uses:

* `np.argmax()`
* `np.mean()`

---

## 9. Must-Remember Formula Summary

* Dense layer: ( XW + b )
* ReLU: ( \max(0, x) )
* Softmax: ( e^x / \sum e^x )
* Cross-entropy: ( -\log(\hat{y}) )
* Gradient shortcut: ( \hat{y} - y )

---

### ✅ Covers NNFS Chapters 1–10 completely
=======
Uses:
- `np.argmax()`
- `np.mean()`

---

## 9. Key Formulas to Remember

- Dense layer:
$$
X \cdot W + b
$$

- ReLU:
$$
\max(0, x)
$$

- Softmax:
$$
\frac{e^x}{\sum e^x}
$$

- Cross-entropy loss:
$$
-\log(\hat{y}_{correct})
$$

- Softmax + loss gradient shortcut:
$$
\hat{y} - y
$$

---
>>>>>>> de56031 (methods)
