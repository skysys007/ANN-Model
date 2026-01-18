# My ANN Model from Scratch 

Hi! Welcome to my repository. I am currently learning Deep Learning, and I decided to challenge myself by building an **Artificial Neural Network (ANN)** completely from scratch using Python.

I’m avoiding libraries like TensorFlow or PyTorch for now because I really want to understand the math and logic happening "under the hood."

## 🎯 The Goal
The main goal of this project is to learn. I want to be able to answer:
* How do neurons actually "learn"?
* What is the math behind Backpropagation?
* How does the code look without the "magic" of high-level frameworks?

## 🚀 Current Progress
**Status:** Just started! 🌱

I have successfully coded the first building block of the network: **The Dense Layer**.
It can currently take inputs, multiply them by weights, add biases, and produce an output (Forward Pass).

### What's Working:
- [x] Setting up the project structure.
- [x] **Class `Layer_Dense`**: It initializes random weights and biases.
- [x] **Forward Pass**: It calculates the dot product of inputs and weights.
- [x] **ReLU Activation Function**: It calculates the dot product of inputs and weights.
- [x] **SoftMax Activation Function**: It calculates the dot product of inputs and weights.

## 🛠️ Tools I'm Using
* **Python**: The language I'm coding in.
* **NumPy**: I'm using this for the matrix math (dot products) because doing that with raw Python lists is too slow!

## 💻 How to Run My Code
If you want to see what I've built so far:

1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/your-username/ANN-Model.git](https://github.com/your-username/ANN-Model.git)
    ```

2.  **Make sure you have NumPy:**
    ```bash
    pip install numpy
    ```

3.  **Test the Layer:**
    I've included a small script to test if the neurons fire correctly.
    ```python
    import numpy as np
    from layer import Layer_Dense

    # Creating some dummy input data (batch of 5 samples)
    X = np.array([[1.0, 2.0, 3.0, 2.5],
                  [2.0, 5.0, -1.0, 2.0],
                  [-1.5, 2.7, 3.3, -0.8],
                  [0.5, -0.2, 1.2, 3.1],
                  [1.1, 0.8, -0.5, 2.2]])

    # Creating a layer with 4 inputs and 5 neurons
    layer1 = Layer_Dense(4, 5)

    # Doing a forward pass
    layer1.forward(X)

    # Printing the result
    print("It works! Here is the output:")
    print(layer1.output)
    ```

## 📝 Learning Roadmap / To-Do
This is my checklist for the project. I'll check these off as I learn and code them.

- [x] **Step 1:** Create the Neurons (Dense Layer)
- [x] **Step 2:** The Forward Pass (Math part)
- [x] **Step 3:** Activation Functions (ReLU - for the hidden layer)
- [x] **Step 4:** Softmax Activation (For the output layer)
- [ ] **Step 5:** Calculating Loss (How wrong is the model?)
- [ ] **Step 6:** Backpropagation (The hard part!)
- [ ] **Step 7:** Optimizer (SGD/Adam)

## 👋 Connect
I'm documenting my learning journey here. If you have any tips for a beginner or see a bug in my math, feel free to open an issue or let me know!
