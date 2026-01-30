# Overfitting vs. Generalization

**Overfitting** is effectively just memorizing the data without any understanding of it. An overfit model will do very well predicting the data that it has already seen, but often significantly worse on unseen data.

![Figure 11.01](../assets/overfitting.svg)

**Figure 11.01 Analysis:**
* **Left Image (Good Generalization):** The decision boundary is smooth and captures the underlying trend, accepting some errors (noise) to ensure it works well on new data.
* **Right Image (Overfitting):** The decision boundary is jagged and complex. It attempts to "memorize" the specific position of every noise point, which will likely cause it to fail on new, unseen data.