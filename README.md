# Handwritten Digit Classification with a Pure NumPy Neural Network

## 🧠 Project Overview

This project implements a multi-layer **Artificial Neural Network (ANN)** from scratch using **NumPy** to classify handwritten digits (0-9) from the **MNIST dataset**.

The core objective is to build a foundational understanding of neural networks by implementing all key components—**forward propagation**, **activation functions (ReLU/Softmax)**, **cross-entropy loss**, **backpropagation**, and **Mini-Batch Gradient Descent**—without relying on high-level deep learning frameworks like TensorFlow, Keras, or PyTorch.

### Key Features
* Implementation of a fully connected neural network architecture.
* Manual implementation of **ReLU** and **Softmax** activation functions.
* Derivation and implementation of the **backpropagation** algorithm.
* Training using **Mini-Batch Gradient Descent**.
* Comprehensive evaluation using **Confusion Matrix**, **Accuracy**, **Precision**, and **F1 Score**.

---

## ⚙️ Implementation Requirements

### 1. Dataset
* **Dataset:** MNIST Handwritten Digits (imported via `sklearn.datasets`).
* **Total Images:** 70,000 (60,000 Training, 10,000 Testing).
* **Features:** $784$ (flattened $28 \times 28$ image).
* **Classes:** 10 (digits 0-9).

### 2. Data Preprocessing
* **Normalization:** Pixel values scaled to the range $[0, 1]$ using **Min-Max scaling**.
* **One-Hot Encoding:** Target labels converted into 10-dimensional one-hot vectors (necessary for SoftMax and cross-entropy loss).

### 3. Model Architecture
The network must have at least one hidden layer. 

[Image of a feedforward neural network architecture with input, hidden, and output layers]

* **Input Layer:** 784 nodes.
* **Hidden Layer(s):** Chosen number of nodes (e.g., 128, 64).
    * **Activation:** **Sigmoid** or **ReLU**.
* **Output Layer:** 10 nodes.
    * **Activation:** **SoftMax**.

### 4. Core Functions Implemented
* **Initialization:** Weights and biases initialized with small random values.
* **Forward Propagation:** Computes layer outputs and final class probabilities.
* **Loss Function:** **Categorical Cross-Entropy Loss**.
* **Backpropagation:** Computes gradients $\frac{\partial L}{\partial \mathbf{W}}$ and $\frac{\partial L}{\partial \mathbf{b}}$.
* **Optimization:** **Stochastic Gradient Descent (SGD)** or **Mini-Batch Gradient Descent** to adjust parameters.

---

## 🚀 Getting Started

### Prerequisites

You need Python 3.x installed along with the required libraries.

```bash
pip install numpy matplotlib scikit-learn
