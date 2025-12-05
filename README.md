# Handwritten Digit Classification with a Pure NumPy Neural Network

This project implements a fully functional Artificial Neural Network (ANN) to classify handwritten digits from the MNIST dataset. The core focus of this project is to understand the mathematical underpinnings of deep learning by building the model using only `NumPy`, without relying on high-level deep learning frameworks like TensorFlow, Keras, or PyTorch.

## 📌 Project Overview

The model takes 28x28 grayscale images of handwritten digits (0-9) and classifies them into one of ten categories. The implementation handles the entire machine learning pipeline, including data preprocessing, forward propagation, backpropagation, and optimization via gradient descent.

## 🛠️ Tech Stack & Dependencies

* **Language:** Python
* **Core Logic:** `NumPy` (Matrix multiplication, activation functions, gradients)
* **Visualization:** `Matplotlib`
* **Utilities:** `scikit-learn` (Used strictly for data fetching, splitting, and metric calculation)

## 🧠 Model Architecture

The network is designed as a Multi-Layer Perceptron (MLP) with the following structure:

* **Input Layer:** 784 nodes (corresponding to flattened 28x28 pixel images).
* **Hidden Layer(s):** Configurable density (e.g., 128 nodes) using **Sigmoid** or **ReLU** activation functions.
* **Output Layer:** 10 nodes using **SoftMax** activation to generate class probabilities for digits 0-9.

## ⚙️ Implementation Details

### 1. Data Preprocessing
* **Normalization:** Pixel values are scaled using Min-Max scaling to ensure stable convergence.
* **Encoding:** Target labels are converted into one-hot encoded vectors (e.g., `3` becomes `[0,0,0,1,0,0,0,0,0,0]`) to align with the SoftMax output layer.

### 2. Core Algorithms
* **Forward Propagation:** Computes the network's output by passing inputs through weights, biases, and activation functions.
* **Loss Calculation:** Utilizes **Cross-Entropy Loss** to measure the difference between predicted probabilities and actual labels.
* **Backpropagation:** Manually derives and implements gradients of the loss function with respect to weights and biases to update parameters.
* **Optimization:** Weights are updated using **Stochastic Gradient Descent (SGD)** or **Mini-Batch Gradient Descent** (batch sizes of 32 or 64).

### 3. Training
The training loop iterates over the dataset for a fixed number of epochs, updating gradients via mini-batches and validating performance against a hold-out validation set (10% of training data).

## 📊 Evaluation Metrics

Upon training completion, the model is evaluated on the test set using the following metrics:

* **Accuracy:** Overall percentage of correct predictions.
* **Confusion Matrix:** Visual representation of true vs. predicted classes.
* **Precision & F1 Score:** To measure class-wise performance balance.

## 🚀 Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/mnist-ann-scratch.git](https://github.com/yourusername/mnist-ann-scratch.git)
    cd mnist-ann-scratch
    ```

2.  **Install dependencies:**
    ```bash
    pip install numpy matplotlib scikit-learn
    ```

3.  **Run the training script:**
    ```bash
    python main.py
    ```

---
*Note: This implementation demonstrates the fundamentals of neural networks including matrix calculus and optimization algorithms, deliberately avoiding pre-built "black box" libraries to showcase algorithmic understanding.*
