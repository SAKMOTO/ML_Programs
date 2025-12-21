<div align="center">

# 🤖 Machine Learning Programs

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-orange.svg)](https://numpy.org/)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)

*A comprehensive collection of Machine Learning algorithms implemented from scratch in Python*

[Getting Started](#-getting-started) • [Algorithms](#-algorithms) • [Usage](#-usage) • [Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [About](#-about)
- [What is Machine Learning?](#-what-is-machine-learning)
- [Algorithms](#-algorithms)
- [Getting Started](#-getting-started)
- [Usage Examples](#-usage-examples)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🌟 About

This repository contains clean, well-documented implementations of popular Machine Learning algorithms built from scratch using Python and NumPy. Each implementation is designed to be educational, helping you understand the mathematical foundations and inner workings of these algorithms.

### Why This Repository?

- 📚 **Educational**: Learn how ML algorithms work under the hood
- 🔍 **Transparent**: No black boxes - see every step of the algorithm
- 💻 **Practical**: Includes working examples and visualizations
- 🎯 **Well-Documented**: Comprehensive docstrings and comments
- 🚀 **Easy to Use**: Simple APIs similar to scikit-learn

---

## 🧠 What is Machine Learning?

Machine Learning (ML) is a subset of Artificial Intelligence that enables computers to learn and improve from experience without being explicitly programmed. ML algorithms build mathematical models based on sample data (training data) to make predictions or decisions.

### Types of Machine Learning:

1. **Supervised Learning**: Learning from labeled data
   - Classification (e.g., spam detection, image recognition)
   - Regression (e.g., price prediction, trend analysis)

2. **Unsupervised Learning**: Finding patterns in unlabeled data
   - Clustering (e.g., customer segmentation)
   - Dimensionality Reduction (e.g., feature extraction)

3. **Reinforcement Learning**: Learning through interaction with an environment
   - Game playing, robotics, autonomous systems

---

## 🔧 Algorithms

This repository currently includes implementations of the following algorithms:

### Supervised Learning

| Algorithm | Type | File | Description |
|-----------|------|------|-------------|
| **Linear Regression** | Regression | [`linear_regression.py`](linear_regression.py) | Predicts continuous values using a linear relationship |
| **Logistic Regression** | Classification | [`logistic_regression.py`](logistic_regression.py) | Binary classification using sigmoid function |
| **Decision Tree** | Classification | [`decision_tree.py`](decision_tree.py) | Tree-based classifier using information gain |
| **Neural Network** | Classification | [`neural_network.py`](neural_network.py) | Basic feedforward neural network with backpropagation |

### Unsupervised Learning

| Algorithm | Type | File | Description |
|-----------|------|------|-------------|
| **K-Means Clustering** | Clustering | [`kmeans_clustering.py`](kmeans_clustering.py) | Partitions data into K distinct clusters |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SAKMOTO/ML_Programs.git
   cd ML_Programs
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run an example**
   ```bash
   python linear_regression.py
   ```

---

## 💡 Usage Examples

### Linear Regression

```python
from linear_regression import LinearRegression
import numpy as np

# Create sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Create and train model
model = LinearRegression(learning_rate=0.01, iterations=1000)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
print(f"Predictions: {predictions}")
```

### Logistic Regression

```python
from logistic_regression import LogisticRegression
import numpy as np

# Create sample data
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Create and train model
model = LogisticRegression(learning_rate=0.1, iterations=1000)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.2%}")
```

### K-Means Clustering

```python
from kmeans_clustering import KMeans
import numpy as np

# Create sample data
X = np.random.randn(300, 2)

# Create and train model
model = KMeans(n_clusters=3, max_iterations=100, random_state=42)
model.fit(X)

# Get cluster assignments
labels = model.predict(X)
print(f"Cluster assignments: {labels}")
```

### Decision Tree

```python
from decision_tree import DecisionTree
import numpy as np

# Create sample data
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# Create and train model
model = DecisionTree(max_depth=5, min_samples_split=2)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.2%}")
```

### Neural Network

```python
from neural_network import NeuralNetwork
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Create sample data
X = np.random.randn(200, 4)
y = np.random.randint(0, 3, 200)
y_encoded = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))

# Create and train model
model = NeuralNetwork(input_size=4, hidden_size=10, output_size=3, learning_rate=0.1)
model.fit(X, y_encoded, epochs=1000, verbose=True)

# Make predictions
predictions = model.predict(X)
print(f"Predictions: {predictions}")
```

---

## 📁 Project Structure

```
ML_Programs/
│
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
│
├── linear_regression.py         # Linear Regression implementation
├── logistic_regression.py       # Logistic Regression implementation
├── kmeans_clustering.py         # K-Means Clustering implementation
├── decision_tree.py             # Decision Tree implementation
└── neural_network.py            # Neural Network implementation
```

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**!

### How to Contribute

1. **Fork the Project**
2. **Create your Feature Branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your Changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the Branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Contribution Ideas

- 🆕 Implement new ML algorithms (SVM, Random Forest, Naive Bayes, etc.)
- 📊 Add more visualization features
- 🧪 Add unit tests
- 📝 Improve documentation
- 🐛 Fix bugs or optimize existing code
- 🎨 Add Jupyter notebooks with tutorials

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**HARUN** - [@SAKMOTO](https://github.com/SAKMOTO)

Project Link: [https://github.com/SAKMOTO/ML_Programs](https://github.com/SAKMOTO/ML_Programs)

---

## 🙏 Acknowledgments

- Thanks to the open-source community for inspiration
- NumPy and scikit-learn for their excellent documentation
- All contributors who help improve this project

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

Made with ❤️ by [SAKMOTO](https://github.com/SAKMOTO)

</div>