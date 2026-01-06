"""
Logistic Regression Implementation
A simple implementation of Logistic Regression from scratch using gradient descent.
"""

import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    """
    Logistic Regression classifier using gradient descent optimization.
    
    Parameters:
    -----------
    learning_rate : float, default=0.01
        The learning rate for gradient descent
    iterations : int, default=1000
        Number of iterations for gradient descent
    """
    
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def sigmoid(self, z):
        """
        Sigmoid activation function.
        
        Parameters:
        -----------
        z : array-like
            Input values
            
        Returns:
        --------
        array-like
            Sigmoid of input
        """
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """
        Fit the logistic regression model to the training data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values (0 or 1)
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.iterations):
            # Linear combination
            linear_pred = np.dot(X, self.weights) + self.bias
            # Apply sigmoid
            predictions = self.sigmoid(linear_pred)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Calculate cost (binary cross-entropy)
            cost = (-1/n_samples) * np.sum(y * np.log(predictions + 1e-15) + 
                                          (1 - y) * np.log(1 - predictions + 1e-15))
            self.cost_history.append(cost)
    
    def predict_proba(self, X):
        """
        Predict probability estimates.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        proba : array, shape (n_samples,)
            Probability of the positive class
        """
        linear_pred = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_pred)
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
        threshold : float, default=0.5
            Decision threshold
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted class labels (0 or 1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def plot_cost_history(self):
        """Plot the cost function history during training."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(self.iterations), self.cost_history)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost Function History')
        plt.grid(True)
        plt.show()


# Example usage
if __name__ == "__main__":
    # Generate sample data for binary classification
    np.random.seed(42)
    
    # Class 0
    X0 = np.random.randn(50, 2) + np.array([2, 2])
    y0 = np.zeros(50)
    
    # Class 1
    X1 = np.random.randn(50, 2) + np.array([5, 5])
    y1 = np.ones(50)
    
    # Combine data
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])
    
    # Create and train model
    model = LogisticRegression(learning_rate=0.1, iterations=1000)
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y)
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Weights: {model.weights}")
    print(f"Bias: {model.bias}")
    print(f"Final Cost: {model.cost_history[-1]:.4f}")
    
    # Visualize decision boundary
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y==0][:, 0], X[y==0][:, 1], color='blue', label='Class 0', alpha=0.5)
    plt.scatter(X[y==1][:, 0], X[y==1][:, 1], color='red', label='Class 1', alpha=0.5)
    
    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0.5], colors='green', linewidths=2)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary')
    plt.legend()
    plt.grid(True)
    plt.show()
