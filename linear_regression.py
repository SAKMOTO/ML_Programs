"""
Linear Regression Implementation
A simple implementation of Linear Regression from scratch using gradient descent.
"""

import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    """
    Linear Regression model using gradient descent optimization.
    
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
    
    def fit(self, X, y):
        """
        Fit the linear regression model to the training data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.iterations):
            # Predictions
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Calculate cost
            cost = (1/(2*n_samples)) * np.sum((y_pred - y)**2)
            self.cost_history.append(cost)
    
    def predict(self, X):
        """
        Predict using the linear model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted values
        """
        return np.dot(X, self.weights) + self.bias
    
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
    # Generate sample data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X.squeeze() + np.random.randn(100)
    
    # Create and train model
    model = LinearRegression(learning_rate=0.01, iterations=1000)
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Print results
    print(f"Weights: {model.weights}")
    print(f"Bias: {model.bias}")
    print(f"Final Cost: {model.cost_history[-1]:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', alpha=0.5, label='Actual data')
    plt.plot(X, y_pred, color='red', linewidth=2, label='Predictions')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.legend()
    plt.grid(True)
    plt.show()
