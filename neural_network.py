"""
Simple Neural Network Implementation
A basic implementation of a feedforward neural network from scratch.
"""

import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """
    A simple feedforward neural network with one hidden layer.
    
    Parameters:
    -----------
    input_size : int
        Number of input features
    hidden_size : int
        Number of neurons in hidden layer
    output_size : int
        Number of output classes
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
        self.loss_history = []
    
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
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_derivative(self, z):
        """
        Derivative of sigmoid function.
        
        Parameters:
        -----------
        z : array-like
            Input values
            
        Returns:
        --------
        array-like
            Derivative of sigmoid
        """
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def softmax(self, z):
        """
        Softmax activation function for output layer.
        
        Parameters:
        -----------
        z : array-like
            Input values
            
        Returns:
        --------
        array-like
            Softmax probabilities
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        Forward propagation through the network.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        A2 : array-like, shape (n_samples, output_size)
            Output probabilities
        cache : dict
            Intermediate values for backpropagation
        """
        # Hidden layer
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.sigmoid(Z1)
        
        # Output layer
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.softmax(Z2)
        
        cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
        return A2, cache
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute cross-entropy loss.
        
        Parameters:
        -----------
        y_true : array-like, shape (n_samples, output_size)
            True labels (one-hot encoded)
        y_pred : array-like, shape (n_samples, output_size)
            Predicted probabilities
            
        Returns:
        --------
        loss : float
            Cross-entropy loss
        """
        n_samples = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / n_samples
        return loss
    
    def backward(self, X, y_true, cache):
        """
        Backward propagation to compute gradients.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        y_true : array-like, shape (n_samples, output_size)
            True labels (one-hot encoded)
        cache : dict
            Cached values from forward propagation
            
        Returns:
        --------
        grads : dict
            Gradients for weights and biases
        """
        n_samples = X.shape[0]
        A1 = cache['A1']
        A2 = cache['A2']
        
        # Output layer gradients
        dZ2 = A2 - y_true
        dW2 = np.dot(A1.T, dZ2) / n_samples
        db2 = np.sum(dZ2, axis=0, keepdims=True) / n_samples
        
        # Hidden layer gradients
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(cache['Z1'])
        dW1 = np.dot(X.T, dZ1) / n_samples
        db1 = np.sum(dZ1, axis=0, keepdims=True) / n_samples
        
        grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
        return grads
    
    def update_parameters(self, grads):
        """
        Update network parameters using gradients.
        
        Parameters:
        -----------
        grads : dict
            Gradients for weights and biases
        """
        self.W1 -= self.learning_rate * grads['dW1']
        self.b1 -= self.learning_rate * grads['db1']
        self.W2 -= self.learning_rate * grads['dW2']
        self.b2 -= self.learning_rate * grads['db2']
    
    def fit(self, X, y, epochs=1000, verbose=True):
        """
        Train the neural network.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples, output_size)
            Training labels (one-hot encoded)
        epochs : int, default=1000
            Number of training epochs
        verbose : bool, default=True
            Whether to print progress
        """
        for epoch in range(epochs):
            # Forward pass
            y_pred, cache = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)
            
            # Backward pass
            grads = self.backward(X, y, cache)
            
            # Update parameters
            self.update_parameters(grads)
            
            if verbose and (epoch + 1) % 100 == 0:
                accuracy = self.score(X, y)
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.2%}")
    
    def predict(self, X):
        """
        Predict class labels.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        predictions : array, shape (n_samples,)
            Predicted class labels
        """
        y_pred, _ = self.forward(X)
        return np.argmax(y_pred, axis=1)
    
    def score(self, X, y):
        """
        Calculate accuracy.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        y : array-like, shape (n_samples, output_size)
            True labels (one-hot encoded)
            
        Returns:
        --------
        accuracy : float
            Classification accuracy
        """
        predictions = self.predict(X)
        true_labels = np.argmax(y, axis=1)
        return np.mean(predictions == true_labels)
    
    def plot_loss_history(self):
        """Plot the loss function history during training."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.loss_history)), self.loss_history)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss History')
        plt.grid(True)
        plt.show()


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    
    # Generate sample data
    X, y = make_classification(n_samples=500, n_features=4, n_informative=3,
                              n_redundant=0, n_classes=3, random_state=42)
    
    # One-hot encode labels
    y_encoded = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    # Create and train model
    model = NeuralNetwork(input_size=4, hidden_size=10, output_size=3, learning_rate=0.1)
    model.fit(X_train, y_train, epochs=1000, verbose=True)
    
    # Evaluate model
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"\nFinal Results:")
    print(f"Training Accuracy: {train_accuracy:.2%}")
    print(f"Test Accuracy: {test_accuracy:.2%}")
    
    # Plot loss history
    model.plot_loss_history()
