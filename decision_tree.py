"""
Decision Tree Classifier Implementation
A simple implementation of Decision Tree for classification from scratch.
"""

import numpy as np
from collections import Counter


class Node:
    """
    A node in the decision tree.
    
    Parameters:
    -----------
    feature : int, default=None
        Feature index used for splitting
    threshold : float, default=None
        Threshold value for the split
    left : Node, default=None
        Left child node
    right : Node, default=None
        Right child node
    value : int, default=None
        Class label if leaf node
    """
    
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        """Check if the node is a leaf node."""
        return self.value is not None


class DecisionTree:
    """
    Decision Tree classifier using recursive splitting.
    
    Parameters:
    -----------
    max_depth : int, default=10
        Maximum depth of the tree
    min_samples_split : int, default=2
        Minimum number of samples required to split a node
    """
    
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
    def entropy(self, y):
        """
        Calculate entropy of a label array.
        
        Parameters:
        -----------
        y : array-like
            Labels
            
        Returns:
        --------
        entropy : float
            Entropy value
        """
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy
    
    def information_gain(self, X_column, y, threshold):
        """
        Calculate information gain for a split.
        
        Parameters:
        -----------
        X_column : array-like
            Feature column
        y : array-like
            Labels
        threshold : float
            Split threshold
            
        Returns:
        --------
        gain : float
            Information gain
        """
        # Parent entropy
        parent_entropy = self.entropy(y)
        
        # Split data
        left_mask = X_column <= threshold
        right_mask = X_column > threshold
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0
        
        # Calculate weighted entropy of children
        n = len(y)
        n_left, n_right = np.sum(left_mask), np.sum(right_mask)
        e_left, e_right = self.entropy(y[left_mask]), self.entropy(y[right_mask])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right
        
        # Information gain
        gain = parent_entropy - child_entropy
        return gain
    
    def best_split(self, X, y):
        """
        Find the best feature and threshold to split on.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Labels
            
        Returns:
        --------
        best_feature : int
            Best feature index
        best_threshold : float
            Best threshold value
        """
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                gain = self.information_gain(X[:, feature], y, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Labels
        depth : int
            Current depth
            
        Returns:
        --------
        node : Node
            Root node of the tree or subtree
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_labels == 1 or 
            n_samples < self.min_samples_split):
            # Create leaf node
            most_common_label = Counter(y).most_common(1)[0][0]
            return Node(value=most_common_label)
        
        # Find best split
        best_feature, best_threshold = self.best_split(X, y)
        
        if best_feature is None:
            # Create leaf node if no valid split found
            most_common_label = Counter(y).most_common(1)[0][0]
            return Node(value=most_common_label)
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold
        
        # Recursively build left and right subtrees
        left_subtree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(best_feature, best_threshold, left_subtree, right_subtree)
    
    def fit(self, X, y):
        """
        Fit the decision tree to the training data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Labels
        """
        self.root = self.build_tree(X, y)
    
    def predict_sample(self, x, node):
        """
        Predict class label for a single sample.
        
        Parameters:
        -----------
        x : array-like
            Single sample
        node : Node
            Current node
            
        Returns:
        --------
        label : int
            Predicted class label
        """
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self.predict_sample(x, node.left)
        else:
            return self.predict_sample(x, node.right)
    
    def predict(self, X):
        """
        Predict class labels for samples.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        predictions : array, shape (n_samples,)
            Predicted class labels
        """
        return np.array([self.predict_sample(x, self.root) for x in X])


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    
    # Generate sample data
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                              n_informative=2, n_clusters_per_class=1,
                              random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    model = DecisionTree(max_depth=5, min_samples_split=2)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy:.2%}")
    
    # Visualize decision boundary
    plt.figure(figsize=(10, 6))
    
    # Create mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdYlBu', edgecolors='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Tree Classification')
    plt.colorbar()
    plt.grid(True)
    plt.show()
