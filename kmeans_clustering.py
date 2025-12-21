"""
K-Means Clustering Implementation
A simple implementation of K-Means clustering algorithm from scratch.
"""

import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    """
    K-Means clustering algorithm.
    
    Parameters:
    -----------
    n_clusters : int, default=3
        The number of clusters to form
    max_iterations : int, default=100
        Maximum number of iterations
    random_state : int, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, n_clusters=3, max_iterations=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None
    
    def initialize_centroids(self, X):
        """
        Initialize centroids randomly from the data points.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        """
        np.random.seed(self.random_state)
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_indices]
    
    def compute_distances(self, X):
        """
        Compute Euclidean distances between data points and centroids.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        distances : array, shape (n_samples, n_clusters)
            Distances from each point to each centroid
        """
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        return distances
    
    def assign_clusters(self, distances):
        """
        Assign each data point to the nearest centroid.
        
        Parameters:
        -----------
        distances : array, shape (n_samples, n_clusters)
            Distances from each point to each centroid
            
        Returns:
        --------
        labels : array, shape (n_samples,)
            Cluster labels for each point
        """
        return np.argmin(distances, axis=1)
    
    def update_centroids(self, X, labels):
        """
        Update centroids as the mean of points in each cluster.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        labels : array, shape (n_samples,)
            Current cluster labels
            
        Returns:
        --------
        centroids : array, shape (n_clusters, n_features)
            Updated centroids
        """
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                centroids[i] = cluster_points.mean(axis=0)
            else:
                centroids[i] = self.centroids[i]
        return centroids
    
    def fit(self, X):
        """
        Fit the K-Means model to the training data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        """
        # Initialize centroids
        self.initialize_centroids(X)
        
        for iteration in range(self.max_iterations):
            # Assign clusters
            distances = self.compute_distances(X)
            new_labels = self.assign_clusters(distances)
            
            # Update centroids
            new_centroids = self.update_centroids(X, new_labels)
            
            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                print(f"Converged at iteration {iteration}")
                break
            
            self.centroids = new_centroids
            self.labels = new_labels
        
        # Calculate final inertia (sum of squared distances to nearest centroid)
        final_distances = self.compute_distances(X)
        self.inertia_ = np.sum(np.min(final_distances, axis=1) ** 2)
    
    def predict(self, X):
        """
        Predict cluster labels for new data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            New data
            
        Returns:
        --------
        labels : array, shape (n_samples,)
            Cluster labels
        """
        distances = self.compute_distances(X)
        return self.assign_clusters(distances)


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    
    # Create three clusters
    cluster1 = np.random.randn(100, 2) + np.array([2, 2])
    cluster2 = np.random.randn(100, 2) + np.array([8, 3])
    cluster3 = np.random.randn(100, 2) + np.array([5, 8])
    
    X = np.vstack([cluster1, cluster2, cluster3])
    
    # Create and train model
    model = KMeans(n_clusters=3, max_iterations=100, random_state=42)
    model.fit(X)
    
    # Get predictions
    labels = model.predict(X)
    
    # Print results
    print(f"Number of clusters: {model.n_clusters}")
    print(f"Inertia: {model.inertia_:.2f}")
    print(f"Final centroids:\n{model.centroids}")
    
    # Visualize clusters
    plt.figure(figsize=(10, 6))
    
    # Plot data points colored by cluster
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
    for i in range(model.n_clusters):
        cluster_points = X[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   c=colors[i], label=f'Cluster {i}', alpha=0.5)
    
    # Plot centroids
    plt.scatter(model.centroids[:, 0], model.centroids[:, 1], 
               c='black', marker='X', s=200, label='Centroids', edgecolors='white', linewidths=2)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('K-Means Clustering')
    plt.legend()
    plt.grid(True)
    plt.show()
