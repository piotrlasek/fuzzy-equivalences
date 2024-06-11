from nltk.cluster import KMeansClusterer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# Define your custom distance function
def euclidean_distance(vector1, vector2):
    """Calculate the Euclidean distance between two vectors."""
    return sum((p-q)**2 for p, q in zip(vector1, vector2)) ** 0.5

def custom_distance(u, v):
    differences = 1 - np.abs(u - v)
    distance = np.sum(differences)
    return distance

# Number of clusters
NUM_CLUSTERS = 4

# Initialize KMeansClusterer with your custom distance function
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=custom_distance, repeats=25)
#kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=euclidean_distance, repeats=25)

X, y = make_blobs(n_samples=100, centers=4, n_features=2, center_box=(-15.0, 15.0), random_state=7)
k = 4  # Number of clusters
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_normalized = (X - X_min) / (X_max - X_min)
X = X_normalized


# Cluster the data-xx-old
assigned_clusters = kclusterer.cluster(X_normalized, assign_clusters=True)

# Output the result
print(assigned_clusters)

# Convert assigned_clusters to a numpy array for easy indexing
assigned_clusters = np.array(assigned_clusters)

# Plot the clustered data-xx-old
plt.figure(figsize=(8, 6))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Colors for clusters

for i in range(NUM_CLUSTERS):
    # Select data-xx-old points that belong to the current cluster
    cluster_data = X_normalized[assigned_clusters == i]

    # Plot the data-xx-old points with the cluster-specific color
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[i], label=f'Cluster {i + 1}')

plt.title('Clustering results with KMeans')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.show()
