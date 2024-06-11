from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Define your custom distance function
def custom_distance(x, y):
    """
    Custom distance metric function. This is where you define how to calculate
    the distance between two points x and y.

    Example: Euclidean distance
    """
    distance = np.sqrt(np.sum((x - y) ** 2))
    print(distance)
    return distance

def d(u, v):
    #print(type(u))
    differences = 1 - np.abs(u - v)
    distance = 1 - np.mean(differences)
    #print(u, v)
    #print(differences)
    print(distance)
    #print("---")
    return distance

# Generate some sample data-xx-old
X = np.array([[1, 2], [2, 2], [2, 3],
              [8, 7], [8, 8], [25, 80]])

X, y = make_blobs(n_samples=10, centers=3, n_features=2, center_box=(-12.0, 12.0), random_state=27)

k = 3  # Number of clusters
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_normalized = (X - X_min) / (X_max - X_min)
X = X_normalized

# Initialize DBSCAN with the custom metric
db = DBSCAN(eps=0.2, min_samples=2, metric=custom_distance)
v

# Fit the model
db.fit(X_normalized)

# Get the cluster labels
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

noise_points = X[labels == -1]
clustered_points = X[labels != -1]
cluster_labels = labels[labels != -1]

# Plot clustered points
plt.scatter(clustered_points[:, 0], clustered_points[:, 1],
            c=cluster_labels, cmap='viridis', label='Clustered Points', alpha=0.8, edgecolor='k')

# Plot noise points
plt.scatter(noise_points[:, 0], noise_points[:, 1],
            c='gray', label='Noise Points', alpha=0.2)

plt.show()

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print('Cluster labels:', labels)
