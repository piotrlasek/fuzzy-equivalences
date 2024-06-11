import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances
import data as data

def gaussian_kernel(distance, sigma):
    return np.exp(-distance**2 / (2 * sigma**2))

def compute_affinity_matrix(X, sigma):
    pairwise_distances = squareform(pdist(X, 'euclidean'))
    return gaussian_kernel(pairwise_distances, sigma)

def diffusion_map(A, t):
    # Perform eigen-decomposition
    eigenvalues, eigenvectors = eigh(A)
    # Sort the eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    # Compute the diffusion map
    return eigenvectors[:, :t] * eigenvalues[:t]


def compute_constrained_distance_matrix(X, cannot_links, sigma, t):
    A = compute_affinity_matrix(X, sigma)
    eigenvalues, diffusion_map_coords = diffusion_map(A, t)

    # Initialize a list to hold all v arrays for each cannot-link constraint
    vs = []

    # Compute v for each cannot-link pair and store in the list
    for c1, c2 in cannot_links:
        v = (diffusion_map_coords[:, c2] - diffusion_map_coords[:, c1]) / \
            (diffusion_map_coords[:, c2] + diffusion_map_coords[:, c1])
        v[c1] = 1
        v[c2] = -1
        vs.append(v)

    # Initialize a matrix to hold the maximum constrained distance for each point pair
    constrained_distance_matrix = np.zeros((len(X), len(X)))

    # For each v array, compute the constrained distance and update the matrix with the maximum distance
    for v in vs:
        constrained_distance = np.abs(np.subtract.outer(v, v))
        constrained_distance_matrix = np.maximum(constrained_distance_matrix, constrained_distance)

    return constrained_distance_matrix

# Example dataset and constraint
X = np.array([
    [1.0, 2.0],  # Point 1
    [2.0, 3.0],  # Point 2
    [3.0, 3.0],  # Point 3
    [5.0, 2.0],  # Point 4
    [6.0, 3.0],  # Point 5 (Constrained with Point 1)
])

n_samples = 100
n_clusters = 3
X, y = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=2, center_box=(-1, 1), random_state=42)

X = data.normalize_01(X)

cannot_links = [(0, 1), (10, 20)]

sigma = 1.0  # Kernel's width for the Gaussian kernel
t = n_samples        # Number of dimensions in the diffusion map (since we have 2D points)

# Calculate the constrained distance matrix for our example
cd = compute_constrained_distance_matrix(X, cannot_links, sigma, t)
cd = cd.astype(float)


m1 = pairwise_distances(X)

fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns
    # Original distance matrix
axs[0].scatter(X[:, 0], X[:, 1], cmap='viridis', edgecolor='k', alpha=0.7)
axs[0].set_title('KMedoids')

plt.tight_layout()
plt.show()

# Show the result
print("Distance Matrix:\n", m1)
print("Constrained Distance Matrix:\n", cd)
