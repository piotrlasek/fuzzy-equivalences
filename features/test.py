import numpy as np
import matplotlib.pyplot as plt
from pydiffmap import diffusion_map as dm
from sklearn.metrics import pairwise_distances
from pydiffmap.visualization import embedding_plot, data_plot


X = np.array([
    [1, 3],  # Center for cluster 1
    [2, 4],  # Center for cluster 1
    [3, 2],  # Center for cluster 2
    [4, 3],  # Center for cluster 2
    [5, 5],  # Center for cluster 2
    # Add more centers if needed
])

cannot_links = [(0, 1)]  # Example cannot-link constraints

def calculate_diffusion_map(X, sigma=1.0):
    # X - data matrix
    # sigma - scale parameter for the affinity / kernel

    # Create an instance of the DiffusionMap class
    mydmap = dm.DiffusionMap.from_sklearn(epsilon=sigma, alpha=1.0, n_evecs=2)
    map = mydmap.fit_transform(X)

    # Return the diffusion matrix
    return map


def modify_distance_matrix_with_cannot_link(X, cannot_links):
    # X - data matrix
    # cannot_links - list of tuples (p, q) representing cannot-link constraints
    # diffusion_distance_matrix - precomputed diffusion distance matrix

    diffusion_distance_matrix = calculate_diffusion_map(X)

    M = X.shape[0]  # Number of data points
    N = len(cannot_links)  # Number of cannot-link constraints

    # Initialize new feature space matrix
    V = np.zeros((M, N))

    # Compute new feature values based on cannot-link constraints
    for c, (p, q) in enumerate(cannot_links):
        for i in range(M):
            if i == p:
                V[i, c] = -1
            elif i == q:
                V[i, c] = 1
            else:
                V[i, c] = (diffusion_distance_matrix[i, q] - diffusion_distance_matrix[i, p]) / \
                          (diffusion_distance_matrix[i, q] + diffusion_distance_matrix[i, p])

    # Compute the new distance matrix using the modified feature space
    modified_distance_matrix = pairwise_distances(V)

    return modified_distance_matrix


# Example usage:
# X = np.array(...)  # Your data matrix
# cannot_links = [(0, 1), (2, 3)]  # Example cannot-link constraints
# sigma = 1.0  # Example sigma value
# modified_distance_matrix = modify_distance_matrix_with_cannot_link(X, cannot_links, sigma)

m1 = pairwise_distances(X)
m2  = modify_distance_matrix_with_cannot_link(X, cannot_links)

fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns
    # Original distance matrix
axs[0].scatter(X[:, 0], X[:, 1], cmap='viridis', edgecolor='k', alpha=0.7)
axs[0].set_title('KMedoids')

plt.tight_layout()
plt.show()
