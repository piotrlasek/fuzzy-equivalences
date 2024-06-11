import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances

def get_relative_constraints(X, y, n_clusters):

    # Calculate the representative points for each cluster (medoids) using global indices
    centers_indices = []
    for i in range(n_clusters):
        indices_in_cluster = np.where(y == i)[0]
        points_in_cluster = X[indices_in_cluster]
        # Compute the pairwise distances between all points in the cluster
        distances = pairwise_distances(points_in_cluster)
        # Find the index of the medoid (point with the smallest sum of distances)
        medoid_index_local = np.argmin(distances.sum(axis=0))
        # Convert local cluster index to global index
        medoid_index_global = indices_in_cluster[medoid_index_local]
        centers_indices.append(medoid_index_global)

    # Generate relative constraints based on medoid centers
    relative_constraints = []
    n_medoids = len(centers_indices)
    for i in range(n_medoids):
        for j in range(i + 1, n_medoids):
            for k in range(j + 1, n_medoids):
                # Calculate distances between the medoid pairs
                dist_ij = np.linalg.norm(X[centers_indices[i]] - X[centers_indices[j]])
                dist_jk = np.linalg.norm(X[centers_indices[j]] - X[centers_indices[k]])
                dist_ki = np.linalg.norm(X[centers_indices[k]] - X[centers_indices[i]])

                # Determine which two medoids are closest and create a relative constraint
                if dist_ij <= dist_jk and dist_ij <= dist_ki:
                    # i and j are closer compared to k
                    relative_constraints.append((centers_indices[i], centers_indices[j], centers_indices[k]))
                elif dist_jk <= dist_ij and dist_jk <= dist_ki:
                    # j and k are closer compared to i
                    relative_constraints.append((centers_indices[j], centers_indices[k], centers_indices[i]))
                else:
                    # k and i are closer compared to j
                    relative_constraints.append((centers_indices[k], centers_indices[i], centers_indices[j]))

    # Print relative constraints
    # print("Relative constraints based on medoid centers:")
    # for constraint in relative_constraints:
    #     print(f"Relative constraint: {constraint}")

    return relative_constraints
