import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn_extra.cluster import KMedoids
import modifydistance as md
import myplots as mp
import data as data
import diffusion as diffusion
from sklearn.metrics import silhouette_score
import relative as relative
from itertools import combinations
from pydiffmap import diffusion_map as dm

# Function to compute the distance matrix
def compute_distance_matrix(X):
    return pairwise_distances(X)

# Generate dataset
n_samples = 200
n_clusters = 3

for i in range(15, 35):

    X, y = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=2, center_box=(-1, 1), random_state=40 + i)

    X = data.normalize_01(X)

    # Calculate cluster centers
    centers_indices = []
    for i in range(n_clusters):
        # Select all indices that belong to the current cluster
        indices_in_cluster = np.where(y == i)[0]
        points_in_cluster = X[indices_in_cluster]

        # Compute the pairwise distances between all points in the cluster
        distances = pairwise_distances(points_in_cluster)
        # Find the index of the medoid (point with the smallest sum of distances)
        medoid_index_local = np.argmin(distances.sum(axis=0))
        # Convert local cluster index to global index
        medoid_index_global = indices_in_cluster[medoid_index_local]
        centers_indices.append(medoid_index_global)

    # Convert centers_indices to a numpy array
    centers_indices = np.array(centers_indices)
    medoid_centers = X[centers_indices]

    distance_matrix = compute_distance_matrix(X)


    # KMedoids clustering
    kmedoids1 = KMedoids(n_clusters=n_clusters, metric="precomputed", random_state=42)
    kmedoids1.fit(distance_matrix)

    # Get the cluster labels
    clusters1 = kmedoids1.labels_
    silhouette1 = silhouette_score(X, clusters1)

    cannot_links = [(kmedoids1.medoid_indices_[i], kmedoids1.medoid_indices_[i+1]) for i in range(0, len(kmedoids1.medoid_indices_)-1)]
    distance_matrix_cannot = diffusion.modify_distance_matrix_with_cannot_link(X, cannot_links, distance_matrix)
    kmedoids2 = KMedoids(n_clusters=n_clusters, metric="precomputed", random_state=42)
    kmedoids2.fit(distance_matrix_cannot)
    #
    # # Get the cluster labels
    clusters2 = kmedoids2.labels_
    silhouette2 = silhouette_score(X, clusters2)

    # Assuming medoid_indices_ is a list of indices for the medoids

    relative_constraints = [combo for combo in combinations(kmedoids1.medoid_indices_, 3)]

    relative_constraints = [(0, 10, 20), (60, 80, 100)]

    distance_matrix_relati = diffusion.modify_distance_matrix_with_relative_constraints(X, relative_constraints, distance_matrix)
    kmedoids3 = KMedoids(n_clusters=n_clusters, metric="precomputed", random_state=42)
    kmedoids3.fit(distance_matrix_relati)

    # # #
    # # # # Get the cluster labels
    clusters3 = kmedoids3.labels_
    # # #
    #
    silhouette3 = silhouette_score(X, clusters3)
    # print(silhouette3)
    #

    # mp.plot_clusters(X, cannot_links, relative_constraints, kmedoids1, kmedoids2, kmedoids3, silhouette1, silhouette2, silhouette3)

    print(i, round(silhouette1, 2), round(silhouette2, 2), round(silhouette3, 2))
    # #
    #

    # print(clusters1)
    # print(clusters2)
