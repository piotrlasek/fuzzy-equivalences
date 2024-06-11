from itertools import combinations

import numpy as np
from sklearn.datasets import make_blobs
from sklearn_extra.cluster import KMedoids

import distances
import kmeans as kmeans
import mysilhouette as silhouette
import distances as dist
import data as data
import pandas as pd
from datetime import datetime
from functools import partial
from sklearn.metrics import silhouette_samples, silhouette_score, pairwise_distances
from scipy.spatial.distance import pdist, squareform

from features import diffusion

max_iters = 5

dataset_res = {}

#dataset_res["Dataset"] = data.datasets

sc = {}
sc["1"] = []
sc["2"] = []
sc["3"] = []

sc_list1 = []
sc_list2 = []
sc_list3 = []
dataset_names = []

def calc_silhouette(X, labels):
    sc = -1
    if len(np.unique(labels)) > 1:
        try:
            sc = silhouette_score(X, labels)

        except:
            sc = -2
    else:
        sc = -1

    return sc

datasets_in = []

for i in range(len(data.datasets)):
    n_samples = int(100 + (100 * i) / 2)

    n_samples = 100 * int(np.ceil((1 + i) / 2))
    n_clusters = int(np.ceil((3 + i) / 2))
    n_feat = 2 + i

    # X, y = make_blobs(n_samples=n_centers, centers=n_centers, n_features=n_features, center_box=(-1, 1), random_state=42)

    X, y = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=n_feat, center_box=(-4, 4), random_state=40 + i)

    X = data.normalize_01(X)

    name = "" + str(n_samples) + "_" + str(n_feat) + "_" + str(n_clusters)
    datasets_in.append((X, n_clusters, name))

for j in range(len(data.datasets)):
    datasets_in.append(data.get_dataset(j))


for i in range(len(data.datasets)):
    # X, k, dataset_name = data.get_dataset(i)
    X, k, dataset_name = datasets_in[i]
    #X = data.normalize_01(X)

    print("---------- " + dataset_name + " (" + str(k) + ") ---------- ")

    distance_matrix = pairwise_distances(X)

    # KMedoids clustering
    kmedoids1 = KMedoids(n_clusters=k, metric="precomputed", random_state=42)
    kmedoids1.fit(distance_matrix)
    labels = kmedoids1.labels_

    centers_indices = []
    for i in range(k):
        # Select all indices that belong to the current cluster
        indices_in_cluster = np.where(labels == i)[0]
        points_in_cluster = X[indices_in_cluster]

        # Compute the pairwise distances between all points in the cluster
        distances = pairwise_distances(points_in_cluster)
        # Find the index of the medoid (point with the smallest sum of distances)
        medoid_index_local = np.argmin(distances.sum(axis=0))
        # Convert local cluster index to global index
        medoid_index_global = indices_in_cluster[medoid_index_local]
        centers_indices.append(medoid_index_global)

    cannot_links = [(kmedoids1.medoid_indices_[i], kmedoids1.medoid_indices_[i + 1]) for i in
                    range(0, len(kmedoids1.medoid_indices_) - 1)]

    relative_constraints = [combo for combo in combinations(kmedoids1.medoid_indices_, 3)]

    # CANNOT
    distance_matrix_cannot = diffusion.modify_distance_matrix_with_cannot_link(X, cannot_links, distance_matrix)
    kmedoids2 = KMedoids(n_clusters=k, metric="precomputed", random_state=42)
    kmedoids2.fit(distance_matrix_cannot)
    clusters2 = kmedoids2.labels_

    # RELATIVE
    distance_matrix_relati = diffusion.modify_distance_matrix_with_relative_constraints(X, relative_constraints, distance_matrix)
    kmedoids3 = KMedoids(n_clusters=k, metric="precomputed", random_state=42)
    kmedoids3.fit(distance_matrix_relati)
    clusters3 = kmedoids3.labels_

    sc = calc_silhouette(X, labels)
    sc2 = calc_silhouette(X, clusters2)
    sc3 = calc_silhouette(X, clusters3)

    dist_res = []

    print("\t{0}\t{1}\t{2}".format( round(sc, 3), round(sc2, 3), round(sc3, 3)))

    sc_list1.append(sc)
    sc_list2.append(sc2)
    sc_list3.append(sc3)
    dataset_names.append(dataset_name)

dataset_res["Dataset"] = dataset_names
dataset_res["kMedoids"] = sc_list1
dataset_res["Cannot"] = sc_list2
dataset_res["Relative"] = sc_list3

    #dataset_res[dataset_name] = dist_res

df = pd.DataFrame(dataset_res)
print(df)
#
currentDateAndTime = datetime.now()
currentTime = currentDateAndTime.strftime("%H%M%S")
#
res_file = "results/rel-" + currentTime + ".xlsx"
#
print("Writing to {0}".format(res_file))
#
df.to_excel(res_file)