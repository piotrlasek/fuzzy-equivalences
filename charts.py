import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

import distances
import kmeans as kmeans
import mysilhouette as silhouette
from scipy.spatial.distance import pdist, squareform
import distances as dist
import plot as plot
from functools import partial
from sklearn.metrics import silhouette_samples, silhouette_score

k = 4
n = 400
max_iters = 10

# Example dataset
X, y = make_blobs(n_samples=n, centers=k, n_features=2, center_box=(-6, 6), random_state=96)

X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_normalized = (X - X_min) / (X_max - X_min)
X = X_normalized

num_columns = 3
num_rows = int(math.floor(len(dist.distances) / num_columns))

fig, asx = plot.make_plot(num_rows, num_columns)

for i, distance_name_tuple in zip(range(len(dist.distances)), dist.distances):
    distance, name = distance_name_tuple
    if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
    #if i in [3]:
        print(str(i) + ": " + distance.__name__)

        distance_with_agg = partial(distance, agg=np.mean)

        dist_matrix = squareform(pdist(X, metric=lambda u, v: distance_with_agg(u, v)))
        labels, centroids = kmeans.my_kmedoids(X, dist_matrix, k, 5)

        #labels, centroids = kmeans.my_kmeans(X, k, distance_with_agg, max_iters=max_iters)
        sc = np.nan

        #sc1 = silhouette.silhouette_score_custom_distance(X, labels, distance)

        dist_matrix = squareform(pdist(X, metric=lambda u, v: distance_with_agg(u, v)))
        try:
            sc = silhouette_score(dist_matrix, labels, metric='precomputed')
        except:
            sc = np.nan

        coords = plot.fig_coords(i, num_columns)
        plot.plot_sub_scatter(fig, asx, coords[0], coords[1], X, labels, sc, centroids, distance, name)

plot.show()
