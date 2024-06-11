from functools import partial

from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import KMeans
import numpy as np
import data
import distances
import kmeans as kmeans


def d_E_3(x, y, agg):
    result = np.where(x + y == 0, 1, (2 * np.minimum(x, y)) / (x + y))
    result = 1 - agg(result)
    return result


X, k, dataset_name = data.get_dataset(0)
X = data.normalize_01(X)

kmeansx = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X)

labels1 = kmeansx.labels_
print(labels1)

pairwise_dists = pdist(X, 'euclidean')
dist_matrix2 = squareform(pairwise_dists)

clusters2, labels2 = kmeans.my_kmeans_dist(dist_matrix2, k)

distance_with_agg = partial(distances.E, agg=distances.np.mean)
dist_matrix3 = squareform(pdist(X, metric=lambda u, v: distance_with_agg(u, v)))
clusters3, labels3 = kmeans.my_kmeans_dist(dist_matrix3, k)

clusters4, labels4 = kmeans.my_kmedoids(X, dist_matrix3, k, 5)

print(labels3)

