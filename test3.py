from functools import partial

import numpy as np
import warnings

from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist, squareform

import distances as distances
from data import normalize_01

warnings.simplefilter("ignore", category=RuntimeWarning)

x = np.array([1, 1, 1])
y = np.array([0, 0, 0])

x = np.array([0.335,0.88,1.0,0.0,1.0,0.5,0.75,1.0,0.265957])
y = np.array([0.335,0.00,1.0,0.0,0.0,0.5,1.00,0.0,0.468085])

def d_E_3(x, y, agg):
    result = np.where(x + y == 0, 1, (2 * np.minimum(x, y)) / (x + y))
    result = 1 - agg(result)
    return result


X = np.row_stack((x, y))
#X = normalize_01(X)

distance_with_agg = partial(d_E_3, agg=distances.A2_01)
dist_matrix1 = squareform(pdist(X, metric=lambda u, v: distance_with_agg(u, v)))

distance_with_agg = partial(distances.E, agg=distances.np.mean)
dist_matrix2 = squareform(pdist(X, metric=lambda u, v: distance_with_agg(u, v)))



print(dist_matrix1)
# for distance in distances.distances:
#     dist, n = distance
#     for aggregation in distances.aggregations:
#         agg_name,  agg = aggregation
#         d = dist(x, y, agg)
#         print("{0}\t{1}\t{2}".format(n, agg_name, round(d, 3)))